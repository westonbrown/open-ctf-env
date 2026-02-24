"""SkyRL-compatible vLLM server for open-ctf-env.

Wraps standard vLLM 0.14+ with the custom HTTP endpoints that SkyRL's
RemoteInferenceClient expects for control plane operations:

  GET  /get_server_info                  - parallelism info (world_size)
  POST /init_weight_update_communicator  - NCCL weight sync init
  POST /init_weight_transfer             - alias (actor naming convention)
  POST /update_weights                   - broadcast weights from trainer
  POST /finalize_weight_update           - post-processing hook (no-op)
  POST /destroy_weights_update_group     - teardown weight sync
  POST /sleep                            - offload model to CPU
  POST /wake_up                          - reload model to GPU
  POST /reset_prefix_cache               - clear KV cache

Standard vLLM OpenAI-compatible endpoints (/v1/chat/completions, etc.)
are served as usual via vLLM's build_app().

Why this exists:
  SkyRL's RemoteInferenceClient calls /get_server_info to discover
  world_size, then uses /init_weight_transfer + /update_weights to
  sync policy weights to the inference engine via NCCL. Standard vLLM
  doesn't have these endpoints — SkyRL's own vllm_server.py targets
  vLLM 0.13 and is missing /get_server_info (only in vllm_server_actor).

  This module unifies both, compatible with vLLM 0.14+.

Usage:
    python -m open_ctf.training.skyrl_vllm_server \\
        --model <PATH> \\
        --host 0.0.0.0 --port 8001 \\
        --dtype bfloat16 \\
        --gpu-memory-utilization 0.2 \\
        --worker-extension-cls skyrl_train.inference_servers.vllm_worker.WorkerWrap \\
        [... other vllm args ...]
"""

import logging
import os
import pickle
import signal
import socket
import sys

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("open_ctf.skyrl_vllm_server")

# ---------------------------------------------------------------------------
# vLLM imports — handle 0.13 → 0.16+ path differences
# ---------------------------------------------------------------------------


def _import_vllm():
    """Import vLLM components with version-adaptive paths.

    Returns a namespace dict with the components we need.
    """
    ns = {}

    # AsyncEngineArgs
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        ns["AsyncEngineArgs"] = AsyncEngineArgs
    except ImportError:
        raise ImportError("Cannot import AsyncEngineArgs — unsupported vLLM version")

    # AsyncLLMEngine
    try:
        from vllm import AsyncLLMEngine
        ns["AsyncLLMEngine"] = AsyncLLMEngine
    except ImportError:
        try:
            from vllm.engine.async_llm_engine import AsyncLLMEngine
            ns["AsyncLLMEngine"] = AsyncLLMEngine
        except ImportError:
            raise ImportError("Cannot import AsyncLLMEngine — unsupported vLLM version")

    # UsageContext
    try:
        from vllm.usage.usage_lib import UsageContext
        ns["UsageContext"] = UsageContext
    except ImportError:
        ns["UsageContext"] = None

    # FastAPI app builder (vLLM's OpenAI-compatible routes)
    try:
        from vllm.entrypoints.openai.api_server import build_app
        ns["build_app"] = build_app
    except ImportError:
        ns["build_app"] = None

    # App state initializer (loads served model info etc.)
    try:
        from vllm.entrypoints.openai.api_server import init_app_state
        ns["init_app_state"] = init_app_state
    except ImportError:
        ns["init_app_state"] = None

    # Server socket helper
    try:
        from vllm.entrypoints.openai.api_server import create_server_socket
        ns["create_server_socket"] = create_server_socket
    except ImportError:
        ns["create_server_socket"] = None

    # CLI arg parser
    try:
        from vllm.utils.argparse_utils import FlexibleArgumentParser
        ns["FlexibleArgumentParser"] = FlexibleArgumentParser
    except ImportError:
        from argparse import ArgumentParser
        ns["FlexibleArgumentParser"] = ArgumentParser

    try:
        from vllm.entrypoints.openai.cli_args import (
            make_arg_parser,
            validate_parsed_serve_args,
        )
        ns["make_arg_parser"] = make_arg_parser
        ns["validate_parsed_serve_args"] = validate_parsed_serve_args
    except ImportError:
        ns["make_arg_parser"] = None
        ns["validate_parsed_serve_args"] = None

    # System utils
    try:
        from vllm.utils.system_utils import set_ulimit
        ns["set_ulimit"] = set_ulimit
    except ImportError:
        ns["set_ulimit"] = lambda: None

    # Env vars
    try:
        import vllm.envs as vllm_envs
        ns["VLLM_HTTP_TIMEOUT_KEEP_ALIVE"] = getattr(
            vllm_envs, "VLLM_HTTP_TIMEOUT_KEEP_ALIVE", 5
        )
    except ImportError:
        ns["VLLM_HTTP_TIMEOUT_KEEP_ALIVE"] = 5

    return ns


def _create_server_socket(addr):
    """Fallback socket creation if vLLM's helper is unavailable."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(addr)
    sock.listen(128)
    sock.setblocking(False)
    return sock


# ---------------------------------------------------------------------------
# SkyRL-compatible server
# ---------------------------------------------------------------------------


class SkyRLVllmServer:
    """vLLM server with SkyRL control-plane endpoints."""

    def __init__(self, args, vllm_ns):
        self.server_args = args
        self._ns = vllm_ns
        self._host = getattr(args, "host", "0.0.0.0") or "0.0.0.0"
        self._port = getattr(args, "port", 8001)

    async def run_server(self):
        import uvicorn

        ns = self._ns
        ns["set_ulimit"]()

        def signal_handler(*_):
            raise KeyboardInterrupt("terminated")

        signal.signal(signal.SIGTERM, signal_handler)

        # Create vLLM engine
        logger.info("Creating AsyncLLMEngine...")
        engine_args = ns["AsyncEngineArgs"].from_cli_args(self.server_args)

        kwargs = {"engine_args": engine_args}
        if ns["UsageContext"] is not None:
            kwargs["usage_context"] = ns["UsageContext"].OPENAI_API_SERVER
        engine = ns["AsyncLLMEngine"].from_engine_args(**kwargs)
        logger.info("Engine created.")

        # Use minimal FastAPI app to avoid vLLM 0.16's lifespan handler
        # which requires full state initialization (openai_serving_completion,
        # engine_client, log_stats etc.) that conflicts with our custom flow.
        # We add our own /v1/completions and /v1/chat/completions endpoints
        # that delegate directly to the engine.
        app = self._minimal_app()

        # Add OpenAI-compatible /v1/completions endpoint
        self._add_completions_endpoint(app, engine)

        # Add SkyRL control-plane + data-plane endpoints
        self._add_skyrl_endpoints(app, engine)

        # Create socket and serve
        create_sock = ns.get("create_server_socket") or _create_server_socket
        sock = create_sock((self._host, self._port))
        logger.info("Listening on %s:%d", self._host, self._port)

        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._port,
            log_level=getattr(self.server_args, "uvicorn_log_level", "info"),
            timeout_keep_alive=ns["VLLM_HTTP_TIMEOUT_KEEP_ALIVE"],
        )
        server = uvicorn.Server(config)
        await server.serve(sockets=[sock])

    def _minimal_app(self):
        """Create a minimal FastAPI app with /health when vLLM build_app fails."""
        app = FastAPI(title="SkyRL-compatible vLLM server")

        @app.get("/health")
        async def _health():
            return JSONResponse({"status": "ok"})

        logger.info("Created minimal FastAPI app (no built-in OpenAI routes).")
        return app

    def _add_completions_endpoint(self, app, engine):
        """Add /v1/completions and /v1/models endpoints using the engine directly."""
        model_name = getattr(self.server_args, "model", "unknown")

        @app.get("/v1/models")
        async def _list_models():
            return {
                "object": "list",
                "data": [
                    {
                        "id": model_name,
                        "object": "model",
                        "owned_by": "vllm",
                        "root": model_name,
                    }
                ],
            }

        @app.post("/v1/completions")
        async def _completions(request: Request):
            """OpenAI-compatible completions endpoint (prompt token IDs).

            SkyRL sends prompt as token IDs (single or batched), not text.
            Handles: str, List[int], List[List[int]], List[str].
            """
            import asyncio
            import uuid

            from vllm import SamplingParams

            data = await request.json()
            prompt = data.get("prompt", [])

            # SkyRL uses max_generate_length; OpenAI uses max_tokens
            max_tokens = data.get("max_tokens") or data.get("max_generate_length", 256)
            temperature = data.get("temperature", 1.0)
            top_p = data.get("top_p", 1.0)
            stop = data.get("stop")
            repetition_penalty = data.get("repetition_penalty")
            top_k = data.get("top_k")
            min_p = data.get("min_p")

            # Parse prompt into a list of gen_inputs
            # Supports: str, List[int], List[List[int]], List[str]
            gen_inputs = []
            if isinstance(prompt, str):
                gen_inputs = [{"prompt": prompt}]
            elif isinstance(prompt, list):
                if len(prompt) == 0:
                    return JSONResponse({"error": "Empty prompt"}, status_code=400)
                if isinstance(prompt[0], int):
                    # Single prompt as List[int]
                    gen_inputs = [{"prompt_token_ids": prompt}]
                elif isinstance(prompt[0], list):
                    # Batched prompts as List[List[int]]
                    gen_inputs = [{"prompt_token_ids": p} for p in prompt]
                elif isinstance(prompt[0], str):
                    # Batched text prompts
                    gen_inputs = [{"prompt": p} for p in prompt]
                else:
                    gen_inputs = [{"prompt": str(prompt)}]
            else:
                gen_inputs = [{"prompt": str(prompt)}]

            sp_kwargs = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            if stop:
                sp_kwargs["stop"] = stop
            if repetition_penalty is not None and repetition_penalty != 1.0:
                sp_kwargs["repetition_penalty"] = repetition_penalty
            if top_k is not None and top_k > 0:
                sp_kwargs["top_k"] = top_k
            if min_p is not None and min_p > 0:
                sp_kwargs["min_p"] = min_p
            params = SamplingParams(**sp_kwargs)

            async def _generate_one(gen_input, idx):
                request_id = f"comp-{uuid.uuid4().hex[:12]}"
                final_output = None
                async for output in engine.generate(
                    prompt=gen_input,
                    sampling_params=params,
                    request_id=request_id,
                ):
                    final_output = output
                return idx, final_output

            # Generate all prompts (concurrently for batches)
            tasks = [_generate_one(gi, i) for i, gi in enumerate(gen_inputs)]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            choices = []
            tokenizer = engine.get_tokenizer()
            choice_idx = 0
            for result in results:
                if isinstance(result, Exception):
                    logger.error("Generation error: %s", result)
                    choices.append({
                        "index": choice_idx,
                        "text": "",
                        "finish_reason": "error",
                    })
                    choice_idx += 1
                    continue

                _order, final_output = result
                if final_output is None:
                    choices.append({
                        "index": choice_idx,
                        "text": "",
                        "finish_reason": "error",
                    })
                    choice_idx += 1
                    continue

                for completion in final_output.outputs:
                    text = ""
                    try:
                        text = tokenizer.decode(
                            list(completion.token_ids),
                            skip_special_tokens=True,
                        )
                    except Exception:
                        text = str(list(completion.token_ids))
                    choices.append({
                        "index": choice_idx,
                        "text": text,
                        "finish_reason": completion.finish_reason or "stop",
                    })
                    choice_idx += 1

            request_id = f"comp-{uuid.uuid4().hex[:12]}"
            return {
                "id": request_id,
                "object": "text_completion",
                "choices": choices,
                "model": model_name,
            }

        logger.info("Added /v1/completions and /v1/models endpoints.")

    def _add_skyrl_endpoints(self, app, engine):
        """Add SkyRL control-plane endpoints to the FastAPI app."""
        host = self._host if self._host != "0.0.0.0" else "127.0.0.1"
        port = self._port

        # Compute world_size = TP * PP
        tp = getattr(self.server_args, "tensor_parallel_size", 1) or 1
        pp = getattr(self.server_args, "pipeline_parallel_size", 1) or 1
        world_size = tp * pp

        logger.info(
            "Adding SkyRL endpoints (world_size=%d, TP=%d, PP=%d)", world_size, tp, pp
        )

        # -- Server info (required by RemoteInferenceClient.get_world_size) --

        @app.get("/get_server_info")
        async def _get_server_info():
            return {
                "ip": host,
                "port": port,
                "url": f"http://{host}:{port}",
                "server_idx": 0,
                "world_size": world_size,
            }

        # -- Weight sync endpoints --

        @app.post("/init_weight_update_communicator")
        async def _init_weight_update_communicator(request: Request):
            # Try collective_rpc if WorkerWrap is loaded; otherwise no-op.
            # LoRA file-based sync doesn't need NCCL communicator.
            try:
                from skyrl_train.weight_sync import BroadcastInitInfo

                data = await request.json()
                init_info = BroadcastInitInfo(**data)
                pickled = pickle.dumps(init_info)
                await engine.collective_rpc(
                    "init_weight_update_communicator",
                    args=(pickled,),
                )
                return {"status": "ok"}
            except Exception as e:
                logger.warning(
                    "init_weight_update_communicator skipped (%s). "
                    "LoRA sync uses file-based path instead.",
                    e,
                )
                return {"status": "ok", "note": "skipped_nccl"}

        # Alias: vllm_server_actor uses /init_weight_transfer
        @app.post("/init_weight_transfer")
        async def _init_weight_transfer(request: Request):
            return await _init_weight_update_communicator(request)

        @app.post("/update_weights")
        async def _update_weights(request: Request):
            try:
                from skyrl_train.weight_sync import BroadcastWeightUpdateRequest

                data = await request.json()
                weight_request = BroadcastWeightUpdateRequest(**data)
                pickled = pickle.dumps(weight_request)
                await engine.collective_rpc(
                    "load_weights",
                    args=(pickled,),
                )
                return {"status": "ok"}
            except Exception as e:
                logger.warning("update_weights skipped (%s).", e)
                return {"status": "ok", "note": "skipped"}

        @app.post("/finalize_weight_update")
        async def _finalize_weight_update(request: Request):
            return {"status": "ok"}

        @app.post("/destroy_weights_update_group")
        async def _destroy_weights_update_group(request: Request):
            try:
                await engine.collective_rpc(
                    "teardown_weight_receiver",
                    args=(),
                )
            except Exception as e:
                logger.warning("destroy_weights_update_group skipped (%s).", e)
            return {"status": "ok"}

        # -- Sleep/wake endpoints (memory management) --

        @app.post("/sleep")
        async def _sleep(request: Request):
            data = await request.json()
            level = data.get("level")
            # Reset prefix cache before sleep (vLLM bug workaround)
            await engine.reset_prefix_cache()
            await engine.sleep(level)
            return {"status": "ok"}

        @app.post("/wake_up")
        async def _wake_up(request: Request):
            data = await request.json()
            tags = data.get("tags")
            await engine.wake_up(tags)
            return {"status": "ok"}

        @app.post("/reset_prefix_cache")
        async def _reset_prefix_cache(request: Request):
            await engine.reset_prefix_cache()
            return {"status": "ok"}

        # -- LoRA weight sync endpoint --

        @app.post("/load_lora")
        async def _load_lora(request: Request):
            """Load LoRA adapter from disk path (file-based weight sync).

            SkyRL's FSDP worker saves LoRA to a shared filesystem path,
            then tells the inference engine to reload.  For remote engines,
            this arrives as an HTTP POST rather than the in-process call
            used by local VllmEngine._load_lora_from_disk().
            """
            data = await request.json()
            lora_path = data.get("lora_path", "")
            logger.info("load_lora: path=%s", lora_path)

            if not lora_path:
                return JSONResponse(
                    {"status": "error", "error": "lora_path required"},
                    status_code=400,
                )

            try:
                # Try vLLM's built-in LoRA loading if available
                if hasattr(engine, "add_lora"):
                    from vllm.lora.request import LoRARequest

                    lora_req = LoRARequest(
                        lora_name="skyrl_lora",
                        lora_int_id=1,
                        lora_path=lora_path,
                    )
                    await engine.add_lora(lora_req)
                    logger.info("LoRA loaded via engine.add_lora()")
                    return {"status": "ok", "method": "add_lora"}

                # Fallback: use collective_rpc to call the worker extension
                try:
                    await engine.collective_rpc(
                        "load_lora_from_disk",
                        args=(lora_path,),
                    )
                    logger.info("LoRA loaded via collective_rpc")
                    return {"status": "ok", "method": "collective_rpc"}
                except Exception as rpc_err:
                    logger.warning(
                        "collective_rpc load_lora_from_disk failed: %s", rpc_err
                    )

                # Last resort: log and skip (training still works,
                # inference model just uses stale weights until restart)
                logger.warning(
                    "LoRA sync skipped — no compatible loader. "
                    "Inference model will use base weights until restart."
                )
                return {
                    "status": "skipped",
                    "reason": "no compatible LoRA loader",
                }

            except Exception as e:
                logger.error("load_lora failed: %s", e, exc_info=True)
                return JSONResponse(
                    {"status": "error", "error": str(e)},
                    status_code=500,
                )

        # -- Data-plane generation endpoint --
        # SkyRL's RemoteInferenceClient sends token-level generation
        # requests to {proxy_url}/inference/v1/generate. Standard vLLM
        # doesn't have this — it's a SkyRL-specific API.

        @app.post("/inference/v1/generate")
        async def _generate(request: Request):
            """SkyRL data-plane: generate tokens from prompt token IDs.

            Request:  {"sampling_params": {...}, "model": "name", "token_ids": [int]}
            Response: {"choices": [{"token_ids": [int], "finish_reason": str, "logprobs": {...}}]}
            """
            import uuid

            from vllm import SamplingParams

            data = await request.json()
            raw_ids = data.get("token_ids", [])

            # SkyRL may send token_ids as:
            #   1) A flat list of ints: [1, 2, 3]
            #   2) A BatchEncoding dict: {"input_ids": [1, 2, 3], ...}
            #   3) A flat list of strings: ["1", "2", "3"]
            # Use try/except instead of isinstance (which can fail
            # if the JSON parser returns a non-standard mapping type).
            try:
                raw_ids = raw_ids["input_ids"]
            except (KeyError, TypeError, IndexError):
                pass
            # Ensure all token IDs are ints (vLLM V1 validates this).
            try:
                token_ids = [int(t) for t in raw_ids]
            except (ValueError, TypeError) as cast_err:
                logger.error(
                    "token_ids cast failed: type(raw_ids)=%s, raw_ids=%r, err=%s",
                    type(raw_ids).__name__,
                    raw_ids if len(str(raw_ids)) < 500 else str(raw_ids)[:500],
                    cast_err,
                )
                return JSONResponse(
                    {"error": f"Invalid token_ids: {cast_err}"}, status_code=400
                )

            sp_dict = data.get("sampling_params", {})
            logger.info(
                "generate: %d token_ids, model=%s, sp_keys=%s",
                len(token_ids),
                data.get("model", "N/A"),
                list(sp_dict.keys()),
            )

            # Map SkyRL params to vLLM SamplingParams
            vllm_kw = {}

            # max tokens (SkyRL uses max_tokens, max_completion_tokens, or max_generate_length)
            for key in ("max_tokens", "max_completion_tokens", "max_generate_length"):
                if key in sp_dict and sp_dict[key] is not None:
                    vllm_kw["max_tokens"] = int(sp_dict[key])
                    break

            # Direct numeric/bool mappings
            for key in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty"):
                if key in sp_dict and sp_dict[key] is not None:
                    vllm_kw[key] = sp_dict[key]

            # min_tokens
            if sp_dict.get("min_tokens") is not None:
                vllm_kw["min_tokens"] = int(sp_dict["min_tokens"])

            # skip_special_tokens (bool)
            if sp_dict.get("skip_special_tokens") is not None:
                vllm_kw["skip_special_tokens"] = bool(sp_dict["skip_special_tokens"])

            # include_stop_str_in_output (bool)
            if sp_dict.get("include_stop_str_in_output") is not None:
                vllm_kw["include_stop_str_in_output"] = bool(
                    sp_dict["include_stop_str_in_output"]
                )

            # Logprobs
            if sp_dict.get("logprobs") is not None:
                vllm_kw["logprobs"] = int(sp_dict["logprobs"])

            # Stop sequences
            stop = sp_dict.get("stop")
            if stop is not None:
                vllm_kw["stop"] = stop

            logger.info("generate: vllm_kw=%s", vllm_kw)

            try:
                params = SamplingParams(**vllm_kw)
            except Exception as exc:
                logger.error("Bad SamplingParams %s: %s", vllm_kw, exc)
                return JSONResponse({"error": str(exc)}, status_code=400)

            request_id = f"skyrl-{uuid.uuid4().hex[:12]}"

            # Generate via the async engine
            final_output = None
            try:
                async for output in engine.generate(
                    prompt={"prompt_token_ids": token_ids},
                    sampling_params=params,
                    request_id=request_id,
                ):
                    final_output = output
            except Exception as exc:
                import traceback

                tb = traceback.format_exc()
                logger.error(
                    "Engine generate failed: type=%s repr=%r\n%s",
                    type(exc).__name__,
                    exc,
                    tb,
                )
                return JSONResponse(
                    {"error": f"{type(exc).__name__}: {exc!r}", "traceback": tb},
                    status_code=500,
                )

            if final_output is None:
                return JSONResponse({"error": "No output"}, status_code=500)

            choices = []
            for completion in final_output.outputs:
                choice = {
                    "token_ids": list(completion.token_ids),
                    "finish_reason": completion.finish_reason or "stop",
                }
                if completion.logprobs:
                    try:
                        choice["logprobs"] = {
                            "content": [
                                {"logprob": lp.logprob}
                                for lp in completion.logprobs
                                if lp is not None
                            ],
                        }
                    except (AttributeError, TypeError):
                        pass
                choices.append(choice)

            return JSONResponse({"choices": choices})

        logger.info("SkyRL endpoints registered (control-plane + data-plane).")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    ns = _import_vllm()

    if ns["make_arg_parser"] is not None:
        parser = ns["FlexibleArgumentParser"](
            description="SkyRL-compatible vLLM server (open-ctf-env)"
        )
        parser = ns["make_arg_parser"](parser)
    else:
        # Fallback: minimal CLI for when vLLM CLI helpers aren't available
        parser = ns["FlexibleArgumentParser"](
            description="SkyRL-compatible vLLM server (open-ctf-env)"
        )
        parser.add_argument("--model", required=True)
        parser.add_argument("--host", default="0.0.0.0")
        parser.add_argument("--port", type=int, default=8001)
        parser.add_argument("--dtype", default="bfloat16")
        parser.add_argument("--max-model-len", type=int, default=4096)
        parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
        parser.add_argument("--max-num-seqs", type=int, default=32)
        parser.add_argument("--enforce-eager", action="store_true")
        parser.add_argument("--trust-remote-code", action="store_true")
        parser.add_argument("--enable-auto-tool-choice", action="store_true")
        parser.add_argument("--tool-call-parser", type=str)
        parser.add_argument("--worker-extension-cls", type=str)
        parser.add_argument("--tensor-parallel-size", type=int, default=1)
        parser.add_argument("--pipeline-parallel-size", type=int, default=1)
        parser.add_argument("--uvicorn-log-level", default="info")
        parser.add_argument("--enable-sleep-mode", action="store_true")
        parser.add_argument("--distributed-executor-backend", default="mp")

    args = parser.parse_args()

    if ns["validate_parsed_serve_args"] is not None:
        try:
            ns["validate_parsed_serve_args"](args)
        except Exception as e:
            logger.warning("validate_parsed_serve_args failed: %s", e)

    server = SkyRLVllmServer(args, ns)

    try:
        import uvloop

        uvloop.run(server.run_server())
    except ImportError:
        import asyncio

        asyncio.run(server.run_server())


if __name__ == "__main__":
    main()
