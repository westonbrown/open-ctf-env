"""SkyRL-based Group Relative Policy Optimization (GRPO) stage.

Thin orchestrator that:
  1. Converts our GRPO JSONL to SkyRL dataset format (messages -> prompt + extras)
  2. Registers OpenCTFTextEnv with SkyRL-Gym
  3. Launches SkyRL training via BasePPOExp

Replaces the custom grpo.py (1305 lines, 6 monkey-patches) with ~200 lines
by delegating to SkyRL for:
  - Fully async Ray-based trainer (eliminates NCCL segfault patch)
  - vLLM in separate process (eliminates dtype/weight sync patches)
  - Module-grouped weight sync (eliminates 3 HF->vLLM translation patches)
  - TIS off-policy correction + dynamic DAPO sampling
  - Process isolation per env (eliminates thread-local episode management)

Uses OpenCTFTextEnv (SkyRL-Gym BaseTextEnv subclass) which executes
tools directly via ToolExecutor (no HTTP server needed).

Default test model: Nanbeige4.1-3B (dense LlamaForCausalLM).
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs" / "skyrl"


def _flash_attn_available() -> bool:
    """Return True if flash-attn exports the symbols Transformers expects."""
    try:
        import flash_attn  # type: ignore
    except Exception:
        return False
    return all(
        hasattr(flash_attn, attr)
        for attr in ("flash_attn_func", "flash_attn_varlen_func")
    )


def _should_force_legacy_inference(model_path: str) -> bool:
    """Return True when SkyRL new inference should be disabled for model config.

    SkyRL's new inference server path currently initializes vLLM renderers in a
    way that can mis-handle some HuggingFace text-wrapper configs (for example
    Qwen3_5TextConfig). When this happens, vLLM raises a config type mismatch in
    multimodal processor setup before training starts.
    """
    try:
        from transformers import AutoConfig
        hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        logger.warning(
            "Could not inspect model config for inference backend selection: %s",
            exc,
        )
        return False

    cfg_cls_name = hf_cfg.__class__.__name__
    if cfg_cls_name.endswith("TextConfig"):
        logger.warning(
            "Model config class %s detected at %s. Forcing SkyRL legacy "
            "inference path (new inference is incompatible with text-wrapper configs).",
            cfg_cls_name,
            model_path,
        )
        return True
    return False


def _resolve_vllm_ready_model_path(model_path: str) -> str:
    """Resolve a vLLM-compatible model handoff path for GRPO.

    Qwen3.5 merged checkpoints used for HF training can expose a text-wrapper
    config (for example ``model_type=qwen3_5_text``). SkyRL+vLLM runtime paths
    expect the vLLM-ready variant (for example sibling ``*_vllm`` directory).
    """
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        logger.warning("Could not inspect model config at %s: %s", model_path, exc)
        return model_path

    cfg_cls_name = cfg.__class__.__name__
    cfg_model_type = str(getattr(cfg, "model_type", ""))
    looks_text_wrapper = cfg_cls_name.endswith("TextConfig") or cfg_model_type.endswith("_text")
    if not looks_text_wrapper:
        return model_path

    candidate = f"{model_path.rstrip('/')}_vllm"
    if os.path.isdir(candidate):
        try:
            from transformers import AutoConfig
            cand_cfg = AutoConfig.from_pretrained(candidate, trust_remote_code=True)
            cand_cls_name = cand_cfg.__class__.__name__
            cand_model_type = str(getattr(cand_cfg, "model_type", ""))
            cand_is_text_wrapper = (
                cand_cls_name.endswith("TextConfig") or cand_model_type.endswith("_text")
            )
            if not cand_is_text_wrapper:
                logger.warning(
                    "Model path %s uses text-wrapper config (%s/%s). "
                    "Auto-switching GRPO runtime model to sibling vLLM-ready path: %s",
                    model_path,
                    cfg_model_type or "<unknown_model_type>",
                    cfg_cls_name,
                    candidate,
                )
                return candidate
        except Exception as exc:
            logger.warning("Failed to validate sibling vLLM model path %s: %s", candidate, exc)

    logger.warning(
        "Model path %s appears to use text-wrapper config (%s/%s) and no validated "
        "sibling '*_vllm' path was found. GRPO runtime may fail in vLLM.",
        model_path,
        cfg_model_type or "<unknown_model_type>",
        cfg_cls_name,
    )
    return model_path


def _parse_lora_rank(lora_cfg: Dict[str, Any]) -> int:
    """Parse LoRA rank from config with a defensive fallback."""
    raw_rank = lora_cfg.get("r", 64)
    try:
        return int(raw_rank)
    except (TypeError, ValueError):
        logger.warning("Invalid lora.r=%r; defaulting to rank 64.", raw_rank)
        return 64


def _normalize_module_filter(raw_modules: Any, *, default: Optional[str]) -> Any:
    """Normalize LoRA target/exclude module filters for SkyRL + PEFT.

    Important: SkyRL's FSDP worker serializes PEFT target_modules via
    ``list(peft_config["target_modules"])`` before LoRA sync. If a plain
    string is used (for example "q_proj,k_proj"), that becomes a character list
    and breaks vLLM adapter loading. We therefore pass structured lists for
    explicit module selections, and reserve a string only for the special
    ``all-linear`` selector.
    """
    if raw_modules is None:
        return default

    if isinstance(raw_modules, str):
        value = raw_modules.strip()
        if not value:
            return default
        if value == "all-linear":
            return value
        if "," in value:
            modules = [part.strip() for part in value.split(",") if part.strip()]
            return modules or default
        return [value]

    if isinstance(raw_modules, (list, tuple, set)):
        modules = [str(module).strip() for module in raw_modules if str(module).strip()]
        return modules or default

    value = str(raw_modules).strip()
    if not value:
        return default
    if value == "all-linear":
        return value
    return [value]


def _parse_lora_modules(lora_cfg: Dict[str, Any]) -> tuple[Any, Any]:
    """Parse LoRA target/exclude module filters from config."""
    target_modules = _normalize_module_filter(
        lora_cfg.get("target_modules"),
        default="all-linear",
    )
    exclude_modules = _normalize_module_filter(
        lora_cfg.get("exclude_modules"),
        default=None,
    )
    return target_modules, exclude_modules


def _normalize_remote_url(url: str) -> str:
    """Normalize remote vLLM URL to SkyRL host:port format."""
    return re.sub(r"^https?://", "", str(url).strip()).rstrip("/")


def _resolve_generator_topology(grpo_cfg: Dict[str, Any], lora_rank: int) -> Dict[str, Any]:
    """Resolve SkyRL generator topology from config.

    SkyRL currently supports LoRA weight sync only when engines are local
    (``run_engines_locally=true``). Remote engine mode does not support
    ``LoraLoadRequest`` in upstream SkyRL.
    """
    vllm_mode = str(grpo_cfg.get("vllm_mode", "colocate")).strip().lower()
    requested_remote_url = grpo_cfg.get("vllm_server_url")
    remote_requested = bool(requested_remote_url)

    local_disagg_modes = {
        "server",
        "disagg",
        "disaggregated",
        "non_colocate",
        "non-colocate",
    }
    valid_modes = {"colocate", "local"} | local_disagg_modes
    if vllm_mode not in valid_modes:
        logger.warning(
            "Unknown grpo.vllm_mode=%r; defaulting to 'colocate'.", vllm_mode
        )
        vllm_mode = "colocate"

    # Upstream SkyRL remote inference mode cannot apply LoRA adapters.
    # Fall back to local non-colocated engines to keep LoRA training supported.
    if remote_requested and lora_rank > 0:
        logger.warning(
            "grpo.vllm_server_url=%r requested with LoRA rank=%d. "
            "SkyRL remote engines do not support LoRA weight sync; "
            "falling back to local non-colocated vLLM engines.",
            requested_remote_url,
            lora_rank,
        )
        remote_requested = False
        vllm_mode = "server"

    run_engines_locally = not remote_requested
    colocate_all = run_engines_locally and vllm_mode in {"colocate", "local"}
    remote_urls = (
        [_normalize_remote_url(requested_remote_url)]
        if remote_requested
        else ["127.0.0.1:8001"]
    )

    return {
        "remote_vllm": remote_requested,
        "run_engines_locally": run_engines_locally,
        "colocate_all": colocate_all,
        "weight_sync_backend": "broadcast" if remote_requested else "nccl",
        "remote_inference_engine_urls": remote_urls,
    }


def _convert_grpo_data(data_path: str, output_dir: str, registry=None) -> str:
    """Convert our GRPO JSONL to SkyRL dataset format.

    SkyRL expects each sample to have:
      - prompt: list of message dicts (system + user)
      - Per-sample extras as flat top-level keys (ground_truth_flag, etc.)

    Our GRPO JSONL has:
      - messages: full trajectory (system, user, assistant, tool, ...)
      - ground_truth_flag: str
      - metadata: dict with optimal_steps, task_type, etc.

    We extract the prompt (system + user messages before first assistant)
    and flatten metadata as top-level keys for SkyRL extras.

    Returns:
        Path to the converted JSONL file.
    """
    import jsonlines

    output_path = os.path.join(output_dir, "skyrl_grpo_data.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    converted = 0
    with jsonlines.open(data_path) as reader, jsonlines.open(output_path, "w") as writer:
        for sample in reader:
            messages = sample.get("messages", [])

            # Extract prompt: system + user messages before first assistant/tool
            prompt = []
            for msg in messages:
                role = msg.get("role", "")
                if role in ("system", "user"):
                    prompt.append({"role": role, "content": msg.get("content", "")})
                else:
                    break

            # Ensure prompt ends with user message (SkyRL requirement)
            if not prompt or prompt[-1]["role"] != "user":
                challenge = sample.get("metadata", {}).get("challenge", "")
                prompt.append({
                    "role": "user",
                    "content": (
                        f"Solve the CTF challenge{f': {challenge}' if challenge else ''}. "
                        "Find and capture the flag."
                    ),
                })

            # Flatten extras as top-level keys (SkyRL reads them as extras).
            # env_class is required — SkyRL dataset pops it to find the registered env.
            metadata = sample.get("metadata", {})

            # Extract target URL from user messages
            target = None
            for msg in messages:
                if msg.get("role") == "user":
                    urls = re.findall(r'http://localhost:\d+', msg.get("content", ""))
                    if urls:
                        target = urls[0]
                        break
            if not target:
                target = metadata.get("target")

            # Registry fallback: if no target in prompt/metadata, check registry
            challenge_id = metadata.get("challenge_id") or metadata.get("challenge")
            if not target and registry and challenge_id:
                try:
                    target = registry.get_target_url(challenge_id)
                except KeyError:
                    pass

            row = {
                "prompt": prompt,
                "env_class": "openctf",
                "ground_truth_flag": sample.get("ground_truth_flag"),
                "optimal_steps": metadata.get("optimal_steps"),
                "challenge_id": challenge_id,
                "task_type": metadata.get("task_type", "ctf"),
                "target": target,
            }

            writer.write(row)
            converted += 1

    logger.info("Converted %d GRPO samples → %s", converted, output_path)
    return output_path


def _build_skyrl_config(
    model_path: str,
    output_dir: str,
    config: Dict[str, Any],
    data_path: str,
) -> Dict[str, Any]:
    """Build a SkyRL config dict matching SkyRLConfig dataclass schema.

    Uses SkyRLConfig dataclass defaults as the base, then overrides with
    our training-specific values. This ensures all required keys exist
    regardless of SkyRL version.
    """
    # Load SkyRL's default config as a base. SkyRL 0.3.1 uses a Hydra
    # YAML config (ppo_base_config.yaml) rather than a Python dataclass.
    skyrl_defaults = {}
    try:
        from dataclasses import asdict
        from skyrl_train.config.config import SkyRLConfig
        skyrl_defaults = asdict(SkyRLConfig())
    except (ImportError, ModuleNotFoundError):
        pass
    if not skyrl_defaults:
        try:
            import importlib.resources as pkg_resources
            from omegaconf import OmegaConf as _OC
            # Load the YAML base config from skyrl_train package
            cfg_dir = Path(
                pkg_resources.files("skyrl_train") / "config"
            )
            base_yaml = cfg_dir / "ppo_base_config.yaml"
            if base_yaml.exists():
                raw = _OC.load(base_yaml)
                skyrl_defaults = _OC.to_container(raw, resolve=False)
                logger.info("Loaded SkyRL base config from %s", base_yaml)
        except Exception as exc:
            logger.warning("Could not load SkyRL base config: %s", exc)

    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    grpo_cfg = config.get("grpo", {})
    output_cfg = config.get("output", {})
    lora_rank = _parse_lora_rank(lora_cfg)
    lora_target_modules, lora_exclude_modules = _parse_lora_modules(lora_cfg)
    topology = _resolve_generator_topology(grpo_cfg, lora_rank=lora_rank)
    remote_vllm = topology["remote_vllm"]
    requested_flash_attn = bool(grpo_cfg.get("flash_attn", False))
    enable_flash_attn = requested_flash_attn and _flash_attn_available()
    if requested_flash_attn and not enable_flash_attn:
        logger.warning(
            "flash_attn requested but unavailable in this environment; falling back to SDPA."
        )
    use_sample_packing = bool(grpo_cfg.get("use_sample_packing", False))
    if use_sample_packing and not enable_flash_attn:
        logger.warning(
            "use_sample_packing requested without flash_attn support; disabling sample packing."
        )
        use_sample_packing = False
    max_prompt_length = grpo_cfg.get("max_prompt_length", model_cfg.get("max_seq_length", 8192))
    vllm_max_model_len = grpo_cfg.get("vllm_max_model_len", model_cfg.get("max_seq_length", 8192))
    vllm_language_model_only = bool(grpo_cfg.get("vllm_language_model_only", False))
    chat_template_name = grpo_cfg.get("chat_template", None)
    default_logprobs = None if remote_vllm else 0
    train_logprobs = grpo_cfg.get("logprobs", default_logprobs)
    eval_logprobs = grpo_cfg.get("eval_logprobs", train_logprobs)

    # SkyRL's multi-turn generator does not support response-logprob bookkeeping
    # when a custom chat template is used. Enforce this at config build time
    # so every benchmark/model path gets the same safe behavior.
    if chat_template_name and train_logprobs is not None:
        logger.warning(
            "Custom chat_template=%r set with logprobs=%r; forcing "
            "generator.sampling_params.logprobs=None for SkyRL compatibility.",
            chat_template_name,
            train_logprobs,
        )
        train_logprobs = None
    if chat_template_name and eval_logprobs is not None:
        logger.warning(
            "Custom chat_template=%r set with eval_logprobs=%r; forcing "
            "generator.eval_sampling_params.logprobs=None for SkyRL compatibility.",
            chat_template_name,
            eval_logprobs,
        )
        eval_logprobs = None

    # Detect transformer layer class for FSDP wrapping.
    # model._no_split_modules returns a set on some architectures (e.g. Llama),
    # which triggers a set-indexing bug in SkyRL's apply_fsdp2.
    # We auto-detect and pass the class name as a string to avoid this.
    _ARCH_TO_LAYER_CLS = {
        "LlamaForCausalLM": "LlamaDecoderLayer",
        "Qwen2ForCausalLM": "Qwen2DecoderLayer",
        "Qwen3ForCausalLM": "Qwen3DecoderLayer",
        "Qwen3_5ForConditionalGeneration": "Qwen3_5DecoderLayer",
        "MistralForCausalLM": "MistralDecoderLayer",
        "GptOssForCausalLM": "GptOssDecoderLayer",
    }
    transformer_layer_cls = None
    try:
        from transformers import AutoConfig
        auto_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        arch = getattr(auto_cfg, "architectures", [None])[0]
        transformer_layer_cls = _ARCH_TO_LAYER_CLS.get(arch)
        if not transformer_layer_cls and arch:
            # Fallback: guess from architecture name
            base = arch.replace("ForCausalLM", "")
            transformer_layer_cls = f"{base}DecoderLayer"
    except Exception:
        pass

    # Reference model path for KL divergence. Defaults to the policy model
    # (standard GRPO), but can be overridden for distillation.
    ref_model_path = config.get("ref_model_path", model_path)
    eps_clip_low = grpo_cfg.get("epsilon_low", 0.2)
    eps_clip_high = grpo_cfg.get("epsilon_high", eps_clip_low)

    skyrl_config = {
        # Data
        "data": {
            "train_data": [data_path],
            "val_data": [],
        },

        # Trainer — includes all fields required by SkyRL 0.3.1 validate_cfg
        "trainer": {
            "strategy": "fsdp2",
            "bf16": True,
            "gradient_checkpointing": True,
            "gradient_checkpointing_use_reentrant": False,
            "seed": 42,
            "sequence_parallel_backend": "ulysses",
            "epochs": grpo_cfg.get("epochs", 1),
            "update_epochs_per_batch": 1,
            "train_batch_size": grpo_cfg.get("batch_size", 1),
            "policy_mini_batch_size": grpo_cfg.get("batch_size", 1),
            "critic_mini_batch_size": grpo_cfg.get("batch_size", 1),
            "micro_train_batch_size_per_gpu": 1,
            "micro_forward_batch_size_per_gpu": 1,
            "max_prompt_length": max_prompt_length,
            "use_sample_packing": use_sample_packing,
            "eval_batch_size": 1,
            "eval_interval": -1,
            "flash_attn": enable_flash_attn,
            "disable_fast_tokenizer": False,
            "update_ref_every_epoch": False,
            "resume_mode": None,
            "resume_path": None,
            "ckpt_path": output_dir,
            "max_ckpts_to_keep": -1,
            "hf_save_interval": -1,
            "ckpt_interval": output_cfg.get("save_steps", 50),
            "export_path": os.path.join(output_dir, "final"),
            "eval_before_train": False,
            "project_name": "open-ctf",
            "run_name": "grpo",
            "logger": (
                "console" if output_cfg.get("report_to", "none") == "none"
                else output_cfg["report_to"]
            ),
            "dump_data_batch": False,
            "dump_eval_results": False,
            "target_modules": None,
            "exclude_modules": None,
            "rope_scaling": None,
            "rope_theta": None,

            "placement": {
                # Non-colocated mode allows trainer and local engines to use
                # separate GPUs while staying on SkyRL's supported LoRA path.
                "colocate_all": topology["colocate_all"],
                "colocate_policy_ref": True,
                "policy_num_nodes": 1,
                "policy_num_gpus_per_node": 1,
                "critic_num_nodes": 1,
                "critic_num_gpus_per_node": 1,
                "ref_num_nodes": 1,
                "ref_num_gpus_per_node": 1,
            },

            "fully_async": {
                "max_staleness_steps": 4,
                "num_parallel_generation_workers": 1,
            },

            "policy": {
                "model": {
                    "path": model_path,
                    "lora": {
                        "rank": lora_rank,
                        "alpha": lora_cfg.get("alpha", 128),
                        "dropout": lora_cfg.get("dropout", 0.0),
                        "target_modules": lora_target_modules,
                        "exclude_modules": lora_exclude_modules,
                        "lora_sync_path": os.path.join(output_dir, "lora_sync"),
                        "init_method": "kaiming",
                    },
                    "config_kwargs": {},
                },
                "model_config_kwargs": {},
                "fsdp_config": {
                    "cpu_offload": False,
                    "reshard_after_forward": True,
                    "fsdp_size": -1,
                    "wrap_policy": (
                        {"transformer_layer_cls_to_wrap": [transformer_layer_cls]}
                        if transformer_layer_cls
                        else {}
                    ),
                },
                "sequence_parallel_size": 1,
                "use_torch_compile": False,
                "record_memory": False,
                "optimizer_config": {
                    "lr": grpo_cfg.get("learning_rate", 5e-6),
                    "adam_betas": [0.9, 0.999],
                    "weight_decay": grpo_cfg.get("weight_decay", 0.0),
                    "max_grad_norm": grpo_cfg.get("max_grad_norm", 5.0),
                    "offload_after_step": True,
                    "num_warmup_steps": 0,
                    "scheduler": "constant_with_warmup",
                },
            },

            "ref": {
                "model": {
                    "path": ref_model_path,
                    "config_kwargs": {},
                },
                "model_config_kwargs": {},
                "sequence_parallel_size": 1,
                "fsdp_config": {
                    # CPU offload ref model when KL beta is 0 (ref logprobs unused).
                    # Saves ~16-32GB GPU memory for 8B+ models on single-GPU setups.
                    "cpu_offload": grpo_cfg.get("beta", 0.0) == 0.0,
                    "reshard_after_forward": True,
                    "fsdp_size": -1,
                },
            },

            "critic": {
                "model": {
                    "path": None,
                    "lora": {
                        "rank": 0,
                        "alpha": 16,
                        "dropout": 0,
                        "target_modules": lora_target_modules,
                        "exclude_modules": lora_exclude_modules,
                        "init_method": "kaiming",
                    },
                },
                "model_config_kwargs": {},
                "sequence_parallel_size": 1,
                "fsdp_config": {
                    "cpu_offload": False,
                    "reshard_after_forward": True,
                    "fsdp_size": -1,
                },
                "optimizer_config": {
                    "lr": 5e-6,
                    "adam_betas": [0.9, 0.999],
                    "weight_decay": 0.01,
                    "max_grad_norm": 1.0,
                    "offload_after_step": True,
                    "num_warmup_steps": 0,
                    "scheduler": "constant_with_warmup",
                },
            },

            "algorithm": {
                "advantage_estimator": grpo_cfg.get("advantage_estimator", "rloo_n"),
                "policy_loss_type": "regular",
                "kl_loss_coef": grpo_cfg.get("beta", 0.0),
                "use_kl_loss": grpo_cfg.get("beta", 0.0) > 0,
                "use_kl_in_reward": False,
                "kl_ctrl": {
                    "type": "fixed",
                    "kl_target": 0.1,
                    "horizon": 10000,
                },
                "kl_estimator_type": "k3",
                "use_kl_estimator_k3": False,
                "use_abs_kl": False,
                "use_entropy_loss": False,
                "entropy_loss_coef": 0.01,
                "advantage_batch_normalize": False,
                "value_head_prefix": "value_head",
                "loss_reduction": "token_mean",
                "grpo_norm_by_std": True,
                "zero_variance_filter": False,
                "lambd": 1.0,
                "gamma": 1.0,
                "eps_clip_low": eps_clip_low,
                "eps_clip_high": eps_clip_high,
                "clip_ratio_c": 3.0,
                "tis_imp_ratio_cap": -1.0,
                "use_tis": False,
                "sapo": {"tau_pos": 1.0, "tau_neg": 1.05},
                "value_clip": 0.2,
                "dynamic_sampling": {
                    "type": None,
                    "max_sample_batches": 30,
                    "min_replace_ratio": 0.3,
                },
                "clip_cov": {
                    "clip_ratio": 0.0002,
                    "clip_cov_lb": 1.0,
                    "clip_cov_ub": 5.0,
                },
                "kl_cov": {
                    "kl_cov_frac": 0.2,
                    "ppo_kl_coef": 1.0,
                },
                "cispo": {
                    "cispo_eps_clip_low": 0,
                    "cispo_eps_clip_high": 5,
                },
            },
        },

        # Generator (vLLM inference) — all fields from SkyRL 0.3.1
        "generator": {
            "model_name": model_path,
            "model_dtype": "bfloat16",
            "run_engines_locally": topology["run_engines_locally"],
            "num_inference_engines": 1,
            "backend": "vllm",
            "weight_sync_backend": topology["weight_sync_backend"],
            "weight_transfer_threshold_cuda_ipc_GB": 1.0,
            "inference_engine_tensor_parallel_size": 1,
            "inference_engine_pipeline_parallel_size": 1,
            "inference_engine_expert_parallel_size": 1,
            "inference_engine_data_parallel_size": 1,
            "n_samples_per_prompt": grpo_cfg.get("num_generations", 8),
            "async_engine": True,
            "batched": False,
            "max_input_length": max_prompt_length,
            "vllm_v1_disable_multiproc": True,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 32768,
            "enforce_eager": True,
            "fully_sharded_loras": False,
            "enable_ray_prometheus_stats": False,
            "gpu_memory_utilization": grpo_cfg.get("gpu_memory_utilization", 0.4),
            "max_num_seqs": 32,
            "remote_inference_engine_urls": topology["remote_inference_engine_urls"],
            "enable_http_endpoint": False,
            "http_endpoint_host": "127.0.0.1",
            "http_endpoint_port": 8000,
            "served_model_name": None,
            "max_turns": grpo_cfg.get("max_tool_calling_iterations", 15),
            "chat_template": {
                "source": "name",
                "name_or_path": chat_template_name,
            },
            # Pass enable_thinking, reasoning_effort, etc. to tokenizer.
            # SkyRL unpacks these in every apply_chat_template() call.
            # For Qwen3/Qwen3.5: {"enable_thinking": true} activates
            # <think>...</think> generation and correct loss masking.
            "chat_template_kwargs": grpo_cfg.get("chat_template_kwargs", {}),
            # max_model_len limits vLLM's KV cache allocation.  Without this,
            # vLLM uses the model's max_position_embeddings (e.g. 262144) which
            # can exceed available GPU memory.  Set to max_input_length + headroom.
            "engine_init_kwargs": {
                "max_model_len": vllm_max_model_len,
                "language_model_only": vllm_language_model_only,
            },
            "override_existing_update_group": "auto",
            "sampling_params": {
                "max_generate_length": grpo_cfg.get("max_completion_length", 8192),
                "repetition_penalty": 1.0,
                "temperature": 1.0,
                "top_p": 0.95,
                "min_p": 0.0,
                "top_k": -1,
                # Local default is 0, remote default is None. If a custom
                # chat template is configured we force None above.
                "logprobs": train_logprobs,
                "stop": None,
            },
            "use_conversation_multi_turn": True,
            "append_eos_token_after_stop_str_in_multi_turn": True,
            "eval_sampling_params": {
                "max_generate_length": grpo_cfg.get("max_completion_length", 8192),
                "repetition_penalty": 1.0,
                "temperature": 0.6,
                "top_p": 0.95,
                "min_p": 0.0,
                "top_k": -1,
                "logprobs": eval_logprobs,
                "stop": None,
            },
            "eval_n_samples_per_prompt": 1,
            "zero_reward_on_non_stop": False,
            "apply_overlong_filtering": False,
            "rope_scaling": None,
            "rope_theta": None,
            "step_wise_trajectories": False,
        },

        # Environment
        "environment": {
            "env_class": "openctf",
            "skyrl_gym": {
                "max_env_workers": 32,
            },
        },
    }

    # Merge with SkyRL defaults to ensure all required keys exist.
    # Our overrides take precedence over defaults.
    if skyrl_defaults:
        # Strip Hydra-specific keys that contain OmegaConf interpolations
        # (e.g. ${deepspeed_config.train}) — we don't use DeepSpeed/Megatron.
        _HYDRA_KEYS = {"defaults", "deepspeed_config", "megatron_config"}
        for k in _HYDRA_KEYS:
            skyrl_defaults.pop(k, None)

        # Strip any values containing OmegaConf interpolation syntax
        def _strip_interpolations(d):
            """Remove dict values that are OmegaConf interpolation strings."""
            if not isinstance(d, dict):
                return d
            cleaned = {}
            for k, v in d.items():
                if isinstance(v, str) and "${" in v:
                    continue  # Skip interpolation references
                elif isinstance(v, dict):
                    cleaned[k] = _strip_interpolations(v)
                else:
                    cleaned[k] = v
            return cleaned
        skyrl_defaults = _strip_interpolations(skyrl_defaults)

        def _deep_merge(base, override):
            """Recursively merge override into base dict."""
            result = dict(base)
            for k, v in override.items():
                if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                    result[k] = _deep_merge(result[k], v)
                else:
                    result[k] = v
            return result
        skyrl_config = _deep_merge(skyrl_defaults, skyrl_config)

    # Remove any remaining Hydra/OmegaConf interpolation keys from final config
    for k in ("defaults", "deepspeed_config", "megatron_config"):
        skyrl_config.pop(k, None)

    # vLLM 0.16 rejects SamplingParams.additional_kwargs; older SkyRL defaults
    # can reintroduce it during deep-merge even when our overrides omit it.
    generator_cfg = skyrl_config.get("generator", {})
    for key in ("sampling_params", "eval_sampling_params"):
        sampling = generator_cfg.get(key)
        if isinstance(sampling, dict):
            sampling.pop("additional_kwargs", None)

    return skyrl_config


def train_grpo(
    model_path: str,
    data_path: str,
    output_dir: str,
    config: Dict[str, Any],
    resume_from: Optional[str] = None,
    challenge_registry: Optional[str] = None,
    agent_class: Optional[str] = None,
) -> str:
    """Run online GRPO training via SkyRL.

    Uses OpenCTFTextEnv (SkyRL-Gym BaseTextEnv subclass) with ToolExecutor
    for direct tool execution (no HTTP server needed).

    The reward function is reconstructed inside each SkyRL env instance
    from ``config["reward"]`` (a serializable dict). This avoids passing
    non-serializable callables through Ray.

    Args:
        model_path: Path to the SFT model (merged directory).
        data_path: Path to JSONL GRPO data with ground_truth_flag.
        output_dir: Directory for checkpoints and final model.
        config: Merged config dict from training.yaml.
        resume_from: Optional checkpoint path to resume from.

    Returns:
        Path to the saved final model directory.
    """
    model_path = _resolve_vllm_ready_model_path(model_path)

    logger.info("=" * 60)
    logger.info("GRPO TRAINING (SkyRL)")
    logger.info("  Model:  %s", model_path)
    logger.info("  Data:   %s", data_path)
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 60)

    # 1. Convert data to SkyRL format
    registry = None
    if challenge_registry:
        from open_ctf.challenges.registry import ChallengeRegistry
        registry = ChallengeRegistry(challenge_registry)
    converted_data = _convert_grpo_data(data_path, output_dir, registry=registry)

    # 2. Build SkyRL config
    skyrl_config = _build_skyrl_config(model_path, output_dir, config, converted_data)

    if resume_from:
        skyrl_config["trainer"]["resume_path"] = resume_from

    # 3. Write config for reference
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "skyrl_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(skyrl_config, f, default_flow_style=False)
    logger.info("SkyRL config written to %s", config_path)

    # 4. Launch SkyRL training
    reward_config = config.get("reward", {})
    grpo_cfg = config.get("grpo", {})
    use_new_inference = bool(grpo_cfg.get("use_new_inference", False))
    if use_new_inference and _should_force_legacy_inference(model_path):
        use_new_inference = False
    # Agent class from CLI flag > config file > None (DefaultStepAgent)
    resolved_agent_class = agent_class or grpo_cfg.get("agent_class")
    resolved_agent_kwargs = grpo_cfg.get("agent_kwargs", {})
    try:
        _run_skyrl_training(
            skyrl_config, reward_config,
            agent_class=resolved_agent_class,
            agent_kwargs=resolved_agent_kwargs,
            use_new_inference=use_new_inference,
        )
    except ImportError as e:
        logger.error(
            "SkyRL not installed. Install with: pip install skyrl-train skyrl-gym ray[default] vllm"
        )
        raise

    logger.info("GRPO training complete. Output: %s", output_dir)

    final_dir = os.path.join(output_dir, "final")
    if os.path.exists(final_dir):
        return final_dir
    return output_dir


def _run_skyrl_training(
    config: Dict[str, Any],
    reward_config: Dict[str, Any],
    agent_class: Optional[str] = None,
    agent_kwargs: Optional[Dict[str, Any]] = None,
    use_new_inference: bool = False,
) -> None:
    """Launch SkyRL training with the given config.

    Uses SkyRL's Python API (BasePPOExp). The environment is registered
    inside a Ray remote task so Ray workers can access it.

    Args:
        config: SkyRL config dict.
        reward_config: Serializable reward weight dict (reconstructed
            into CTFReward inside each env instance).
        agent_class: Dotted path to a StepAgent class (Ray-safe string).
        agent_kwargs: Dict of primitives for the StepAgent constructor.

    Key: exp.run() already calls asyncio.run() internally -- do NOT wrap
    in another asyncio.run().
    """
    import ray
    from omegaconf import OmegaConf

    # Convert dict to OmegaConf DictConfig (SkyRL expects this)
    cfg = OmegaConf.create(config)

    # Import SkyRL utilities
    from skyrl_train.entrypoints.main_base import validate_cfg
    from skyrl_train.utils import initialize_ray

    # Validate config against SkyRLConfig schema
    validate_cfg(cfg)

    # SkyRL's Ray actors should run vLLM V1 with in-process worker handling.
    # Keep multiprocess disabled to avoid Ray actor engine bootstrap issues.
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    # Use SkyRL's new HTTP inference layer when requested.
    # This avoids legacy Ray-wrapped vLLM LoRA startup issues on Qwen3.5.
    os.environ["_SKYRL_USE_NEW_INFERENCE"] = "1" if use_new_inference else "0"
    if use_new_inference:
        logger.info("Enabled SkyRL new inference layer (_SKYRL_USE_NEW_INFERENCE=1)")

    # Initialize Ray cluster
    initialize_ray(cfg)

    # Register env and run training inside a Ray remote task.
    # This ensures the env registration is visible to Ray workers.
    @ray.remote(num_cpus=1)
    def _skyrl_entrypoint(cfg_dict, reward_config, agent_class, agent_kwargs, use_new_inference):
        import os as _os
        _os.environ["VLLM_USE_V1"] = "1"
        _os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        _os.environ["_SKYRL_USE_NEW_INFERENCE"] = "1" if use_new_inference else "0"

        from omegaconf import OmegaConf
        from skyrl_gym.envs import register
        from open_ctf.envs.skyrl.openctf_env import OpenCTFTextEnv
        from skyrl_train.entrypoints.main_base import BasePPOExp

        cfg = OmegaConf.create(cfg_dict)

        # Register OpenCTFTextEnv with serializable kwargs only.
        # reward_config is a plain dict of floats -- JSON-safe for SkyRL's
        # EnvSpec._check_can_jsonify(). The env reconstructs CTFReward
        # from this config in __init__().
        # agent_class is a dotted path string (Ray-safe).
        # agent_kwargs is a dict of primitives (Ray-safe).
        env_kwargs = {
            "reward_config": reward_config,
            # Keep env-side guard aligned with generator max_turns, even though
            # SkyRL also injects max_turns per-sample via env_extras.
            "max_turns": int(getattr(cfg.generator, "max_turns", 15)),
        }
        if agent_class:
            env_kwargs["agent_class"] = agent_class
        if agent_kwargs:
            env_kwargs["agent_kwargs"] = agent_kwargs

        register(
            id="openctf",
            entry_point=OpenCTFTextEnv,
            kwargs=env_kwargs,
        )

        exp = BasePPOExp(cfg)
        exp.run()  # Already calls asyncio.run() internally

    # Convert back to dict for serialization through Ray
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    ray.get(
        _skyrl_entrypoint.remote(
            cfg_dict,
            reward_config,
            agent_class,
            agent_kwargs or {},
            use_new_inference,
        )
    )
