# Open CTF Environment

Open platform for post-training security LLMs on CTF challenge trajectories. Anyone can plug in their own agent, benchmark, model, and reward function — then run the full pipeline (SFT + online GRPO + GEPA) on any training infrastructure. Designed for remote training instances (cloud GPUs, on-prem clusters) running online RL with SFT-finetuned models against live challenge environments.

## Architecture

| Stage | Framework | What it does |
|-------|-----------|--------------|
| **SFT** | LlamaFactory or TRL | Supervised fine-tuning on expert CTF traces (LoRA). `--backend trl` for Qwen3.5+, `llamafactory` for others. |
| **GRPO** | SkyRL | Online reinforcement learning with live tool execution via ToolExecutor |
| **GEPA** | DSPy | Prompt evolution -- no weight updates, Pareto-based candidate selection. Reflection LM defaults to same model (local, no cloud APIs). |
| **ToolExecutor** | subprocess | Direct tool execution (shell, Python, files, flag submission) -- no HTTP layer |
| **ChallengeRegistry** | YAML | Maps 40 CyBench challenges to infra (docker/static, ports). Uses [cybench-patched](https://github.com/0ca/cybench-patched.git) for ARM + build fixes. |
| **ChallengeManager** | Docker | Container lifecycle management for service-based challenges |
| **CTFAgent** | Protocol | Pluggable agent interface for eval/GEPA -- bring any agent (default: BoxPwnr) |
| **StepAgent** | Protocol | Pluggable tool-execution agent for GRPO training loop (default: DefaultStepAgent) |

## Key File Locations

```
src/open_ctf/
  agent/
    protocol.py       CTFAgent + StepAgent protocols, AgentResult + StepResult dataclasses
    default_agent.py  DefaultStepAgent — default tool parsing + execution for GRPO
    boxpwnr_adapter.py BoxPwnr adapter implementing CTFAgent
    runner.py         BoxPwnr AgentRunner (low-level)
  challenges/
    registry.py       ChallengeRegistry — YAML-backed challenge lookup
    manager.py        ChallengeManager — Docker container lifecycle
  cli/
    train.py          SFT/GRPO/GEPA/merge orchestration
    evaluate.py       Model evaluation (--agent flag for pluggable agents)
    challenges.py     Challenge container management (setup/status/teardown)
    (+ convert, split, validate, export)
  data/             BoxPwnr trace converter + dataset splitter
  envs/
    tool_executor.py  Direct tool execution engine (13 tools, subprocess-based)
    skyrl/            SkyRL-Gym environment bridge (OpenCTFTextEnv delegates to StepAgent)
  formatters/       Model-specific chat template formatters (GLM-4, Qwen3, Qwen3.5, Devstral)
  rewards/reward.py CTF reward function (8 signals + hallucination penalty)
  training/
    sft.py                LlamaFactory SFT integration (backend=llamafactory)
    sft_trl.py            TRL SFTTrainer SFT integration (backend=trl, for Qwen3.5+)
    grpo.py               SkyRL GRPO integration (per-challenge target routing)
    gepa.py               GEPA prompt optimizer (DSPy + CTFAgentDSPyAdapter)
    tools.py              Tool wrappers for ToolExecutor with episode management
    step_reward.py        CTF reward adapter for SkyRL per-step rewards

configs/
  challenges/
    cybench.yaml      40 CyBench challenges (25 docker + 15 static) — uses cybench-patched for ARM compat
  llamafactory/     Per-model SFT configs (nanbeige_3b, glm47_flash, devstral_24b, qwen3_8b, gptoss_20b, qwen35_27b)
  skyrl/            Per-model GRPO configs (nanbeige_3b, glm47_flash, devstral_24b, qwen3_8b, gptoss_20b, qwen35_27b)

docker/
  Dockerfile        Multi-stage build (targets: base, sft, grpo)

data/               Training data (generated from BoxPwnr traces)
tests/              Reward, ToolExecutor, challenge registry, StepAgent protocol tests
```

## Testing

```bash
pytest tests/
open-ctf-validate          # Full pipeline validation (no GPU needed)
```

## Docker

```bash
docker compose run --rm sft        # Stage 1: SFT
docker compose run --rm merge      # Merge LoRA adapter
docker compose run --rm grpo       # Stage 2: GRPO
docker compose run --rm validate   # Validate pipeline
docker compose run --rm export     # Export to GGUF
```

## Model-Agnostic Design

Models are configured via YAML files, not hardcoded. To add a new model:

1. Create `configs/llamafactory/<model>.yaml` (SFT hyperparameters)
2. Create `configs/skyrl/<model>.yaml` (GRPO hyperparameters)
3. Add a formatter in `src/open_ctf/formatters/` if the chat template is non-standard

Existing models: Nanbeige4.1-3B (default), GLM-4.7-Flash (30B MoE), Devstral-Small-2-24B, Qwen3-8B, GPT-OSS-20B, **Qwen3.5-27B** (dense, hybrid attention, current target).

## Deployment

This environment is designed to run on **any training infrastructure** — cloud GPU instances, on-prem clusters, or local machines. The typical workflow:

1. **Provision a GPU instance** (RunPod, Lambda, etc.) with Docker support
2. **Clone this repo + [cybench-patched](https://github.com/0ca/cybench-patched.git)** for the challenge benchmark
3. **Run `open-ctf-challenges setup`** to build and start challenge containers (40 CyBench challenges, ports 32801-32844)
4. **SFT** your base model on expert traces → **merge** LoRA → **GRPO** with live tool execution against challenges → **GEPA** for prompt optimization
5. **Export** to GGUF for local deployment

The pipeline supports online RL where the model generates tool calls, the StepAgent executes them against live challenge services, and the reward function scores the results — all in a single training loop. Remote training instances run the full environment (model + challenges + tools) co-located for minimal latency.

## CLI Commands

| Command | Purpose |
|---------|---------|
| `open-ctf-train sft` | Stage 1: SFT (`--backend trl` for Qwen3.5+, `llamafactory` for others) |
| `open-ctf-train merge` | Merge LoRA adapter into base model |
| `open-ctf-train grpo` | Stage 2: Online GRPO via SkyRL (`--agent` for pluggable StepAgent) |
| `open-ctf-train gepa` | Stage 3: GEPA prompt optimization — reflection LM defaults to same model, no cloud APIs (`--agent`, `--challenge-registry`, `--budget`, `--reflection-model`) |
| `open-ctf-convert` | Convert BoxPwnr traces to training format |
| `open-ctf-split` | Split datasets into SFT and GRPO sets |
| `open-ctf-eval` | Evaluate models on CyBench (--agent for pluggable agents) |
| `open-ctf-challenges` | Manage challenge Docker containers (setup/status/teardown) |
| `open-ctf-validate` | Validate pipeline without GPU |
| `open-ctf-export` | Export to GGUF |

## Production Readiness Checklist

### Completed

- [x] OpenEnv HTTP server removed — replaced with direct ToolExecutor (subprocess)
- [x] All compliance findings fixed (31 across 5 frameworks: BoxPwnr, DSPy, SkyRL, LlamaFactory, OpenThoughts)
- [x] Data quality: reasoning_content converted, bad entries removed, role violations fixed
- [x] File renames: sft_llamafactory→sft, grpo_skyrl→grpo, openenv_reward→step_reward
- [x] 502 tests passing (2 pre-existing failures, 18 skipped)
- [x] pyproject.toml dependency versions updated + uv.lock regenerated (234 packages)
- [x] LlamaFactory version constraint conflict resolved (SFT extra lets LlamaFactory own pins)
- [x] SkyRL config added for Devstral-24B
- [x] Docker defaults fixed (GRPO data path, stale OpenEnv comment)
- [x] SkyRL deep dive: _all_text includes tool output (reward sees flag verification)
- [x] SkyRL deep dive: observations use role="user" (SkyRL apply_chat_template compat)
- [x] SkyRL deep dive: RLOO-N advantage estimator + 8 samples/prompt (OpenThoughts-aligned)
- [x] SkyRL deep dive: error classification around executor.step()
- [x] SkyRL deep dive: bare JSON regex handles nested braces
- [x] SkyRL deep dive: per_step_reward → 0.0 (binary terminal reward, no dilution)
- [x] SkyRL deep dive: top_p 0.95, weight_decay 0.0, max_grad_norm 5.0, max_generate_length 8192
- [x] SkyRL deep dive: prefix caching, chunked prefill, max_num_seqs 32 in all configs
- [x] Pluggable StepAgent wired into GRPO training loop (2026-02-24)
- [x] vLLM 0.16 compat shims for SkyRL 0.3.1 import paths (2026-02-24)
- [x] Root-caused SkyRL remote LoRA incompatibility; replaced runtime monkey-patch path with topology fix (`run_engines_locally=true`, `colocate_all=false`) (2026-02-24)
- [x] StepAgent smoke test: 21/21 pre-GPU + GRPO training pass (2026-02-24)
- [x] Thinking mode support: `<think>` stripping in parse_tool_calls(), chat_template propagation (2026-02-24)
- [x] Qwen3.5-27B configs: SFT (LlamaFactory), GRPO (SkyRL), training YAML all created (2026-02-24)
- [x] Dependency versions bumped: transformers ≥5.2.0, accelerate ≥1.4.0, torch ≥2.5.0 (2026-02-24)
- [x] SkyRL colocate-mode patches: 5 new patches (#25-#30) for vLLM 0.16 + Ray 2.54 compat (2026-02-24)
- [x] Nanbeige4.1-3B tool response format fix: tool output now wrapped in `<tool_response>` tags matching model's native ChatML template expectation (2026-02-25)
- [x] Nanbeige4.1-3B thinking mode fix: `chat_template_kwargs: {keep_all_think: false}` configured — model auto-strips thinking from history, keeps on last turn (2026-02-25)
- [x] Double tool schema injection fix: `_inject_tool_schemas` now detects "Available tools:" in GRPO data system prompts, preventing duplicate tool schema blocks (~800 token savings per prompt) (2026-02-25)

### Active Deployment TODO

- [x] Confirm root cause for Qwen3.5 instability when linear-attention fast-path deps are missing.
- [x] Add strict startup guard (`grpo.require_fast_linear_attention=true`) so missing `fla`/`causal-conv1d` fails fast.
- [x] Make reward wiring robust (default CTFReward config instead of silent binary fallback).
- [x] Add deterministic challenge-root resolution and Docker preflight checks in `ChallengeManager`.
- [x] Remove runtime monkey-patch path from GRPO entrypoint and rely on deterministic build-time compatibility patches.
- [ ] Run periodic GRPO soak tests on fresh instances to validate: rewards evolve, checkpoints save, and LoRA weight sync stays healthy.

### Environment Bring-Up Notes

- Online GRPO requires nested Docker support when docker-backed benchmark challenges are used.
- `open-ctf-challenges setup` now fails fast with actionable errors if Docker networking/layer import is unavailable.
- For remote challenge hosts or port-forwarded setups, use challenge target maps to avoid stale `localhost` routing.

### Validation Checklist

- [ ] Run SFT smoke + merge on a small sample and verify tool-call generation quality.
- [ ] Run GRPO smoke (2-5 steps) and confirm non-binary reward dynamics in logs.
- [ ] Confirm periodic checkpoints are written and LoRA weight sync succeeds.
- [ ] Run GEPA on a CyBench subset with `--challenge-registry` routing enabled.
- [ ] Evaluate baseline vs SFT vs SFT+GRPO (and GEPA prompt, if used) on the same challenge split.

### GRPO v7 Bug Fixes — Tool Parsing & Reasoning (2026-02-25)

Three bugs caused the Nanbeige4.1-3B model to receive malformed multi-turn context during GRPO training, degrading tool-use learning and reasoning quality. All three were fixed in v7.

**Bug 1 (Critical): Tool response format mismatch**
- **Root cause**: `DefaultStepAgent` returned tool output as plain text (`[Tool: name]\noutput`), but Nanbeige4.1-3B's ChatML template expects tool results wrapped in `<tool_response>...</tool_response>` tags under the user role. Without the tags, the tokenizer treated tool results as human queries instead of tool responses, breaking multi-turn tool-use handling and thinking-block management.
- **Fix**: `src/open_ctf/agent/default_agent.py` — wrap tool output in `<tool_response>` tags.
- **Impact**: Model now correctly distinguishes tool results from user messages in context, enabling proper multi-turn tool chaining.

**Bug 2 (Medium): Thinking mode not configured**
- **Root cause**: `chat_template_kwargs` was empty in `training_120gb_dense.yaml`. Nanbeige4.1-3B has dedicated `<think>`/`</think>` tokens (IDs 166103/166104) with template logic controlled by `keep_all_think`. Without explicit configuration, the tokenizer's default behavior was unpredictable — thinking blocks could be retained in all history turns, wasting context window on reasoning traces from prior steps.
- **Fix**: `src/open_ctf/configs/training_120gb_dense.yaml` — set `chat_template_kwargs: {keep_all_think: false}` (strip thinking from history, keep on last turn).
- **Impact**: Context window savings (~500-1000 tokens per multi-turn episode) and cleaner training signal.

**Bug 3 (Low): Double tool schema injection**
- **Root cause**: `_inject_tool_schemas()` in `openctf_env.py` checked for `"# Available Tools"` in system prompts, but GRPO training data system prompts already contain tool schemas with the header `"Available tools:"` (lowercase, no `#`). The check missed these, injecting a second copy of the schema (~800 tokens wasted per prompt).
- **Fix**: `src/open_ctf/envs/skyrl/openctf_env.py` — multi-variant detection: checks for `"# Available Tools"`, `"Available tools:"`, `"Available tools\n"`, and `"<tools>"`.
- **Impact**: ~800 token savings per prompt, more room for actual tool-use context.

**Result**: v7 (with fixes) survived past step 33 where v6 crashed, and achieved the best reward (+0.21 at step 32) vs v6's deterioration to -2.13 at step 33.

### Known Gaps (Non-Blocking)

- `export_gguf.py` has no unit tests (requires llama.cpp binary)
- `open-ctf-agent` CLI entry point not explicitly validated in test_cli.py
- docs/ still has some stale "OpenEnv" references (code is correct, only docs affected)
- Smoke test reward comparison test (fail < success) needs better test data construction — minimal success case scores lower than failure case with format bonuses
- Issue #24/28: Colocate mode crashes GB10 (OOM from vLLM + FSDP on 120GB unified memory). Use non-colocated topology.
- Issue #30: BatchEncoding incompatibility — fixed via deterministic SkyRL source patching of `apply_chat_template` call sites and remote client token-list coercion (no runtime monkey patch).

## Pluggable Platform Design ("Bring Any X")

### Bring Any Agent

Two agent protocols for two contexts:

**CTFAgent** — for eval/GEPA (agent owns generation + tool execution):

```python
from open_ctf.agent import CTFAgent, AgentResult

class MyAgent:
    def solve(self, challenge, target, ground_truth_flag="",
              max_steps=30, timeout=300) -> AgentResult:
        ...  # your logic
        return AgentResult(success=True, flag="FLAG{...}")

assert isinstance(MyAgent(), CTFAgent)  # structural subtyping
```

Built-in: `BoxPwnrAgent` wraps BoxPwnr's Solver. Use `--agent boxpwnr` (default) or `--agent custom:module.Class`.

Custom agents work with GEPA via `CTFAgentDSPyAdapter`, which wraps any CTFAgent as a DSPy Module so GEPA can evolve the system prompt:

```bash
# GEPA with custom agent
open-ctf-train gepa \
    --model openai/ctf-agent \
    --data data/grpo.jsonl \
    --output outputs/gepa \
    --agent my_module.MyAgent \
    --challenge-registry configs/challenges/cybench.yaml
```

**StepAgent** — for GRPO training (SkyRL owns generation, agent owns tool parsing + execution):

```python
from open_ctf.agent import StepAgent, StepResult

class MyStepAgent:
    def reset(self, target="", ground_truth_flag="", max_steps=30, **kw):
        self.target = target
        # ... setup your tools ...

    def step(self, action: str) -> StepResult:
        # Parse tool calls YOUR way
        # Execute tools YOUR way
        return StepResult(observations=[...], done=False)

    def close(self):
        pass

assert isinstance(MyStepAgent(), StepAgent)  # structural subtyping
```

Built-in: `DefaultStepAgent` uses `parse_tool_calls()` + `SubprocessExecutor` (exact same logic as before). Swap via config or CLI:

```yaml
# training.yaml
grpo:
  agent_class: "my_module.MyStepAgent"
  agent_kwargs: {}
```

```bash
open-ctf-train grpo --agent my_module.MyStepAgent --model ... --data ... --output ...
```

**Architecture**: `OpenCTFTextEnv` delegates tool parsing + execution to the `StepAgent` but **still owns**: reward computation (SkyRL contract), tool schema injection (prompt formatting), and SkyRL protocol compliance. The env reads agent state (`tool_calls_history`, `tool_outputs`, `all_text`, `episode_done`) for reward computation via `getattr()` with graceful fallback.

### Bring Any Benchmark

Challenge registries are YAML-driven. Add a new benchmark by creating `configs/challenges/<name>.yaml`:
```yaml
challenges:
  - id: "my-challenge"
    category: web
    difficulty: easy
    infra_type: docker  # or "static"
    port: 8080
```

Then: `open-ctf-challenges setup --registry configs/challenges/<name>.yaml`

### Per-Challenge Target Routing (GRPO)

GRPO data conversion (`_convert_grpo_data`) extracts target URLs from user messages (`http://localhost:NNNNN`) and falls back to the ChallengeRegistry if provided:

```bash
open-ctf-train grpo --model outputs/sft-qwen35-merged --data data/grpo_cybench40.jsonl \
    --config src/open_ctf/configs/training_qwen35_27b.yaml \
    --challenge-registry configs/challenges/cybench.yaml \
    --output outputs/grpo-qwen35
```

## Dependency Version Constraints

| Extra | Key Packages | Notes |
|-------|-------------|-------|
| **sft** | LlamaFactory ≥0.9.0 | LlamaFactory owns transformers/peft/accelerate/datasets versions (pins ≤4.57.1) |
| **grpo** | SkyRL-gym ≥0.1.0, Ray ≥2.40.0, torch ≥2.5.0 | transformers ≥5.2.0 (Qwen3.5), peft ≥0.15.0, accelerate ≥1.4.0 (FSDP2) |
| **merge** | torch ≥2.5.0, transformers ≥5.2.0 | peft ≥0.15.0, accelerate ≥1.4.0 |
| **gepa** | DSPy ≥3.1.0, GEPA ≥0.0.26 | Lightweight, no GPU deps |

**Docker separates SFT and GRPO environments** so they can resolve different transformers versions independently.

### Why transformers ≥5.2.0

Qwen3.5-27B uses `Qwen3_5ForConditionalGeneration` (hybrid attention: Gated DeltaNet + Full Attention), added in transformers 5.2.0. Earlier versions lack the model class and will fail at `AutoModelForCausalLM.from_pretrained()`. The SFT extra intentionally does NOT pin transformers — LlamaFactory owns that pin and its `qwen3` template handles Qwen3.5 correctly via ChatML.

### Why `flash-linear-attention` + `causal-conv1d` are required (not just `flash_attn`)

Qwen3.5's linear-attention blocks require `fla` and `causal_conv1d`. If either is missing, Transformers falls back to torch kernels (`torch_chunk_gated_delta_rule`). On RunPod B200 GRPO this fallback produced `torch.AcceleratorError: CUDA illegal memory access`. Prevent recurrence on new instances:

1. Install GRPO deps via project extras (`pip install -e ".[grpo]"`) instead of ad-hoc package installs.
2. Verify imports before launch: `python -c "import fla, causal_conv1d; print('ok')"`
3. Keep `grpo.require_fast_linear_attention: true` (default) so startup fails fast if deps are missing.

## DGX Spark (GB10) GRPO Deployment Lessons

Hard-won lessons from deploying SkyRL-based online GRPO on DGX Spark (Grace Blackwell GB10, 120GB unified memory).

### Working Container

| Component | Value |
|-----------|-------|
| **Container** | `open-ctf-test` based on `vllm-node-tf5` |
| **Base Image** | NGC PyTorch 26.01 (`nvcr.io/nvidia/pytorch:26.01-py3`) + vLLM 0.16 compiled for sm_121a |
| **Key Packages** | skyrl-train==0.3.1, skyrl-gym==0.1.1, ray==2.54.0, vllm==0.16.0rc2, torch==2.10.0, peft==0.18.1 |
| **Model** | Nanbeige4.1-3B (3B dense LlamaForCausalLM, ChatML, Hermes tool format) |
| **SFT Merged** | `/workspace/open-ctf-env/outputs/sft-nanbeige3b-merged` (7.4GB) |
| **GRPO Data** | `/workspace/open-ctf-env/data/grpo_cybench40.jsonl` (87 samples) |
| **Patches** | 11 patches via `bash docker/patches/apply_all_patches.sh` (must re-apply after container restart) |

### Issue Table

| # | Issue | Root Cause | Fix |
|---|-------|-----------|-----|
| 1 | **SkyRL version comparison bug** | `str(torch.__version__) >= "2.6"` in `distributed/utils.py` — string comparison makes `"2.10" < "2.6"` lexicographically, so PyTorch 2.10+ gets the wrong parameter name for `_new_process_group_helper()` (`pg_options` instead of `backend_options`), causing `TypeError` | Use `(int(major), int(minor)) >= (2, 6)` tuple comparison |
| 2 | **vLLM `/get_server_info` missing** | Standard vLLM does NOT have a `/get_server_info` endpoint. SkyRL's `RemoteInferenceClient` requires it for `world_size` discovery. Even SkyRL's own `vllm_server.py` is missing it (only present in `vllm_server_actor.py`) | Created `src/open_ctf/training/skyrl_vllm_server.py` as compatibility wrapper that adds the endpoint |
| 3 | **GB10 unified memory + Ray OOM** | GB10 has 120GB unified memory shared between CPU and GPU. vLLM GPU pre-allocation appears as system memory usage, triggering Ray's default OOM threshold (95%) with false kills | Set `RAY_memory_monitor_refresh_ms=0` to disable the OOM monitor |
| 4 | **Policy model hardcoded fp32 init** | `FSDPPolicyWorkerBase.init_model` in `fsdp_worker.py:127` has `bf16=False` hardcoded ("Model initialization should always be in fp32 during training"). This forces `torch_dtype=torch.float32` in `HFModelWrapper.from_pretrained()`. A 3B model in fp32 = ~12GB vs ~6GB in bf16. Combined with FSDP state dict copy (~12GB peak), vLLM allocation, and Ray overhead = OOM on GB10 | Change `bf16=False` to `bf16=self.cfg.trainer.bf16` in `fsdp_worker.py`. For 3B SLMs with LoRA on GB10, bf16 init is safe and saves ~6GB peak. The ref model already uses configurable `bf16=self.cfg.trainer.bf16` (line 357) — only the policy model was hardcoded |
| 5 | **vLLM `gpu-memory-utilization` pre-allocates** | `--gpu-memory-utilization 0.2` = 24GB on GB10 (0.2 * 120GB). Far too much for a 3B model with 2 seqs and 4096 context | Use `0.1` or less for small models on GB10 |
| 6 | **`eval_before_train` requires val_data** | SkyRL crashes with `TypeError: object of type 'NoneType' has no len()` if `eval_before_train: true` but `val_data: []` | Set `eval_before_train: false` when no validation data is provided |
| 7 | **vLLM 0.16+ import paths changed** | `AsyncLLMEngine` moved to `vllm.v1.engine.async_llm.AsyncLLM`; `FlexibleArgumentParser` moved to `vllm.utils.argparse_utils`; `build_app`/`init_app_state` remain in `vllm.entrypoints.openai.api_server` | `skyrl_vllm_server.py` uses version-adaptive imports with try/except fallbacks |
| 8 | **Ray zombie processes accumulate** | Multiple test runs in the same container leave zombie Ray processes that hold memory | Run `ray stop --force` or restart the container between runs to reclaim memory |
| 9 | **External vLLM topology** | SkyRL's internal vLLM import path pulls vLLM internals that may not match the installed version | Set `_SKYRL_USE_NEW_INFERENCE=1` to use `external_server_urls` (HTTP-only path, no vLLM internal imports). Start vLLM with `--worker-extension-cls skyrl_train.inference_servers.vllm_worker.WorkerWrap` for NCCL weight sync |
| 10 | **`BatchEncoding` token-id corruption in generate payload** | Some tokenizers return `BatchEncoding` for `apply_chat_template(..., tokenize=True)`. Naively coercing with `list(batch_encoding)` yields keys (`["input_ids","attention_mask"]`), which get sent as `token_ids` and trigger HTTP 400 validation errors. | Add `_to_token_ids(...)` helper in `skyrl_gym_generator.py` and normalize remote client prompt IDs via `input_ids` extraction (mapping/tensor/nested-safe), then concatenate with accumulated ids. |
| 11 | **Multiprocessing spawn guard** | SkyRL-Gym env workers use Python's `multiprocessing` module. Without `if __name__ == '__main__':` guard in the main script, the `spawn` start method (required for CUDA) raises `RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase` | Wrap the GRPO test/entry script in `if __name__ == '__main__':` and call `multiprocessing.set_start_method("spawn", force=True)` before importing CUDA/Ray |
| 12 | **Missing `/inference/v1/generate` endpoint** | SkyRL's `RemoteInferenceClient` sends generation requests to `{proxy_url}/inference/v1/generate` — a custom data-plane API that standard vLLM does not have. Additionally, when only `external_server_urls` is set (not `external_proxy_url`), SkyRL creates an `InferenceRouter` on port 8080 which may conflict with other services | Added `/inference/v1/generate` to `skyrl_vllm_server.py` (translates SkyRL's `{token_ids, sampling_params}` to vLLM's engine.generate()). Set `external_proxy_url` = `vllm_server_url` in `_build_skyrl_config()` to trigger SkyRL's "fully external" path and skip InferenceRouter creation |
| 13 | **NCCL weight sync deadlock** | `init_weight_sync_state()` creates an NCCL process group with `world_size=2` (trainer + inference). `BroadcastTransferStrategy.create_sender()` blocks forever waiting for rank 1 (vLLM inference engine) to join the NCCL group. With remote engines, the vLLM server can't join (no WorkerWrap) | Patched `worker.py:init_weight_sync_state()` to detect LoRA + remote engines and skip NCCL init entirely. Sets `self._weight_transfer_sender = None`. LoRA uses file-based sync (save adapters → HTTP `/load_lora`), not NCCL broadcast. Patch: `docker/patches/patch_skyrl_weight_sync.py` |
| 14 | **BatchEncoding closure pickle error** | Closure-based wrappers around `apply_chat_template` are not picklable when Ray serializes tokenizer state across worker processes | Patched `skyrl_gym_generator.py` source directly to wrap `apply_chat_template` call sites with list coercion. Removed runtime closure-based monkey patching. Patch: `docker/patches/patch_skyrl_batchencoding.py` |
| 15 | **`/v1/completions` doesn't handle batched token IDs** | SkyRL sends `prompt` as `List[List[int]]` (batched token IDs). The endpoint only checked `all(isinstance(t, int) for t in prompt)` which fails for nested lists, falling through to text mode | Rewrote endpoint to detect: `str`, `List[int]`, `List[List[int]]`, `List[str]`. Generates all prompts concurrently via `asyncio.gather()`. Also extracts `max_generate_length` as alias for `max_tokens` |
| 16 | **vLLM `max_model_len` too small** | Server started with `--max-model-len 4096` but SkyRL generates prompts up to `max_prompt_length + max_generate_length` = 4352 tokens | Use `--max-model-len 8192` for the vLLM server |
| 17 | **FlashAttention "Cannot access data pointer"** | `flash_attn_interface.py` raises `RuntimeError: Cannot access data pointer of Tensor that doesn't have storage` during `fwd_logprobs_values_reward` forward pass. Likely FSDP2 tensor sharding interacting with FA2 on GB10 sm_121a | Set `flash_attn: false` and `use_sample_packing: false` in SkyRL config. Uses SDPA/eager attention instead. Minor perf impact for small models |
| 18 | **flash_attn stub incomplete for vLLM V1** | vLLM V1 engine imports `flash_attn.ops.triton.rotary.apply_rotary` for rotary embeddings. Our flash_attn stub only had `__init__.py` and `bert_padding.py` | Extended flash_attn stub package: added `ops/__init__.py`, `ops/triton/__init__.py`, `ops/triton/rotary.py` with PyTorch fallback `apply_rotary()` implementation |
| 19 | **SkyRL advantage_estimator `rloo_n` not supported** | SkyRL 0.3.1 only supports `reinforce++`, `rloo`, `gae`, `grpo` — not `rloo_n` from SkyRL 0.4+ | Changed `advantage_estimator: rloo_n` to `rloo` in `training_qwen3_8b.yaml` |
| 20 | **OmegaConf interpolation leak from SkyRL defaults** | SkyRL base config (`ppo_base_config.yaml`) has `${deepspeed_config.train}`, `${oc.env:HOME}` interpolation refs that leak into merged config, causing `InterpolationKeyError` | Strip Hydra keys (`defaults`, `deepspeed_config`, `megatron_config`) before merge; recursive `_strip_interpolations()` removes any `${` values; explicit `val_data: []` overrides default |
| 21 | **SkyRL logger `none` not supported** | SkyRL tracking backends: `wandb`, `mlflow`, `swanlab`, `tensorboard`, `console`. Setting `logger: none` from our `report_to: none` config crashes | Map `"none"` → `"console"` in grpo.py logger config |
| 22 | **8B model FSDP2 OOM on GB10 with external vLLM** | Qwen3-8B bf16 + FSDP2 (policy ~32GB peak + ref ~32GB peak) + vLLM (~23GB) + overhead = ~120GB+. **Critical**: On GB10 unified memory, `cpu_offload: True` does NOT help — CPU and GPU share the same 120GB physical memory pool. CPU offload only saves memory on systems with separate CPU DRAM + GPU HBM. | **For 8B on GB10**: Must eliminate ref model entirely when `beta=0.0` (set `ref_num_gpus_per_node: 0` or ref path to None). Alternatively use colocate mode to share model weights between trainer and vLLM (saves ~16GB). Or reduce to 3B model. vLLM at `--gpu-memory-utilization 0.15 --max-model-len 4096` minimizes inference footprint |
| 23 | **vLLM 0.16 serving module restructuring** | SkyRL 0.3.1 imports from vLLM 0.13-0.15 paths (`serving_chat`, `serving_completion`, `serving_models`, `protocol`) that were restructured into subdirectory packages in vLLM 0.16 (`chat_completion/serving`, `completion/serving`, `models/serving`, `engine/protocol`) | Created `docker/patches/patch_vllm_compat_shims.py` — generates 4 compatibility shim modules at old import paths that re-export from the new locations. Added as patch #5 in `apply_all_patches.sh` |
| 24 | **vLLM V1 engine core init fails in Ray actor (colocate)** | In colocate mode, SkyRL creates `AsyncVLLMInferenceEngine` inside a Ray actor. vLLM 0.16 V1 engine uses `AsyncMPClient` which spawns child processes via `multiprocessing`. Inside a Ray actor, this triggers `RuntimeError: Engine core initialization failed` because the child process can't initialize in the Ray subprocess context | Set `VLLM_USE_V1=0` and `VLLM_ENABLE_V1_MULTIPROCESSING=0` inside the Ray actor environment (not just the main process). Must be set before vLLM is imported. Add to Ray runtime env: `env_vars={"VLLM_USE_V1": "0", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}`. Alternatively force V0 engine via SkyRL `engine_init_kwargs`. **UPDATE (2026-02-24)**: vLLM 0.16 fully removed V0 engine — `VLLM_USE_V1=0` has no effect. See Issue #28. |
| 25 | **MixedPrecisionPolicy missing from fsdp_utils.py** | SkyRL 0.3.1's `fsdp_strategy.py` imports `MixedPrecisionPolicy` from `skyrl_train.distributed.fsdp_utils`, but `fsdp_utils.py` only imports `CPUOffloadPolicy, FSDPModule, fully_shard` from `torch.distributed.fsdp`. The `MixedPrecisionPolicy` import is missing from all 3 conditional branches (composable, legacy, None) | Added `MixedPrecisionPolicy` to all conditional import branches in `fsdp_utils.py`. Patch: `docker/patches/patch_skyrl_fsdp_mixed_precision.py` |
| 26 | **`ray.experimental.collective.util` removed in Ray 2.54** | SkyRL 0.3.1 imports `get_address_and_port()` from `ray.experimental.collective.util` (was in Ray 2.51.1). Ray 2.54.0 removed the `collective` module entirely | Created compatibility shim at `ray/experimental/collective/util.py` with socket-based `get_address_and_port()` implementation. Patch: `docker/patches/patch_ray_collective_compat.py` |
| 27 | **vLLM 0.16 removed `model_config` from serving constructors** | vLLM 0.16 changed `OpenAIServingModels`, `OpenAIServingChat`, `OpenAIServingCompletion` constructors — dropped the `model_config` parameter. SkyRL 0.3.1's `vllm_engine.py` passes it as positional/keyword arg | Remove `model_config` from constructor calls in `vllm_engine.py`. Uses try/except fallback for backward compatibility. Patch: `docker/patches/patch_vllm_serving_api.py` |
| 28 | **vLLM 0.16 V0 engine completely removed** | `VLLM_USE_V1=0` has no effect in vLLM 0.16 — V0 engine was fully removed. `AsyncLLMEngine` always resolves to `vllm.v1.engine.async_llm.AsyncLLM`. Issue #24's fix (`VLLM_USE_V1=0`) doesn't work for colocate mode with vLLM 0.16 | Must use vLLM 0.16's V1 engine directly. The `max_model_len` must be passed via `engine_init_kwargs` to prevent 262K default KV cache allocation. Added `"max_model_len": model_cfg.get("max_seq_length", 8192)` to `engine_init_kwargs` in grpo.py |
| 29 | **`peft` not installed in vLLM container** | The `vllm-node-tf5` base image (NGC PyTorch + vLLM compiled) does not include `peft`. SkyRL 0.3.1 `fsdp_worker.py` imports `peft` for LoRA application | `pip install peft` in container setup. Add to Dockerfile or install script |
| 30 | **BatchEncoding closure not picklable in colocate mode** | In colocate mode, SkyRL's dataloader uses multiprocessing workers. Closure-based wrappers around `apply_chat_template` are not picklable and crash worker startup. | Removed runtime closure wrappers and patched SkyRL source call sites directly with `_to_token_ids(apply_chat_template(...))` plus deterministic remote client prompt-id normalization. Patch: `docker/patches/patch_skyrl_batchencoding.py` |
| 31 | **vLLM 0.16 `SamplingParams` rejects `additional_kwargs`** | SkyRL config passes `additional_kwargs: None` in `sampling_params` dict. vLLM 0.16's `SamplingParams.__init__()` raises `TypeError: Unexpected keyword argument 'additional_kwargs'` — this field was removed or renamed in 0.16 | Removed `"additional_kwargs": None` from both `sampling_params` and `eval_sampling_params` in `grpo.py` |
| 32 | **Qwen3.5 linear-attention fast-path deps missing (`fla`, `causal_conv1d`)** | Installing `flash_attn` alone does not satisfy Qwen3.5's Gated DeltaNet path. When `flash-linear-attention` (`fla`) and `causal-conv1d` are absent, Transformers falls back to `torch_chunk_gated_delta_rule`, which triggered `torch.AcceleratorError: CUDA illegal memory access` in GRPO runs. | Added explicit deps in `pyproject.toml` (`grpo` extra + `tool.uv` constraints), added strict runtime preflight in `grpo.py` (`require_fast_linear_attention=true` default), and documented pre-launch import check (`import fla, causal_conv1d`) for new instances. |

### SkyRL + External vLLM Topology

The recommended (and only working) deployment topology for GB10:

```
┌──────────────────────────────────────────────────────────┐
│                   DGX Spark (GB10, 120GB unified)        │
│                                                          │
│  ┌──────────────────┐     ┌────────────────────────────┐ │
│  │  vLLM Server     │     │  SkyRL GRPO Trainer         │ │
│  │  (0.16.0rc2)     │◄───►│  (skyrl-train 0.3.1)       │ │
│  │  Port 8001       │HTTP │  + Ray (2.54.0)             │ │
│  │  gpu_mem=0.15    │     │  + FSDP2 (policy+ref, bf16) │ │
│  │  ~18GB           │     │  + DefaultStepAgent          │ │
│  │  enforce-eager   │     │  + SubprocessExecutor        │ │
│  └──────────────────┘     │  + CTFReward (8 signals)    │ │
│                           └────────────────────────────┘ │
│                                                          │
│  ENV: _SKYRL_USE_NEW_INFERENCE=1                         │
│  ENV: RAY_memory_monitor_refresh_ms=0                    │
│  ENV: VLLM_ENABLE_V1_MULTIPROCESSING=0                   │
│  ⚠️  NEVER use colocate mode on GB10 (OOM → system freeze)│
└──────────────────────────────────────────────────────────┘
```

### E2E GRPO Validated (2026-02-24)

#### Run 1: External vLLM, 5 Steps (no tool execution)

```
Step 1: 36.78s ✅  (generate=20s, fwd_logprobs=3.65s, policy_train=6.2s)
Step 2: 32.29s ✅
Step 3: 34.69s ✅
Step 4: 43.42s ✅
Step 5: 37.09s ✅
Total: 3m19s for 5 steps with 2 trajectories each
```

#### Run 2: Full Online RL with Tool Execution, 3 Steps (2026-02-24)

**This is the definitive validation** — model generates tool calls, DefaultStepAgent parses them, SubprocessExecutor runs shell commands against live CyBench challenges, CTFReward scores results, and weights update.

```
Step 1: 354s ✅  (generate=90s, fwd_logprobs=36s, policy_train=120s, sync=1.1s)
  → avg_final_rewards: 0.25 (1/4 generations found flag!)
  → avg_response_length: 4,400 tokens
Step 2: 479s ✅  (generate=253s, fwd_logprobs=51s, policy_train=175s)
  → avg_final_rewards: 0.00, avg_response_length: 5,899 tokens
Step 3: 317s ✅  (generate=~200s, fwd_logprobs=~50s, policy_train=~65s)
  → avg_final_rewards: 0.00, avg_response_length: 4,959 tokens
Total: 19m32s for 3 steps with 4 trajectories each
Checkpoint: global_step_4 saved (model + optimizer + LoRA adapter)
```

Full pipeline: generate → tool execution → reward → postprocess → convert_to_training_input → fwd_logprobs_values_reward → compute_advantages → policy_train → sync_weights → checkpoint

#### Agent & Tools Used in GRPO Training

**GRPO does NOT use BoxPwnr.** The agent roles are split:

| Component | Role | What It Does |
|-----------|------|--------------|
| **SkyRL** | Generation | Sends prompts to vLLM, receives completions |
| **vLLM** | Inference | Generates token-by-token completions (tool calls) |
| **DefaultStepAgent** | Tool parsing | Parses Hermes JSON / GLM4 XML / bare JSON tool calls from model output |
| **SubprocessExecutor** | Tool execution | Runs 13 tools directly via subprocess (shell_command, python_code, read_file, etc.) |
| **CTFReward** | Reward | 8 signals + hallucination penalty (flag capture, efficiency, format, etc.) |
| **FSDP2** | Training | LoRA weight updates via backward pass |
| **File-based sync** | Weight sync | Saves LoRA adapter → HTTP `/load_lora` to vLLM |

**BoxPwnr** is used only for **eval/GEPA** (it owns generation + execution, which conflicts with SkyRL's generation role). During GRPO, SkyRL owns generation and delegates tool execution to DefaultStepAgent.

**The 13 tools** available during training (from `SubprocessExecutor`):
`shell_command`, `exec_command`, `write_stdin`, `python_code`, `read_file`, `grep`, `file_search`, `apply_patch`, `flag_found`, `web_search`, `list_sessions`, `close_session`, `execute_command`

#### Pluggable StepAgent Smoke Test (2026-02-24)

Comprehensive 7-area smoke test with DefaultStepAgent integration:

```
Tests 1-5 (pre-GPU, no vLLM needed):
  ✅ StepAgent import + DefaultStepAgent is StepAgent
  ✅ Step no-tool-call returns nudge
  ✅ Shell tool call executes (subprocess)
  ✅ Hermes/bare JSON tool parsing
  ✅ CTFReward import + flag capture scoring
  ✅ CyBench registry loads (40 challenges, 25 docker)
  ✅ CyBench connectivity (1/3 reachable)
  ✅ OpenCTFTextEnv creates + delegates to DefaultStepAgent
  ✅ Env init injects tool schema
  ✅ Env step returns observations + nudge
  ✅ Env metrics populated (steps, tool_calls)
  Result: 20/21 pass (1 reward test data construction issue — non-blocking)

Test 7 (GRPO training loop — external vLLM mode):
  ✅ 3/3 steps completed (19m32s total)
  ✅ Reward signal: 0.25 on step 1 (1/4 generations found flag)
  ✅ Full loop: generate → tool exec → reward → train → sync_weights → checkpoint
  ✅ DefaultStepAgent + SubprocessExecutor + CTFReward all working
  ⚠️ Colocate mode crashes GB10 (OOM) — use external vLLM only
```

### GLM-4.7-Flash (30B MoE) Online GRPO — Code Path Trace (2026-02-24)

Traced the full code path for online GRPO with GLM-4.7-Flash to verify it works. **Verdict: YES — already validated for 65 steps / 1 epoch (mean reward 0.245).**

#### Important Clarifications

1. **SkyRL's `Glm4MoeLiteForCausalLM = DeepseekV3ForCausalLM` alias is in the JAX/Flax TX backend** (`skyrl/tx/models/deepseekv3.py`). Our configs use `strategy: fsdp2` (PyTorch backend). The alias is **irrelevant** to our code path.
2. **The FSDP2 backend is model-agnostic** — loads via standard HuggingFace `AutoModelForCausalLM.from_pretrained()`. No special MoE handling; it just loads whatever HF gives it.
3. **Tool call parsing happens in the StepAgent (delegated from OpenCTFTextEnv), not vLLM** — DefaultStepAgent uses `parse_tool_calls()` which supports 4 formats (Hermes JSON, GLM4 XML, Qwen3.5 qwen3_coder XML, bare JSON). Thinking blocks (`<think>...</think>`) are stripped before parsing. No `--tool-call-parser` flag needed on vLLM for training. Custom StepAgent implementations can parse differently.

#### Code Path

```
train_grpo("outputs/sft-merged/", "data/grpo.jsonl", ...)
  │
  ├─ _convert_grpo_data()               Extract prompt + target + ground_truth_flag
  │                                      → outputs/skyrl_grpo_data.jsonl
  │
  ├─ _build_skyrl_config()              Read configs/skyrl/glm47_flash.yaml
  │   ⚠️ "Glm4MoeLiteForCausalLM" NOT in _ARCH_TO_LAYER_CLS
  │   → falls back to guessed layer class name for FSDP2 wrap
  │
  ├─ _run_skyrl_training()              Ray remote → register OpenCTFTextEnv
  │                                      (passes agent_class + agent_kwargs)
  │
  └─ BasePPOExp.run()                   [SkyRL FSDP2 backend]
      ├─ AutoModelForCausalLM.from_pretrained("outputs/sft-merged/")
      │   ⚠️ bf16=False hardcoded → patch fixes to self.cfg.trainer.bf16
      │   Apply LoRA (rank=64, 7 modules, excludes router)
      │   Wrap in FSDP2
      │
      ├─ Launch vLLM (colocate, bf16, gpu_mem_util=0.8)
      │
      └─ Per step:
          vLLM generates 8 completions (raw text)
          → OpenCTFTextEnv.step() delegates to StepAgent.step()
          → DefaultStepAgent parses GLM4 XML tool calls
          → SubprocessExecutor runs shell_command, python_code, etc.
          → OpenCTFTextEnv reads agent state for CTFReward (8 signals)
          → RLOO-N advantage estimation (across 8 samples)
          → FSDP2 backward pass (LoRA weights update)
          → File-based LoRA sync to vLLM (/load_lora)
```

#### Why It Bypasses All Unsloth/TRL MoE Bugs

| Bug we hit with Unsloth/TRL | Why SkyRL avoids it |
|------------------------------|---------------------|
| Unsloth `efficient_log_softmax` shape mismatch (GLM shared_head) | SkyRL doesn't use Unsloth — own training loop |
| Unsloth `grouped_mm` dtype (float32 vs bfloat16) | SkyRL uses PyTorch native ops, not Unsloth Triton kernels |
| 4-bit QLoRA breaks MoE routing | Config uses BF16 LoRA only (`quantization_bit: null`) |
| TRL `sync_weights` sends HF param names to vLLM (fused name mismatch) | SkyRL uses file-based LoRA sync (`/load_lora`), not NCCL param broadcast |
| NaN gradients with batch_size > 1 | Config sets `batch_size: 1` |
| GB10 Triton shared memory 99KB < 104KB needed | SkyRL FSDP2 uses PyTorch native, not Triton MoE kernels |

#### 12 Patches for SkyRL 0.3.1 + vLLM 0.16 + GB10

| Patch | What it fixes | Blocking without it? |
|-------|--------------|---------------------|
| `patch_skyrl_bf16_policy_init.py` | `bf16=False` hardcoded in policy/critic init → fp32 memory spike + OOM; class-scoped idempotent rewrite | **YES** |
| `patch_skyrl_weight_sync.py` | LoRA sync routed to `/update_weights` (400) + strict JSON parsing; now routes to `/v1/load_lora_adapter` and tolerates non-JSON 2xx | **YES** (server mode) |
| `patch_skyrl_version_comparison.py` | `"2.10" < "2.6"` string compare → wrong NCCL params | **YES** |
| `patch_skyrl_batchencoding.py` | Prevent `BatchEncoding` coercion from sending string keys as `token_ids` | **YES** |
| `patch_vllm_compat_shims.py` | vLLM 0.16 restructured `serving_chat/completion/models/protocol` paths | **YES** (colocate mode) |
| `patch_llamafactory_tool_calls.py` | `len(None)` on missing tool_calls | SFT only |
| `patch_torchaudio_stub.py` | NGC torchaudio ABI mismatch | Cosmetic |
| `patch_skyrl_fsdp_mixed_precision.py` | `MixedPrecisionPolicy` missing from fsdp_utils exports | **YES** |
| `patch_ray_collective_compat.py` | `ray.experimental.collective.util` removed in Ray 2.54 | **YES** |
| `patch_vllm_serving_api.py` | vLLM 0.16 removed `model_config` from serving constructors | **YES** (colocate mode) |
| `patch_flash_attn_stub.py` | flash_attn not compiled for GB10 sm_121a | **YES** |
| (grpo.py code change) | `max_model_len` not passed to vLLM engine → 262K default OOM | **YES** |

#### MoE-Specific Config (Legacy GLM-4.7-Flash Path, Not Current Qwen3.5 RunPod Baseline)

| Setting | Value | Why |
|---------|-------|-----|
| `trainer.bf16` | `true` | MoE routers need BF16 (fp32 wastes memory) |
| `trainer.train_batch_size` | `1` | batch > 1 → NaN from MoE router on padded positions |
| `trainer.placement.colocate_all` | `true` | CPU offload between gen/train; avoids NCCL weight naming issues |
| `policy.lora.target_modules` | `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj` | 7 modules; **excludes router/shared_expert_gate** (LoRA on router → NaN) |
| `generator.gpu_memory_utilization` | `0.8` | High but safe for colocate on ≥120GB GPU |
| `algorithm.advantage_estimator` | `rloo_n` | RLOO-N with 8 samples/prompt (OpenThoughts-aligned) |
| `algorithm.kl_loss_coef` | `0.0` | Terminal reward only, no KL penalty |

#### Remaining Risks for Multi-Epoch

| Risk | Severity | Detail |
|------|----------|--------|
| FSDP2 layer class guess | Medium | `Glm4MoeLiteForCausalLM` not in `_ARCH_TO_LAYER_CLS` → suboptimal wrap policy. Worked for 65 steps; may cause memory fragmentation over thousands. |
| Mean reward 0.245 | Medium | Low signal. Binary terminal reward (flag found or not). Check if reward trends upward over steps. |
| vLLM KV cache fragmentation | Low | Repeated LoRA reloads over 1000+ steps could fragment KV cache. Monitor vLLM memory. |
| Ray zombie processes | Low | Long multi-epoch runs may accumulate. Run `ray stop --force` between epochs if needed. |

#### Proven Working

- **SFT (GLM-4.7-Flash)**: 821 samples, 5 epochs, loss 0.40, 92.6% accuracy (outputs/sft_v2/final/)
- **GRPO (GLM-4.7-Flash)**: 65 steps, 6m49s, mean reward 0.245 (external vLLM, offline reward)
- **GRPO (Nanbeige4.1-3B, online RL)**: 3 steps, 19m32s, reward 0.25/0.0/0.0 (external vLLM, live tool execution, flag captured in step 1) — **2026-02-24**
- **CyBench baseline**: 7/40 (17.5%) with base GLM-4.7-Flash Q8_0

### Applying Patches

All SkyRL patches are stored in `docker/patches/` and can be applied with:

```bash
bash docker/patches/apply_all_patches.sh
```

11 patches total (2026-02-24): 5 SkyRL, 3 vLLM, 1 Ray, 1 flash_attn stub, 1 LlamaFactory. See patch table above for details.

### DGX Spark (GB10) Full Environment Setup (Reproducible)

**CRITICAL**: Follow this exact sequence to avoid the issues we hit. Colocate mode OOMs GB10 — always use external vLLM server mode.

#### Hardware

| Component | Value |
|-----------|-------|
| **System** | NVIDIA DGX Spark (Grace Blackwell GB10) |
| **GPU** | NVIDIA GB10, compute capability sm_121a |
| **Memory** | 120GB **unified** (CPU + GPU share same physical pool) |
| **CPU** | ARM Grace (aarch64) |
| **Disk** | 4TB NVMe |
| **Network** | Private cluster network (Tailscale/LAN) |
| **SSH** | `ssh <user>@<host>` |

#### Container Setup

```bash
# Base image: NGC PyTorch 26.01 + vLLM 0.16 compiled for sm_121a
# Built from eugr/spark-vllm-docker with ./build-and-copy.sh --pre-tf -t vllm-node-tf5
docker run -d --name open-ctf-test --gpus all \
    --shm-size=64g \
    -v /path/to/open-ctf-env:/workspace/open-ctf-env \
    -p 8001:8001 \
    vllm-node-tf5 \
    sleep infinity
```

#### Package Versions (Validated 2026-02-24)

| Package | Version | Notes |
|---------|---------|-------|
| **torch** | 2.10.0a0+nv26.01 | NGC pre-built for GB10 sm_121a |
| **vllm** | 0.16.0rc2 | Compiled from source for sm_121a (V0 engine REMOVED) |
| **skyrl-train** | 0.3.1 | `pip install skyrl-train==0.3.1` |
| **skyrl-gym** | 0.1.1 | `pip install skyrl-gym==0.1.1` |
| **ray** | 2.54.0 | Latest (SkyRL wants 2.51.1, needs compat patches) |
| **peft** | 0.18.1 | `pip install peft` (NOT in base image) |
| **transformers** | 5.2.0+ | From NGC base |
| **accelerate** | 1.4.0+ | From NGC base |
| **CUDA** | 12.8 | NGC pre-installed |

#### Dependency Install (Inside Container)

```bash
# 1. Install SkyRL + deps
pip install skyrl-train==0.3.1 skyrl-gym==0.1.1 peft

# 2. Install open-ctf-env in editable mode
cd /workspace/open-ctf-env
pip install -e ".[grpo]"

# 3. Apply all 11 patches (REQUIRED — SkyRL 0.3.1 doesn't work with vLLM 0.16 / Ray 2.54 without them)
bash docker/patches/apply_all_patches.sh

# 4. Kill any zombie Ray processes from previous runs
ray stop --force 2>/dev/null
```

#### Running Online GRPO (Step-by-Step)

**Step 1: Start vLLM server** (background, ~60s to load model)
```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 nohup python3 -m open_ctf.training.skyrl_vllm_server \
    --model /workspace/open-ctf-env/outputs/sft-nanbeige3b-merged \
    --host 0.0.0.0 --port 8001 \
    --max-model-len 8192 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.15 \
    --max-num-seqs 2 \
    --enforce-eager \
    --trust-remote-code \
    > /tmp/vllm_server.log 2>&1 &

# Wait for "Started server process" in log
tail -f /tmp/vllm_server.log | grep -m1 "Started server"
```

**Step 2: Run GRPO training**
```bash
cat > /tmp/run_grpo.py << 'EOF'
import multiprocessing, os
multiprocessing.set_start_method("spawn", force=True)
os.environ["RAY_memory_monitor_refresh_ms"] = "0"
os.environ["_SKYRL_USE_NEW_INFERENCE"] = "1"

from open_ctf.training.grpo import train_grpo
train_grpo(
    model_path="/workspace/open-ctf-env/outputs/sft-nanbeige3b-merged",
    data_path="/workspace/open-ctf-env/data/grpo_cybench40.jsonl",
    output_dir="/tmp/grpo_output",
    max_steps=3,  # smoke test; set higher for real training
    vllm_url="http://localhost:8001",
)
EOF

cd /workspace/open-ctf-env
python3 /tmp/run_grpo.py 2>&1 | tee /tmp/grpo_training.log
```

**Step 3: Verify results**
```bash
grep -E "(Step [0-9]|avg_final_rewards|Training done)" /tmp/grpo_training.log
```

#### Memory Budget (GB10 Unified 120GB)

| Component | Memory | Notes |
|-----------|--------|-------|
| vLLM server (3B bf16) | ~18GB | `--gpu-memory-utilization 0.15` = 0.15 × 120GB |
| SkyRL FSDP policy (3B bf16) | ~12GB | With LoRA, bf16 init (patch #2) |
| SkyRL FSDP ref model | ~6GB | Same model, frozen |
| Ray overhead | ~5GB | GCS, dashboard, raylet |
| Training activations | ~10GB | batch_size=1, grad checkpointing |
| **Total** | **~51GB** | Comfortable for 3B on 120GB |

**WARNING**: Colocate mode (vLLM inside Ray actor) tries to allocate vLLM + FSDP in the same memory pool and **crashes GB10** (caused system freeze requiring physical reboot). Always use external vLLM server mode on GB10.

#### Critical Gotchas

1. **Unified memory**: `cpu_offload: True` does NOT save memory on GB10 — CPU and GPU share the same 120GB pool
2. **Colocate mode = OOM**: vLLM pre-allocates GPU memory; combined with FSDP it exceeds 120GB. Use external server mode only
3. **Patches required on every container restart**: The 11 patches modify installed packages in `/usr/local/lib/python3.12/dist-packages/`. Container restart resets them. Always re-run `apply_all_patches.sh`
4. **Ray zombie cleanup**: Run `ray stop --force` before each GRPO run to prevent memory accumulation
5. **max_model_len**: Must match between vLLM server (`--max-model-len 8192`) and SkyRL config. Without this, vLLM uses model's `max_position_embeddings` (262K for Nanbeige) and OOMs
6. **`VLLM_ENABLE_V1_MULTIPROCESSING=0`**: Required for vLLM on GB10, even in server mode. V1 multiprocessing spawns child processes that fail on GB10
7. **Disk space warnings**: Ray reports `/tmp/ray/` as >95% full — this is cosmetic (138GB free on 3.7TB, but Ray sees the partition percentage). Harmless

### Key Environment Variables

| Variable | Value | Purpose |
|----------|-------|---------|
| `_SKYRL_USE_NEW_INFERENCE` | `1` | Use external HTTP path instead of vLLM internal imports |
| `RAY_memory_monitor_refresh_ms` | `0` | Disable Ray OOM monitor (false kills on unified memory) |
| `VLLM_ENABLE_V1_MULTIPROCESSING` | `0` | Prevent vLLM subprocess spawn failure on GB10 |

## Qwen3.5-27B (Current Target Model — 2026-02-24)

Pivoting from GLM-4.7-Flash (MoE, broken vLLM LoRA) to Qwen3.5-27B (dense, hybrid attention).

### Architecture

| Property | Value |
|----------|-------|
| **Type** | Dense (NOT MoE) — eliminates all fused_moe_lora bugs |
| **Parameters** | 27B all active |
| **Layers** | 64 — hybrid: [3x Linear (Gated DeltaNet) + 1x Full Attention] x 16 |
| **Context** | 256K native, 1M with YaRN |
| **VRAM (BF16)** | ~54GB weights + KV cache |
| **SWE-bench** | 72.4 |
| **License** | Apache-2.0 |

### Tool Call Format (qwen3_coder)

Qwen3.5 uses XML-style tool calls (different from Qwen3's Hermes JSON):

```xml
<tool_call>
<function=shell_command>
<parameter=command>curl http://target/</parameter>
</function>
</tool_call>
```

`parse_tool_calls()` in `openctf_env.py` supports this format (added 2026-02-24).

### Thinking Mode (`<think>...</think>`)

Qwen3.5 generates thinking tokens before responses/tool calls. Thinking mode is configured consistently across all three training stages:

**SFT (LlamaFactory):**
- `template: qwen3` → `ReasoningTemplate` auto-handles `<think>` tokens
- `enable_thinking: true` → thinking included in loss (model learns to reason)
- Thinking automatically stripped from history turns by template

**GRPO (SkyRL):**
- `chat_template: qwen3_without_thinking` — strips thinking from history, keeps on final turn
- `chat_template_kwargs: {enable_thinking: true}` — activates `<think>` generation
- `parse_tool_calls()` strips `<think>...</think>` before parsing (prevents false matches)
- Reward function sees `all_text` including thinking (WPA signal preserved)
- `--reasoning-parser qwen3` on vLLM extracts thinking in API responses

**Key pattern (from OpenThoughts-Agent research):**
The `qwen3_without_thinking` template strips thinking from history to save context window, but preserves it on the final turn so the loss mask includes thinking tokens. This trains the model to reason while keeping multi-turn context efficient.

### Qwen3.5-35B-A3B vs 27B Dense

The 27B dense model outperforms the 35B-A3B MoE variant on metrics that matter for CTF:

| Metric | 27B Dense | 35B-A3B MoE |
|--------|-----------|-------------|
| TIR-Bench (tool calling) | 59.8 | 55.5 (+4.3) |
| IFEval (instruction following) | 95.0 | 91.9 (+3.1) |
| SWE-bench (coding) | 72.4 | 69.2 (+3.2) |
| VRAM (BF16) | ~54 GB | ~70 GB (35B loaded, 3B active) |

The MoE variant would also reintroduce `fused_moe_lora` bugs. Stick with 27B dense.

### Framework Requirements

| Framework | Requirement |
|-----------|-------------|
| **vLLM** | ≥0.16.0 nightly (0.15.1 stable does NOT support `qwen3_5` arch) |
| **transformers** | ≥5.2.0 (for `Qwen3_5ForConditionalGeneration`) |
| **LlamaFactory** | `qwen3` template works (same ChatML) |
| **SkyRL** | Architecture-agnostic (uses AutoModelForCausalLM) — vLLM is the blocker |

### vLLM Serving

```bash
vllm serve /workspace/models/qwen35-27b \
  --max-model-len 8192 --dtype bfloat16 \
  --gpu-memory-utilization 0.50 --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 --language-model-only \
  --attention-backend FLASH_ATTN  # Required for B200 (FlashInfer degraded)
```

### Why Not GLM-4.7-Flash

| Issue | GLM-4.7-Flash | Qwen3.5-27B |
|-------|--------------|-------------|
| MoE LoRA crashes | fused_moe_lora assertion + PassManager fail | **Dense — no MoE** |
| 4-bit QLoRA | NOT supported (MoE) | **Should work** (dense) |
| SFT merge quality | Garbage output | Standard PEFT merge |
| vLLM compatibility | Broken on B200 | Day-0 nightly support |

### Reference B200 Deployment Topology

- **Pod**: 2x NVIDIA B200 SXM (183GB each, 366GB total)
- **Model**: Downloaded to `/workspace/models/qwen35-27b/` (52GB, 11 shards)
- **Strategy**: vLLM server mode on GPU 0, GRPO trainer on GPU 1
- **Config files**: `src/open_ctf/configs/training_qwen35_27b.yaml` (GRPO), `configs/llamafactory/qwen35_27b.yaml` (LlamaFactory SFT)

## Lessons Learned (2026-02-24)

### vLLM on B200 — Two Required Flags

Qwen3.5-27B uses `Qwen3_5ForConditionalGeneration` (VL model class even for text-only). Two B200-specific flags are required:

1. **`--language-model-only`** — skips vision encoder (vLLM tries to profile multimodal embeddings → crash)
2. **`--enforce-eager`** — bypasses CUDA graph capture (B200 sm_100a PTX compiled with unsupported toolchain → `cudaErrorUnsupportedPtxVersion`)

Working launch command:
```bash
CUDA_VISIBLE_DEVICES=0 vllm serve /workspace/models/qwen35-27b \
  --host 0.0.0.0 --port 9000 \
  --max-model-len 8192 --dtype bfloat16 \
  --gpu-memory-utilization 0.50 --max-num-seqs 8 \
  --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --language-model-only --enforce-eager
```

GPU memory: 91.7 / 183 GB (50% util). Leaves GPU 1 entirely free for training.

### SFT Framework — Use Vanilla TRL, Not LlamaFactory/Unsloth

Neither LlamaFactory nor Unsloth can train Qwen3.5-27B today:

| Framework | Blocker |
|-----------|---------|
| **LlamaFactory v0.9.4** | Pins `transformers<=4.57.1`. Qwen3.5 needs `>=5.2.0`. PR #9569 (v5 compat) merged on main but not released. `qwen3_5` model_type not in registry. |
| **Unsloth Feb 2026** | Targets `transformers==5.1.0` (needs >=5.2.0). Qwen3.5 not in fine-tuning catalog. Gated DeltaNet hybrid attention incompatible with Unsloth's custom Triton kernels. |

**Recommendation**: Use vanilla HuggingFace `SFTTrainer` (TRL) + `peft` + `transformers>=5.2.0` directly. No wrappers needed. VRAM budget on B200: ~94 GB total (model 54 + LoRA 1.5 + optimizer 3 + activations 25 + logits 5 + overhead 5). Single GPU sufficient.

### SFT Data Quality (`data/sft.jsonl`)

| Metric | Value | Concern |
|--------|-------|---------|
| Samples | 820 | Small for 27B model (~16.7M tokens/epoch) |
| Platforms | 8 (xbow 28%, portswigger 22%, picoctf 19%, cybench 16%) | Good diversity |
| Unique challenges | 497 | Excellent coverage |
| Tool calls | 25,588 (79.8% shell_command) | Good tool distribution |
| `<think>` blocks | 247/820 (30.1%) | **Mixed signal with enable_thinking=true** |
| Malformed think tags | 28 (orphan `</think>`) | Minor, should clean |
| Orphan env_feedback messages | 299 in 150 samples | LlamaFactory tolerates, adds noise |
| Missing tool responses | 366 | Trace truncation artifacts |
| Fails/errors | 0 (all successful solves) | **No failure recovery training** |
| Fits in 8192 context | 49% | **51% truncated at cutoff_len** |
| Fits in 16384 context | 67% | Better but still 33% loss |
| Fits in 32768 context | 85% | Needs more VRAM |

**Key decision**: Increase `cutoff_len` to 16384 or 32768 to reduce truncation. B200 has headroom.

## Code Style

- Python 3.10+
- Linting and formatting: `ruff check .` / `ruff format .`
- Type hints encouraged but not enforced
