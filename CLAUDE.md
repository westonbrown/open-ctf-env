# Open CTF Environment — Project Reference

Last updated: 2026-03-01

## 1) Overview

Open platform for post-training security LLMs on CTF challenge trajectories. Plug in any agent, benchmark, model, and reward function — then run SFT + online GRPO + GEPA on any GPU infrastructure. Current target: **Qwen3.5-27B** (dense, hybrid attention, 256K context) on 2x B200 SXM via SkyRL.

**Verdict on framework**: Stay on SkyRL. Don't pivot to ROCK or AgentGym-RL. Cherry-pick operational ideas. Migration cost (200-400 hrs) not justified — SkyRL works E2E today with 19 patches for vLLM 0.16 + Ray 2.54.

**Cross-source consensus**: Terminal agent RL is a **systems problem first, algorithm problem second**. Environment quality gates, rollout filtering, and observability dominate success — not novel loss functions.

## 2) Architecture (7 Layers)

```
L7  CLI Orchestration (train.py, evaluate.py, challenges.py)
L6  Training Stages: SFT (TRL) | GRPO (SkyRL) | GEPA (DSPy)
L5  SkyRL Integration (runtime.py → OpenCTFTextEnv, delegates to StepAgent)
L4  Agent Protocols (StepAgent for GRPO, CTFAgent for eval/GEPA)
L3  Runtime Bridge (tool_calls parse mode | native shell-out mode)
L2  Execution Engine (ToolExecutor 13 tools | CTFReward 8 signals)
L1  Challenge Infrastructure (ChallengeRegistry YAML | ChallengeManager Docker)
```

**Key files**: `src/open_ctf/agent/protocol.py` (agent contracts), `src/open_ctf/envs/skyrl/openctf_env.py` (SkyRL bridge), `src/open_ctf/training/online_rl/runtime.py` (GRPO orchestration, 2480 lines), `src/open_ctf/rewards/reward.py` (8-signal reward), `src/open_ctf/envs/tool_executor.py` (13 tools), `src/open_ctf/agent/framework_runtime_bridge.py` (BYO adapter bridge), `configs/challenges/cybench.yaml` (40 CyBench challenges).

### Data Flow
```
BoxPwnr traces → open-ctf-convert → JSONL → SFT (LoRA) → merge →
  GRPO: vLLM generate → StepAgent parse+exec → CTFReward → RLOO → FSDP2 backward → LoRA sync
    → (optional) GEPA prompt evolution → export GGUF
```

### Per-Step GRPO Execution
```
SkyRL vLLM generates text → env.step(action)
  ├─ agent.step(action) → StepResult(observations, done)
  ├─ per_step_reward() → format/phase/loop shaping (+/-0.02)
  ├─ if done: terminal_ctf_reward() → 8-signal score (0.0-2.0)
  └─ return (observations, reward, done, info)
```

## 3) Five BYO Seams

### Seam 1: BYO Agent
| Integration | Protocol | Config |
|---|---|---|
| GRPO training | StepAgent (`reset/step/close/tools`) | `online_rl.agent_class: "module.MyAgent"` |
| Eval / GEPA | CTFAgent (`solve() → AgentResult`) | `--agent custom:module.MyAgent` |
| External framework | Runtime bridge native mode | `OPEN_CTF_AGENT_MODE=native OPEN_CTF_AGENT_CMD="python adapter.py"` |

Adapters exist for: BoxPwnr (default), LangGraph. Protocol v1.0 with capability negotiation. Template stubs exist for Autogen, Strands, and ADK but are not functional.

### Seam 2: BYO Model
Create configuration at `configs/training/<model>.yaml`. Optional formatter in `src/open_ctf/formatters/`. Existing: Nanbeige4.1-3B, GLM-4.7-Flash, Devstral-24B, Qwen3-8B, GPT-OSS-20B, Qwen3.5-27B.

### Seam 3: BYO Benchmark
Create `configs/challenges/<name>.yaml` → pass `--challenge-registry`. Target-map overrides: `OPEN_CTF_TARGET_MAP_PATH=targets.json`.

### Seam 4: BYO Reward
Override weights in config or replace entirely — any callable matching `__call__(completions, **kwargs) -> List[float]`.

### Seam 5: BYO Infrastructure
Runs on any GPU infra. GPU 0: vLLM server. GPU 1: SkyRL FSDP2 trainer. 19 patches: `bash docker/patches/apply_all_patches.sh`.

## 4) Reward System (CTFReward)

| Signal | Weight | Measures |
|--------|--------|----------|
| Flag | 0.40 | Exact match (1.0), pattern (0.1), none (0.0) |
| Efficiency | 0.15 | `min(optimal_steps / actual_steps, 1.0)` |
| Format | 0.10 | Valid tool calls + reasoning structure |
| Recovery | 0.09 | Pivot from stuck states |
| Progression | 0.08 | RECON→ENUM→EXPLOIT phase ordering |
| Cognitive | 0.08 | Words-per-action density (optimal: 42 WPA) |
| Exploration | 0.05 | Novelty with temporal decay (γ=0.95) |
| Uniqueness | 0.05 | Information entropy of unique commands |
| *Hallucination* | *-0.20* | *Wrong flag decays all signals to 30%* |

Auto-normalization prevents silent binary fallback (v2 bug fix). Per-step shaping: +0.02 valid tool call, -0.02 no tool call, +0.03 phase progression, -0.03 repeated command.

**Research warning (ROLL team)**: "Reward engineering is fragile." For early GRPO, use simplified binary reward (1.0 flag correct, 0.0 otherwise). Switch to 8-signal after >10% flag capture rate.

## 5) Target Model: Qwen3.5-27B

| Property | Value |
|---|---|
| Type | Dense (NOT MoE) — eliminates fused_moe_lora bugs |
| Parameters | 27B all active |
| Layers | 64 hybrid: [3x Linear (Gated DeltaNet) + 1x Full Attention] x 16 |
| Context | 256K native, 1M with YaRN |
| VRAM (BF16) | ~54GB weights + KV cache |
| SWE-bench | 72.4 |
| Tool format | `qwen3_coder` XML: `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` |
| Thinking | `<think>...</think>` — stripped from history, kept on final turn |

**Framework requirements**: vLLM ≥0.16.0, transformers ≥5.2.0. **SFT via vanilla TRL** (Unsloth lacks Qwen3.5 support).

**vLLM launch** (B200):
```bash
vllm serve /workspace/models/qwen35-27b --max-model-len 8192 --dtype bfloat16 \
  --gpu-memory-utilization 0.50 --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 --language-model-only --enforce-eager
```

**Why not GLM-4.7-Flash**: MoE LoRA crashes (fused_moe_lora assertion), 4-bit QLoRA unsupported, SFT merge produces garbage, broken on B200.

## 6) Framework Comparison: SkyRL vs ROLL vs AgentGym-RL

| Feature | SkyRL (ours) | ROLL (Alibaba) | AgentGym-RL (Fudan) |
|---|---|---|---|
| Status | Working E2E + 19 patches | Production (3000+ GPUs, 1M+ trajectories) | Research (ICLR 2026 Oral) |
| Backend | FSDP2 + vLLM ≥0.14 | DeepSpeed/Megatron/FSDP2 + vLLM/SGLang | FSDP only + vLLM ≤0.6.3 |
| Credit assignment | GRPO/RLOO (episode) + basic step | **GiGPO** (chunk) + **IPA** (poorly formalized) | GRPO/RLOO (episode only) |
| Key strength | CTF integration, LoRA during RL | Scale, async rollouts (2.72x), positive-only mode | Progressive horizon (ScalingInter-RL) |
| Key weakness | 19 patches needed, no chunk credit | Migration cost 200-400 hrs | vLLM ≤0.6.3, Qwen-only, no LoRA |

### Bitter Lessons from Alibaba (1M+ trajectories)
1. **40% false positive rate** — synthetic rewards for NOT solving problems
2. **Models read/modified test files** — exploited harness instead of solving
3. **Environment diversity starvation** — overfit to exact Docker configs
4. **Token-level credit fails on long horizons** — noisy gradients in 2000+ token episodes
5. **Mixed positive+negative collapses policy** — negative signal overwhelms sparse positive
6. **Batch stragglers** — slow rollouts hold entire batch (20x median latency)
7. **Systematic reward hacking** — tool loops, brute-force retries, destructive operations

### Data Quantity Gap
| Framework | Training Trajectories | Ours |
|---|---|---|
| ROME (ROLL) | 1,000,000+ | 339 SFT + 33 GRPO |
| AgentGym-RL | ~50,000 | — |

**1170x gap.** "Data quantity >> algorithmic sophistication." Prioritize generating more traces over reward engineering.

## 7) Tiered Improvement Roadmap

### T0: Operational Robustness (This Sprint)
| ID | Feature | Effort | Impact |
|---|---|---|---|
| 0a | Rollout quality filter + reason codes (RolloutStatus enum) | Low | High |
| 0b | Progressive horizon schedule (start 12 turns, expand to 60) | Low | High |
| 0c | Training preflight gate (`open-ctf-validate --mode grpo-preflight`) | Low | Medium |
| 0d | Straggler + loop observability (per-rollout timing, repeat detection) | Low | Medium |

### T1: Algorithmic Improvements (Next Sprint)
| ID | Feature | Effort | Impact | Prerequisite |
|---|---|---|---|---|
| 1a | Positive-only RL (metric-gated, >10% flag rate) | Low | **High** | 0a |
| 1b | Simplified reward for early training (binary 0/1) | Low | High | — |
| 1c | Data gate checks (gold solve + no-op verification) | Medium | Medium | 0c |
| 1d | Chunk diagnostics logging (per-chunk timing/reward) | Medium | Medium | 0d |

### T2: Research Track (Future)
| ID | Feature | Effort | Impact | Prerequisite |
|---|---|---|---|---|
| 2a | GiGPO chunk-level credit (+12% ALFWorld) | High | High | 1d |
| 2b | Initialized resampling (truncated successful traces as scaffolding) | Medium | Medium | 1a |
| 2c | SkyRL config surface audit (off-policy, async, overlong filter) | Low | Low | T0+T1 stable |
| 2d | Environment hygiene + diversity (randomize ports/IDs/cookies) | Medium | Medium | — |
| 2e | Async rollout-training queue (ROLL Flash, 2.72x claimed) | High | Medium | 0d proves bottleneck |

## 8) Issue Table (SkyRL 0.3.1 + vLLM 0.16 + Ray 2.54)

| # | Issue | Root Cause | Fix |
|---|---|---|---|
| 1 | Version comparison bug | `"2.10" < "2.6"` string compare | Tuple comparison patch |
| 2 | `/get_server_info` missing | Standard vLLM lacks endpoint | Custom `skyrl_vllm_server.py` |
| 3 | GB10 unified memory Ray OOM | False kills from vLLM pre-allocation | `RAY_memory_monitor_refresh_ms=0` |
| 4 | Policy model fp32 init | `bf16=False` hardcoded, doubles memory | Patch to use `self.cfg.trainer.bf16` |
| 10 | BatchEncoding corruption | `list(batch_encoding)` yields keys not IDs | `_to_token_ids()` helper |
| 13 | NCCL weight sync deadlock | Trainer waits for vLLM to join NCCL group | Skip NCCL for LoRA + remote engines |
| 17 | FlashAttention storage error | FSDP2 tensor sharding + FA2 on sm_121a | `flash_attn: false`, use SDPA |
| 22 | 8B FSDP2 OOM on GB10 | Policy+ref+vLLM > 120GB unified | Eliminate ref model when `beta=0.0` |
| 23 | vLLM 0.16 serving restructure | Import paths changed to subdirectories | 4 compatibility shim modules |
| 25 | MixedPrecisionPolicy missing | Not exported from `fsdp_utils.py` | Add to all conditional import branches |
| 26 | Ray collective removed | `ray.experimental.collective.util` gone in 2.54 | Socket-based compat shim |
| 28 | vLLM V0 engine removed | `VLLM_USE_V1=0` no-op in 0.16 | Pass `max_model_len` via `engine_init_kwargs` |
| 30 | BatchEncoding pickle error | Closure wrappers not picklable in Ray | Patch source call sites directly |
| 32 | Qwen3.5 linear-attn deps | Missing `fla`/`causal_conv1d` → CUDA illegal access | Startup guard + explicit deps |
| 33 | vLLM memory profiling assert | GB10: free memory increases during profiling | Convert assertion to warning + adjustment |
| 34 | `native_tool_schemas` wrong | Set to `false`, should be `true` | Config fix |
| 35 | OmegaConf breaks `apply_chat_template` | DictConfig fails `isinstance(dict)` check | Patch #19: resolve to plain Python types |
| 36 | Terminal reward never fires | `max_turns` (20) != `max_tool_calling_iterations` (10) | Clamp env max_turns to agent loop limit |
| 37 | **`flag_found` parameter mismatch** | Models generate `flag_found(flag="...")` but executor reads `arguments.get("content", "")` — flag value silently dropped, reward=0 despite correct flag | `_extract_flag_value()` helper tries `content`, `flag`, `value`, `submission`, then first string value |
| 38 | SkyRL uses `/v1/completions` not `/v1/chat/completions` | vLLM `--tool-call-parser qwen3_coder` only works on chat completions; text completions are unconstrained → model falls back to Python-style `func(arg="val")` | Fallback parser catches Python-style calls; proper fix: patch SkyRL generator or set `native_tool_schemas: false` |
| 39 | FSDP2 `sharded_sd` memory leak | `fsdp2_load_full_state_dict()` holds GPU tensor references in `sharded_sd` dict after `load_state_dict(assign=True)` → offload/reload cycle OOMs | `del sharded_sd; gc.collect(); torch.cuda.empty_cache()` after load |
| 40 | SkyRL overrides `enforce_eager=True` to `False` | `main_base.py:79` sets `engine_kwargs["enforce_eager"] = False` → cudagraph warmup crashes Qwen3.5 LoRA | Patch line 79 to `pass` |
| 41 | `CUDA_VISIBLE_DEVICES` defeats Ray GPU isolation | Injecting `CUDA_VISIBLE_DEVICES=0,1` into Ray runtime_env overrides per-actor GPU assignment → both actors on GPU 0 | `os.environ.pop("CUDA_VISIBLE_DEVICES")` before `initialize_ray()` when SkyRL manages both actors |

### Critical Bug: Issue #37 — `flag_found` Parameter Mismatch (4 days to diagnose)

**Symptom**: Model correctly solves CTF challenges (finds flag, calls `flag_found`) but terminal reward is ~0.07 instead of ~0.74. Training produces zero policy gradient. Both GRPO generations appear to fail despite actually capturing the flag.

**Root cause**: `tool_executor.py` only accepted `arguments.get("content", "")` for flag submission. But models naturally generate `flag_found(flag="HTB{...}")` — using `"flag"` as the parameter name. The tool registry defines `content` as the schema parameter, but no model follows this when generating freely. The submitted flag was silently an empty string, which never matches ground truth.

**Impact**: All GRPO training runs prior to this fix produced zero flag reward signal. RLOO advantage was near-zero. Policy gradients were zero. Training was mechanically functional but learning nothing about flag capture. The hallucination penalty (-0.20) also fired because the empty-string submissions looked like wrong flags.

**Fix**: `_extract_flag_value()` in `tool_executor.py` — tries canonical name `content`, then `flag`, `value`, `submission`, then falls back to first string value in arguments dict. Generalizable across all model formats.

**Validation**: v6 (before fix): reward=0.078, flag_found=False. v7 (after fix): reward=0.743, flag_found=True. Both generations solved Flag Command in 9-11 tool calls.

**19 patches total** (10 SkyRL, 3 vLLM, 1 Ray, 1 flash_attn stub, 1 torchaudio stub, 3 additional). Patch #5 (fsdp_mixed_precision) is fixed upstream and safely no-ops. Must re-apply on container restart: `bash docker/patches/apply_all_patches.sh`.

Patches not listed in the issue table above (infrastructure/operational, no corresponding bug #):
- `patch_torchaudio_stub.py`: NGC PyTorch ABI incompatibility stub
- `patch_skyrl_filter_stepwise.py`: `filter_generator_output` drops `is_last_step`/`trajectory_ids` (KeyError in postprocess)
- `patch_skyrl_collect_lora_fsdp2.py`: `collect_lora_params` uses FSDP1-only `summon_full_params`, blocks on FSDP2
- `patch_skyrl_stepwise_truncation.py`: `step_wise_trajectories` truncation truncates steps not trajectories
- `patch_skyrl_stepwise_index_guard.py`: Step-wise reward index guard (skip out-of-range `resp_end_idx` writes)
- `patch_skyrl_empty_reward_fallback.py`: Empty `per_token_reward` fallback (prevents ValueError when step has 0 tokens)
- `patch_skyrl_loss_diagnostics.py`: `loss_mask` diagnostics + all-zero fallback (root-causes `policy_loss=0.0`)

## 9) Deployment Topology

```
┌──────────────────────────────────────────────────────────────┐
│  GPU 0: vLLM Server              GPU 1: SkyRL GRPO Trainer   │
│  (policy generation, bf16)       (FSDP2 + LoRA, bf16)       │
│  Port 8001/9000                  + DefaultStepAgent           │
│  gpu_mem_util=0.25-0.50          + SubprocessExecutor (13 tools) │
│                                  + CTFReward (8 signals)     │
│  ◄── HTTP /v1/completions ───►   + File-based LoRA sync     │
└──────────────────────────────────────────────────────────────┘
ENV: _SKYRL_USE_NEW_INFERENCE=1, RAY_memory_monitor_refresh_ms=0
     VLLM_ENABLE_V1_MULTIPROCESSING=0 (GB10 only)
```

**GB10 critical gotchas**: Unified memory → `cpu_offload` ineffective. Colocate mode = OOM (system freeze). Always external vLLM server mode. Patches reset on container restart.

**B200 required flags**: `--language-model-only` (skip vision encoder), `--enforce-eager` (bypass CUDA graph capture for sm_100a).

## 10) Training Data (v8, 2026-02-26)

### SFT: `data/sft.jsonl` — 285 samples, 14MB
- 19 unique CyBench challenges, 10,005 tool calls (78% shell_command)
- All tool responses wrapped in `<tool_response>` tags, 0 consecutive assistant messages
- 285/285 flag_found matches ground_truth_flag
- Context: 55.5% fit 8K, 79.1% fit 16K, 95.0% fit 32K

### GRPO: `data/online_rl_quality.jsonl` — 33 samples, 2.7MB
- 9 unique docker Easy/Medium challenges, 33/33 byte-exact flag match vs live DGX

### Flag Verification
All 40 CyBench flags triple-checked: benchmark repo → metadata.json → docker-compose sources. 10 flags corrected from BoxPwnr traces (fake placeholders, unicode mismatches, hallucinated flags).

## 11) Smoke Test Results

### v16 (pipeline validated, live tool execution)
7/21 steps complete, training continuing. Model makes valid HTTP calls, gets real HTML.

| Step | Challenge | avg_reward | Notes |
|---|---|---|---|
| 1 | Delulu | 0.404 | Best reward |
| 4 | Labyrinth Linguist | 0.347 | SSTI reasoning visible |
| 7 | Flag Command | ~0.40 | 0/2 found flag — can't reason through JS→API chain |

**Policy gradient growing**: 0.0002 (step 1) → 0.248 (step 5). `policy_loss=0.042` (was 0.0 in all prior versions).

### Key fixes that unblocked training
1. SDPA attention patch (prevent 105+ GiB OOM from eager attention)
2. GPU placement monkey-patch for `CUDA_VISIBLE_DEVICES` propagation
3. Port offset (`OPEN_CTF_TARGET_PORT_OFFSET=10200`) for DGX→RunPod tunnel
4. `max_tool_response_chars: 1200 → 2500` (Flag Command HTML truncated at JS import)
5. Terminal reward clamping (`max_turns` ≤ `max_tool_calling_iterations`)

### v7 (Qwen3.5-27B FP8, 2x H200, Flag Command single-challenge, flag parameter fix)
**100% flag capture rate.** Both GRPO generations solved Flag Command with live tool execution.

| Gen | Turns | Tool Calls | Flag Found | Reward | Key Actions |
|---|---|---|---|---|---|
| 0 | 10/20 | 9 (8 shell + 1 flag_found) | Yes | 0.743 | curl HTML → read JS → /api/options → POST secret → flag |
| 1 | 11/20 | 11 (10 shell + 1 flag_found) | Yes | 0.725 | Same chain, 2 extra exploratory curls |

**Critical fix applied**: `_extract_flag_value()` — accepts `content`, `flag`, `value`, `submission` parameter names for `flag_found` tool (Issue #37). Without this fix, v6 showed 0.078 reward and flag_found=False despite model actually finding the flag.

**Known limitations**: `policy_loss=0.0` because both generations solved with similar reward (~0.02 RLOO advantage). Need multi-challenge batches or `num_generations >= 4` for meaningful gradient signal.

### Failure analysis: 3B model vs Flag Command
Confirmed across 4 configurations (SkyRL GRPO, BoxPwnr chat, LangChain tools). Model finds HTML, fetches JS, sees `availableOptions['secret']` — but **never calls `GET /api/options`**. Fundamental reasoning capacity limit at 3B. Need ≥27B for reliable multi-step CTF exploitation.

## 12) Code Review Findings

### God Objects
- **`runtime.py`** (2480 lines): `_build_skyrl_config()` 841 lines, `_convert_online_rl_data()` 573 lines. Decompose into 4+3 sub-functions.
- **`default_agent.py`** (780 lines): `step()` ~300 lines with 6 early-returns. Decompose into 3 extraction paths.

### Logic Bugs
| Location | Bug | Severity |
|---|---|---|
| `tool_executor.py` flag_found parameter | **FIXED** — `arguments.get("content", "")` only; models use `flag` param → silent empty submission, zero reward | **Critical** |
| `reward.py` HTTP error counting | Counts historical errors, not current step | Medium |
| `openctf_env.py` rollout status | `normalize_rollout_status()` silently returns `"ok"` on unknown values | Medium |
| `default_agent.py` flag detection | Ground-truth in tool output prematurely marks episode done | Medium |
| `reward.py` base64 regex | `[A-Za-z0-9+/]{20,}` matches normal URLs | Low |

### Dead Code
- `reward.py` IQ scoring: ~300 lines disabled at `weight=0.0`

### What's Good
- Tool registry single-source-of-truth eliminates duplicated definitions
- RolloutStatus enum replaces stringly-typed values across 14+ call sites
- RuntimeProtocol v1.0 with capability negotiation for forward-compatible BYO
- Workspace isolation prevents concurrent-rollout file collision
- Auto-normalization prevents silent reward binary fallback
- 845 tests passing

## 13) CLI Commands

| Command | Purpose |
|---|---|
| `open-ctf-train sft` | SFT via TRL |
| `open-ctf-train merge` | Merge LoRA → full weights |
| `open-ctf-train rl` | Online GRPO via SkyRL (`--agent` for pluggable StepAgent) |
| `open-ctf-train gepa` | GEPA prompt evolution |
| `open-ctf-convert` | BoxPwnr traces → training format |
| `open-ctf-eval` | Evaluate on CyBench (`--agent custom:module.Class`) |
| `open-ctf-challenges` | Docker container lifecycle (setup/status/teardown) |
| `open-ctf-validate` | Full pipeline validation (no GPU) |

## 14) Key Lessons

1. **3B models cannot solve CTF** — valid tool calls but no multi-step reasoning. ≥27B required.
2. **Silent reward degradation kills training** — weights summing ≠ 1.0 caused binary fallback → zero gradient. Now auto-normalized.
3. **Docker networking is non-obvious** — `localhost` inside container ≠ host. Use `target_host_override` or port offset.
4. **Data quantity >> algorithms** — our 372 samples vs ROLL's 1M+ is a 1170x gap. Generate more traces first.
5. **Positive-only RL first** — mixed positive+negative with sparse rewards collapses policy (ROLL finding).
6. **Simplified reward for early training** — 8-signal reward over-engineered when model barely generates tool calls.
7. **GB10 unified memory pitfalls** — `cpu_offload` saves nothing, colocate mode causes system freeze.
8. **Patches required on every restart** — 19 patches modify installed packages. Always re-apply.
9. **Terminal reward must align with agent loop** — `max_turns` vs `max_tool_calling_iterations` mismatch = reward never fires.
10. **Tool response truncation matters** — 1200 chars hid the JS import revealing the API endpoint. 2500+ needed.
11. **CRITICAL: Tool parameter names must be flexible** — Models generate `flag_found(flag="...")` not `flag_found(content="...")`. The tool registry schema says `content` but no model follows this during free-form generation. 4 days to diagnose because the flag was silently dropped (empty string submission). Always accept multiple parameter name aliases for critical tools. See Issue #37.
12. **SkyRL uses text completion, not chat completion** — vLLM's `--tool-call-parser` and `--enable-auto-tool-choice` only work on `/v1/chat/completions`. SkyRL's multi-turn generator uses `/v1/completions` (raw text), so the model generates unconstrained text. The fallback parser catches Python-style calls, but the model doesn't learn the XML format it was SFT'd on. See Issue #38.
13. **FSDP2 state dict lifecycle** — After `model.load_state_dict(sharded_sd, assign=True)`, the `sharded_sd` dict still holds GPU tensor references. Must `del sharded_sd; gc.collect(); torch.cuda.empty_cache()` before offload/reload cycles or OOM is guaranteed. See Issue #39.
14. **Ray GPU isolation requires clean env** — Never inject `CUDA_VISIBLE_DEVICES` into Ray runtime_env when SkyRL manages both vLLM and trainer actors. Clear the env var before `initialize_ray()` and let Ray assign GPUs per-actor. See Issue #41.

## 15) References

- [Let It Flow / ROLL (arxiv 2512.24873)](https://arxiv.org/abs/2512.24873)
- [ROLL Blog: Bitter Lesson](https://faithful-almanac-add.notion.site/The-Bitter-Lesson-Behind-Building-Agentic-RL-in-Terminal-Environments)
- [AgentGym-RL (ICLR 2026)](https://github.com/WooooDyy/AgentGym-RL)
- [GiGPO (arxiv 2505.10978)](https://arxiv.org/abs/2505.10978)
- [verl-agent GiGPO impl](https://github.com/langfengQ/verl-agent)
- [SkyRL-Agent (arxiv 2511.16108)](https://arxiv.org/abs/2511.16108)

## BoxPwnr RL Proxy Integration Status [2026-03-01]

We successfully refactored the offline Ray RL Generation Evaluation pipeline to fully orchestrate natively deployed instances of the **BoxPwnr ChatCompletionToolsStrategy** instead of the simulated Mock LangGraph runner. The framework allows actual model representations across multi-turn trajectories.

### What We Did:
1. Created `scripts/run_boxpwnr_proxy.py` to transparently intercept inference payloads targeting the local offline vLLM port via `ProxyStepAgent`.
2. Implemented `DirectExecutor` bounds logic dynamically injecting execution environments down into the Ray generator actors through the OpenCTF framework APIs.
3. Updated the metrics serialization inside `ProxyStepAgent` to gracefully parse highly malformed LLM context outputs that utilize erratic schema closures when calling `shell_command`.

### What We Discovered:
* **The Integration is Functional**: The system fully connects and the underlying model is attempting to solve challenges sequentially. The Ray framework captures these execution traces correctly and maps them into JSON sequence trajectories.
* **Format Parsers**: Due to model stochasticity and lack of perfect parsing structures, native tool usages were originally zeroing out within `CTFReward` because `proxy_step_agent.py` monitored interactions using strict BoxPwnr tags (`<COMMAND>`) rather than the native Chat Template `json` format (`<tool_call>...`). We patched the proxy boundary string extraction using a non-greedy regex (`(?:```)?(?:<)?tool_call>?\s*(\{.*?\})\s*(?:</tool_call>|```|$)`) to handle malformed trailing data.
  * **Final Score Check**: The pipeline correctly extracted exactly **106 valid tool invocations** consisting of `shell_command`, `web_search`, and `python_code` commands directly output into the PPO training weights, establishing positive variance bonuses for exploration (`0.13`), progression (`0.8`), and recovery (`0.75`).
