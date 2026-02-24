# Open CTF Environment

Post-training pipeline for security LLMs on CTF challenge trajectories. Converts BoxPwnr agent traces into training data, fine-tunes with SFT + online GRPO, optimizes prompts with GEPA, and exports to GGUF for local deployment.

## Architecture

| Stage | Framework | What it does |
|-------|-----------|--------------|
| **SFT** | LlamaFactory | Supervised fine-tuning on expert CTF traces (LoRA) |
| **GRPO** | SkyRL | Online reinforcement learning with live tool execution via ToolExecutor |
| **GEPA** | DSPy | Prompt evolution -- no weight updates, Pareto-based candidate selection |
| **ToolExecutor** | subprocess | Direct tool execution (shell, Python, files, flag submission) -- no HTTP layer |
| **ChallengeRegistry** | YAML | Maps 40 CyBench challenges to infra requirements (docker/static, ports) |
| **ChallengeManager** | Docker | Container lifecycle management for service-based challenges |
| **CTFAgent** | Protocol | Pluggable agent interface -- bring any agent (default: BoxPwnr) |

## Key File Locations

```
src/open_ctf/
  agent/
    protocol.py       CTFAgent protocol + AgentResult dataclass
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
    skyrl/            SkyRL-Gym environment bridge (OpenCTFTextEnv, tool schemas)
  formatters/       Model-specific chat template formatters (GLM-4, Qwen3, Devstral)
  rewards/reward.py CTF reward function (8 signals + hallucination penalty)
  training/
    sft.py                LlamaFactory SFT integration
    grpo.py               SkyRL GRPO integration (per-challenge target routing)
    gepa.py               GEPA prompt optimizer
    tools.py              Tool wrappers for ToolExecutor with episode management
    step_reward.py        CTF reward adapter for SkyRL per-step rewards

configs/
  challenges/
    cybench.yaml      40 CyBench challenges (15 docker + 25 static)
  llamafactory/     Per-model SFT configs (nanbeige_3b, glm47_flash, devstral_24b)
  skyrl/            Per-model GRPO configs (nanbeige_3b, glm47_flash, devstral_24b)

docker/
  Dockerfile        Multi-stage build (targets: base, sft, grpo)

data/               Training data (generated from BoxPwnr traces)
tests/              Reward function tests, ToolExecutor tests, challenge registry tests
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

Existing models: Nanbeige4.1-3B (default), GLM-4.7-Flash (30B MoE), Devstral-Small-2-24B.

## CLI Commands

| Command | Purpose |
|---------|---------|
| `open-ctf-train sft` | Stage 1: SFT via LlamaFactory |
| `open-ctf-train merge` | Merge LoRA adapter into base model |
| `open-ctf-train grpo` | Stage 2: Online GRPO via SkyRL |
| `open-ctf-train gepa` | Stage 3: GEPA prompt optimization |
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
- [x] 490 tests passing (0 failures, 18 skipped)
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

### Next: DGX Deployment & Live Validation

#### Phase 1 — Transfer & Install
- [ ] **Rsync to DGX Spark**: `rsync -avz open-ctf-env/ abrown@100.91.175.48:/home/abrown/open-ctf-env/`
- [ ] **Docker build**: Build `open-ctf-env:latest` container on DGX (base + sft + grpo targets)
- [ ] **Dep install**: Verify skyrl-gym, skyrl-train, ray, vllm, llamafactory all resolve in container
- [ ] **CyBench setup**: Ensure all 40 CyBench challenges are running on DGX (`docker compose up -d` in validation-benchmarks)

#### Phase 2 — SFT Smoke Test
- [ ] **SFT run**: `open-ctf-train sft --model Nanbeige/Nanbeige4.1-3B --data data/sft.jsonl --output outputs/sft` (1 epoch, verify loss decreases)
- [ ] **Merge**: `open-ctf-train merge --adapter outputs/sft/final --base-model Nanbeige/Nanbeige4.1-3B --output outputs/sft-merged`
- [ ] **Sanity check**: Quick inference on merged model (can it generate valid tool calls?)

#### Phase 3 — Online GRPO with Live CyBench
- [ ] **GRPO run**: `open-ctf-train grpo --model outputs/sft-merged --data data/grpo.jsonl --output outputs/grpo` with ToolExecutor hitting real CyBench challenges
- [ ] **Monitor**: Verify reward signal (flag capture rate > 0), KL divergence, grad norms stable
- [ ] **Checkpoint**: Save checkpoint at step 50, 100 for comparison

#### Phase 4 — GEPA Prompt Optimization
- [ ] **GEPA run**: `open-ctf-train gepa` with BoxPwnr agent against CyBench subset
- [ ] **Validate**: Compare GEPA-optimized prompt vs baseline on 5 challenges (solve rate delta)

#### Phase 5 — Full CyBench Benchmark (40 Challenges)
- [ ] **Baseline**: Run BoxPwnr + base Nanbeige-3B on all 40 CyBench challenges (establish floor)
- [ ] **Post-SFT**: Run BoxPwnr + SFT model on all 40 (measure SFT lift)
- [ ] **Post-GRPO**: Run BoxPwnr + GRPO model on all 40 (measure GRPO lift)
- [ ] **Post-GEPA**: Run BoxPwnr + GRPO model + GEPA prompt on all 40 (measure full pipeline lift)
- [ ] **Report**: Compile results table (solve rate, avg steps, unique tools per challenge)

### Known Gaps (Non-Blocking)

- `export_gguf.py` has no unit tests (requires llama.cpp binary)
- `open-ctf-agent` CLI entry point not explicitly validated in test_cli.py
- docs/ still has some stale "OpenEnv" references (code is correct, only docs affected)

## Pluggable Platform Design ("Bring Any X")

### Bring Any Agent

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
open-ctf-train grpo --model outputs/sft-merged --data data/grpo.jsonl \
    --challenge-registry configs/challenges/cybench.yaml \
    --output outputs/grpo
```

## Dependency Version Constraints

| Extra | Key Packages | Notes |
|-------|-------------|-------|
| **sft** | LlamaFactory ≥0.9.0 | LlamaFactory owns transformers/peft/accelerate/datasets versions (pins ≤4.57.1) |
| **grpo** | SkyRL-gym ≥0.1.0, Ray ≥2.40.0, torch ≥2.5.0 | transformers ≥4.45.0, peft ≥0.15.0 |
| **merge** | torch ≥2.4.0, transformers ≥4.45.0 | peft ≥0.15.0, accelerate ≥0.34.0 |
| **gepa** | DSPy ≥3.1.0, GEPA ≥0.0.26 | Lightweight, no GPU deps |

**Docker separates SFT and GRPO environments** so they can resolve different transformers versions independently.

## Code Style

- Python 3.10+
- Linting and formatting: `ruff check .` / `ruff format .`
- Type hints encouraged but not enforced
