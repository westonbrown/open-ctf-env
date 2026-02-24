# Architecture

Open CTF Environment is a **3-stage training pipeline** for fine-tuning LLMs on CTF tasks using BoxPwnr agent traces: LlamaFactory SFT, SkyRL online GRPO, and GEPA prompt evolution.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    3-STAGE PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Data Collection                                            │
│  ├── BoxPwnr Agent → CyBench Challenges                    │
│  ├── Raw traces: conversation.json + stats.json             │
│  ├── BoxPwnrConverter (lossless, 8 tools preserved)         │
│  └── DatasetSplitter → SFT + GRPO datasets                 │
│                                                              │
│  Stage 1: SFT (LlamaFactory)                               │
│  ├── YAML-driven config (no Python changes per experiment)  │
│  ├── Native tool formats (chatml/qwen, glm4_7/glm4_moe)    │
│  ├── Sequence packing + 4D attention masks                  │
│  ├── DeepSpeed ZeRO for multi-GPU                          │
│  └── Output: LoRA adapter → merge → full checkpoint         │
│                                                              │
│  Stage 2: Online GRPO (SkyRL)                               │
│  ├── Ray-based async trainer (FSDP2)                        │
│  ├── vLLM inference engine (separate process)               │
│  ├── OpenCTFTextEnv → HTTP → OpenEnv server                │
│  ├── CTFReward (8 signals + hallucination penalty)          │
│  └── DAPO loss, no KL penalty                               │
│                                                              │
│  Stage 3: GEPA (DSPy)                                       │
│  ├── No weight updates, only system prompt evolution        │
│  ├── Pareto-based candidate selection                       │
│  └── ~6% better than GRPO with 4-35x fewer rollouts        │
│                                                              │
│  OpenEnv Server (unchanged across all stages)               │
│  ├── FastAPI HTTP (reset/step/state/health)                 │
│  ├── 8 tools (shell, python, files, flag submission)        │
│  ├── Docker challenge containers                            │
│  └── Per-session state isolation                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Module Structure

```
src/open_ctf/
├── cli/                         # CLI entry points
│   ├── train.py                 # open-ctf-train (sft, grpo, gepa, merge)
│   ├── convert_traces.py        # open-ctf-convert
│   ├── split_dataset.py         # open-ctf-split
│   ├── evaluate.py              # open-ctf-eval
│   ├── validate_pipeline.py     # open-ctf-validate
│   └── export_gguf.py           # open-ctf-export
├── configs/                     # Training YAML configs
│   ├── training.yaml            # Default (Nanbeige4.1-3B)
│   ├── training_120gb_dense.yaml # DGX Spark dense config
│   ├── training_120gb_moe.yaml  # DGX Spark MoE config
│   └── training_140gb_moe.yaml  # H200 MoE config
├── data/
│   ├── converter.py             # BoxPwnr trace → ChatML conversion
│   └── splitter.py              # SFT/GRPO dataset splitting
├── envs/openenv/
│   ├── server.py                # FastAPI environment server
│   └── client.py                # HTTP client for tool execution
├── formatters/
│   ├── base.py                  # ModelFormatter abstract base
│   ├── qwen3.py                 # ChatML + Hermes tool format
│   ├── glm4.py                  # GLM-4.7 observation role + tool format
│   └── devstral.py              # Mistral INST tags + strict alternation
├── rewards/
│   └── reward.py                # CTFReward (8 signals + penalty)
└── training/
    ├── sft.py                   # LlamaFactory SFT orchestrator
    ├── grpo.py                  # SkyRL GRPO orchestrator
    ├── gepa.py                  # GEPA prompt optimizer (DSPy)
    ├── step_reward.py           # CTFReward adapter for SkyRL
    └── tools.py                 # 8 tool schemas + episode management

skyrl_envs/
├── openctf_env.py               # OpenCTFTextEnv (SkyRL BaseTextEnv bridge)
└── tool_groups.py               # Tool schema definitions for SkyRL

configs/
├── llamafactory/                # Per-model SFT configs
│   ├── nanbeige_3b.yaml
│   ├── glm47_flash.yaml
│   └── devstral_24b.yaml
└── skyrl/                       # Per-model GRPO configs
    ├── nanbeige_3b.yaml
    └── glm47_flash.yaml
```

## Training Data Flow

```
BoxPwnr Agent
    │ conversation.json + stats.json
    ▼
BoxPwnrConverter
    │ Preserve 8 tool types, handle tool_calls + chat formats, extract reasoning/flags
    ▼
DatasetSplitter
    │ Success → SFT, Multi-turn + flag → GRPO
    ├─── sft.jsonl ────────────────────────► LlamaFactory SFT
    │                                            │ LoRA adapter
    │                                            ▼
    │                                        Merge (PEFT)
    │                                            │ Full checkpoint
    └─── grpo.jsonl ───────────────────────► SkyRL Online GRPO
                                                 │ (via OpenCTFTextEnv → OpenEnv)
                                                 ▼
                                            GRPO model → GEPA → Final
```

## CTF Reward Function

The reward function (`src/open_ctf/rewards/reward.py`) scores completions on 8 signals plus a hallucination penalty:

| Signal | Weight | What It Measures |
|--------|--------|------------------|
| **Flag Capture** | 0.20 | `metadata.success` > exact match > pattern match (0.1) |
| **Efficiency** | 0.25 | `min(optimal / actual, 1.0)` steps |
| **Format** | 0.15 | Valid tool call JSON structure |
| **Progression** | 0.10 | RECON → ENUM → EXPLOIT phase ordering |
| **Exploration** | 0.08 | Novel tool usage weighted toward early trajectory |
| **Uniqueness** | 0.07 | Command diversity (detects stuck loops) |
| **Recovery** | 0.08 | Successful pivot after errors |
| **Cognitive** | 0.07 | Reasoning depth (words per action) |
| **Hallucination** | -0.10 | Penalty for `flag_found` calls with wrong flag |

All process signals are **ungated** -- they provide gradient signal regardless of flag capture. This prevents reward sparsity during early GRPO training.

## Model Formatters

Each model family has a formatter that handles chat template differences:

| Formatter | Models | Template | Tool Format |
|-----------|--------|----------|-------------|
| `Qwen3Formatter` | Nanbeige4.1-3B, Qwen3 | ChatML (`<\|im_start\|>`) | Hermes (`<tool_call>`) |
| `GLM4Formatter` | GLM-4.7-Flash | GLM4 (observation role) | GLM4MOE (XML function calls) |
| `DevstralFormatter` | Devstral-Small-2-24B | Mistral (`[INST]`) | Mistral tool format |

`ModelFormatter.from_model_id()` auto-detects the appropriate formatter from the model name.

## Online RL Architecture (Stage 2: GRPO)

### SkyRL Integration

SkyRL runs GRPO with Ray actors for async training:

1. **Generator** (Ray actor): vLLM inference engine produces N completions per prompt.
2. **Environment** (`OpenCTFTextEnv`): Each generation gets its own env instance.
   - `init(prompt)` → POST `/reset` to OpenEnv
   - `step(action)` → Parse tool calls from LLM text → POST `/step` to OpenEnv → Compute reward
   - `close()` → POST `/close` to OpenEnv
3. **Trainer** (Ray actor): FSDP2 computes GRPO loss, updates policy weights.
4. **Placement**: `colocate_all: true` (default) offloads weights to CPU between gen/train. Eliminates weight sync issues for MoE.

### OpenEnv Server

FastAPI server providing Gym-style tool execution:

- `POST /reset` → Reset episode, close PTY sessions
- `POST /step` → Execute tool, return observation + reward + done
- `POST /state` → Current environment state
- `GET /health` → Health check

### Tool Set

8 tools shared across training data, reward function, and OpenEnv:

| Tier | Tools | Description |
|------|-------|-------------|
| **Execution** | `shell_command`, `python_code` | Shell scripts, Python exploits |
| **File Ops** | `read_file`, `grep`, `file_search`, `apply_patch` | Read, search, modify files |
| **Meta** | `flag_found`, `web_search` | Flag submission, CVE lookup |

## Configuration Architecture

### Layered Configs

```
src/open_ctf/configs/training.yaml        ← Default (Nanbeige4.1-3B, model-agnostic)
src/open_ctf/configs/training_120gb_dense.yaml ← DGX Spark tuned for dense models
src/open_ctf/configs/training_120gb_moe.yaml   ← DGX Spark tuned for MoE
src/open_ctf/configs/training_140gb_moe.yaml   ← H200 tuned for MoE (production)

configs/llamafactory/<model>.yaml          ← LlamaFactory SFT per-model config
configs/skyrl/<model>.yaml                 ← SkyRL GRPO per-model config
```

The `training.yaml` files define model/lora/sft/grpo parameters consumed by the CLI. The LlamaFactory and SkyRL configs are framework-specific formats generated or used directly.

### Key Settings (32K Context)

| Parameter | SFT | GRPO |
|-----------|-----|------|
| Context length | 32768 (`cutoff_len`) | 32768 (`max_prompt_length`) |
| Max completion | N/A | 32768 (`max_generate_length`) |
| LoRA rank | 64 | 64 |
| Learning rate | 2e-4 | 5e-6 |
| Epochs | 5 | 1 |
| Loss | Cross-entropy | DAPO |
| Packing | Yes (3x throughput) | N/A |
| Generations per prompt | N/A | 4 |
| Max tool turns | N/A | 15 |
| Trainer strategy | DeepSpeed / single GPU | FSDP2 + Ray |

## Container Strategy

Two Docker images, one per training stage:

| Image | Base | Purpose | Size |
|-------|------|---------|------|
| `Dockerfile` (target: sft) | NGC PyTorch | LlamaFactory SFT + merge + validate + export | ~15GB |
| `Dockerfile` (target: grpo) | NGC PyTorch | SkyRL GRPO + Ray + vLLM | ~20GB |

OpenEnv server runs in its own container (or directly on host).

## Evaluation Pipeline

```
Trained Model → BoxPwnr Agent → CyBench Challenges → Traces → Metrics
                    │                    │
                    └── shell, python ───►│
                    └── flag_found ──────►│
                                          ▼
                                     Solve Rate, Avg Turns, Avg Time
```

Evaluation uses the same BoxPwnr scaffold as data collection. The only variable is model weights.

## CLI Entry Points

| Command | Module | Purpose |
|---------|--------|---------|
| `open-ctf-train sft` | `cli.train` | Stage 1: LlamaFactory SFT |
| `open-ctf-train merge` | `cli.train` | Merge LoRA adapter |
| `open-ctf-train grpo` | `cli.train` | Stage 2: SkyRL GRPO |
| `open-ctf-train gepa` | `cli.train` | Stage 3: GEPA |
| `open-ctf-convert` | `cli.convert_traces` | BoxPwnr trace → ChatML |
| `open-ctf-split` | `cli.split_dataset` | SFT/GRPO splitting |
| `open-ctf-eval` | `cli.evaluate` | CyBench evaluation |
| `open-ctf-validate` | `cli.validate_pipeline` | Pipeline validation (no GPU) |
| `open-ctf-export` | `cli.export_gguf` | GGUF export |

## Key Design Decisions

### 1. LlamaFactory for SFT (Replaces Custom sft.py)

**Problem**: Custom `sft.py` (378 lines) had Unsloth dependency, manual message normalization, and broke on new hardware.

**Solution**: LlamaFactory handles tool formats, packing, multi-GPU, and LoRA natively via YAML config. 11 built-in tool format classes cover all our model families.

### 2. SkyRL for GRPO (Replaces Custom grpo.py)

**Problem**: Custom `grpo.py` (1305 lines) required 6 monkey-patches for GLM-4.7-Flash MoE on Blackwell GB10 (prefix check, dtype cast, weight sync translation, NCCL segfault, etc.).

**Solution**: SkyRL uses Ray process isolation -- vLLM runs in a separate process from training. This eliminates all 6 patches: no shared-process dtype collisions, no weight sync bugs, no NCCL group conflicts.

### 3. Model-Agnostic Design

**Problem**: Hardcoded model assumptions (MoE batch_size=1, BF16-only, specific template) made it difficult to test new architectures.

**Solution**: All model-specific settings live in YAML configs. To add a model: create `configs/llamafactory/<model>.yaml` and `configs/skyrl/<model>.yaml`. No Python code changes.

### 4. Framework-Independent Components

OpenEnv server, CTFReward, data converters, and GEPA are framework-independent:
- OpenEnv: HTTP API, consumed by SkyRL (via OpenCTFTextEnv), GEPA (via DSPy), or any other RL framework.
- CTFReward: Pure Python callable, wrappable by any trainer.
- BoxPwnrConverter/Splitter: Standard JSONL I/O.
- GEPA: Uses OpenEnv + CTFReward directly, no training framework dependency.

### 5. MoE-Aware Configuration

MoE models (GLM-4.7-Flash) have unique constraints documented in their config files:
- No 4-bit quantization (BitsAndBytes + MoE incompatibility)
- batch_size=1 (MoE routing NaN with padding)
- `colocate_all: true` (safe default for weight sync)
- Router layers excluded from LoRA targets

## Hardware Compatibility

| Hardware | SFT (Dense 3B) | SFT (MoE 30B) | GRPO (Dense 3B) | GRPO (MoE 30B) |
|----------|----------------|----------------|------------------|-----------------|
| DGX Spark GB10 (128GB) | QLoRA 4-bit | BF16 LoRA | Colocate mode | Colocate mode |
| H100 80GB | QLoRA 4-bit | BF16 LoRA | Colocate mode | Server mode (2 GPU) |
| H200 141GB | QLoRA 4-bit | BF16 LoRA | Colocate mode | Server mode (1-2 GPU) |

## Extension Points

### Adding New Models

1. Create `configs/llamafactory/<model>.yaml` with SFT hyperparameters
2. Create `configs/skyrl/<model>.yaml` with GRPO hyperparameters
3. Add a formatter in `src/open_ctf/formatters/` if the chat template is non-standard
4. Add detection logic to `formatters/base.py` factory method

### Adding New Reward Signals

1. Add the signal computation to `src/open_ctf/rewards/reward.py`
2. Add its weight to the configurable weights dict
3. Ensure weights sum to 1.0

### Adding New Benchmarks

The `OpenEnv` server abstracts execution. To add a new CTF platform:
1. Spin up target containers
2. Register the endpoint on OpenEnv
3. Point `OPEN_CTF_ENV_URL` to the new server
4. No changes to training pipeline, reward function, or tool schemas
