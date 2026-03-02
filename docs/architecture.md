# Architecture

Open CTF Environment is a **3-stage post-training pipeline** for fine-tuning LLMs on CTF challenge trajectories using [TRL](https://github.com/huggingface/trl) (SFT), [SkyRL](https://github.com/NovaSky-AI/SkyRL) (online GRPO), and [GEPA](https://arxiv.org/abs/2507.19457) (prompt evolution).

## System Overview

```mermaid
flowchart TD
    %% Node Definitions
    Traces[/"Raw Agent Traces"/]
    Synth[/"Synthetic Data"/]
    Convert[["Converter & Splitter"]]
    
    SFTData[("SFT Dataset")]
    GRPOData[("GRPO Dataset")]
    
    SFT("Stage 1: SFT (TRL)")
    Merge[["Merge LoRA"]]
    GRPO("Stage 2: Online GRPO (SkyRL)")
    GEPA("Stage 3: GEPA (DSPy)")
    
    Model(("Final CTF Agent"))
    Eval{{"CyBench Evaluation"}}

    %% Data Flow
    Traces --> Convert
    Convert -->|"Successes"| SFTData
    Convert -->|"All + Flags"| GRPOData
    Synth -.-> SFTData & GRPOData

    %% Training Flow
    SFTData --> SFT
    SFT --> Merge
    Merge --> GRPO
    GRPOData --> GRPO
    GRPO --> GEPA
    
    %% Output
    GEPA --> Model
    Model --> Eval
```

## Module Structure

```
src/open_ctf/
├── agent/
│   ├── protocol.py              # StepAgent + CTFAgent protocols, AgentResult, validate_step_agent
│   ├── default_agent.py         # DefaultStepAgent — default GRPO tool parser + executor
│   ├── framework_runtime_bridge.py  # BYO adapter bridge (tool_calls + native mode)
│   ├── runtime_protocol.py      # RuntimeProtocol v1.0 with capability negotiation
│   ├── rollout_status.py        # RolloutStatus enum (replaces stringly-typed values)
│   ├── proxy_step_agent.py      # ProxyStepAgent for external RL proxy integration
│   ├── boxpwnr_adapter.py      # Example CTFAgent adapter (BoxPwnr reference agent)
│   └── runner.py                # Example agent runner (BoxPwnr reference agent)
├── challenges/
│   ├── registry.py              # ChallengeRegistry — YAML-backed challenge lookup
│   └── manager.py               # ChallengeManager — Docker container lifecycle
├── cli/
│   ├── train.py                 # open-ctf-train (sft, rl, gepa, merge)
│   ├── convert_traces.py        # open-ctf-convert
│   ├── split_dataset.py         # open-ctf-split
│   ├── evaluate.py              # open-ctf-eval (--agent for pluggable agents)
│   ├── challenges.py            # open-ctf-challenges (setup/status/teardown)
│   ├── validate_pipeline.py     # open-ctf-validate
│   └── export_gguf.py           # open-ctf-export
├── data/
│   ├── converter.py             # Agent trace → ChatML conversion (default: BoxPwnr format)
│   └── splitter.py              # SFT/GRPO dataset splitting
├── envs/
│   ├── tool_executor.py         # SubprocessExecutor + RemoteBatchExecutor
│   └── skyrl/
│       ├── openctf_env.py       # OpenCTFTextEnv (SkyRL BaseTextEnv subclass)
│       └── tool_groups.py       # 13 tool schema definitions for SkyRL
├── synthetic_data_generation/   # Offline synthetic trace generation
│   ├── manifest.py              # WorldManifest dataclass — loads YAML configs defining hosts, files, services, tool responses
│   ├── executor.py              # SimulatedEnvironmentExecutor — mocks all 13 agent tools using manifest data
│   └── generator.py             # LiteLLMAgentAdapter + SyntheticGenerator — runs teacher LLM in ReAct loop, exports traces
├── formatters/
│   ├── base.py                  # ModelFormatter abstract base + auto-detection
│   ├── qwen3.py                 # ChatML + Hermes tool format
│   ├── glm4.py                  # GLM-4.7 observation role + MoE tool format
│   └── devstral.py              # Mistral INST tags + strict alternation
├── rewards/
│   └── reward.py                # CTFReward (8 signals + hallucination penalty)
└── training/
    ├── sft/
    │   └── trl.py               # TRL SFT backend
    ├── online_rl/
    │   ├── runtime.py           # SkyRL runtime + config conversion
    │   ├── step_reward.py       # Per-step shaping reward adapter
    │   └── trajectory_logger.py # Rollout + reward telemetry
    └── gepa.py                  # GEPA prompt optimizer (DSPy + CTFAgentDSPyAdapter)

configs/
├── challenges/
│   └── cybench.yaml             # 40 CyBench challenges (25 docker + 15 static)
├── training/                    # Unified training configurations
│   ├── nanbeige_3b.yaml
│   ├── glm47_flash.yaml
│   └── devstral_24b.yaml
└── skyrl/                       # Per-model GRPO configs
    ├── nanbeige_3b.yaml
    ├── glm47_flash.yaml
    └── devstral_24b.yaml
```

## Training Data Flow

```mermaid
flowchart LR
    %% Definitions
    Traces[/"Raw Traces"/]
    Synth[/"Synthetic Gens"/]
    Converter[["TraceConverter"]]
    Splitter[["DatasetSplitter"]]
    
    SFT_DB[("sft.jsonl")]
    GRPO_DB[("grpo_cybench40.jsonl")]

    SFT("SFT Stage")
    Merge[["PEFT Merge"]]
    GRPO("GRPO Stage")
    GEPA("GEPA Stage")
    Model(("Final Export"))

    %% Flow
    Traces --> Converter --> Splitter
    Splitter -- "Successes" --> SFT_DB
    Splitter -- "All + Flags" --> GRPO_DB
    Synth -.-> SFT_DB & GRPO_DB

    SFT_DB --> SFT --> Merge --> GRPO
    GRPO_DB --> GRPO
    GRPO --> GEPA --> Model
```

## CTF Reward Function

The reward function (`src/open_ctf/rewards/reward.py`) scores trajectories with **8 process signals + 1 penalty** (plus optional interaction-quality bonus). Process signals are ungated so training still gets useful gradient before first-flag success becomes common.

| Signal | Weight | What It Measures |
|--------|--------|------------------|
| **Flag Capture** | 0.40 | Correct flag submission / terminal solve |
| **Efficiency** | 0.15 | Steps vs `optimal_steps` target |
| **Format Compliance** | 0.10 | Valid, parseable tool-call structure |
| **Recovery** | 0.09 | Pivoting out of repeated/stuck action loops |
| **Progression** | 0.08 | RECON → ENUM → EXPLOIT phase ordering |
| **Cognitive** | 0.08 | Reasoning density (words per action band) |
| **Exploration** | 0.05 | Early novelty in tool usage (decayed over time) |
| **Uniqueness** | 0.05 | Non-redundant command/output information |
| **Hallucination Penalty** | -0.20 | Wrong flag / fabricated-success behavior |
| **Interaction Quality (bonus)** | 0.00 default | Optional additive signal for productive interactions |

## Model Formatters

Each model family has a formatter that handles chat template differences:

| Formatter | Models | Template | Tool Format |
|-----------|--------|----------|-------------|
| `Qwen3Formatter` | Nanbeige4.1-3B, Qwen3 | ChatML (`<\|im_start\|>`) | Hermes (`<tool_call>`) |
| `GLM4Formatter` | GLM-4.7-Flash | GLM4 (observation role) | GLM4MOE (XML function calls) |
| `DevstralFormatter` | Devstral-Small-2-24B | Mistral (`[INST]`) | Mistral tool format |

`ModelFormatter.from_model_id()` auto-detects the appropriate formatter from the model name.

## Online RL Architecture (Stage 2: GRPO)

### Why BaseTextEnv, Not SkyRL-Agent

We deliberately use `skyrl-gym`'s low-level `BaseTextEnv` interface rather than the higher-level `skyrl-agent` framework (`AutoAgentRunner`, `ReActAgent`). The reasons:

1. **13 CTF tools** don't exist in `skyrl-agent`'s `TOOL_REGISTRY` — wrapping our `SubprocessExecutor` inside their tool class hierarchy would add an unnecessary abstraction layer.
2. **Token format parity**: Tool schemas in the GRPO system prompt must exactly match what the model saw during SFT (verified by `test_tokenizer_drift`). SkyRL-Agent's own prompt construction would break this.
3. **Multi-signal reward**: SkyRL-Agent's verifiers are binary pass/fail. Our `CTFReward` computes 6 continuous process signals.
4. **Per-challenge routing**: CTF challenges require Docker container lifecycle management and per-challenge target URLs — not supported by SkyRL-Agent's task abstraction.

### SkyRL Integration

```mermaid
flowchart LR
    subgraph skyrl["SkyRL BasePPOExp"]
        direction LR
        vLLM("vLLM Generator<br/>(Prefix Cache + Cont. Batching)")
        Parser[["Tool Parser"]]
        Exec[["ToolExecutor"]]
        Env("OpenCTFTextEnv<br/>(Reward Computation)")
        Trainer("FSDP2 Trainer<br/>(DAPO + RLOO-N)")

        vLLM -->|"Generates Call"| Parser
        Parser --> Exec
        Exec -->|"Stdout/Flag"| Env
        Env -->|"Observation"| Trainer
        Trainer -.->|"Weight Sync"| vLLM
    end
```

### Execution Sequence

To keep GRPO scalable, environments operate autonomously and execute commands directly as subprocesses, bypassing the HTTP bottleneck. The training iteration loops are securely coordinated between the model generators and physical containers.

```mermaid
sequenceDiagram
    participant Model as vLLM Generator
    participant Env as OpenCTFTextEnv
    participant Tool as ToolExecutor
    participant Target as CyBench Container
    participant Trainer as FSDP2 Trainer
    
    loop Trajectory Generation (Per Turn)
        Model->>Env: Generate Tool Call (e.g. nmap)
        Env->>Tool: Extract & Parse Action
        Tool->>Target: Execute Subprocess
        Target-->>Tool: Return Stdout / Stderr
        Tool-->>Env: Raw Execution Output
        Env-->>Model: Formatted Observation
    end
    
    Env->>Trainer: Calculate CTFReward (8 Signals)
    Trainer->>Trainer: Compute DAPO Loss & Step
    Trainer-->>Model: Sync Updated Weights via Ray
```

- **Generator** (Ray actor): vLLM inference engine produces 8 completions per prompt.
- **Environment** (`OpenCTFTextEnv`): Each generation gets its own env instance with a `SubprocessExecutor` for tool execution. No HTTP server — direct subprocess calls.
- **Trainer** (Ray actor): FSDP2 computes DAPO loss with config-driven advantage estimation (for example RLOO or RLOO-N).
- **Placement**: Current Qwen3.5 RunPod/B200 baseline uses `run_engines_locally: true` + `colocate_all: false` (trainer and vLLM on separate GPUs).

### Tool Execution

13 tools shared across training data, reward function, and the ToolExecutor:

| Tier | Tools | Description |
|------|-------|-------------|
| **Execution** | `shell_command`, `exec_command`, `write_stdin`, `python_code`, `execute_command` | Shell scripts, interactive PTY sessions, Python exploits |
| **File Ops** | `read_file`, `grep`, `file_search`, `apply_patch` | Read, search, modify files in the container |
| **Meta** | `flag_found`, `web_search`, `list_sessions`, `close_session` | Flag submission, web search, session management |

## Configuration Architecture

### Per-Framework Configs

```
configs/training/<model>.yaml    ← TRL SFT (native YAML format)
configs/skyrl/<model>.yaml           ← SkyRL GRPO (native YAML format)
configs/challenges/<benchmark>.yaml  ← Challenge registry (custom YAML)
```

TRL and SkyRL configs use each framework's native format directly — no translation layer.

### Key Settings (Qwen3.5-27B, 48K Context)

| Parameter | SFT | GRPO |
|-----------|-----|------|
| Context length | 32768 (`cutoff_len`) | 32768 (`max_prompt_length`) |
| Max completion | N/A | 8192 (`max_generate_length`) |
| LoRA rank | 64 | 64 |
| LoRA targets | Attention + shared expert only | Same |
| Learning rate | 2e-4 | 5e-6 |
| Epochs | 5 | 1 |
| Batch size | 1 (MoE routing NaN fix) | 1 |
| Loss | Cross-entropy | DAPO |
| Packing | Yes (3x throughput) | N/A |
| Samples per prompt | N/A | 8 |
| Max tool turns | N/A | 50 |
| Trainer strategy | Single GPU / DeepSpeed | FSDP2 + Ray |
| Advantage estimator | N/A | RLOO / RLOO-N (config-dependent) |

## Container Strategy

Two Docker targets from a single multi-stage Dockerfile, separated to avoid dependency conflicts:

| Target | Base | Purpose |
|--------|------|---------|
| `sft` | `nvcr.io/nvidia/pytorch:25.11-py3` | TRL SFT + merge + validate + export |
| `grpo` | `nvcr.io/nvidia/pytorch:25.11-py3` | SkyRL GRPO + Ray + vLLM |

```bash
docker build -t open-ctf:sft  --target sft  -f docker/Dockerfile .
docker build -t open-ctf:sft-trl --target sft-trl -f docker/Dockerfile .
docker build -t open-ctf:grpo --target grpo -f docker/Dockerfile .
```

## Evaluation Pipeline

```mermaid
flowchart LR
    Model(("Trained Model")) --> Agent[["CTFAgent (Pluggable)"]]
    Agent <--> Challenges[("CyBench 40")]
    Challenges --> Metrics[["Compute Metrics:<br/>Solve rate, Turns, Time"]]
```
Evaluation uses any agent implementing the `CTFAgent` protocol. The only variable is model weights — architecture, tools, and evaluation harness are held constant. Use `--agent boxpwnr` (default example) or `--agent custom:module.Class` to plug in your own agent.

## Key Design Decisions

### 1. TRL for SFT

**Problem**: Fine-tuning across different LLM families securely usually requires many custom scripts.

**Solution**: The TRL backend natively supports model-specific configurations uniformly and scales appropriately. No Python code changes per experiment.

### 2. SkyRL for GRPO

**Problem**: Custom GRPO script required 6 monkey-patches for GLM-4.7-Flash MoE on Blackwell GB10 (prefix check, dtype cast, weight sync translation, NCCL segfault, etc.).

**Solution**: SkyRL uses Ray process isolation — vLLM runs in a separate process from training. This removes the old in-repo monkey-patch-heavy GRPO loop. Remaining upstream compatibility fixes are isolated in `docker/patches/`.

### 3. BaseTextEnv Over SkyRL-Agent

**Problem**: SkyRL-Agent's `ReActAgent` and `TOOL_REGISTRY` are designed for SWE-Bench tasks. CTF challenges need 13 custom tools, multi-signal rewards, per-challenge Docker routing, and strict SFT-GRPO token format parity.

**Solution**: Use `skyrl-gym`'s `BaseTextEnv` directly. Tool execution via `SubprocessExecutor`, reward via `CTFReward`, schemas injected into prompts deterministically. Plugs into SkyRL-Train's `BasePPOExp` without the agent abstraction layer.

### 4. Model-Agnostic YAML Design

**Problem**: Hardcoded model assumptions (MoE batch_size=1, BF16-only, specific template) made it difficult to test new architectures.

**Solution**: All model-specific settings live in YAML configs. To add a model: create `configs/training/<model>.yaml`. No Python code changes.

### 5. MoE-Aware Configuration

MoE models (GLM-4.7-Flash) have unique constraints documented in their config files:
- No 4-bit quantization (BitsAndBytes + MoE routing incompatibility)
- `batch_size=1` (MoE routing NaN with padding tokens)
- `colocate_all: true` for legacy colocated MoE profiles (Qwen3.5/B200 baseline uses non-colocated local engines)
- Router layers excluded from LoRA targets

## Hardware Compatibility

| Hardware | SFT (Dense 3B) | SFT (MoE 30B) | GRPO (Dense 3B) | GRPO (MoE 30B) |
|----------|----------------|----------------|------------------|-----------------|
| DGX Spark GB10 (128GB) | QLoRA 4-bit | BF16 LoRA | Colocate mode | Colocate mode |
| H200 (141GB) | QLoRA 4-bit | BF16 LoRA | Colocate mode | Zero-offload possible |
| B200 (192GB) | QLoRA 4-bit | BF16 LoRA | Non-colocated local engines | Non-colocated local engines |

## Extension Points

### Adding New Models

1. Create `configs/training/<model>.yaml` with SFT hyperparameters
2. Create `configs/skyrl/<model>.yaml` with GRPO hyperparameters
3. Add a formatter in `src/open_ctf/formatters/` if the chat template is non-standard
4. Add detection logic to `formatters/base.py` factory method

### Adding New Reward Signals

1. Add the signal computation to `src/open_ctf/rewards/reward.py`
2. Add its weight to the configurable weights dict
3. Ensure weights sum to 1.0

### Adding New Benchmarks

Challenge registries are YAML-driven. To add a new benchmark:
1. Create `configs/challenges/<name>.yaml` with challenge definitions
2. Run `open-ctf-challenges setup --registry configs/challenges/<name>.yaml`
3. No changes to training pipeline, reward function, or tool schemas

### Adding Custom Agents (BYO Agent)

Open CTF supports two agent protocols depending on the integration point:

1. **StepAgent** (GRPO training): Set `online_rl.agent_class` in your training config to a dotted path to your class. The agent handles tool parsing and execution while SkyRL owns generation.
2. **CTFAgent** (eval/GEPA): Pass `--agent custom:module.MyAgent` to `open-ctf-eval` or `open-ctf-train gepa`.
3. **Runtime bridge** (external frameworks): Use `OPEN_CTF_AGENT_MODE=native` with an adapter script. See `examples/adapters/template_runtime_adapter.py`.

For the complete BYO agent guide, see `src/open_ctf/agent/README.md`.
