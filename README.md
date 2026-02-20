# Open CTF Environment

[![Version](https://img.shields.io/badge/version-0.4.0-blue)](https://github.com/westonbrown/open-ctf-env)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

An open-source pipeline for **post-training security LLMs on CTF challenge trajectories**. Collect agent traces with [BoxPwnr](https://github.com/0ca/BoxPwnr), fine-tune with SFT + GRPO, evaluate on [CyBench](https://cybench.github.io/), and deploy locally via GGUF quantization.

> Presented at **[un]prompted — The AI Security Practitioner Conference**
> March 3-4, 2026 | Salesforce Tower, San Francisco

## Thesis

Base open-weight models understand security concepts but cannot execute multi-step exploits. A 24B model can plan a 5-phase attack but fails to enumerate user IDs. A 20B model gets stuck thinking on step 1. We investigate whether **trajectory-aware post-training** (SFT on expert traces, then GRPO with a multi-signal CTF reward function) can close this plan-execute gap — producing a locally deployable security agent from [GLM-4.7-Flash](https://huggingface.co/THUDM/GLM-4.7-Flash) (30B MoE, ~3.6B active parameters).

## How It Works

```mermaid
flowchart LR
    subgraph collect["1. Collect Traces"]
        scaffold["BoxPwnr"] -- "prompt + tools" --> llm["Base Model"]
        llm -- "tool calls" --> scaffold
        scaffold -- "shell, python, files" --> challenges["CTF\nTargets"]
        challenges -- "stdout, flags" --> scaffold
    end

    subgraph convert["2. Convert"]
        traces["conversation.json\n+ stats.json"] --> converter["BoxPwnrConverter"]
        converter --> sft_data["SFT Data\n(441 successes)"]
        converter --> grpo_data["GRPO Data\n(779 traces + flags)"]
    end

    subgraph train["3. Fine-Tune (Three Stages)"]
        sft["SFT\nFormat & Domain"] --> merge["Merge\nLoRA"]
        merge --> grpo["Offline GRPO\nStatic Scoring"]
        grpo --> online["Online RL\nVERL Gym Eval"]
    end

    subgraph deploy["4. Evaluate + Deploy"]
        eval_model["Fine-Tuned\nCTF Agent"] --> eval_bench["CyBench Eval"]
        eval_model --> export["GGUF Export"]
    end

    collect --> convert --> train
    online --> deploy
    online -. "VERL trajectory feedback" .-> convert
```

The same scaffold (BoxPwnr) runs both the baseline and fine-tuned models against identical challenges. The only variable is the model weights — architecture, tools, and evaluation harness are held constant. Successful trajectories feed back as training data for the next iteration (rejection sampling).

GRPO runs in two modes:
1. **Offline GRPO** (via Unsloth + TRL): Generate completions from static prompts, score with CTFReward for fast iteration.
2. **Online RL** (via VERL): Execute tool calls dynamically against live Docker challenges via the `VerlRolloutEnvironment`. The agent reasons and interacts with real `stdout`, driven by a distributed Ray cluster architecture.

*Note: The environment and tooling are completely agnostic. You can easily swap out base models (e.g., from GLM-4.7 to Qwen or Devstral) or change hyperparameters dynamically without rewriting the integration layers.*

## Baseline Results

GLM-4.7-Flash Q8_0 (30B MoE, ~3.6B active) evaluated on [CyBench](https://cybench.github.io/) 40-challenge suite via BoxPwnr on NVIDIA DGX Spark (GB10). 40 turns max, 161 total runs across retries.

| Model | CyBench Solve Rate | Avg Turns (solved) | Avg Time (solved) |
|-------|-------------------|-----------|----------|
| GLM-4.7-Flash Q8_0 (base) | **7/40 (17.5%)** | 10.9 | 3m 54s |
| + SFT | TBD | TBD | TBD |
| + SFT + GRPO | TBD | TBD | TBD |

### By Difficulty

| Difficulty | Solved | Rate |
|------------|--------|------|
| Very Easy | 5/8 | 62% |
| Easy | 2/12 | 17% |
| Medium | 0/16 | 0% |
| Hard | 0/4 | 0% |

### By Category

| Category | Solved | Rate | Notes |
|----------|--------|------|-------|
| Forensics | 2/4 | 50% | LootStash, Urgent |
| Misc | 2/10 | 20% | Flag Command, eval-me |
| Web | 1/6 | 17% | avatar |
| Crypto | 2/15 | 13% | Dynastic, Primary Knowledge |
| Rev | 0/5 | 0% | — |

### Solved Challenges

| Challenge | Category | Turns | Time | Input Tokens |
|-----------|----------|-------|------|-------------|
| Flag Command | misc | 6 | 0:39 | 40K |
| Primary Knowledge | crypto | 4 | 0:54 | 14K |
| LootStash | forensics | 13 | 1:03 | 97K |
| avatar | web | 9 | 1:12 | 27K |
| Urgent | forensics | 10 | 3:03 | 78K |
| Dynastic | crypto | 3 | 7:24 | 8K |
| eval-me | misc | 31 | 13:08 | 1.2M |

### Failure Modes (33 unsolved)

| Mode | Count | Description |
|------|-------|-------------|
| Hit turn limit | 16 | Exhausted 40 turns without finding flag |
| Incomplete run | 13 | Process terminated before completion (Docker/infra) |
| API error | 2 | Model API failures mid-run |
| Early stall | 2 | Stalled within first 2 turns |

### Key Observations

- **Difficulty cliff at Medium.** 62% solve rate on Very Easy drops to 0% on Medium+. The model can follow simple exploitation paths but lacks multi-step reasoning for complex challenges.
- **Crypto is the weakest category** (13%) despite 15 challenges. Most crypto challenges require implementing custom solvers — the model writes Python code but makes algorithmic errors.
- **Failed runs use 2x more tokens** than solved runs (372K vs 216K avg input). The model spends tokens on unproductive exploration rather than converging on the exploit.
- **93% command success rate** (819/878). Tool execution isn't the bottleneck — strategy is.
- **It Has Begun** (Very Easy forensics) is a near-miss: the model extracted both flag halves but tried 8 wrong combinations without getting the format right.

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.4+ with CUDA support
- Docker and Docker Compose
- NVIDIA GPU with 24GB+ VRAM (60GB+ for GLM-4.7-Flash BF16 LoRA)

### Setup

### Setup

You can either install the package locally or use our **Unified Dockerfile** which bundles SFT, Offline GRPO, and Online VERL RL flawlessly.

**Option A: Local Install**
```bash
git clone https://github.com/westonbrown/open-ctf-env.git
cd open-ctf-env

# Install PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install open-ctf-env with full training and VERL online RL dependencies
pip install -e ".[dev,train,rl]"

# Setup references
git clone https://github.com/0ca/BoxPwnr.git references/boxpwnr
git clone https://github.com/verl-project/verl.git references/verl
cp env.example .env
```

**Option B: Unified Docker (Recommended for DGX/Heavy Setup)**
```bash
git clone https://github.com/westonbrown/open-ctf-env.git
cd open-ctf-env
docker build -t open-ctf-env:latest .
```

### Generate Training Data

```bash
# Convert BoxPwnr traces to training format
open-ctf-convert \
    --input /path/to/BoxPwnr-Traces \
    --output data/all_traces.jsonl \
    --output-failure data/failed_traces.jsonl \
    --dedup

# Split into SFT (successes) + GRPO (all traces with flags)
cat data/all_traces.jsonl data/failed_traces.jsonl > data/combined.jsonl
open-ctf-split \
    --input data/combined.jsonl \
    --sft-output data/sft.jsonl \
    --grpo-output data/grpo.jsonl
```

### Train

```bash
```bash
# Stage 1: SFT (tool format + domain knowledge)
open-ctf-train sft \
    --model unsloth/GLM-4.7-Flash \
    --data data/sft.jsonl \
    --output outputs/sft

# Merge LoRA adapter into base
open-ctf-train merge \
    --adapter outputs/sft/final \
    --output outputs/sft/merged

# Stage 2: Offline GRPO (exploitation efficiency)
open-ctf-train grpo \
    --model outputs/sft/merged \
    --data data/grpo.jsonl \
    --output outputs/grpo
    
# Stage 3: Online Frontier RL (VERL)
# Evaluates and updates the model interactively against live BoxPwnr containers. 
# Easily swap algorithms (ppo, grpo) and models dynamically:
docker run --gpus all -v /var/run/docker.sock:/var/run/docker.sock \
    open-ctf-env:latest python3 src/open_ctf/training/online_rl.py
```

### Evaluate

```bash
# Compare base vs fine-tuned on CyBench
open-ctf-eval \
    --model outputs/grpo/final \
    --baseline unsloth/GLM-4.7-Flash \
    --challenges cybench
```

### Deploy

```bash
# Export to GGUF
open-ctf-export \
    --adapter outputs/grpo/final \
    --base-model unsloth/GLM-4.7-Flash \
    --output models/ctf-agent.gguf \
    --quant Q4_K_M

# Serve with Ollama
echo 'FROM ./models/ctf-agent.gguf
PARAMETER num_ctx 32768' > Modelfile
ollama create ctf-agent -f Modelfile
```

### Validate Pipeline (no GPU needed)

```bash
open-ctf-validate
```

## Training Data

Data is generated from [BoxPwnr-Traces](https://github.com/0ca/BoxPwnr-Traces) — real agent trajectories across 6 CTF platforms, collected by running frontier models (Claude Sonnet 4.5, GPT-5, Grok 4, Gemini 3) against Dockerized challenges.

| Dataset | Traces | Size | Description |
|---------|--------|------|-------------|
| `data/sft.jsonl` | 441 | 37MB | Successful solves — expert demonstrations for SFT |
| `data/grpo.jsonl` | 779 | 46MB | All traces with cross-referenced flags for GRPO |

**Sources:** HackTheBox (518), PicoCTF (393), PortSwigger (358), CyBench (142), HackBench (3) — 1,414 raw traces total. After dedup and filtering, 779 remain (441 successes). Each trace is a multi-turn conversation (avg 74 messages) with structured tool calls in ChatML format. The dataset grows iteratively via rejection sampling.

### Data Flow

```mermaid
flowchart LR
    sources["1,414 Raw Traces\nHTB · PicoCTF · PortSwigger\nCyBench · HackBench"] --> conv["BoxPwnrConverter\n(lossless)"]
    conv --> dedup["Dedup +\nnormalize"]
    dedup --> flags["Cross-reference\nflags"]
    flags --> sft["SFT: 441\n(success only)"]
    flags --> grpo["GRPO: 779\n(all + flags)"]
```

## Reward Function

The CTF reward for GRPO training uses **6 signals + 1 penalty**. All process signals are **ungated** — they provide gradient signal in offline GRPO where the model generates completions without environment interaction and rarely captures the exact flag.

| Signal | Weight | Description |
|--------|--------|-------------|
| **Flag Capture** | 0.20 | `metadata.success` > exact match > pattern match (0.1) |
| **Efficiency** | 0.25 | `min(optimal / actual, 1.0)` |
| **Format** | 0.20 | Valid tool call JSON structure |
| **Progression** | 0.15 | RECON → ENUM → EXPLOIT phase ordering |
| **Exploration** | 0.10 | Novel tool usage weighted toward early trajectory |
| **Uniqueness** | 0.10 | Command diversity (detects stuck loops) |
| **Hallucination** | -0.10 | Penalty for `flag_found` calls with wrong flag |

## Training Configuration

Edit `src/open_ctf/configs/training.yaml`. Key settings:

| Parameter | SFT | GRPO |
|-----------|-----|------|
| Model | `unsloth/GLM-4.7-Flash` | SFT merged output |
| LoRA rank | 64 | 64 |
| Learning rate | 2e-4 | 5e-6 |
| Epochs | 3 | 1 |
| Loss | Cross-entropy | DAPO |
| Packing | Yes (3x throughput) | N/A |
| vLLM generation | N/A | 8 completions/prompt |

**Hardware:** H100/H200 recommended for training. Q4_K_M GGUF inference fits in ~15GB (RTX 4090). Single-GPU configs available (`training_dgx.yaml`).

## Architecture

### Project Structure

```
open-ctf-env/
├── data/                        # Training data (generated)
│   ├── sft.jsonl                # 441 successful traces
│   └── grpo.jsonl               # 779 traces with flags
├── src/open_ctf/
│   ├── cli/                     # CLI entry points (train, convert, split, etc.)
│   ├── configs/                 # training.yaml, training_dgx.yaml
│   ├── data/                    # Trace converter + dataset splitter
│   │   ├── converter.py         # BoxPwnr → ChatML (lossless, 13 tools)
│   │   └── splitter.py          # Success → SFT, All → GRPO
│   ├── rewards/reward.py        # CTFReward (6 signals + penalty)
│   ├── training/
│   │   ├── sft.py               # SFTTrainer (Unsloth + HF fallback)
│   │   ├── grpo.py              # GRPOTrainer (offline modes)
│   │   ├── tools.py             # 13 TRL tool wrappers for offline validations
│   │   └── online_rl.py         # VERL Rollout Environment for live Docker Gym interaction
│   └── openenv/                 # OpenEnv server (online GRPO)
│       ├── server.py            # HTTP environment server (13 tools)
│       └── models.py            # Action, Observation, State dataclasses
├── scripts/
│   ├── run_cybench_benchmark.py # Full CyBench benchmark runner
│   └── spawn_all_cybench.py     # Docker setup for all 40 challenges
├── tests/
│   ├── test_rewards.py          # Reward function tests
│   └── test_openenv.py          # OpenEnv integration tests
└── references/
    ├── boxpwnr/                 # BoxPwnr agent framework
    └── OpenEnv/                 # OpenEnv RL environment framework
```

## CLI Commands

| Command | Purpose |
|---------|---------|
| `open-ctf-train` | SFT, GRPO, and LoRA merge |
| `open-ctf-convert` | Convert BoxPwnr traces to training format |
| `open-ctf-split` | Split datasets into SFT and GRPO sets |
| `open-ctf-agent` | Run agent against CyBench challenges |
| `open-ctf-eval` | Evaluate and compare models on CyBench |
| `open-ctf-validate` | Validate pipeline without GPU |
| `open-ctf-export` | Export LoRA adapter to GGUF |

## BoxPwnr Tool Set

Training data, the reward function, the OpenEnv server, and the TRL tool wrappers all share the same 13-tool vocabulary. Every tool the model learns during SFT is available for execution during online GRPO — no tool gap between training and inference.

| Tier | Tools | Description |
|------|-------|-------------|
| **Execution** | `shell_command`, `exec_command`, `write_stdin`, `python_code`, `execute_command` | Shell scripts, interactive PTY sessions, Python |
| **File Ops** | `read_file`, `grep`, `file_search`, `apply_patch` | Read, search, patch files in the container |
| **Meta** | `flag_found`, `web_search`, `list_sessions`, `close_session` | Flag submission, web search, session management |

The converter maps legacy tmux tools from older traces to their PTY equivalents (`tmux_send_and_read` → `write_stdin`, `tmux_cancel_command` → `close_session`).

## Roadmap

### Phase 1: Pipeline + Infrastructure (Done)
- [x] Lossless trace converter (tool-calling + chat-command formats)
- [x] Training data: 441 SFT + 779 GRPO traces from BoxPwnr across 6 platforms
- [x] 3-stage training pipeline: SFT + Offline GRPO + Online RL
- [x] Multi-signal CTF reward function (6 signals + hallucination penalty)
- [x] OpenEnv server + VERL Rollout integration for interactive Gym operations
- [x] TRL prefix-preserving patch for GLM-4.7-Flash
- [x] BoxPwnr agent integration with 13 native security tools
- [x] CyBench benchmark runner with per-challenge metrics
- [x] GGUF export pipeline
- [x] Validation pipeline (`open-ctf-validate`)
- [x] Unified Dockerfile mapping bitsandbytes, ray, and unsloth for DGX Spark

### Phase 2: Baseline + Train + Evaluate (In Progress)

**Baseline Collection**
- [x] CyBench 40-challenge baseline (GLM-4.7-Flash Q8_0 via BoxPwnr) — 7/40 solved (17.5%), 161 traces across retries
- [ ] Collect new traces from baseline to supplement training data

**Train**
- [ ] Stage 1: SFT (441 success traces, BF16 LoRA, 3 epochs)
- [ ] Merge LoRA adapter
- [ ] Stage 2: Offline GRPO (779 traces with flags, DAPO, 8 generations)
- [ ] Stage 3: Online RL (VERL) against live BoxPwnr Docker challenges

**Evaluate**
- [ ] Compare base vs SFT vs GRPO vs VERL on same CyBench 40-challenge suite
- [ ] Ablation: reward signal contribution analysis

**Release (Target: March 3)**
- [ ] Export final model to GGUF, validate on single-GPU hardware
- [ ] Publish results table
- [ ] Upload weights to HuggingFace
- [ ] Tag v1.0.0 release

## Contributing

```bash
pip install -e ".[dev]"
pytest tests/ -v
open-ctf-validate  # No GPU needed
```

## Related Work

- [CyBench](https://cybench.github.io/) — Cybersecurity benchmark, 40 challenges, ICLR 2025 Oral ([paper](https://arxiv.org/abs/2408.08926), [repo](https://github.com/andyzorigin/cybench))
- [BoxPwnr](https://github.com/0ca/BoxPwnr) — LLM-powered CTF solver (our data collection and evaluation engine)
- [OpenEnv](https://github.com/OpenEnvs/OpenEnv) — Gymnasium-style RL environments for LLM agents (online GRPO backend)
- [Unsloth](https://github.com/unslothai/unsloth) — Efficient fine-tuning with MoE Grouped GEMM
- [TRL](https://github.com/huggingface/trl) — Transformer Reinforcement Learning (GRPOTrainer + DAPO)
- [DeepSeek R1](https://arxiv.org/abs/2501.12948) — SFT → GRPO pipeline inspiration
- [Dreadnode Worlds](https://dreadnode.io/blog/worlds) — "Reasoning traces are the critical delta" finding

## License

MIT License — See [LICENSE](./LICENSE) for details.
