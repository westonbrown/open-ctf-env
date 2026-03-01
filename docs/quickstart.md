# Quick Start

Get up and running with Open CTF Environment in minutes.

## Prerequisites

- Docker and Docker Compose
- If running inside another container (for example RunPod), Docker must have privileges to create networks and run nested containers (`docker network create` and `docker run` must work)
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An LLM backend (Ollama, llama.cpp, vLLM, or any OpenAI-compatible API)

## Installation

```bash
git clone https://github.com/westonbrown/open-ctf-env.git
cd open-ctf-env

# Install core only
pip install -e .

# For SFT training (TRL)
pip install -e ".[sft]"

# For GRPO training (SkyRL + Ray)
pip install -e ".[grpo]"

# For GEPA prompt optimization (DSPy)
pip install -e ".[gepa]"

# For agent runner
pip install -e ".[agent]"
```

## Setup

```bash
# Clone BoxPwnr reference (agent framework with CyBench support)
git clone https://github.com/0ca/BoxPwnr.git references/boxpwnr

# Clone CyBench benchmarks (40 professional CTF challenges)
git clone https://github.com/andyzorigin/cybench.git benchmarks/cybench

# Copy environment config
cp env.example .env
# Edit .env with your API keys and preferences
```

## Verify Installation

```bash
open-ctf-validate
```

This checks data format, reward functions, training scripts, tool registry, and model formatters without requiring a GPU.

## Common Workflows

### Run the Agent

```bash
# Against a CyBench challenge
open-ctf-agent \
    --platform cybench \
    --target "[Very Easy] Dynastic" \
    --model openrouter/openai/gpt-oss-120b \
    --max-turns 30

# With a local Ollama model
open-ctf-agent \
    --platform cybench \
    --target "[Easy] TimeKORP" \
    --model ollama/qwen3:8b
```

### Convert Traces to Training Data

```bash
# Convert and split traces
open-ctf-convert \
    --input targets/ \
    --output data/all_traces.jsonl \
    --output-failure data/failed_traces.jsonl \
    --dedup

cat data/all_traces.jsonl data/failed_traces.jsonl > data/combined.jsonl
open-ctf-split --input data/combined.jsonl
```

### Train a Model (3-Stage Pipeline)

```bash
# Stage 1: SFT via TRL
open-ctf-train sft \
    --model Nanbeige/Nanbeige4.1-3B \
    --data data/sft.jsonl \
    --output outputs/sft

# Merge LoRA adapter into base model
open-ctf-train merge \
    --adapter outputs/sft \
    --base-model Nanbeige/Nanbeige4.1-3B \
    --output outputs/sft-merged

# Stage 2: Online GRPO via SkyRL
open-ctf-train rl \
    --model outputs/sft-merged \
    --data data/online_rl.jsonl \
    --output outputs/online_rl \
    --config configs/training/training.yaml

# Stage 3: GEPA prompt optimization (no weight updates)
open-ctf-train gepa \
    --model openai/ctf-agent \
    --data data/online_rl.jsonl \
    --output outputs/gepa \
    --reflection-model openai/ctf-reflection \
    --challenge-registry configs/challenges/cybench.yaml
```

Note: `open-ctf-train rl` runs a preflight validation gate and, by default, requires `<data>.manifest.json` produced by `src/open_ctf/cli/generate_online_rl.py`.

### Export for Deployment

```bash
open-ctf-export \
    --adapter outputs/online_rl/final \
    --base-model Nanbeige/Nanbeige4.1-3B \
    --output models/ctf-agent.gguf \
    --quant Q4_K_M
```

## Docker Workflows

```bash
# Stage 1: SFT
docker compose run --rm sft

# Stage 1: SFT (TRL backend for newer models, e.g. Qwen3.5)
docker compose run --rm sft-trl

# Merge LoRA
docker compose run --rm merge

# Stage 2: Online GRPO
docker compose run --rm grpo

# Validate pipeline
docker compose run --rm validate

# Export to GGUF
docker compose run --rm export
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPEN_CTF_PROVIDER` | Model provider | `ollama` |
| `OPEN_CTF_MODEL` | LLM model ID | `ollama/qwen3:8b` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OPEN_CTF_OUTPUT_DIR` | Output directory | `./outputs` |

## Next Steps

- [Training Guide](training.md) -- Full 3-stage pipeline details
- [Deployment Guide](deployment.md) -- Deploy trained models
- [Data Collection Guide](data-collection.md) -- Collect real training traces
- [Architecture](architecture.md) -- Module overview and data flow
