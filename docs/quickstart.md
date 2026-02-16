# Quick Start

Get up and running with Open CTF Environment in minutes.

## Prerequisites

- Docker and Docker Compose
- Python 3.10+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- An LLM backend (Ollama, llama.cpp, vLLM, or any OpenAI-compatible API)

## Installation

```bash
git clone https://github.com/westonbrown/open-ctf-env.git
cd open-ctf-env

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .

# For training with Unsloth
uv pip install -e ".[train]"

# For agent runner
uv pip install -e ".[agent]"
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

## Run the Demo

```bash
./scripts/demo/run_demo.sh
```

This starts a challenge, runs the agent, and shows results in one command.

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
open-ctf-convert \
    --input targets/ \
    --output data/sft_train.jsonl \
    --success-only --dedup
```

### Train a Model

```bash
# SFT stage
open-ctf-train sft \
    --model unsloth/GLM-4.7-Flash \
    --data data/sft_train.jsonl \
    --output outputs/sft

# GRPO stage (after SFT)
open-ctf-train grpo \
    --model outputs/sft/final \
    --data data/grpo_train.jsonl \
    --output outputs/grpo
```

### Export for Deployment

```bash
open-ctf-export \
    --adapter outputs/sft/final \
    --base-model unsloth/GLM-4.7-Flash \
    --output models/ctf-agent.gguf \
    --quant Q4_K_M
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPEN_CTF_PROVIDER` | Model provider | `ollama` |
| `OPEN_CTF_MODEL` | LLM model ID | `ollama/qwen3:8b` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OPEN_CTF_OUTPUT_DIR` | Output directory | `./outputs` |

## Next Steps

- [Training Guide](training.md) - Full 2-stage pipeline details
- [Deployment Guide](deployment.md) - Deploy trained models
- [Architecture](architecture.md) - Module overview and data flow
