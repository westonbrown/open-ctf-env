# GEPA Prompt Evolution — Flag Command

End-to-end example of **Stage 3 (GEPA)** in the Open CTF training pipeline.

GEPA evolves the agent's system prompt — no weight updates — by reflecting on
execution traces and using Pareto selection across challenges.

```
SFT (weights) -> GRPO (weights) -> GEPA (prompt only) -> Deploy
```

## What this example does

Runs GEPA against a single HackTheBox **"[Very Easy] Flag Command"** web
challenge. The model must discover a multi-step chain:

```
curl HTML page  ->  find JS import  ->  read main.js  ->
  discover /api/options  ->  find secret command  ->  flag
```

GEPA evolves the seed prompt so the model reliably follows this chain.

## Prerequisites

```bash
# 1. Install GEPA dependencies
pip install -e ".[gepa]"

# 2. Start vLLM with any supported model
vllm serve /path/to/model --port 8001 --dtype bfloat16 \
  --gpu-memory-utilization 0.50 --trust-remote-code

# 3. Start the challenge container
open-ctf-challenges setup --challenge '[Very Easy] Flag Command'
```

## Quick start

All commands run from the repo root.

**Shell** (handles env vars and preflight automatically):
```bash
bash examples/gepa_flag_command/run.sh --model openai/<your-model-id>
```

**Python** (programmatic API, same result):
```bash
export OPENAI_API_BASE=http://localhost:8001/v1
export OPENAI_API_KEY=dummy

python examples/gepa_flag_command/run.py --model openai/<your-model-id>
```

**CLI** (direct):
```bash
OPENAI_API_BASE=http://localhost:8001/v1 \
OPENAI_API_KEY=dummy \
open-ctf-train gepa \
    --model openai/<your-model-id> \
    --data examples/gepa_flag_command/challenge.jsonl \
    --output outputs/gepa_flag_command \
    --budget light \
    --challenge-registry configs/challenges/cybench.yaml
```

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | — | Model ID for `dspy.LM` (e.g. `openai/qwen35-27b`) |
| `--budget` | `light` | GEPA iterations: `light` / `medium` / `heavy` |
| `--port` / `--vllm-port` | `8001` | vLLM server port |
| `--target-port` | `32810` | Challenge container port |
| `--output` | `outputs/gepa_flag_command` | Output directory |

## Output

```
outputs/gepa_flag_command/
  optimized_prompt.txt   # The evolved system prompt
  gepa_results.json      # Scores per candidate
  gepa_logs/             # Full optimizer traces
  optimized_agent/       # Saved DSPy module (reusable)
```

## How GEPA works

1. Evaluate agent on challenges with the seed prompt, score with CTFReward
2. Reflection LM analyzes traces and proposes improved instructions
3. New candidate prompts evaluated on next minibatch
4. Pareto selection keeps prompts best on at least one challenge
5. Repeat until budget exhausted

Both agent and reflection LMs point at the same vLLM server (different
temperature settings). No second server or cloud API needed.

## Files

| File | Purpose |
|------|---------|
| `challenge.jsonl` | Single-challenge JSONL data (Flag Command) |
| `run.sh` | Shell wrapper with preflight checks and env setup |
| `run.py` | Python script using the `run_gepa()` API directly |
