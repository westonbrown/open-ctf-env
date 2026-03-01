# Data Collection Guide

This guide shows how to collect **real training data** from BoxPwnr agent traces solving CyBench challenges.

## Overview

To train a production model, collect hundreds or thousands of traces by running BoxPwnr against CyBench's 40 professional CTF challenges.

**Pipeline:**
```
BoxPwnr Agent → CyBench Challenges → Raw Traces → Converter → Training Data
```

## Prerequisites

1. **BoxPwnr installed** (in `references/boxpwnr/` after setup)
2. **CyBench benchmarks cloned** (in `benchmarks/cybench/`)
3. **Docker running** (for challenge containers)
4. **LLM API access** (GPT-4o, Claude Sonnet, or local models)

## Step 1: Setup CyBench

```bash
# Clone CyBench if not already done
cd benchmarks
git clone https://github.com/andyzorigin/cybench.git cybench
cd ..

# Verify BoxPwnr has CyBench platform
cd references/boxpwnr
uv sync
uv run boxpwnr --platform cybench --list
```

This lists 40 challenges across categories: Crypto, Web, Pwn, Reversing, Forensics, Misc, Blockchain.

## Step 2: Run BoxPwnr on CyBench Challenges

### Single Challenge

```bash
cd references/boxpwnr

# Run a Very Easy challenge (good starting point)
uv run boxpwnr --platform cybench \
  --target "[Very Easy] Dynastic" \
  --model gpt-4o \
  --max-cost 2.0 \
  --keep-container

# Output directory: targets/cybench/[Very Easy] Dynastic/
# Files created:
# - conversation.json  (full agent trace)
# - stats.json         (metadata, success/failure, steps)
# - report.md          (human-readable summary)
```

### Multiple Challenges

```bash
# Run individual challenges sequentially
for target in "[Very Easy] Dynastic" "[Very Easy] Primary Knowledge" "[Easy] TimeKORP"; do
  uv run boxpwnr --platform cybench \
    --target "$target" \
    --model gpt-4o \
    --max-turns 30 \
    --max-cost 2.0
done
```

### Cost Management

| Model | Cost/1M Tokens | Typical Challenge Cost | 100 Challenges |
|-------|----------------|------------------------|----------------|
| GPT-4o | $2.50 / $10.00 | $0.50-$2.00 | $50-$200 |
| Claude Sonnet 4 | $3.00 / $15.00 | $0.75-$3.00 | $75-$300 |
| Claude Haiku | $0.25 / $1.25 | $0.10-$0.50 | $10-$50 |
| Local (Ollama) | $0 | $0 | $0 |

**Tip:** Start with Haiku or local models for bulk collection, then use GPT-4o for harder challenges.

## Step 3: Verify Trace Quality

```bash
# Check trace structure
cat targets/cybench/[Very Easy] Dynastic/conversation.json | jq '.messages | length'

# Check success status
cat targets/cybench/[Very Easy] Dynastic/stats.json | jq '.success'

# Check flag found
cat targets/cybench/[Very Easy] Dynastic/stats.json | jq '.flag_found'
```

**Good trace characteristics:**
- 10-100 messages (multi-turn problem solving)
- Mix of reasoning + tool calls (shell, python, etc.)
- `success: true` and `flag_found: true` for SFT data
- Both successes AND failures for GRPO data

## Step 4: Convert Traces to Training Data

### Convert Successful Traces (SFT)

```bash
cd ../..  # Back to open-ctf-env root

# Convert all successful CyBench traces
open-ctf-convert \
  --input targets/cybench/ \
  --output data/sft_v6.jsonl \
  --success-only \
  --dedup

# Check output
wc -l data/sft_v6.jsonl
head -1 data/sft_v6.jsonl | jq .
```

### Convert All Traces (SFT + GRPO)

```bash
# Successful traces → SFT data
# All traces → GRPO data (failures help with exploration)
open-ctf-convert \
  --input targets/cybench/ \
  --output data/sft_v6.jsonl \
  --output-failure data/grpo_all.jsonl \
  --dedup
```

### Split into SFT and GRPO Datasets

```bash
open-ctf-split \
  --input data/sft_v6.jsonl \
  --sft-output data/sft_final.jsonl \
  --online-rl-output data/online_rl_final.jsonl \
  --max-online-rl-tokens 32768
```

## Step 5: (Optional) Mass Scale with Synthetic Data Generation
Because hitting real Docker containers or Live APIs (Step 2) is expensive and slow, we provide an offline **Synthetic Data Generator** powered by the latest 2026 World State modeling techniques (incorporating Google MapTrace's spatial constraints and Kubernetes fault injections).

Instead of dealing with real exploit latency, you configure `YAML` manifests of the environment, and a Teacher LLM rapidly creates massive, uniquely randomized datasets of successful and failed agent trajectories.

```bash
open-ctf-synthetic-data \
    --config configs/synthetic_data_generation/default.yaml \
    --teacher-model "openrouter/openai/gpt-4o" \
    --num-traces 500 \
    --sft-out data/sft_synthetic.jsonl
```

**Why do this?**
- **Speed**: Over 1,000x faster than rolling out inside a slow Docker container.
- **Topological Logic**: You can enforce `world_state_dynamics.enforce_topology=true` to force agents to navigate VLAN segmentation securely, natively teaching them networking constructs.
- **Data Uniqueness**: Every clone of the environment injects a uniquely generated `FLAG{uuid}` across all mocked files, processes, and network DBs, entirely preventing sequence memorization during training.

## Step 6: Validate Data Format

```bash
# Validate ChatML format and tool calls
open-ctf-validate

# Check sample records
head -1 data/sft_final.jsonl | jq '.messages[0:3]'
head -1 data/online_rl_final.jsonl | jq '.ground_truth_flag'
```

## Data Collection Strategies

### Strategy 1: Difficulty Progression

Collect data in order of increasing difficulty:

| Phase | Difficulty | Expected Success Rate | Target Traces |
|-------|-----------|----------------------|---------------|
| 1 | Very Easy | 60-80% | 50-100 |
| 2 | Easy | 40-60% | 50-100 |
| 3 | Medium | 20-40% | 30-50 |
| 4 | Hard | 5-20% | 20-30 |

### Strategy 2: Category Diversity

Ensure coverage across vulnerability types:

| Category | Target % | Types |
|----------|----------|-------|
| Web | 30% | SQLi, XSS, IDOR, SSRF, LFI, RCE |
| Crypto | 20% | Classical, modern, implementation flaws |
| Pwn | 20% | Buffer overflow, ROP, heap |
| Reversing | 15% | Static/dynamic analysis |
| Misc | 10% | Steganography, OSINT, scripting |
| Forensics | 5% | Memory, disk, network |

### Strategy 3: Model Diversity

Different models have different strengths:

| Model | Best For | Notes |
|-------|----------|-------|
| GPT-4o | Web, crypto | Better reasoning |
| Claude Sonnet | Pwn, reversing | Better code analysis |
| Claude Haiku | Misc, forensics | Cheap bulk collection |
| Local models | Retry failures | Free exploration |

## Expected Dataset Sizes

| Training Stage | Minimum | Recommended | Optimal |
|----------------|---------|-------------|---------|
| **SFT** | 100 traces | 500 traces | 1,000+ traces |
| **GRPO** | 50 trajectories | 200 trajectories | 500+ trajectories |

**For conference demo:** 200 SFT + 100 GRPO (~$100-200 in API costs).
**Production quality:** 1,000 SFT + 500 GRPO (~$500-1,000 in API costs).

## Troubleshooting

**BoxPwnr challenge fails immediately:**
```bash
# Check Docker is running
docker ps

# Try with --keep-container to debug
uv run boxpwnr --platform cybench \
  --target "[Very Easy] Dynastic" \
  --model gpt-4o \
  --keep-container
```

**Trace conversion fails:**
```bash
# Check conversation.json structure
cat targets/cybench/[challenge]/conversation.json | jq '.messages[0]'

# Validate required fields
cat targets/cybench/[challenge]/stats.json | jq '{success, flag_found, optimal_steps}'
```

**Empty dataset after conversion:**
```bash
# Check input directory has traces
ls -la targets/cybench/

# Try without --success-only to include failures
open-ctf-convert --input targets/ --output data/test.jsonl
```

## Next Steps

After collecting data:

1. [Training Guide](training.md) -- Train your model with the 3-stage pipeline
2. [Deployment Guide](deployment.md) -- Deploy the trained model
3. [Architecture](architecture.md) -- Understand the pipeline internals

## Resources

- **CyBench Paper**: https://arxiv.org/abs/2408.08926
- **CyBench Repository**: https://github.com/andyzorigin/cybench
- **BoxPwnr Documentation**: https://github.com/0ca/BoxPwnr
- **BoxPwnr Traces**: https://github.com/0ca/BoxPwnr-Traces
- **Open CTF Environment**: https://github.com/westonbrown/open-ctf-env
