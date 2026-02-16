# Data Collection Guide

This guide shows how to collect **real training data** from BoxPwnr agent traces solving CyBench challenges.

## Overview

The sample data (`data/sample/*.jsonl`) contains only 20 SFT and 16 GRPO examples for testing the pipeline. For production training, you need to collect hundreds or thousands of traces by running BoxPwnr against CyBench's 40 professional CTF challenges.

**Pipeline:**
```
BoxPwnr Agent → CyBench Challenges → Raw Traces → Converter → Training Data
```

## Prerequisites

1. **BoxPwnr installed** (already in `references/boxpwnr/` after setup)
2. **CyBench benchmarks cloned** (in `benchmarks/cybench/`)
3. **Docker running** (for challenge containers)
4. **LLM API access** (GPT-4, Claude Sonnet, or local models)

## Step 1: Setup CyBench

```bash
# Clone CyBench if not already done
cd benchmarks
git clone https://github.com/andyzorigin/cybench.git cybench
cd ..

# Verify BoxPwnr has CyBench platform
cd references/boxpwnr
uv sync  # Install dependencies
uv run boxpwnr --platform cybench --list
```

This should list 40 challenges across categories:
- **Crypto**: Cryptography challenges
- **Web**: Web security (SQLi, IDOR, XSS, etc.)
- **Pwn**: Binary exploitation
- **Reversing**: Reverse engineering
- **Forensics**: Digital forensics
- **Misc**: Miscellaneous
- **Blockchain**: Smart contract security

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

### Batch Collection (Recommended)

Create a challenge list for systematic data collection:

```bash
# challenges.txt - Start with Very Easy and Easy challenges
cat > challenges.txt << 'EOF'
benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Very Easy] Dynastic
benchmark/hackthebox/cyber-apocalypse-2024/crypto/[Very Easy] Primary Knowledge
benchmark/hackthebox/cyber-apocalypse-2024/misc/[Very Easy] Stop Drop and Roll
benchmark/hackthebox/cyber-apocalypse-2024/web/[Easy] TimeKORP
benchmark/project-sekai-ctf/sekaictf-2023/web/chunky
EOF

# Run batch collection
uv run boxpwnr --platform cybench \
  --targets-file challenges.txt \
  --model gpt-4o \
  --max-cost 10.0 \
  --output-dir ../../targets/cybench_batch_001
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
- ✅ 10-100 messages (multi-turn problem solving)
- ✅ Mix of reasoning + tool calls (shell, python, etc.)
- ✅ `success: true` and `flag_found: true` for SFT data
- ✅ Both successes AND failures for GRPO data

## Step 4: Convert Traces to Training Data

### Convert Successful Traces (SFT)

```bash
cd ../..  # Back to open-ctf-env root

# Convert all successful CyBench traces
open-ctf-convert \
  --input targets/cybench_batch_001/ \
  --output data/sft_cybench.jsonl \
  --success-only \
  --dedup

# Check output
wc -l data/sft_cybench.jsonl
head -1 data/sft_cybench.jsonl | jq .
```

### Convert All Traces (SFT + GRPO)

```bash
# Successful traces → SFT
# All traces → GRPO (failures help with exploration)
open-ctf-convert \
  --input targets/cybench_batch_001/ \
  --output data/sft_cybench.jsonl \
  --output-failure data/grpo_all_cybench.jsonl \
  --dedup
```

### Split into SFT and GRPO Datasets

```bash
# Split based on length and complexity
open-ctf-split \
  --input data/sft_cybench.jsonl \
  --sft-output data/sft_final.jsonl \
  --grpo-output data/grpo_final.jsonl \
  --max-grpo-tokens 32768

# GRPO dataset will contain:
# - Long multi-turn trajectories (good for efficiency learning)
# - Ground truth flags for reward computation
# - Optimal step counts
```

## Step 5: Validate Data Format

```bash
# Validate ChatML format and tool calls
open-ctf-validate

# Check sample records
head -1 data/sft_final.jsonl | jq '.messages[0:3]'
head -1 data/grpo_final.jsonl | jq '.ground_truth_flag'
```

## Data Collection Strategies

### Strategy 1: Difficulty Progression

Collect data in order of increasing difficulty:

```bash
# Phase 1: Very Easy (quick wins, build dataset foundation)
# Expected success rate: 60-80%
# Collect: 50-100 traces

# Phase 2: Easy (more complex, multi-step)
# Expected success rate: 40-60%
# Collect: 50-100 traces

# Phase 3: Medium (challenging, diverse techniques)
# Expected success rate: 20-40%
# Collect: 30-50 traces

# Phase 4: Hard (expert-level, include failures for GRPO)
# Expected success rate: 5-20%
# Collect: 20-30 traces
```

### Strategy 2: Category Diversity

Ensure coverage across vulnerability types:

```bash
# Web (30%): SQLi, XSS, IDOR, SSRF, LFI, RCE
# Crypto (20%): Classical crypto, modern crypto
# Pwn (20%): Buffer overflow, ROP, heap exploitation
# Reversing (15%): Static/dynamic analysis
# Misc (10%): Steganography, OSINT, scripting
# Forensics (5%): Memory, disk, network forensics
```

### Strategy 3: Model Diversity

Use different models for different challenges:

```bash
# GPT-4o: Web, crypto (better reasoning)
# Claude Sonnet: Pwn, reversing (better code analysis)
# Claude Haiku: Misc, forensics (bulk collection)
# Local models: Retry failed challenges (exploration)
```

## Expected Dataset Sizes

| Training Stage | Minimum | Recommended | Optimal |
|----------------|---------|-------------|---------|
| **SFT** | 100 traces | 500 traces | 1,000+ traces |
| **GRPO** | 50 trajectories | 200 trajectories | 500+ trajectories |

**For conference demo/paper:**
- Minimum viable: 200 SFT + 100 GRPO (~$100-200 in API costs)
- Production quality: 1,000 SFT + 500 GRPO (~$500-1,000 in API costs)

## Troubleshooting

### BoxPwnr Challenge Fails Immediately

```bash
# Check Docker is running
docker ps

# Try with --keep-container to debug
uv run boxpwnr --platform cybench \
  --target "[Very Easy] Dynastic" \
  --model gpt-4o \
  --keep-container \
  --debug
```

### Trace Conversion Fails

```bash
# Check conversation.json structure
cat targets/cybench/[challenge]/conversation.json | jq '.messages[0]'

# Validate it has required fields
cat targets/cybench/[challenge]/stats.json | jq '{success, flag_found, optimal_steps}'
```

### Empty Dataset After Conversion

```bash
# Check input directory has traces
ls -la targets/cybench_batch_001/

# Check for success-only filter
open-ctf-convert --input targets/ --output data/test.jsonl
# (without --success-only to include failures)
```

## Real-World Example

Here's a complete workflow for collecting 100 high-quality traces:

```bash
# 1. Setup
cd references/boxpwnr
uv sync

# 2. Create challenge list (select 100 diverse challenges)
uv run boxpwnr --platform cybench --list > all_challenges.txt
# Edit to select 100 challenges across categories/difficulties

# 3. Run batch collection (overnight job)
nohup uv run boxpwnr --platform cybench \
  --targets-file selected_100.txt \
  --model claude-3-5-haiku-20241022 \
  --max-cost 50.0 \
  --output-dir ../../targets/cybench_production \
  > batch_run.log 2>&1 &

# 4. Monitor progress
tail -f batch_run.log

# 5. Convert successful traces
cd ../..
open-ctf-convert \
  --input targets/cybench_production/ \
  --output data/sft_production.jsonl \
  --success-only \
  --dedup

# 6. Split for training
open-ctf-split \
  --input data/sft_production.jsonl \
  --sft-output data/sft_train.jsonl \
  --grpo-output data/grpo_train.jsonl

# 7. Train
open-ctf-train sft --data data/sft_train.jsonl --output outputs/sft
open-ctf-train grpo --model outputs/sft/final --data data/grpo_train.jsonl --output outputs/grpo
```

## Next Steps

After collecting data:

1. [Training Guide](training.md) - Train your model with SFT + GRPO
2. [Deployment Guide](deployment.md) - Deploy the trained model
3. [Architecture](architecture.md) - Understand the pipeline internals

## Resources

- **CyBench Paper**: https://arxiv.org/abs/2408.08926
- **CyBench Repository**: https://github.com/andyzorigin/cybench
- **BoxPwnr Documentation**: https://github.com/0ca/BoxPwnr
- **Open CTF Environment**: https://github.com/westonbrown/open-ctf-env
