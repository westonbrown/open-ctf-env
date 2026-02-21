# Training Data

Generated from [BoxPwnr-Traces](https://github.com/0ca/BoxPwnr-Traces) via `open-ctf-convert` + `open-ctf-split`, then filtered for training quality.

## Datasets

| File | Traces | Size | Description |
|------|--------|------|-------------|
| `sft.jsonl` | 1,120 | 97MB | Successful solves for SFT |
| `grpo.jsonl` | 1,369 | 82MB | All traces with cross-referenced flags for GRPO |

## Platform Distribution

| Platform | SFT | GRPO |
|----------|-----|------|
| HackTheBox | 388 | 538 |
| XBOW | 231 | 247 |
| PortSwigger | 178 | 130 |
| PicoCTF | 154 | 160 |
| CyBench | 127 | 182 |
| 2712 (HTB CTF) | 22 | 52 |
| TryHackMe | 17 | 57 |
| HackBench | 3 | 3 |

## Filters Applied

Three filters are applied after conversion and splitting to remove samples that would degrade training:

1. **No-assistant traces removed (GRPO only)** -- 242 entries with only `[system, user]` messages (no agent actions). These are failed runs where the agent never started and provide no trajectory signal for GRPO generation.

2. **Token outliers removed (>100K estimated tokens)** -- 13 SFT entries with extremely long tool outputs (worst: ~1.4M estimated tokens from a single 5.4M-character shell output). These exceed any reasonable `max_seq_length` and waste compute on truncated data.

3. **Placeholder flag removed** -- 1 GRPO entry with `ground_truth_flag: "FLAG{placeholder}"` that would poison the reward signal.

## Flag Quality

| Metric | SFT | GRPO |
|--------|-----|------|
| Real flags | 1,090 (97.3%) | 1,265 (92.4%) |
| CHECK (PortSwigger) | 30 (2.7%) | 27 (2.0%) |
| None (failed traces) | 0 | 77 (5.6%) |
| Placeholders | 0 | 0 |

PortSwigger labs use a "CHECK" marker instead of traditional CTF flags. The reward function handles these via `metadata.success` (platform-confirmed solve) rather than string matching.

## Known Characteristics

- **50% of entries end with a `tool` message** (no final assistant response). This is inherent to BoxPwnr trace format where the agent submits the flag via `flag_found` tool call and the conversation ends on the tool response. Filtering these would halve the dataset.
- **BoxPwnr system prompt template** contains `<FLAG>content_of_flag_here</FLAG>` as an instruction example in user messages. This is the agent format specification, not a data quality issue.
- **`reasoning_content` field** present on ~37% of entries (from models with thinking tokens). GLM-4.7-Flash renders these as `<think>` blocks.

## Regenerating

```bash
git clone --depth 1 https://github.com/0ca/BoxPwnr-Traces.git /tmp/BoxPwnr-Traces

open-ctf-convert \
    --input /tmp/BoxPwnr-Traces \
    --output /tmp/all_traces.jsonl \
    --output-failure /tmp/failed_traces.jsonl

cat /tmp/all_traces.jsonl /tmp/failed_traces.jsonl > /tmp/combined.jsonl

open-ctf-split \
    --input /tmp/combined.jsonl \
    --sft-output data/sft.jsonl \
    --grpo-output data/grpo.jsonl
```

Then apply filters (see `scripts/filter_training_data.py` or the three criteria above).
