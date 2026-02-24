# Training Data

Generated from [BoxPwnr-Traces](https://github.com/0ca/BoxPwnr-Traces) via `open-ctf-convert` + `open-ctf-split`, then filtered for training quality.

## Datasets

| File | Traces | Size | Description |
|------|--------|------|-------------|
| `sft.jsonl` | 820 | 62.5MB | Successful solves for supervised fine-tuning |
| `grpo_cybench40.jsonl` | 87 | 7.3MB | CyBench-only traces with ground truth flags for online GRPO |
| `grpo_offline_683.jsonl` | 676 | 38.8MB | Cross-platform traces for offline GRPO / reward validation |
| `dataset_info.json` | — | — | LlamaFactory dataset metadata (maps dataset names to files) |

### SFT vs GRPO Split

- **SFT** (`sft.jsonl`): Successful solves across all platforms. Used for supervised fine-tuning with LlamaFactory.
- **GRPO online** (`grpo_cybench40.jsonl`): CyBench challenges only — these have Docker infrastructure for live tool execution during online GRPO. Each trace includes `ground_truth_flag` and `optimal_steps` for the reward function.
- **GRPO offline** (`grpo_offline_683.jsonl`): Cross-platform traces for offline advantage estimation or reward model validation. Not used for live rollouts.

## Format

Each line is a JSON object with OpenAI-format messages:

```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "Solve: http://target"},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "name": "shell_command", "content": "..."}
  ],
  "metadata": {
    "source": "boxpwnr", "platform": "cybench", "challenge": "...",
    "success": true, "total_turns": 12, "model": "..."
  },
  "ground_truth_flag": "FLAG{...}",
  "optimal_steps": 8
}
```

## Known Characteristics

- **~50% of entries end with a `tool` message** (no final assistant response). This is inherent to the BoxPwnr trace format — the agent submits the flag via `flag_found` and the conversation ends on the tool response. Filtering these would halve the dataset.
- **BoxPwnr system prompt** contains `<FLAG>content_of_flag_here</FLAG>` as an instruction example. This is the agent format spec, not a data quality issue.
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

Then apply filters (see `scripts/filter_training_data.py` or the criteria below).

## Filters Applied

Three filters are applied after conversion and splitting to remove samples that would degrade training:

1. **No-assistant traces removed (GRPO only)** — entries with only `[system, user]` messages (no agent actions). These are failed runs where the agent never started and provide no trajectory signal.

2. **Token outliers removed (>100K estimated tokens)** — entries with extremely long tool outputs that exceed any reasonable `max_seq_length` and waste compute on truncated data.

3. **Placeholder flag removed** — entries with `ground_truth_flag: "FLAG{placeholder}"` that would poison the reward signal.
