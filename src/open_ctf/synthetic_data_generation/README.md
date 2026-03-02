# Synthetic Data Generation

Offline trajectory generation for CTF training data. Uses a teacher LLM + simulated environment to produce training-ready JSONL traces without running real Docker containers.

## Architecture

```
WorldManifest (YAML)  →  SimulatedEnvironmentExecutor  →  LiteLLMAgentAdapter  →  JSONL traces
    (scenario def)          (mock tool execution)          (teacher LLM loop)     (SFT/GRPO ready)
```

### Modules

| File | Purpose |
|------|---------|
| `manifest.py` | `WorldManifest` dataclass — loads YAML configs defining hosts, files, services, tool responses |
| `executor.py` | `SimulatedEnvironmentExecutor` — mocks all 13 agent tools (shell, read_file, curl, etc.) using manifest data |
| `generator.py` | `LiteLLMAgentAdapter` + `SyntheticGenerator` — runs teacher LLM in a ReAct loop, exports traces |

### How It Works

1. **Manifest** defines the simulated world: target hosts, discoverable files, scripted tool responses, and a ground truth flag
2. **Executor** intercepts every tool call and returns manifest-driven responses instead of running real commands
3. **Teacher LLM** (via LiteLLM) generates reasoning + tool calls in a multi-turn loop until it finds the flag or runs out of turns
4. Each episode **randomizes the flag** (UUID-based) to prevent memorization during training
5. Output traces match the **sft_v6 OpenAI ChatML format**: `{messages, metadata, ground_truth_flag, optimal_steps}`

## Quick Start

```bash
# CLI (uses pyproject.toml entry point)
open-ctf-synthetic-data \
    --config configs/synthetic_data_generation/default.yaml \
    --teacher-model "openrouter/openai/gpt-4o" \
    --num-traces 100 \
    --sft-out data/synthetic_sft.jsonl

# Or with Azure
export AZURE_API_KEY="..."
export AZURE_API_BASE="https://your-endpoint.cognitiveservices.azure.com"
export AZURE_API_VERSION="2025-04-01-preview"
open-ctf-synthetic-data \
    --config configs/synthetic_data_generation/default.yaml \
    --teacher-model "azure/gpt-5.2-codex" \
    --num-traces 500 \
    --sft-out data/synthetic_sft.jsonl
```

### Programmatic Usage

```python
from open_ctf.synthetic_data_generation import WorldManifest, SimulatedEnvironmentExecutor, SyntheticGenerator
from open_ctf.synthetic_data_generation.generator import LiteLLMAgentAdapter

# Load manifests
manifests = [WorldManifest.from_yaml(f"configs/synthetic_data_generation/{f}") for f in [
    "default.yaml",
    "incident_response_k8s.yaml",
    "pentest_lateral_movement.yaml",
    "threat_emulation_apt.yaml",
]]

# Generate traces
adapter = LiteLLMAgentAdapter(model_name="azure/gpt-5.2-codex")
gen = SyntheticGenerator(manifests=manifests, agent_adapter=adapter)
traces = gen.batch_generate_traces(max_trajectories=100, max_turns=30)
gen.export_jsonl(traces, "data/synthetic_sft.jsonl")
```

### Test Script

```bash
# Test a single manifest
python scripts/test_synth_model.py --model "azure/gpt-5.2-codex" --manifests "default.yaml"

# Test all manifests
python scripts/test_synth_model.py --model "azure/gpt-5.2-codex" --manifests all
```

## Custom Agent Adapters

Subclass `BaseAgentAdapter` to plug in any agent framework:

```python
from open_ctf.synthetic_data_generation.generator import BaseAgentAdapter

class MyAgentAdapter(BaseAgentAdapter):
    def run_episode(self, executor, manifest, max_turns):
        # Your agent logic here — call executor.step(tool_name, args) for each action
        # Return: {"messages": [...], "metadata": {...}, "ground_truth_flag": "...", "optimal_steps": int}
        pass
```

## Executor Mock Coverage

The `SimulatedEnvironmentExecutor` mocks these tools:

| Tool | Mock Behavior |
|------|--------------|
| `shell_command` | Checks manifest `tool_responses` (regex priority matching), then falls back to built-in handlers for nmap, ls, cat, curl, grep, find, ps, env, etc. |
| `read_file` | Returns file content from manifest `files` section |
| `python_code` | Checks manifest responses, credential patterns, then returns file content if code references manifest files |
| `grep` | Pattern searches across manifest files |
| `file_search` | Glob matching against manifest file paths |
| `flag_found` / `submit_flag` | Validates submitted flag against the (randomized) ground truth |
| `web_search` | Returns stub "no results" |
| `list_sessions` / `close_session` / `write_stdin` | Session stubs |

### Response Matching Priority

When a shell command is executed, the executor matches against manifest `tool_responses` using a 3-tier priority system:

1. **Regex match** — `re.search(pattern, command)` — highest priority
2. **Substring match** — `pattern in command`
3. **Token fragment match** — all tokens in pattern found in command

Within each tier, the **longest pattern wins** (more specific = higher priority).

### Credential-Aware Fallback

If a command uses secrets found in manifest files (passwords, API keys, tokens), the executor returns the most relevant scripted response — typically the flag-bearing one. This handles mysql, ssh, psql, crackmapexec, wmic, etc. generically without hardcoded tool-specific logic.

## Teacher Model Selection

Any model supported by [LiteLLM](https://docs.litellm.ai/docs/providers) works. Tested models:

| Model | Solve Rate | Notes |
|-------|-----------|-------|
| `azure/gpt-5.2-codex` | 90-100% | Best quality, long trajectories |
| `openrouter/openai/gpt-4o` | ~75% | Good quality, requires OPENROUTER_API_KEY |
| `openrouter/stepfun/step-3.5-flash:free` | ~25% | Free but weak reasoning |

Set the model via `--teacher-model` CLI arg or `LiteLLMAgentAdapter(model_name="...")`.
