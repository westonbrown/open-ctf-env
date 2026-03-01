# BYO Agent Runtime (ROCK-Aligned)

## Goal
Support real BYO agent frameworks while keeping SkyRL/OpenCTF training stable:
- `tool_calls` mode: existing fast/stable parser + local tool executor path.
- `native` mode: external framework runtime process owns observations/done.

Bridge fallback framework is generic (`generic`); training profiles can set
`OPEN_CTF_AGENT_FRAMEWORK` explicitly (for example `boxpwnr_langgraph`).

## What Was Added
- Generic runtime bridge: `src/open_ctf/agent/framework_runtime_bridge.py`
  - `OPEN_CTF_AGENT_MODE=tool_calls|native`
  - `OPEN_CTF_AGENT_FRAMEWORK=<name>`
  - `OPEN_CTF_AGENT_CMD=<external adapter command>` (required for `native`)
  - `OPEN_CTF_AGENT_CMD_TIMEOUT=<seconds>`
  - `OPEN_CTF_AGENT_WORKDIR=<path>`
- Richer runtime request context:
  - `prompt_messages`
  - `challenge: {id, category, difficulty, infra_type}`
  - `objective`

## WARNING: Reward-Critical Implicit Attributes

If you implement a custom `StepAgent` for GRPO training, your agent **must** expose
these 5 attributes or 7 of 8 reward signals will silently degrade to zero:

| Attribute | Type | Default | Reward signals affected |
|---|---|---|---|
| `tool_calls_history` | `List[Dict[str, str]]` | `[]` | format, efficiency, exploration, uniqueness, recovery |
| `tool_outputs` | `List[str]` | `[]` | progression, cognitive, flag detection |
| `all_text` | `str` | `""` | cognitive (words-per-action), hallucination detection |
| `episode_done` | `bool` | `False` | flag signal (exact match gating) |
| `turns` | `int` | `0` | efficiency signal |

Initialize them in `reset()` and update them in `step()`. The env reads them via
`getattr(agent, attr, default)` after each step to compute rewards.

Call `validate_step_agent(agent)` after constructing your agent to check for
missing attributes:

```python
from open_ctf.agent.protocol import validate_step_agent

agent = MyCustomAgent()
warnings = validate_step_agent(agent)
for w in warnings:
    print(f"WARNING: {w}")
```

The env calls this automatically at startup and logs any warnings.

## Canonical Script Location
- `src/open_ctf/agent/framework_runtime_bridge.py` lives in the agent package.
- Rationale:
  - It is an executable runtime entrypoint (invoked by `runtime_cmd`) and a library module.
  - Training configs reference it as `python src/open_ctf/agent/framework_runtime_bridge.py`.

## Why This Matches ROCK
ROCK runs a real agent process (`run_cmd`) and bridges model/runtime state externally.  
OpenCTF now supports the same architectural seam via `native` mode while preserving current SkyRL training behavior in `tool_calls` mode.

ROCK references:
- `references/ROCK/rock/sdk/sandbox/agent/rock_agent.py`
- `references/ROCK/tests/integration/sdk/sandbox/agent/rock_agent/langgraph_config.yaml`
- `references/ROCK/docs/versioned_docs/version-1.3.0/References/Python SDK References/model-service.md`

## Framework Adapter Contract
In `native` mode, your external command receives runtime request JSON on stdin and can return:

1. Full OpenCTF runtime protocol response (`protocol_version`, `capabilities`, etc), or
2. Simplified object that the bridge wraps into passthrough protocol:

```json
{
  "done": false,
  "episode_done": false,
  "observations": [{"role": "user", "content": "..." }],
  "state": {"k": "v"},
  "info": {"rollout_status": "ok"},
  "tool_calls": [{"name": "shell_command", "arguments": {"command": "echo hi"}}]
}
```

This makes wrappers straightforward for LangGraph or custom runtimes.

## Second ROCK Review (2026-02-28)
- ROCK pattern validated again: external agent process (`run_cmd`) is the correct integration seam.
- OpenCTF keeps the same seam via `OPEN_CTF_AGENT_MODE=native` + `OPEN_CTF_AGENT_CMD`.
- Decision: do not pivot runtime architecture; keep current SkyRL env/trainer stack and expose framework adapters through the bridge.
- Reason: this preserves current training reliability while enabling BYO runtime parity.

## Adapter Templates + Contract Tests
- Adapter scripts are provided in `examples/adapters/`:
  - `boxpwnr_native_runtime_adapter.py` (functional — strict request validation + fail-fast)
  - `langgraph_native_runtime_adapter.py` (functional — recommended for LangGraph)
  - `langgraph_runtime_adapter.py` (functional — lightweight template adapter)
  - `autogen_runtime_adapter.py` (template stub — not tested end-to-end)
  - `strands_runtime_adapter.py` (template stub — not tested end-to-end)
  - `adk_runtime_adapter.py` (template stub — not tested end-to-end)
- Contract smoke tests:
  - `tests/test_framework_runtime_adapters.py`
  - `tests/test_framework_runtime_bridge.py`
  - `tests/test_runtime_protocol.py`
  - `tests/test_boxpwnr_native_runtime_adapter.py`
  - `tests/test_runtime_bridge_defaults.py`
- These tests validate protocol wrapping, passthrough shape, and framework metadata across adapter types.

## Native Smoke Path
- `examples/smoke_test_2ch.sh` now supports `--native-boxpwnr`.
- This sets:
  - `OPEN_CTF_AGENT_MODE=native`
  - `OPEN_CTF_AGENT_CMD=python examples/adapters/boxpwnr_native_runtime_adapter.py`
- Adapter returns protocol `tool_calls_response` (not passthrough), so OpenCTF local tool execution stays in-loop for RL training updates.

## Default Config Pattern
```yaml
agent_kwargs:
  runtime_cmd: "python src/open_ctf/agent/framework_runtime_bridge.py"
  runtime_timeout_seconds: 20
  runtime_passthrough: false
  runtime_fallback_to_parser: false
  runtime_env:
    OPEN_CTF_AGENT_FRAMEWORK: "boxpwnr_langgraph"
    OPEN_CTF_AGENT_MODE: "tool_calls"
```

Switch to native mode:
```yaml
agent_kwargs:
  runtime_passthrough: true
  runtime_env:
    OPEN_CTF_AGENT_FRAMEWORK: "langgraph"
    OPEN_CTF_AGENT_MODE: "native"
    OPEN_CTF_AGENT_CMD: "python examples/adapters/langgraph_native_runtime_adapter.py"
```
