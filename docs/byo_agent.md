# BYO Agent Guide

Open CTF uses a **two-protocol system** for pluggable agents. Which protocol you implement depends on your integration point:

| Protocol | Used By | You Own | SkyRL Owns |
|----------|---------|---------|------------|
| **StepAgent** | GRPO training | Tool parsing + execution | Generation (vLLM) |
| **CTFAgent** | Evaluation, GEPA | Generation + execution | Nothing |

## Quick Start: StepAgent (GRPO Training)

Implement `reset`, `step`, `close`, and a `tools` property:

```python
from open_ctf.agent.protocol import StepAgent, StepResult, validate_step_agent

class MyStepAgent:
    """Minimal StepAgent for GRPO training."""

    def reset(self, target="", ground_truth_flag="", max_steps=30, **kwargs):
        self.target = target
        # These 5 attributes are read by the env for reward scoring.
        # Missing any of them silently degrades 7 of 8 reward signals.
        self.tool_calls_history = []
        self.tool_outputs = []
        self.all_text = ""
        self.episode_done = False
        self.turns = 0

    def step(self, action: str) -> StepResult:
        self.turns += 1
        # Parse tool calls from `action` (raw LLM output)
        # Execute tools your way
        # Update self.tool_calls_history, self.tool_outputs, self.all_text
        return StepResult(
            observations=[{"role": "user", "content": "[Tool: shell_command]\n$ ls\nflag.txt"}],
            done=False,
        )

    def close(self):
        pass

    @property
    def tools(self):
        return None  # None = use environment defaults (13 CTF tools)

# Validate after construction
agent = MyStepAgent()
warnings = validate_step_agent(agent)
for w in warnings:
    print(f"WARNING: {w}")

assert isinstance(agent, StepAgent)  # Structural subtyping check
```

## Quick Start: CTFAgent (Eval / GEPA)

Implement a single `solve` method:

```python
from open_ctf.agent.protocol import CTFAgent, AgentResult

class MyEvalAgent:
    """Minimal CTFAgent for evaluation."""

    def solve(self, challenge, target, ground_truth_flag="",
              max_steps=30, timeout=300) -> AgentResult:
        # Your full agent loop: generate + parse + execute + repeat
        return AgentResult(success=True, flag="FLAG{found_it}", steps=5)

assert isinstance(MyEvalAgent(), CTFAgent)
```

## Integration Options

| Method | Mode | Config | Best For |
|--------|------|--------|----------|
| **Direct class** | `tool_calls` | `agent_class: "my_module.MyAgent"` | Custom parsing/execution in Python |
| **Runtime bridge (tool_calls)** | `tool_calls` | `OPEN_CTF_AGENT_MODE=tool_calls` | Default — OpenCTF parses + executes |
| **Runtime bridge (native)** | `native` | `OPEN_CTF_AGENT_MODE=native` | External framework owns execution |

### Direct Class (Recommended for Custom Agents)

Point the training config at your StepAgent class:

```yaml
online_rl:
  agent_class: my_module.MyStepAgent
```

### Runtime Bridge (Native Mode)

For frameworks like LangGraph that run as external processes:

```yaml
online_rl:
  agent_kwargs:
    runtime_passthrough: true
    runtime_env:
      OPEN_CTF_AGENT_MODE: "native"
      OPEN_CTF_AGENT_CMD: "python examples/adapters/my_adapter.py"
```

Your adapter reads JSON from stdin and writes a response to stdout. See `examples/adapters/template_runtime_adapter.py` for a copy-and-customize template.

## Reward-Critical Attributes

The env reads these 5 attributes from your StepAgent via `getattr(agent, attr, default)` after each step. **If any are missing, 7 of 8 reward signals silently degrade to zero.**

| Attribute | Type | Default | Reward Signals Affected |
|-----------|------|---------|------------------------|
| `tool_calls_history` | `list[dict[str, str]]` | `[]` | format, efficiency, exploration, uniqueness, recovery |
| `tool_outputs` | `list[str]` | `[]` | progression, cognitive, flag detection |
| `all_text` | `str` | `""` | cognitive (words-per-action), hallucination detection |
| `episode_done` | `bool` | `False` | flag signal (exact match gating) |
| `turns` | `int` | `0` | efficiency signal |

Initialize them in `reset()` and update them in `step()`.

## Custom Tool Schemas

Return `None` from the `tools` property to use the environment's default 13 CTF tools. To override:

```python
@property
def tools(self):
    return [
        {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "Custom tool",
                "parameters": {
                    "type": "object",
                    "properties": {"input": {"type": "string"}},
                    "required": ["input"],
                },
            },
        }
    ]
```

## Testing and Validation

### Contract Validation

```python
from open_ctf.agent.protocol import validate_step_agent

agent = MyStepAgent()
warnings = validate_step_agent(agent)
assert not warnings, f"Agent validation failed: {warnings}"
```

### Run Existing Tests

```bash
python -m pytest tests/test_agent_protocol.py tests/test_framework_runtime_bridge.py \
    tests/test_framework_runtime_adapters.py tests/test_runtime_protocol.py -v
```

### Test a Native Adapter

```bash
echo '{"action":"test","turn":1,"runtime_state":{}}' | python examples/adapters/template_runtime_adapter.py
```

## Further Reading

- **Runtime bridge details**: `docs/byo_agent_runtime.md`
- **Adapter examples**: `examples/adapters/`
- **Training configuration**: `docs/training.md`
- **Protocol source**: `src/open_ctf/agent/protocol.py`
- **Default agent implementation**: `src/open_ctf/agent/default_agent.py`
