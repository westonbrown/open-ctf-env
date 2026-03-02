# Native Adapter Entrypoints

These scripts are starter templates for `OPEN_CTF_AGENT_MODE=native` with:
- `runtime_cmd: "python src/open_ctf/agent/framework_runtime_bridge.py"`
- `runtime_passthrough: true`
- `OPEN_CTF_AGENT_CMD: "python examples/adapters/<adapter>.py"`

## Quick Start

Run an adapter with a test request:

```bash
echo '{"action":"test","turn":1,"runtime_state":{}}' \
  | python examples/adapters/template_runtime_adapter.py
```

Use in training by setting the runtime env in your config:

```yaml
agent_kwargs:
  runtime_passthrough: true
  runtime_env:
    OPEN_CTF_AGENT_MODE: "native"
    OPEN_CTF_AGENT_CMD: "python examples/adapters/template_runtime_adapter.py"
```

## Available Adapters

| Adapter | Status | Purpose |
|---------|--------|---------|
| `template_runtime_adapter.py` | Template | Generic copy-and-customize starting point for any framework |
| `langgraph_native_runtime_adapter.py` | Functional | LangGraph example with strict request validation |
| `langgraph_runtime_adapter.py` | Functional | Lightweight LangGraph template adapter |
| `boxpwnr_native_runtime_adapter.py` | Functional | BoxPwnr reference agent example |

## Creating a New Adapter

1. Copy the template:
   ```bash
   cp examples/adapters/template_runtime_adapter.py examples/adapters/my_adapter.py
   ```

2. Implement the `handle_step` function with your framework's logic. The function receives the full runtime request and returns a response dict.

3. Test it:
   ```bash
   echo '{"action":"test","turn":1,"runtime_state":{}}' | python examples/adapters/my_adapter.py
   ```

4. Wire it into training config:
   ```yaml
   agent_kwargs:
     runtime_passthrough: true
     runtime_env:
       OPEN_CTF_AGENT_MODE: "native"
       OPEN_CTF_AGENT_CMD: "python examples/adapters/my_adapter.py"
   ```

## Configuration Modes

### tool_calls mode (default)

OpenCTF parses tool calls from LLM output and executes them locally. Your adapter is not needed.

```yaml
agent_kwargs:
  runtime_env:
    OPEN_CTF_AGENT_MODE: "tool_calls"
```

### native mode

Your external adapter owns tool execution. Each adapter reads JSON from stdin and prints JSON to stdout.

```yaml
agent_kwargs:
  runtime_passthrough: true
  runtime_env:
    OPEN_CTF_AGENT_MODE: "native"
    OPEN_CTF_AGENT_CMD: "python examples/adapters/my_adapter.py"
```

## Minimal Response Shape

```json
{
  "done": false,
  "episode_done": false,
  "observations": [{"role": "user", "content": "..."}],
  "state": {},
  "info": {"rollout_status": "ok"}
}
```

## Testing

Run the contract smoke tests to verify protocol compliance:

```bash
python -m pytest tests/test_framework_runtime_adapters.py \
    tests/test_framework_runtime_bridge.py \
    tests/test_runtime_protocol.py -v
```

## Further Reading

- BYO agent guide: `src/open_ctf/agent/README.md`
- Runtime bridge details: `docs/byo_agent_runtime.md`
- Protocol source: `src/open_ctf/agent/protocol.py`
