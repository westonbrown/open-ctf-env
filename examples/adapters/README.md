# Native Adapter Entrypoints

These scripts are starter templates for `OPEN_CTF_AGENT_MODE=native` with:
- `runtime_cmd: "python src/open_ctf/agent/framework_runtime_bridge.py"`
- `runtime_passthrough: true`
- `OPEN_CTF_AGENT_CMD: "python examples/adapters/<adapter>.py"`

Current adapters:
- `langgraph_native_runtime_adapter.py` (strict, production-ready native adapter)
- `langgraph_runtime_adapter.py` (lightweight template adapter)
- `boxpwnr_native_runtime_adapter.py` (compatibility wrapper for BoxPwnr profile)

Each adapter reads OpenCTF runtime request JSON from stdin and prints a
response JSON object (bridge wraps it into protocol v1.0 passthrough).

Minimal response shape:
```json
{
  "done": false,
  "episode_done": false,
  "observations": [{"role": "user", "content": "..."}],
  "state": {},
  "info": {"rollout_status": "ok"}
}
```

BoxPwnr native mode example:
```bash
OPEN_CTF_AGENT_MODE=native \
OPEN_CTF_AGENT_FRAMEWORK=boxpwnr_langgraph \
OPEN_CTF_AGENT_CMD="python examples/adapters/langgraph_native_runtime_adapter.py"
```
