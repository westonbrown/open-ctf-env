"""Guardrails for default BoxPwnr runtime bridge wiring."""

from __future__ import annotations

from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs" / "training"
SMOKE_SCRIPT = REPO_ROOT / "examples" / "smoke_test_2ch.sh"

EXPECTED_AGENT = "open_ctf.agent.default_agent.DefaultStepAgent"
EXPECTED_RUNTIME_CMD = "python src/open_ctf/agent/framework_runtime_bridge.py"
EXPECTED_FRAMEWORK = "boxpwnr_langgraph"
EXPECTED_MODE = "tool_calls"


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def test_all_training_configs_default_to_boxpwnr_bridge_profile() -> None:
    paths = sorted(CONFIG_DIR.glob("training*.yaml"))
    assert paths, "No training*.yaml configs found"

    # Smoke config intentionally uses direct local parsing (no bridge subprocess)
    # for lower latency and more reliable tool call recovery.
    BRIDGE_OPT_OUT_CONFIGS = {"training_smoke_2ch.yaml"}

    checked = 0
    for path in paths:
        cfg = _load_yaml(path)
        online_rl = cfg.get("online_rl")
        if not isinstance(online_rl, dict):
            continue
        checked += 1

        assert online_rl.get("agent_class") == EXPECTED_AGENT, path.name
        agent_kwargs = online_rl.get("agent_kwargs")
        assert isinstance(agent_kwargs, dict), path.name

        if path.name in BRIDGE_OPT_OUT_CONFIGS:
            # Opted-out configs must NOT have runtime_cmd and should enable
            # local fallback parsing.
            assert agent_kwargs.get("runtime_cmd") is None, path.name
            assert agent_kwargs.get("runtime_fallback_to_parser") is True, path.name
            continue

        assert agent_kwargs.get("runtime_cmd") == EXPECTED_RUNTIME_CMD, path.name
        assert agent_kwargs.get("runtime_passthrough") is False, path.name
        assert agent_kwargs.get("runtime_fallback_to_parser") is False, path.name

        runtime_env = agent_kwargs.get("runtime_env")
        assert isinstance(runtime_env, dict), path.name
        assert runtime_env.get("OPEN_CTF_AGENT_FRAMEWORK") == EXPECTED_FRAMEWORK, path.name
        assert runtime_env.get("OPEN_CTF_AGENT_MODE") == EXPECTED_MODE, path.name

    assert checked > 0, "No online_rl sections found in training configs"


def test_smoke_script_supports_native_boxpwnr_mode() -> None:
    text = SMOKE_SCRIPT.read_text(encoding="utf-8")
    assert "--native-boxpwnr" in text
    assert "OPEN_CTF_AGENT_MODE" in text
    assert "OPEN_CTF_AGENT_CMD" in text
    assert "langgraph_native_runtime_adapter.py" in text
