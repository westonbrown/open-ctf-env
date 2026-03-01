"""Tests for native_tool_schemas default and warmup_ratio computation.

Validates:
- native_tool_schemas defaults to True even when chat_template is set
- native_tool_schemas=False is honored when explicitly set
- warmup_ratio is correctly computed into num_warmup_steps
"""

import pytest

from open_ctf.training.online_rl.runtime import _build_skyrl_config


@pytest.fixture
def base_config():
    """Minimal valid config for _build_skyrl_config."""
    return {
        "model": {"max_seq_length": 4096},
        "lora": {"r": 32, "alpha": 64, "dropout": 0.0, "target_modules": ["q_proj"]},
        "online_rl": {
            "batch_size": 1,
            "epochs": 1,
            "num_generations": 2,
        },
        "output": {"save_steps": 50},
    }


class TestNativeToolSchemasDefault:
    """Test 6: native_tool_schemas defaults to True."""

    def test_defaults_true_without_chat_template(self, base_config):
        """When no chat_template is set, native_tool_schemas should be True."""
        result = _build_skyrl_config("/model", "/out", base_config, "/data.jsonl")
        assert result["generator"]["native_tool_schemas"] is True

    def test_defaults_true_with_chat_template(self, base_config):
        """When chat_template IS set, native_tool_schemas should STILL be True.

        This is the bug fix: previously defaulted to False when chat_template
        was set, which broke native tool support for Qwen3.5.
        """
        base_config["online_rl"]["chat_template"] = "qwen3_without_thinking"
        result = _build_skyrl_config("/model", "/out", base_config, "/data.jsonl")
        assert result["generator"]["native_tool_schemas"] is True

    def test_tools_injected_with_chat_template(self, base_config):
        """Tools should be in chat_template_kwargs even with a custom template."""
        base_config["online_rl"]["chat_template"] = "qwen3_without_thinking"
        result = _build_skyrl_config("/model", "/out", base_config, "/data.jsonl")
        kwargs = result["generator"]["chat_template_kwargs"]
        tools = kwargs.get("tools", [])
        assert len(tools) > 0, "Tools should be injected even with custom chat_template"
        tool_names = [t["function"]["name"] for t in tools]
        assert "shell_command" in tool_names


class TestNativeToolSchemasExplicitFalse:
    """Test 7: native_tool_schemas=False honored."""

    def test_explicit_false_honored(self, base_config):
        """Config explicitly sets native_tool_schemas: false -> stays False."""
        base_config["online_rl"]["native_tool_schemas"] = False
        result = _build_skyrl_config("/model", "/out", base_config, "/data.jsonl")
        assert result["generator"]["native_tool_schemas"] is False

    def test_no_tools_in_kwargs_when_false(self, base_config):
        """When native_tool_schemas is False, tools should NOT be in kwargs."""
        base_config["online_rl"]["native_tool_schemas"] = False
        result = _build_skyrl_config("/model", "/out", base_config, "/data.jsonl")
        kwargs = result["generator"]["chat_template_kwargs"]
        assert "tools" not in kwargs


class TestWarmupRatio:
    """Test 8: warmup_ratio computed correctly."""

    def test_warmup_ratio_computed(self, base_config):
        """warmup_ratio: 0.1 with total_episodes: 100 -> num_warmup_steps = 10."""
        base_config["online_rl"]["warmup_ratio"] = 0.1
        base_config["online_rl"]["total_episodes"] = 100
        result = _build_skyrl_config("/model", "/out", base_config, "/data.jsonl")
        policy_warmup = result["trainer"]["policy"]["optimizer_config"]["num_warmup_steps"]
        critic_warmup = result["trainer"]["critic"]["optimizer_config"]["num_warmup_steps"]
        assert policy_warmup == 10
        assert critic_warmup == 10

    def test_warmup_ratio_minimum_one(self, base_config):
        """Very small warmup_ratio should produce at least 1 warmup step."""
        base_config["online_rl"]["warmup_ratio"] = 0.001
        base_config["online_rl"]["total_episodes"] = 10
        result = _build_skyrl_config("/model", "/out", base_config, "/data.jsonl")
        policy_warmup = result["trainer"]["policy"]["optimizer_config"]["num_warmup_steps"]
        # 0.001 * 10 = 0.01 -> max(1, 0) = 1
        assert policy_warmup == 1

    def test_warmup_ratio_zero_means_no_warmup(self, base_config):
        """warmup_ratio: 0.0 should produce 0 warmup steps."""
        base_config["online_rl"]["warmup_ratio"] = 0.0
        result = _build_skyrl_config("/model", "/out", base_config, "/data.jsonl")
        policy_warmup = result["trainer"]["policy"]["optimizer_config"]["num_warmup_steps"]
        assert policy_warmup == 0

    def test_no_warmup_ratio_defaults_zero(self, base_config):
        """No warmup_ratio in config -> num_warmup_steps = 0."""
        # Ensure no warmup_ratio is set
        base_config["online_rl"].pop("warmup_ratio", None)
        result = _build_skyrl_config("/model", "/out", base_config, "/data.jsonl")
        policy_warmup = result["trainer"]["policy"]["optimizer_config"]["num_warmup_steps"]
        assert policy_warmup == 0

    def test_explicit_num_warmup_steps_without_ratio(self, base_config):
        """When warmup_ratio is absent, explicit num_warmup_steps is used."""
        base_config["online_rl"].pop("warmup_ratio", None)
        base_config["online_rl"]["num_warmup_steps"] = 5
        result = _build_skyrl_config("/model", "/out", base_config, "/data.jsonl")
        policy_warmup = result["trainer"]["policy"]["optimizer_config"]["num_warmup_steps"]
        assert policy_warmup == 5

    def test_warmup_ratio_with_default_total_episodes(self, base_config):
        """warmup_ratio uses total_episodes default of 100 if not set."""
        base_config["online_rl"]["warmup_ratio"] = 0.03
        # No total_episodes set — defaults to 100
        result = _build_skyrl_config("/model", "/out", base_config, "/data.jsonl")
        policy_warmup = result["trainer"]["policy"]["optimizer_config"]["num_warmup_steps"]
        # 0.03 * 100 = 3
        assert policy_warmup == 3
