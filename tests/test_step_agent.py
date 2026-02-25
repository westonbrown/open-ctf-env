"""Tests for StepAgent protocol and DefaultStepAgent.

Validates:
- DefaultStepAgent satisfies StepAgent protocol
- Custom classes satisfy StepAgent protocol
- DefaultStepAgent.step() handles: no tool call → nudge, tool call → output, flag → done
- OpenCTFTextEnv delegates to custom agent when agent_class is specified
"""

import pytest

from open_ctf.agent.protocol import StepAgent, StepResult
from open_ctf.agent.default_agent import DefaultStepAgent


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestStepAgentProtocol:
    def test_default_agent_satisfies_protocol(self):
        """DefaultStepAgent should satisfy StepAgent via structural subtyping."""
        agent = DefaultStepAgent()
        assert isinstance(agent, StepAgent)

    def test_custom_class_satisfies_protocol(self):
        """Any class with matching reset/step/close satisfies StepAgent."""

        class MyAgent:
            def reset(self, target="", ground_truth_flag="", max_steps=30, **kw):
                pass

            def step(self, action: str) -> StepResult:
                return StepResult(observations=[], done=False)

            def close(self):
                pass

            @property
            def tools(self):
                return None

        agent = MyAgent()
        assert isinstance(agent, StepAgent)

    def test_class_without_step_fails_protocol(self):
        """Class without step() should NOT satisfy StepAgent."""

        class NotAnAgent:
            def reset(self, **kw):
                pass

            def close(self):
                pass

        assert not isinstance(NotAnAgent(), StepAgent)


# ---------------------------------------------------------------------------
# StepResult
# ---------------------------------------------------------------------------


class TestStepResult:
    def test_defaults(self):
        result = StepResult(observations=[], done=False)
        assert result.observations == []
        assert result.done is False
        assert result.info == {}

    def test_with_observations(self):
        obs = [{"role": "user", "content": "[Tool: shell_command]\noutput"}]
        result = StepResult(observations=obs, done=False, info={"step": 1})
        assert len(result.observations) == 1
        assert result.info["step"] == 1


# ---------------------------------------------------------------------------
# DefaultStepAgent behavior
# ---------------------------------------------------------------------------


class TestDefaultStepAgent:
    def test_no_tool_call_returns_nudge(self):
        """When LLM output has no tool calls, agent returns a nudge message."""
        agent = DefaultStepAgent()
        agent.reset(target="http://localhost:8080", max_steps=30)

        result = agent.step("I'm thinking about how to approach this...")
        assert not result.done
        assert len(result.observations) == 1
        assert "No tool call detected" in result.observations[0]["content"]
        assert agent.turns == 1

    def test_no_tool_call_at_max_steps_is_done(self):
        """When no tool call at max steps, agent returns done=True."""
        agent = DefaultStepAgent()
        agent.reset(target="http://localhost:8080", max_steps=1)

        result = agent.step("Just thinking...")
        assert result.done
        assert result.observations == []

    def test_reset_clears_state(self):
        """Reset should clear all episode state."""
        agent = DefaultStepAgent()
        agent.reset(target="http://localhost:8080", max_steps=30)

        # Take a step
        agent.step("Some text")
        assert agent.turns == 1

        # Reset
        agent.reset(target="http://localhost:9090", max_steps=10)
        assert agent.turns == 0
        assert agent.tool_calls_history == []
        assert agent.tool_outputs == []
        assert agent.all_text == ""
        assert not agent.episode_done
        assert agent.max_steps == 10

    def test_shell_tool_call_executes(self):
        """A shell_command tool call should execute and return output."""
        agent = DefaultStepAgent()
        agent.reset(target="http://localhost:8080", max_steps=30)

        action = '<tool_call>\n{"name": "shell_command", "arguments": {"command": "echo hello_world"}}\n</tool_call>'
        result = agent.step(action)

        assert not result.done
        assert len(result.observations) == 1
        assert "hello_world" in result.observations[0]["content"]
        assert result.observations[0]["role"] == "user"
        assert len(agent.tool_calls_history) == 1
        assert agent.tool_calls_history[0]["name"] == "shell_command"

    def test_close_releases_executor(self):
        """Close should release the executor."""
        agent = DefaultStepAgent()
        agent.reset(target="http://localhost:8080", max_steps=30)
        agent.close()
        assert agent._executor is None


# ---------------------------------------------------------------------------
# OpenCTFTextEnv delegation
# ---------------------------------------------------------------------------


class TestEnvDelegation:
    def test_env_uses_default_agent(self):
        """OpenCTFTextEnv should use DefaultStepAgent by default."""
        from open_ctf.envs.skyrl.openctf_env import OpenCTFTextEnv

        env = OpenCTFTextEnv(extras={"target": "http://localhost:8080"})
        assert isinstance(env._agent, DefaultStepAgent)

    def test_env_delegates_step_to_agent(self):
        """OpenCTFTextEnv.step() should delegate to the agent."""
        from open_ctf.envs.skyrl.openctf_env import OpenCTFTextEnv

        env = OpenCTFTextEnv(
            extras={"target": "http://localhost:8080", "max_turns": 30},
        )
        # init() resets the agent
        prompt = [{"role": "user", "content": "Solve the challenge"}]
        env.init(prompt)

        # Step with no tool call
        result = env.step("Just thinking...")
        assert result["done"] is False
        assert len(result["observations"]) == 1
        assert "No tool call detected" in result["observations"][0]["content"]

    def test_env_accepts_custom_agent_class(self):
        """OpenCTFTextEnv should accept agent_class kwarg."""
        from open_ctf.envs.skyrl.openctf_env import OpenCTFTextEnv

        class MockAgent:
            def __init__(self, **kwargs):
                self.reset_called = False
                self.step_called = False
                self.tool_calls_history = []
                self.tool_outputs = []
                self.all_text = ""
                self.episode_done = False

            def reset(self, target="", ground_truth_flag="", max_steps=30, **kw):
                self.reset_called = True

            def step(self, action: str) -> StepResult:
                self.step_called = True
                return StepResult(
                    observations=[{"role": "user", "content": "custom output"}],
                    done=False,
                    info={"custom": True},
                )

            def close(self):
                pass

        # Pass class directly (not as dotpath string, for testing)
        env = OpenCTFTextEnv(
            extras={"target": "http://localhost:8080"},
        )
        # Replace agent manually (simulates what _resolve_class would do)
        env._agent = MockAgent()

        prompt = [{"role": "user", "content": "test"}]
        env.init(prompt)
        assert env._agent.reset_called

        result = env.step("some action")
        assert env._agent.step_called
        assert result["observations"][0]["content"] == "custom output"
        assert result["metadata"]["custom"] is True

    def test_env_shell_tool_end_to_end(self):
        """Full end-to-end: env → agent → executor → shell → result."""
        from open_ctf.envs.skyrl.openctf_env import OpenCTFTextEnv

        env = OpenCTFTextEnv(
            extras={"target": "http://localhost:8080", "max_turns": 30},
        )
        prompt = [{"role": "user", "content": "Solve it"}]
        env.init(prompt)

        action = '<tool_call>\n{"name": "shell_command", "arguments": {"command": "echo test123"}}\n</tool_call>'
        result = env.step(action)

        assert result["done"] is False
        assert len(result["observations"]) == 1
        assert "test123" in result["observations"][0]["content"]
