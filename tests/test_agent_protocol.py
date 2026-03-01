"""Tests for CTFAgent protocol and BoxPwnr adapter."""

import pytest

from open_ctf.agent.protocol import AgentResult, CTFAgent


class TestAgentResult:
    def test_defaults(self):
        result = AgentResult(success=False)
        assert result.flag is None
        assert result.steps == 0
        assert result.messages == []
        assert result.duration_seconds == 0.0
        assert result.metadata == {}

    def test_full_construction(self):
        result = AgentResult(
            success=True,
            flag="FLAG{test}",
            steps=5,
            messages=[{"role": "user", "content": "hello"}],
            duration_seconds=42.5,
            metadata={"model": "test"},
        )
        assert result.success is True
        assert result.flag == "FLAG{test}"
        assert result.steps == 5
        assert len(result.messages) == 1
        assert result.duration_seconds == 42.5


class TestCTFAgentProtocol:
    def test_custom_class_satisfies_protocol(self):
        """Any class with solve() matching the signature is a CTFAgent."""

        class MyAgent:
            def solve(self, challenge, target, ground_truth_flag="",
                      max_steps=30, timeout=300):
                return AgentResult(success=True, flag="FLAG{custom}")

        agent = MyAgent()
        assert isinstance(agent, CTFAgent)

    def test_class_without_solve_fails_protocol(self):
        """Class without solve() should NOT satisfy CTFAgent."""

        class NotAnAgent:
            def run(self):
                pass

        assert not isinstance(NotAnAgent(), CTFAgent)

    def test_boxpwnr_adapter_satisfies_protocol(self):
        """BoxPwnrAgent should satisfy CTFAgent protocol."""
        from open_ctf.agent.boxpwnr_adapter import BoxPwnrAgent

        agent = BoxPwnrAgent(model="test-model")
        assert isinstance(agent, CTFAgent)

    def test_boxpwnr_adapter_construction(self):
        """BoxPwnrAgent should store constructor args."""
        from open_ctf.agent.boxpwnr_adapter import BoxPwnrAgent

        agent = BoxPwnrAgent(
            model="ollama/test",
            platform="local",
            strategy="chat",
        )
        assert agent.model == "ollama/test"
        assert agent.platform == "local"
        assert agent.strategy == "chat"

    def test_boxpwnr_adapter_propagates_success_from_runner(self, monkeypatch):
        from open_ctf.agent.boxpwnr_adapter import BoxPwnrAgent
        import open_ctf.agent.runner as runner_mod

        class StubRunner:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, target):
                return {
                    "status": "success",
                    "total_turns": 7,
                    "flag": "FLAG{ok}",
                }

        monkeypatch.setattr(runner_mod, "AgentRunner", StubRunner)
        agent = BoxPwnrAgent(model="test-model", platform="cybench")
        result = agent.solve(
            challenge="demo",
            target="http://localhost:32805",
            max_steps=20,
            timeout=120,
        )
        assert result.success is True
        assert result.steps == 7
        assert result.flag == "FLAG{ok}"

    def test_boxpwnr_adapter_non_success_status_is_failure(self, monkeypatch):
        from open_ctf.agent.boxpwnr_adapter import BoxPwnrAgent
        import open_ctf.agent.runner as runner_mod

        class StubRunner:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, target):
                return {
                    "status": "failed",
                    "total_turns": 4,
                    "user_flag_value": "FLAG{wrong}",
                }

        monkeypatch.setattr(runner_mod, "AgentRunner", StubRunner)
        agent = BoxPwnrAgent(model="test-model", platform="cybench")
        result = agent.solve(
            challenge="demo",
            target="http://localhost:32805",
            max_steps=20,
            timeout=120,
        )
        assert result.success is False
        assert result.steps == 4
        assert result.flag == "FLAG{wrong}"
