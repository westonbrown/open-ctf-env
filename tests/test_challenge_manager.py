"""Tests for ChallengeManager — mocked Docker subprocess calls."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from open_ctf.challenges.registry import ChallengeRegistry
from open_ctf.challenges.manager import ChallengeManager


@pytest.fixture
def registry_yaml(tmp_path):
    """Create a test registry."""
    data = {
        "challenges": [
            {
                "id": "eval-me",
                "name": "[Very Easy] eval-me",
                "category": "misc",
                "difficulty": "very_easy",
                "infra_type": "docker",
                "port": 32805,
            },
            {
                "id": "Dynastic",
                "name": "[Very Easy] Dynastic",
                "category": "crypto",
                "difficulty": "very_easy",
                "infra_type": "static",
            },
        ]
    }
    path = tmp_path / "test_challenges.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return str(path)


@pytest.fixture
def bench_dir(tmp_path):
    """Create a mock benchmark directory structure."""
    # Docker challenge with docker-compose.yaml
    eval_dir = tmp_path / "benchmark" / "eval-me"
    eval_dir.mkdir(parents=True)
    (eval_dir / "docker-compose.yaml").write_text("version: '3'\nservices:\n  app:\n    image: test")

    # Static challenge (no docker files)
    crypto_dir = tmp_path / "benchmark" / "Dynastic"
    crypto_dir.mkdir(parents=True)
    (crypto_dir / "challenge.py").write_text("# crypto challenge")

    return str(tmp_path)


@pytest.fixture
def manager(registry_yaml, bench_dir):
    """Create a ChallengeManager instance."""
    registry = ChallengeRegistry(registry_yaml)
    return ChallengeManager(registry=registry, bench_dir=bench_dir)


class TestChallengeManager:
    @patch("subprocess.run")
    def test_setup_returns_url(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        url = manager.setup("eval-me")
        assert url == "http://localhost:32805"

    @patch("subprocess.run")
    def test_setup_calls_docker_compose(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        manager.setup("eval-me")

        # Should have called docker compose up -d
        calls = mock_run.call_args_list
        compose_call = [c for c in calls if "compose" in str(c)]
        assert len(compose_call) >= 1

    def test_setup_static_raises(self, manager):
        with pytest.raises(ValueError, match="static"):
            manager.setup("Dynastic")

    @patch("subprocess.run")
    def test_teardown_calls_docker_compose_down(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        manager.teardown("eval-me")

        calls = mock_run.call_args_list
        down_call = [c for c in calls if "down" in str(c)]
        assert len(down_call) >= 1

    @patch("subprocess.run")
    def test_health_check_returns_true(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=0, stdout="200", stderr="")
        assert manager.health_check("eval-me") is True

    @patch("subprocess.run")
    def test_health_check_returns_false_on_timeout(self, mock_run, manager):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="curl", timeout=5)
        assert manager.health_check("eval-me") is False

    @patch("subprocess.run")
    def test_get_running(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        manager.setup("eval-me")
        assert "eval-me" in manager.get_running()

    @patch("subprocess.run")
    def test_setup_all(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        results = manager.setup_all()
        # Only docker challenges should be launched
        assert "eval-me" in results
        assert "Dynastic" not in results

    @patch("subprocess.run")
    def test_teardown_all(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        manager.setup("eval-me")
        assert len(manager.get_running()) == 1
        manager.teardown_all()
        assert len(manager.get_running()) == 0

    @patch("subprocess.run")
    def test_setup_with_init_script(self, mock_run, manager, bench_dir):
        """If init_script.sh exists, it should be executed before docker compose."""
        # Create init_script.sh
        init_path = Path(bench_dir) / "benchmark" / "eval-me" / "init_script.sh"
        init_path.write_text("#!/bin/bash\necho 'building'")

        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
        manager.setup("eval-me")

        # Should have at least 2 calls: init_script + docker compose
        assert mock_run.call_count >= 2

    @patch("subprocess.run")
    def test_setup_failure_raises(self, mock_run, manager):
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="container error")
        with pytest.raises(RuntimeError, match="docker compose up failed"):
            manager.setup("eval-me")
