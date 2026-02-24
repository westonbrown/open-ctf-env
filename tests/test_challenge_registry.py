"""Tests for ChallengeRegistry."""

import os
import pytest
import yaml

from open_ctf.challenges.registry import ChallengeInfo, ChallengeRegistry


@pytest.fixture
def registry_yaml(tmp_path):
    """Create a test registry YAML."""
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
            {
                "id": "TimeKORP",
                "name": "[Easy] TimeKORP",
                "category": "web",
                "difficulty": "easy",
                "infra_type": "docker",
                "port": 32810,
            },
        ]
    }
    path = tmp_path / "test_challenges.yaml"
    with open(path, "w") as f:
        yaml.dump(data, f)
    return str(path)


class TestChallengeRegistry:
    def test_load_challenges(self, registry_yaml):
        reg = ChallengeRegistry(registry_yaml)
        assert len(reg) == 3

    def test_get_existing(self, registry_yaml):
        reg = ChallengeRegistry(registry_yaml)
        info = reg.get("eval-me")
        assert info.id == "eval-me"
        assert info.category == "misc"
        assert info.infra_type == "docker"
        assert info.port == 32805

    def test_get_unknown_raises(self, registry_yaml):
        reg = ChallengeRegistry(registry_yaml)
        with pytest.raises(KeyError, match="nonexistent"):
            reg.get("nonexistent")

    def test_list_docker_challenges(self, registry_yaml):
        reg = ChallengeRegistry(registry_yaml)
        docker = reg.list_docker_challenges()
        assert len(docker) == 2
        ids = {c.id for c in docker}
        assert ids == {"eval-me", "TimeKORP"}

    def test_list_static_challenges(self, registry_yaml):
        reg = ChallengeRegistry(registry_yaml)
        static = reg.list_static_challenges()
        assert len(static) == 1
        assert static[0].id == "Dynastic"

    def test_get_target_url_docker(self, registry_yaml):
        reg = ChallengeRegistry(registry_yaml)
        url = reg.get_target_url("eval-me")
        assert url == "http://localhost:32805"

    def test_get_target_url_custom_host(self, registry_yaml):
        reg = ChallengeRegistry(registry_yaml)
        url = reg.get_target_url("eval-me", host="192.168.1.100")
        assert url == "http://192.168.1.100:32805"

    def test_get_target_url_static_returns_none(self, registry_yaml):
        reg = ChallengeRegistry(registry_yaml)
        url = reg.get_target_url("Dynastic")
        assert url is None

    def test_contains(self, registry_yaml):
        reg = ChallengeRegistry(registry_yaml)
        assert "eval-me" in reg
        assert "nonexistent" not in reg

    def test_list_all(self, registry_yaml):
        reg = ChallengeRegistry(registry_yaml)
        all_challenges = reg.list_all()
        assert len(all_challenges) == 3

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            ChallengeRegistry("/nonexistent/path.yaml")


class TestChallengeInfo:
    def test_dataclass_defaults(self):
        info = ChallengeInfo(id="test", category="web", difficulty="easy", infra_type="docker")
        assert info.name == ""
        assert info.port is None
        assert info.ground_truth_flag is None

    def test_full_construction(self):
        info = ChallengeInfo(
            id="test",
            category="web",
            difficulty="easy",
            infra_type="docker",
            name="Test Challenge",
            port=8080,
            ground_truth_flag="FLAG{test}",
        )
        assert info.port == 8080
        assert info.ground_truth_flag == "FLAG{test}"
