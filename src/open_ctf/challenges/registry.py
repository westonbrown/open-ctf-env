"""Challenge registry — maps challenge IDs to infrastructure requirements."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ChallengeInfo:
    """Metadata for a single benchmark challenge."""
    id: str
    category: str
    difficulty: str
    infra_type: str  # "docker" or "static"
    name: str = ""
    port: Optional[int] = None
    ground_truth_flag: Optional[str] = None


class ChallengeRegistry:
    """Registry of benchmark challenges loaded from YAML config.

    Usage:
        registry = ChallengeRegistry("configs/challenges/cybench.yaml")
        info = registry.get("eval-me")
        url = registry.get_target_url("eval-me")  # -> "http://localhost:32805"
    """

    def __init__(self, config_path: str):
        self._challenges: Dict[str, ChallengeInfo] = {}
        self._load(config_path)

    def _load(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Challenge registry not found: {config_path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        for entry in data.get("challenges", []):
            info = ChallengeInfo(
                id=entry["id"],
                category=entry.get("category", ""),
                difficulty=entry.get("difficulty", ""),
                infra_type=entry.get("infra_type", "static"),
                name=entry.get("name", ""),
                port=entry.get("port"),
                ground_truth_flag=entry.get("ground_truth_flag"),
            )
            self._challenges[info.id] = info

        logger.info("Loaded %d challenges from %s", len(self._challenges), config_path)

    def get(self, challenge_id: str) -> ChallengeInfo:
        """Get challenge info by ID. Raises KeyError if not found."""
        if challenge_id not in self._challenges:
            raise KeyError(f"Challenge not found: {challenge_id}")
        return self._challenges[challenge_id]

    def list_all(self) -> List[ChallengeInfo]:
        """Return all challenges."""
        return list(self._challenges.values())

    def list_docker_challenges(self) -> List[ChallengeInfo]:
        """Return challenges that need Docker containers."""
        return [c for c in self._challenges.values() if c.infra_type == "docker"]

    def list_static_challenges(self) -> List[ChallengeInfo]:
        """Return file-based challenges (no server needed)."""
        return [c for c in self._challenges.values() if c.infra_type == "static"]

    def get_target_url(self, challenge_id: str, host: str = "localhost") -> Optional[str]:
        """Get the target URL for a challenge, or None for static challenges."""
        info = self.get(challenge_id)
        if info.infra_type == "docker" and info.port:
            return f"http://{host}:{info.port}"
        return None

    def __len__(self) -> int:
        return len(self._challenges)

    def __contains__(self, challenge_id: str) -> bool:
        return challenge_id in self._challenges
