"""Challenge registry — maps challenge IDs to infrastructure requirements."""

import logging
import re
from dataclasses import dataclass, field
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
    aliases: List[str] = field(default_factory=list)
    path_hint: Optional[str] = None


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

    @staticmethod
    def _normalize(value: str) -> str:
        """Normalize a challenge identifier/name for fuzzy matching."""
        return re.sub(r"[^a-z0-9]+", "", value.lower())

    @staticmethod
    def _tokenize(value: str) -> set[str]:
        """Tokenize text for loose challenge matching across naming schemes."""
        stop_tokens = {
            "the",
            "a",
            "an",
            "and",
            "very",
            "easy",
            "medium",
            "hard",
            "challenge",
            "ctf",
        }
        return {
            tok
            for tok in re.split(r"[^a-z0-9]+", value.lower())
            if tok and tok not in stop_tokens and not tok.isdigit()
        }

    def _load(self, config_path: str) -> None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Challenge registry not found: {config_path}")

        with open(path) as f:
            data = yaml.safe_load(f)

        for entry in data.get("challenges", []):
            aliases = entry.get("aliases", [])
            if aliases is None:
                aliases = []
            if not isinstance(aliases, list):
                aliases = [str(aliases)]
            info = ChallengeInfo(
                id=entry["id"],
                category=entry.get("category", ""),
                difficulty=entry.get("difficulty", ""),
                infra_type=entry.get("infra_type", "static"),
                name=entry.get("name", ""),
                port=entry.get("port"),
                ground_truth_flag=entry.get("ground_truth_flag"),
                aliases=[str(x) for x in aliases],
                path_hint=entry.get("path_hint"),
            )
            self._challenges[info.id] = info

        logger.info("Loaded %d challenges from %s", len(self._challenges), config_path)

    def resolve_id(self, challenge_id: str) -> Optional[str]:
        """Resolve challenge ID/name/alias to a canonical registry ID."""
        if challenge_id in self._challenges:
            return challenge_id

        query = self._normalize(challenge_id)
        if not query:
            return None

        query_tokens = self._tokenize(challenge_id)
        scores_by_id: Dict[str, int] = {}

        for cid, info in self._challenges.items():
            keys = [cid, info.name, *info.aliases]
            for key in keys:
                key_norm = self._normalize(key)
                if not key_norm:
                    continue

                score = 0
                if query == key_norm:
                    score = max(score, 100)
                elif query in key_norm:
                    score = max(score, 88)
                elif key_norm in query:
                    score = max(score, 82)

                key_tokens = self._tokenize(key)
                if query_tokens and key_tokens:
                    overlap = len(query_tokens & key_tokens)
                    if overlap:
                        precision = overlap / len(key_tokens)
                        recall = overlap / len(query_tokens)
                        score = max(score, int(65 + 30 * max(precision, recall)))
                        if key_tokens.issubset(query_tokens):
                            score = max(score, 86 + min(4, len(key_tokens)))
                        if query_tokens.issubset(key_tokens):
                            score = max(score, 84 + min(4, len(query_tokens)))

                if score > scores_by_id.get(cid, 0):
                    scores_by_id[cid] = score

        if not scores_by_id:
            return None

        ranked = sorted(scores_by_id.items(), key=lambda item: item[1], reverse=True)
        best_id, best_score = ranked[0]
        if best_score < 75:
            return None

        # Ambiguity guard: when top candidates are effectively tied, force explicit alias/path_hint.
        if len(ranked) > 1 and ranked[1][1] >= best_score - 2:
            logger.warning(
                "Ambiguous challenge resolution for %r: top candidates=%s",
                challenge_id,
                ranked[:3],
            )
            return None

        return best_id

    def get(self, challenge_id: str) -> ChallengeInfo:
        """Get challenge info by ID. Raises KeyError if not found."""
        resolved_id = self.resolve_id(challenge_id)
        if resolved_id is None:
            raise KeyError(f"Challenge not found: {challenge_id}")
        return self._challenges[resolved_id]

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
        return self.resolve_id(challenge_id) is not None
