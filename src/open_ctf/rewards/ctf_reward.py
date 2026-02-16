"""Combined CTF reward function for GRPO training.

Scoring components:
  - Flag capture (0.30): Exact match or pattern match
  - Skill grammar (0.20): RECON -> ENUM -> EXPLOIT phase ordering
  - Efficiency (0.35): Fewer steps = higher reward
  - Format compliance (0.15): Valid tool call structure

Noise (+-0.05) is added to every reward to guarantee variance for GRPO gradients.
"""

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Skill grammar patterns -- BoxPwnr tool names mapped to attack phases
# ---------------------------------------------------------------------------

SKILL_PATTERNS: Dict[str, List[str]] = {
    "recon": [
        r"nmap", r"masscan", r"ping", r"whois", r"dig",
        r"nslookup", r"traceroute", r"rustscan",
    ],
    "enum": [
        r"gobuster", r"ffuf", r"dirb", r"nikto",
        r"whatweb", r"wpscan", r"enum4linux", r"feroxbuster",
        r"nuclei", r"smbclient",
    ],
    "exploit": [
        r"sqlmap", r"hydra", r"python_code",
        r"python.*-c", r"flag_found",
        r"requests\.(get|post|put|delete)",
        r"subprocess\.", r"pty\.spawn",
        r"john", r"hashcat", r"msfconsole", r"metasploit",
        r"reverse.shell", r"nc\s+-[elp]", r"bash\s+-i",
    ],
}

# Tools/commands that appear across multiple phases -- scored by context
_MULTI_PHASE_TOOLS = {"curl", "wget", "ssh"}


def _classify_tool_call(name: str, arguments: str) -> Optional[str]:
    """Classify a tool call by function name and arguments into a phase."""
    name_lower = name.lower()
    args_lower = arguments.lower() if arguments else ""
    combined = f"{name_lower} {args_lower}"

    # Direct tool name matches first
    for phase, patterns in SKILL_PATTERNS.items():
        if any(re.search(p, name_lower) for p in patterns):
            return phase

    # Multi-phase tools: classify by argument content
    if any(t in name_lower or t in args_lower for t in _MULTI_PHASE_TOOLS):
        # Exploitation indicators
        if any(kw in args_lower for kw in ("sqlmap", "hydra", "-d ", "post", "exploit")):
            return "exploit"
        # Enumeration indicators
        if any(kw in args_lower for kw in ("gobuster", "ffuf", "dirb", "/admin", "robots")):
            return "enum"
        # Default: recon (simple curl/wget is reconnaissance)
        return "recon"

    # Fallback: check argument text against all patterns
    for phase, patterns in SKILL_PATTERNS.items():
        if any(re.search(p, combined) for p in patterns):
            return phase

    return None


# ---------------------------------------------------------------------------
# CTFReward
# ---------------------------------------------------------------------------


class CTFReward:
    """Combined CTF reward for GRPO training.

    Designed to be passed directly to ``GRPOTrainer(reward_funcs=[reward])``.

    The ``__call__`` signature matches TRL's expectation:
        reward_fn(completions, prompts=None, **kwargs) -> list[float]

    Extra metadata (``ground_truth_flag``, ``optimal_steps``) is forwarded via
    ``**kwargs`` by the trainer when the dataset contains those columns.
    """

    # GRPOTrainer accesses reward_func.__name__ for logging.
    __name__ = "ctf_reward"

    def __init__(
        self,
        flag_weight: float = 0.30,
        grammar_weight: float = 0.20,
        efficiency_weight: float = 0.35,
        format_weight: float = 0.15,
        noise_range: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        total = flag_weight + grammar_weight + efficiency_weight + format_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Reward weights must sum to 1.0, got {total:.4f} "
                f"(flag={flag_weight}, grammar={grammar_weight}, "
                f"efficiency={efficiency_weight}, format={format_weight})"
            )
        self.flag_weight = flag_weight
        self.grammar_weight = grammar_weight
        self.efficiency_weight = efficiency_weight
        self.format_weight = format_weight
        self.noise_range = noise_range
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        completions: List[Any],
        prompts: Optional[List[Any]] = None,
        **kwargs: Any,
    ) -> List[float]:
        """Score a batch of completions.

        Args:
            completions: List of completions. Each element is either a raw
                string or a list of message dicts (ChatML).
            prompts: (unused) kept for TRL compatibility.
            **kwargs: May contain ``ground_truth_flag`` and ``optimal_steps``
                lists forwarded from the dataset columns.

        Returns:
            List of float reward values, one per completion.
        """
        ground_truth_flags: List[Optional[str]] = kwargs.get(
            "ground_truth_flag", [None] * len(completions)
        )
        optimal_steps_list: List[Optional[int]] = kwargs.get(
            "optimal_steps", [None] * len(completions)
        )

        rewards: List[float] = []
        for idx, completion in enumerate(completions):
            text, tool_calls = self._extract(completion)
            gt_flag = (
                ground_truth_flags[idx]
                if idx < len(ground_truth_flags)
                else None
            )
            opt_steps = (
                optimal_steps_list[idx]
                if idx < len(optimal_steps_list)
                else None
            )

            score = (
                self.flag_weight * self._flag_score(text, gt_flag)
                + self.grammar_weight * self._grammar_score(tool_calls)
                + self.efficiency_weight * self._efficiency_score(len(tool_calls), opt_steps)
                + self.format_weight * self._format_score(tool_calls)
            )

            # Guarantee variance for GRPO
            score += self._rng.uniform(-self.noise_range, self.noise_range)
            rewards.append(score)

        return rewards

    # ------------------------------------------------------------------
    # Component scorers
    # ------------------------------------------------------------------

    def _flag_score(self, text: str, ground_truth: Optional[str]) -> float:
        """Exact flag match -> 1.0, realistic pattern -> 0.1, else 0.0."""
        if ground_truth and ground_truth in text:
            return 1.0
        # Pattern match: FLAG{<alphanumeric>} (excludes placeholders like FLAG{...})
        if re.search(r"FLAG\{[a-zA-Z0-9_-]+\}", text):
            return 0.1
        return 0.0

    def _grammar_score(self, tool_calls: List[Dict[str, str]]) -> float:
        """Score RECON -> ENUM -> EXPLOIT phase ordering (0.0 - 1.0).

        Classifies each tool call by its function name and arguments,
        then checks for correct phase ordering.
        """
        phases_seen: List[str] = []
        for tc in tool_calls:
            phase = _classify_tool_call(tc["name"], tc["arguments"])
            if phase and (not phases_seen or phases_seen[-1] != phase):
                phases_seen.append(phase)

        if not phases_seen:
            return 0.0

        # Phase presence (up to 0.6)
        presence = 0.0
        if "recon" in phases_seen:
            presence += 0.2
        if "enum" in phases_seen:
            presence += 0.2
        if "exploit" in phases_seen:
            presence += 0.2

        # Order adherence (up to 0.4)
        order = 0.0
        indices = {
            p: phases_seen.index(p) for p in ("recon", "enum", "exploit") if p in phases_seen
        }
        if "recon" in indices and "enum" in indices:
            if indices["recon"] < indices["enum"]:
                order += 0.2
        if "enum" in indices and "exploit" in indices:
            if indices["enum"] < indices["exploit"]:
                order += 0.2

        return min(1.0, presence + order)

    def _efficiency_score(self, actual_steps: int, optimal_steps: Optional[int]) -> float:
        """min(optimal / actual, 1.0). Returns 0.5 (neutral) without metadata."""
        if optimal_steps is None:
            return 0.5  # Neutral score when metadata unavailable
        if actual_steps == 0:
            return 0.0
        return min(optimal_steps / actual_steps, 1.0)

    def _format_score(self, tool_calls: List[Dict[str, str]]) -> float:
        """Score based on valid structured tool calls (not string matching)."""
        if not tool_calls:
            return 0.0

        valid = 0
        for tc in tool_calls:
            # Valid tool call has both a name and parseable arguments
            if tc["name"] and tc["arguments"]:
                try:
                    json.loads(tc["arguments"])
                    valid += 1
                except (json.JSONDecodeError, TypeError):
                    # Arguments present but not valid JSON
                    valid += 0.5
        return min(valid / len(tool_calls), 1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(completion: Any) -> Tuple[str, List[Dict[str, str]]]:
        """Extract flat text and structured tool calls from a completion.

        Returns:
            (text, tool_calls) where tool_calls is a list of
            {"name": str, "arguments": str} dicts.
        """
        if isinstance(completion, str):
            return completion, []
        if isinstance(completion, dict):
            # Single message dict (not wrapped in a list)
            content = completion.get("content") or ""
            tool_calls = []
            for tc in completion.get("tool_calls", []):
                func = tc.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", "")
                if isinstance(args, dict):
                    args = json.dumps(args)
                tool_calls.append({"name": name, "arguments": args or ""})
            return str(content), tool_calls
        if isinstance(completion, list):
            text_parts: List[str] = []
            tool_calls: List[Dict[str, str]] = []
            for msg in completion:
                if not isinstance(msg, dict):
                    text_parts.append(str(msg))
                    continue
                content = msg.get("content") or ""
                text_parts.append(str(content))
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    name = func.get("name", "")
                    args = func.get("arguments", "")
                    if isinstance(args, dict):
                        args = json.dumps(args)
                    tool_calls.append({"name": name, "arguments": args or ""})
            return "\n".join(text_parts), tool_calls
        return str(completion), []
