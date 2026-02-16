"""Combined CTF reward function for GRPO training.

Scoring components (4 signals, sum to 1.0):
  - Flag capture (0.50): metadata.success > exact match > pattern match
  - Efficiency (0.30): Fewer steps = higher reward, **gated on flag capture**
  - Uniqueness (0.10): Command diversity (repeated commands = stuck)
  - Format compliance (0.10): Valid tool call structure

Design principles:
  - **Principle of least action**: Flag (boundary condition) + efficiency (shortest
    path) are the only ground-truth-verified signals. Together they encode "reach
    the goal via the shortest path."
  - **Efficiency gating**: Efficiency is multiplied by flag_score so only successful
    traces receive credit for being fast. Prevents rewarding quitting early.
  - **No regex in process signals**: Uniqueness and format use structural analysis
    (command deduplication, JSON parsing) instead of regex pattern matching.
  - **metadata.success is authoritative in both directions**: True = solved,
    False = not solved (blocks exact-match fallback). Fixes 151 misscored
    successes where ground_truth_flag was a placeholder, and 22 failures where
    the flag string appeared in text but wasn't properly submitted.

Noise (+-0.05) is added to every reward to guarantee variance for GRPO gradients.
"""

import json
import random
import re
from typing import Any, Dict, List, Optional, Tuple

# Placeholder flag strings that should not be used for matching.
# "CHECK" is used by PortSwigger challenges (appears in system prompt).
_FLAG_PLACEHOLDERS = frozenset({
    "content_of_flag_here",
    "FLAG{...}",
    "",
    "CHECK",
})


class CTFReward:
    """Combined CTF reward for GRPO training.

    Designed to be passed directly to ``GRPOTrainer(reward_funcs=[reward])``.

    The ``__call__`` signature matches TRL's expectation:
        reward_fn(completions, prompts=None, **kwargs) -> list[float]

    Extra metadata (``ground_truth_flag``, ``optimal_steps``, ``metadata``)
    is forwarded via ``**kwargs`` by the trainer when the dataset contains
    those columns.
    """

    # GRPOTrainer accesses reward_func.__name__ for logging.
    __name__ = "ctf_reward"

    def __init__(
        self,
        flag_weight: float = 0.50,
        uniqueness_weight: float = 0.10,
        efficiency_weight: float = 0.30,
        format_weight: float = 0.10,
        noise_range: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        total = flag_weight + uniqueness_weight + efficiency_weight + format_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Reward weights must sum to 1.0, got {total:.4f} "
                f"(flag={flag_weight}, uniqueness={uniqueness_weight}, "
                f"efficiency={efficiency_weight}, format={format_weight})"
            )
        self.flag_weight = flag_weight
        self.uniqueness_weight = uniqueness_weight
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
            **kwargs: May contain ``ground_truth_flag``, ``optimal_steps``,
                and ``metadata`` lists forwarded from dataset columns.

        Returns:
            List of float reward values, one per completion.
        """
        n = len(completions)
        ground_truth_flags: List[Optional[str]] = kwargs.get(
            "ground_truth_flag", [None] * n
        )
        optimal_steps_list: List[Optional[int]] = kwargs.get(
            "optimal_steps", [None] * n
        )
        # metadata.success is the authoritative signal from BoxPwnr's
        # platform.validate_flag(). Extract from nested metadata dicts.
        metadata_list: List[Optional[Dict]] = kwargs.get("metadata", [None] * n)
        success_list: List[Optional[bool]] = kwargs.get("success", [None] * n)

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
            # Resolve metadata.success: try top-level 'success' first,
            # then fall back to nested metadata dict
            meta_success = None
            if idx < len(success_list) and success_list[idx] is not None:
                meta_success = bool(success_list[idx])
            elif idx < len(metadata_list) and isinstance(metadata_list[idx], dict):
                meta_success = metadata_list[idx].get("success")

            flag_sc = self._flag_score(text, gt_flag, meta_success)
            score = (
                self.flag_weight * flag_sc
                + self.uniqueness_weight * self._uniqueness_score(tool_calls)
                + self.efficiency_weight * self._efficiency_score(len(tool_calls), opt_steps) * flag_sc
                + self.format_weight * self._format_score(tool_calls)
            )

            # Guarantee variance for GRPO
            score += self._rng.uniform(-self.noise_range, self.noise_range)
            rewards.append(score)

        return rewards

    # ------------------------------------------------------------------
    # Component scorers
    # ------------------------------------------------------------------

    def _flag_score(
        self,
        text: str,
        ground_truth: Optional[str],
        metadata_success: Optional[bool] = None,
    ) -> float:
        """Score flag capture. metadata.success is authoritative in both directions.

        Priority order:
          1. metadata.success=True  -> 1.0 (platform confirmed solve)
          2. metadata.success=False -> 0.0 or 0.1 (platform says no solve;
             skip exact match, only allow partial pattern credit)
          3. metadata.success=None  -> exact match > pattern > 0.0
        """
        # 1. Authoritative positive: platform confirmed solve
        if metadata_success is True:
            return 1.0

        # 2. Authoritative negative: platform says not solved.
        #    Don't fall through to exact match (flag may appear in text
        #    without being properly submitted, or may be system prompt text).
        if metadata_success is False:
            if re.search(r"FLAG\{[a-zA-Z0-9_-]+\}", text):
                return 0.1
            return 0.0

        # 3. No metadata — use string matching as fallback.
        #    Skip placeholder flags and flags containing template text.
        if ground_truth and self._is_real_flag(ground_truth) and ground_truth in text:
            return 1.0

        # 4. Pattern match: FLAG{<alphanumeric>} (excludes FLAG{...})
        if re.search(r"FLAG\{[a-zA-Z0-9_-]+\}", text):
            return 0.1

        return 0.0

    @staticmethod
    def _is_real_flag(flag: str) -> bool:
        """Return True if the flag is a real value, not a placeholder or template."""
        if not flag:
            return False
        if flag in _FLAG_PLACEHOLDERS:
            return False
        # Detect template text that was accidentally captured as ground_truth_flag
        # (e.g., multiline strings containing "content_of_flag_here")
        if "content_of_flag_here" in flag:
            return False
        return True

    def _uniqueness_score(self, tool_calls: List[Dict[str, str]]) -> float:
        """Score command diversity (0.0 - 1.0). No regex.

        Measures the ratio of unique commands to total commands.
        Successful traces have ~97% unique commands (each step tries
        something new). Failed traces repeat ~30% of commands (stuck
        in loops). Correlation with success: r=0.381 (3.7x better
        than the regex-based grammar score it replaces).

        Returns 0.0 for no tool calls, 0.5 for tool calls without
        extractable commands (neutral).
        """
        if not tool_calls:
            return 0.0

        commands: List[str] = []
        for tc in tool_calls:
            cmd = self._extract_command(tc)
            if cmd:
                commands.append(cmd)

        if not commands:
            return 0.5  # Neutral for non-command tool calls

        return len(set(commands)) / len(commands)

    @staticmethod
    def _extract_command(tc: Dict[str, str]) -> str:
        """Extract the command string from a tool call's arguments.

        Handles common BoxPwnr argument schemas:
          - {"command": "..."} (shell_command, exec_command)
          - {"code": "..."} (python_code)
          - {"content": "..."} (flag_found)
          - {"query": "..."} (web_search, grep)
          - {"path": "..."} (read_file)
          - Plain string arguments
        """
        args_str = tc.get("arguments", "")
        if not args_str:
            return ""

        # Try JSON first
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else args_str
            if isinstance(args, dict):
                # Try common argument keys in priority order
                for key in ("command", "code", "content", "query", "path",
                            "pattern", "search_query", "stdin"):
                    val = args.get(key)
                    if val and isinstance(val, str):
                        return val.strip()
                # Fallback: use first non-empty string value
                for val in args.values():
                    if isinstance(val, str) and val.strip():
                        return val.strip()
            elif isinstance(args, str):
                return args.strip()
        except (json.JSONDecodeError, TypeError):
            # Not JSON — use raw string
            if isinstance(args_str, str):
                return args_str.strip()

        return ""

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
