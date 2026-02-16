"""Combined CTF reward function for GRPO training.

Scoring components (6 signals, sum to 1.0 + penalty):
  - Flag capture (0.20): metadata.success > exact match > pattern match
  - Efficiency (0.25): Fewer steps = higher reward, **ungated** for GRPO signal
  - Progression (0.15): RECON→ENUM→EXPLOIT phase ordering, **ungated**
  - Exploration (0.10): Novel tool usage weighted toward early trajectory
  - Uniqueness (0.10): Command diversity (repeated commands = stuck)
  - Format compliance (0.20): Valid tool call structure
  - Hallucination penalty: -0.10 for false flag submissions (structural)

Design principles:
  - **Offline-GRPO compatible**: In GRPO, the model generates completions without
    interacting with a real environment. The model can never produce the exact
    ground_truth_flag, so flag_score rarely exceeds 0.1. Efficiency, progression,
    format, exploration, and uniqueness are ALL ungated to provide meaningful
    gradient signal regardless of flag capture.
  - **Principle of least action**: Flag (boundary condition) + efficiency (shortest
    path) encode "reach the goal via the shortest path."
  - **No regex in process signals**: Progression uses set-based binary lookup (not
    regex pattern matching). Uniqueness, exploration, and format use structural
    analysis (command deduplication, position weighting, JSON parsing).
  - **metadata.success is authoritative in both directions**: True = solved,
    False = not solved (blocks exact-match fallback).
  - **Anti-hallucination**: Structural penalty when ``flag_found`` tool is called
    but the flag is wrong. Catches false submissions without regex.

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

# ---------------------------------------------------------------------------
# Phase classification for skill progression (set-based, NO regex).
# Classify commands by first token (binary name) of shell commands.
# ---------------------------------------------------------------------------
_RECON_BINARIES = frozenset({
    "nmap", "masscan", "ping", "traceroute", "whois", "dig", "nslookup",
    "host", "arp-scan", "netdiscover", "ftp", "smbclient", "smbmap",
    "rpcclient", "snmpwalk", "enum4linux",
})
_ENUM_BINARIES = frozenset({
    "curl", "wget", "gobuster", "ffuf", "dirb", "dirsearch", "nikto",
    "wpscan", "whatweb", "ls", "cat", "head", "tail", "find", "grep",
    "egrep", "fgrep", "strings", "file", "id", "whoami", "ps", "env",
    "uname", "hostname", "ip", "ifconfig", "netstat", "ss", "wc",
    "sort", "uniq", "less", "more", "xxd", "hexdump", "objdump",
    "readelf", "cd", "echo", "which", "sed", "awk", "apt", "apt-get",
    "pip", "pip3", "export", "mkdir", "cp", "mv", "rm", "printf",
    "crackmapexec", "fls", "tesseract", "unzip", "tar", "gunzip",
})
_EXPLOIT_BINARIES = frozenset({
    "sqlmap", "hydra", "john", "hashcat", "python", "python3", "python2",
    "ruby", "perl", "gcc", "g++", "make", "nc", "ncat", "netcat",
    "ssh", "scp", "msfconsole", "msfvenom", "chmod", "chown",
    "gdb", "ltrace", "strace", "pwntools", "sshpass", "bash", "node",
    "java", "docker", "php", "socat",
})
# Tool names (not shell commands) that map directly to phases.
_TOOL_NAME_PHASES = {
    "web_search": "recon",
    "read_file": "enum", "grep": "enum", "file_search": "enum",
    "python_code": "exploit", "apply_patch": "exploit",
    "flag_found": "flag",
}


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
        flag_weight: float = 0.20,
        efficiency_weight: float = 0.25,
        progression_weight: float = 0.15,
        exploration_weight: float = 0.10,
        uniqueness_weight: float = 0.10,
        format_weight: float = 0.20,
        hallucination_penalty: float = 0.10,
        noise_range: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        total = (flag_weight + efficiency_weight + progression_weight
                 + exploration_weight + uniqueness_weight + format_weight)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Reward weights must sum to 1.0, got {total:.4f}"
            )
        self.flag_weight = flag_weight
        self.efficiency_weight = efficiency_weight
        self.progression_weight = progression_weight
        self.exploration_weight = exploration_weight
        self.uniqueness_weight = uniqueness_weight
        self.format_weight = format_weight
        self.hallucination_penalty = hallucination_penalty
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
            # All components ungated for offline GRPO:
            # Model generates completions without environment interaction,
            # so flag_sc rarely exceeds 0.1. Gating efficiency/progression
            # on flag_sc starves the gradient signal (85% of weight → 0).
            # Flag capture still provides a bonus when it occurs.
            score = (
                self.flag_weight * flag_sc
                + self.efficiency_weight * self._efficiency_score(len(tool_calls), opt_steps)
                + self.progression_weight * self._progression_score(tool_calls)
                + self.exploration_weight * self._exploration_score(tool_calls)
                + self.uniqueness_weight * self._uniqueness_score(tool_calls)
                + self.format_weight * self._format_score(tool_calls)
                + self._hallucination_score(tool_calls, flag_sc)
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
        """min(optimal / actual, 1.0). Returns 0.5 (neutral) without metadata.

        Ungated: provides gradient signal in offline GRPO even without flag capture.
        """
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

    def _progression_score(self, tool_calls: List[Dict[str, str]]) -> float:
        """Score RECON→ENUM→EXPLOIT phase presence and ordering.

        Uses set-based binary lookup on the first token of shell commands
        (no regex). Tool names like ``python_code`` or ``read_file`` are
        classified directly by name.

        Scoring: 0.6 for phase presence + 0.4 for correct ordering.
        """
        if not tool_calls:
            return 0.0

        # Build deduplicated phase sequence
        phases: List[str] = []
        for tc in tool_calls:
            phase = self._classify_phase(tc)
            if phase and (not phases or phases[-1] != phase):
                phases.append(phase)

        if not phases:
            return 0.0

        has_recon = "recon" in phases
        has_enum = "enum" in phases
        has_exploit = "exploit" in phases

        # Phase presence (0.0 - 0.6)
        presence = 0.2 * has_recon + 0.2 * has_enum + 0.2 * has_exploit

        # Order adherence (0.0 - 0.4)
        order = 0.0
        if has_recon and has_enum:
            if phases.index("recon") < phases.index("enum"):
                order += 0.2
        if has_enum and has_exploit:
            if phases.index("enum") < phases.index("exploit"):
                order += 0.2

        return min(presence + order, 1.0)

    @staticmethod
    def _classify_phase(tc: Dict[str, str]) -> Optional[str]:
        """Classify a tool call into a CTF phase. Set-based, no regex."""
        name = tc.get("name", "")

        # Direct tool name classification
        if name in _TOOL_NAME_PHASES:
            return _TOOL_NAME_PHASES[name]

        # For shell_command / exec_command, classify by first token
        if name in ("shell_command", "exec_command", "execute_command",
                    "shell", "bash", "Bash"):
            cmd = CTFReward._extract_command(tc)
            if not cmd:
                return None
            # Get binary name (first token, strip path)
            first_token = cmd.split()[0].rsplit("/", 1)[-1].lower()
            if first_token in _RECON_BINARIES:
                return "recon"
            if first_token in _ENUM_BINARIES:
                return "enum"
            if first_token in _EXPLOIT_BINARIES:
                return "exploit"

        return None

    def _exploration_score(self, tool_calls: List[Dict[str, str]]) -> float:
        """Reward novel tool names weighted toward early trajectory.

        Early exploration (trying new tools at the start) is more valuable
        than late exploration. Score = Σ (1 - t/T) for each novel tool,
        normalized to [0, 1]. No regex.
        """
        if not tool_calls:
            return 0.0

        T = len(tool_calls)
        seen: set = set()
        score = 0.0
        max_possible = 0.0

        for t, tc in enumerate(tool_calls):
            decay = 1.0 - (t / T) if T > 1 else 1.0
            max_possible += decay
            name = tc.get("name", "")
            if name and name not in seen:
                seen.add(name)
                score += decay

        return score / max_possible if max_possible > 0 else 0.0

    def _hallucination_score(
        self, tool_calls: List[Dict[str, str]], flag_sc: float
    ) -> float:
        """Penalty for false flag submissions. Structural, no regex.

        Fires when ``flag_found`` tool was called but flag_score < 1.0,
        meaning the submitted flag was wrong or unverified.
        Returns a negative value (penalty) or 0.0.
        """
        if flag_sc >= 1.0:
            return 0.0  # Correct flag — no penalty

        for tc in tool_calls:
            if tc.get("name") == "flag_found":
                return -self.hallucination_penalty

        return 0.0

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
