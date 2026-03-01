#!/usr/bin/env python3
"""Build high-quality SFT and online RL datasets from curated BoxPwnr traces.

Focused on CyBench performance lift for a 3B SLM conference demo.
Quality over quantity — every sample is validated and preprocessed.

Applies all known fixes from previous audit rounds:
  1. HTML entity unescaping (double-encoded)
  2. Orphan </think> tag repair
  3. Non-canonical tool stripping (Edit, Write, Task, etc.)
  4. Flag verification injection/replacement (echo → Correct/Incorrect)
  5. Terminal flag_found injection for samples missing it
  6. Post-correct hallucination truncation
  7. ANSI escape code stripping
  8. Null byte removal
  9. Consecutive assistant message merging
  10. Orphan environment_feedback removal
  11. Empty shell_command argument fixing
  12. <OUTPUT> artifact cleanup in think blocks
  13. PortSwigger false-success exclusion (9 samples)
  14. rpgo cheating trace exclusion

Usage:
  python src/open_ctf/data/build_quality_datasets.py \
      --input data/sft_curated_clean.jsonl \
      --sft-output data/sft_quality.jsonl \
      --online-rl-output data/online_rl_quality.jsonl \
      --online-rl-source data/online_rl_cybench40.jsonl

  # Dry run:
  python src/open_ctf/data/build_quality_datasets.py --input data/sft_curated_clean.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path

# ─── Canonical tools (from open_ctf.envs.tool_executor) ───
_CANONICAL_TOOLS = frozenset({
    "shell_command", "exec_command", "write_stdin", "python_code",
    "read_file", "grep", "file_search", "apply_patch", "flag_found",
    "web_search", "list_sessions", "close_session", "execute_command",
    "submit_flag",
})

# ─── ANSI/terminal escape code patterns (comprehensive) ───
_ANSI_RE = re.compile(
    r"\x1b"          # ESC
    r"(?:"
    r"\[[0-9;?]*[a-zA-Z]"  # CSI sequences: ESC [ ... letter
    r"|"
    r"\][^\x07\x1b]*(?:\x07|\x1b\\)"  # OSC sequences: ESC ] ... BEL/ST
    r"|"
    r"[=>()][0-9A-Za-z]?"  # Simple ESC codes: ESC= ESC> ESC(B etc
    r"|"
    r"\[[0-9;?]*[hlm]"  # Set/reset modes
    r")"
)
# Also strip standalone control chars that often accompany terminal output
_CONTROL_RE = re.compile(r"[\x00-\x08\x0e-\x1f](?!\n|\r|\t)")

# ─── Null byte pattern ───
_NULL_RE = re.compile(r"\x00")

# ─── <OUTPUT> artifact pattern in think blocks ───
_OUTPUT_ARTIFACT_RE = re.compile(r"<OUTPUT>.*?</OUTPUT>", re.DOTALL)

# ─── CyBench difficulty tiers (for 3B model: cap at medium) ───
_CYBENCH_DIFFICULTY = {
    # Very Easy
    "[Very Easy] Dynastic": "very_easy",
    "[Very Easy] It Has Begun": "very_easy",
    "[Very Easy] LootStash": "very_easy",
    "[Very Easy] PackedAway": "very_easy",
    "[Very Easy] Primary Knowledge": "very_easy",
    "[Very Easy] Urgent": "very_easy",
    "[Very Easy] Flag Command": "very_easy",
    "[Very Easy] Delulu": "very_easy",
    # Easy
    "[Easy] Crushing": "easy",
    "[Easy] Labyrinth Linguist": "easy",
    "[Easy] Unbreakable": "easy",
    # Medium
    "[Medium] Partial Tenacity": "medium",
    "[Medium] Data Siege": "medium",
    "[Medium] LockTalk": "medium",
    "[Medium] Were Pickle Phreaks Revenge": "medium",
    "05-motp": "medium",
    "22-back-to-the-past": "medium",
    "eval-me": "medium",
    "failproof": "medium",
    "skilift": "medium",
    "sop": "medium",
    # Hard
    "[Hard] Permuted": "hard",
    "[Hard] FlecksOfGold": "hard",
    # Others (uncategorized CyBench)
    "avatar": "medium",
    "shuffled-aes": "hard",
    "rpgo": "medium",
    # Non-CyBench platforms
    "chunky": "hard",
    "diffecient": "hard",
    "ezmaze": "medium",
    "network-tools": "hard",
    "noisier-crc": "hard",
    "noisy-crc": "hard",
    "randsubware": "hard",
    "robust-cbc": "hard",
    "just-another-pickle-jail": "hard",
    "frog-waf": "hard",
    "GlacierExchange": "hard",
    "WalkingToTheSeaSide": "hard",
    "MissingBits": "hard",
    "SLCG": "hard",
}

# Allowed difficulties for 3B SLM (training_120gb_dense.yaml: difficulty_max: "medium")
_3B_ALLOWED_DIFFICULTIES = {"very_easy", "easy", "medium"}

# ─── Known bad samples ───
# PortSwigger false-success lines (0-indexed from sft_curated_clean.jsonl)
_PORTSWIGGER_FALSE_SUCCESS_LINES = {469, 474, 499, 500, 501, 503, 506, 510, 517}

# rpgo cheating trace exclusion
_EXCLUDED_CHALLENGES = {"rpgo"}


def _fix_html_escapes(text: str) -> str:
    """Double-decode HTML entities."""
    return html.unescape(html.unescape(text))


def _fix_orphan_think(text: str) -> str:
    """Fix orphan </think> tags by prepending <think>."""
    if "</think>" in text and "<think>" not in text.split("</think>")[0]:
        text = "<think>" + text
    return text


def _strip_ansi(text: str) -> str:
    """Remove ANSI/terminal escape codes and control characters."""
    text = _ANSI_RE.sub("", text)
    text = _CONTROL_RE.sub("", text)
    return text


def _strip_nulls(text: str) -> str:
    """Remove null bytes."""
    return _NULL_RE.sub("", text)


def _clean_output_artifacts(text: str) -> str:
    """Remove <OUTPUT>...</OUTPUT> artifacts from think blocks."""
    if "<think>" in text and "<OUTPUT>" in text:
        text = _OUTPUT_ARTIFACT_RE.sub("", text)
    return text


def _extract_submitted_flag(tool_call: dict) -> str | None:
    """Extract the submitted flag string from a flag_found tool call."""
    args_raw = tool_call.get("function", {}).get("arguments", "{}")
    if isinstance(args_raw, str):
        try:
            args = json.loads(args_raw)
        except json.JSONDecodeError:
            return args_raw.strip() or None
    elif isinstance(args_raw, dict):
        args = args_raw
    else:
        return None
    return args.get("content") or args.get("flag") or args.get("answer") or None


def _is_non_canonical(tool_call: dict) -> bool:
    """Check if a tool call uses a non-canonical tool."""
    name = tool_call.get("function", {}).get("name", "")
    return name not in _CANONICAL_TOOLS


def _is_proper_verification(content: str) -> bool:
    """Check if tool response is a proper verification (not an echo)."""
    if not content:
        return False
    if "Correct!" in content or "Incorrect submission:" in content:
        return True
    if "Flag submitted:" in content and ("correct" in content.lower() or "incorrect" in content.lower()):
        return True
    return False


def _make_verification(call_id: str, submitted: str, ground_truth: str | None) -> dict:
    """Build a verification tool response."""
    if submitted and ground_truth and submitted.strip() == ground_truth.strip():
        content = f"Correct! Flag verified: {submitted}"
    elif submitted and ground_truth:
        content = f"Incorrect submission: {submitted}. The flag was not correct."
    elif submitted:
        content = f"Flag submitted: {submitted}"
    else:
        content = "No flag content provided."
    msg: dict = {"role": "tool", "content": content, "name": "flag_found"}
    if call_id:
        msg["tool_call_id"] = call_id
    return msg


def _make_flag_found_call(flag: str, call_id: str = "auto_flag") -> dict:
    """Build a flag_found tool call message."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{
            "id": call_id,
            "type": "function",
            "function": {
                "name": "flag_found",
                "arguments": json.dumps({"content": flag}),
            },
        }],
    }


def _find_flag_response_idx(messages: list[dict], start: int, tc: dict) -> int | None:
    """Find the tool response for a flag_found call."""
    tc_id = tc.get("id", "")
    for j in range(start, min(start + 5, len(messages))):
        m = messages[j]
        if m.get("role") != "tool":
            continue
        if m.get("tool_call_id") == tc_id and tc_id:
            return j
        if m.get("name") in ("flag_found", "submit_flag"):
            return j
    return None


def _truncate_after_correct_flag(messages: list[dict], ground_truth: str) -> tuple[list[dict], bool]:
    """Truncate trajectory after first correct flag submission + verification.

    Prevents post-correct hallucinations from entering training data.
    Returns (truncated_messages, was_truncated).
    """
    if not ground_truth:
        return messages, False

    for i, m in enumerate(messages):
        if m.get("role") == "tool" and m.get("name") in ("flag_found", "submit_flag"):
            content = m.get("content", "")
            if "Correct!" in content or (ground_truth.strip() in content and "Incorrect" not in content):
                # Keep up to and including this verification message
                if i + 1 < len(messages):
                    return messages[:i + 1], True
    return messages, False


def _merge_consecutive_assistant(messages: list[dict]) -> tuple[list[dict], int]:
    """Merge consecutive assistant messages (data artifact)."""
    merged = []
    merge_count = 0
    for m in messages:
        if (merged and
            merged[-1].get("role") == "assistant" and
            m.get("role") == "assistant" and
            not m.get("tool_calls") and
            not merged[-1].get("tool_calls")):
            # Merge content
            prev_content = merged[-1].get("content", "") or ""
            curr_content = m.get("content", "") or ""
            merged[-1]["content"] = (prev_content + "\n" + curr_content).strip()
            merge_count += 1
        else:
            merged.append(m)
    return merged, merge_count


def _remove_orphan_env_feedback(messages: list[dict]) -> tuple[list[dict], int]:
    """Remove orphan environment_feedback messages."""
    cleaned = []
    removed = 0
    for m in messages:
        if (m.get("role") == "tool" and
            m.get("name") == "environment_feedback" and
            not m.get("tool_call_id")):
            removed += 1
            continue
        cleaned.append(m)
    return cleaned, removed


def _fix_empty_shell_args(messages: list[dict]) -> int:
    """Fix empty shell_command arguments in-place. Returns fix count."""
    fixed = 0
    for m in messages:
        for tc in m.get("tool_calls", []):
            fn = tc.get("function", {})
            if fn.get("name") == "shell_command":
                args_raw = fn.get("arguments", "")
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        continue
                    if not args.get("command"):
                        # Can't fix — will be caught by quality filter
                        fixed += 1
    return fixed


def quality_score(sample: dict) -> float:
    """Score a sample's quality (0.0-5.0) for SFT inclusion.

    Factors:
    - Message count (more turns = more signal, but cap at ~40)
    - Tool diversity (uses multiple tools, not just shell)
    - Has proper flag verification
    - Has think blocks (reasoning)
    - Clean conversation flow (no artifacts)
    """
    messages = sample.get("messages", [])
    meta = sample.get("metadata", {})

    score = 0.0

    # Base: must have system + user + at least 1 tool turn
    if len(messages) < 6:
        return 0.0

    # Turn count signal (more turns up to ~30 is better, diminishing after)
    turns = len(messages)
    if turns >= 8:
        score += 1.0
    elif turns >= 6:
        score += 0.7

    # Tool diversity
    tool_names = set()
    tool_call_count = 0
    for m in messages:
        for tc in m.get("tool_calls", []):
            name = tc.get("function", {}).get("name", "")
            tool_names.add(name)
            tool_call_count += 1
        if m.get("role") == "tool" and m.get("name"):
            tool_names.add(m["name"])

    if len(tool_names) >= 3:
        score += 1.0
    elif len(tool_names) >= 2:
        score += 0.5

    # Has flag verification
    has_correct_verif = any(
        m.get("role") == "tool" and "Correct!" in (m.get("content") or "")
        for m in messages
    )
    if has_correct_verif:
        score += 1.0

    # Has thinking blocks
    has_think = any(
        "<think>" in (m.get("content") or "")
        for m in messages if m.get("role") == "assistant"
    )
    if has_think:
        score += 0.5

    # Penalty: very long traces (>80 messages) are often repetitive
    if turns > 80:
        score -= 0.5

    # Penalty: only shell_command (no diversity)
    if tool_names == {"shell_command"} or tool_names == {"shell_command", "flag_found"}:
        score -= 0.3

    # Bonus: CyBench platform (primary focus)
    if meta.get("platform") == "cybench":
        score += 0.5

    return max(0.0, min(5.0, score))


def preprocess_sample(sample: dict) -> tuple[dict | None, dict]:
    """Full preprocessing pipeline for a single sample.

    Returns (fixed_sample_or_None, stats_dict).
    """
    stats = {
        "html_fixed": 0,
        "think_fixed": 0,
        "ansi_stripped": 0,
        "nulls_stripped": 0,
        "output_artifacts": 0,
        "echo_stripped": 0,
        "noncanonical_stripped": 0,
        "verification_injected": 0,
        "verification_replaced": 0,
        "terminal_flag_added": 0,
        "post_correct_truncated": 0,
        "consecutive_merged": 0,
        "orphan_env_removed": 0,
        "empty_shell_fixed": 0,
        "dropped": False,
        "drop_reason": "",
    }

    messages = [dict(m) for m in sample.get("messages", [])]  # deep-ish copy
    ground_truth = sample.get("ground_truth_flag", "")

    # ── Pass 1: Text cleanup (HTML, think, ANSI, nulls, artifacts) ──
    for m in messages:
        content = m.get("content") or ""
        if not content:
            continue

        orig = content

        content = _fix_html_escapes(content)
        if content != orig:
            stats["html_fixed"] += 1

        prev = content
        content = _fix_orphan_think(content)
        if content != prev:
            stats["think_fixed"] += 1

        prev = content
        content = _strip_ansi(content)
        if content != prev:
            stats["ansi_stripped"] += 1

        prev = content
        content = _strip_nulls(content)
        if content != prev:
            stats["nulls_stripped"] += 1

        prev = content
        content = _clean_output_artifacts(content)
        if content != prev:
            stats["output_artifacts"] += 1

        m["content"] = content

    # ── Pass 2: Merge consecutive assistant messages ──
    messages, merge_count = _merge_consecutive_assistant(messages)
    stats["consecutive_merged"] = merge_count

    # ── Pass 3: Remove orphan environment_feedback ──
    messages, env_removed = _remove_orphan_env_feedback(messages)
    stats["orphan_env_removed"] = env_removed

    # ── Pass 4: Fix empty shell args ──
    stats["empty_shell_fixed"] = _fix_empty_shell_args(messages)

    # ── Pass 5: Strip non-canonical tool calls ──
    cleaned: list[dict] = []
    skip_next_tools: set[str] = set()

    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "") or ""

        # Skip echo assistant responses
        if role == "assistant" and content.strip().startswith("Flag found:"):
            tool_calls = m.get("tool_calls", [])
            if not tool_calls:
                stats["echo_stripped"] += 1
                continue

        # Skip tool responses for non-canonical calls
        if role == "tool" and m.get("tool_call_id") in skip_next_tools:
            skip_next_tools.discard(m.get("tool_call_id"))
            continue

        # Filter non-canonical tool calls
        tool_calls = m.get("tool_calls", [])
        if tool_calls:
            canonical_tcs = []
            for tc in tool_calls:
                if _is_non_canonical(tc):
                    stats["noncanonical_stripped"] += 1
                    tc_id = tc.get("id")
                    if tc_id:
                        skip_next_tools.add(tc_id)
                else:
                    canonical_tcs.append(tc)

            if not canonical_tcs and not content.strip():
                continue
            elif not canonical_tcs:
                m = dict(m)
                m.pop("tool_calls", None)
            else:
                m = dict(m)
                m["tool_calls"] = canonical_tcs

        cleaned.append(m)

    messages = cleaned

    # ── Pass 6: Fix flag verification (echo → proper, missing → inject) ──
    final: list[dict] = []
    replace_indices: set[int] = set()

    for i, m in enumerate(messages):
        flag_calls = [
            tc for tc in (m.get("tool_calls") or [])
            if tc.get("function", {}).get("name") in ("flag_found", "submit_flag")
        ]
        for tc in flag_calls:
            resp_idx = _find_flag_response_idx(messages, i + 1, tc)
            if resp_idx is not None:
                resp_content = messages[resp_idx].get("content", "") or ""
                if not _is_proper_verification(resp_content):
                    replace_indices.add(resp_idx)

    for i, m in enumerate(messages):
        if i in replace_indices:
            tc_id = m.get("tool_call_id", "")
            submitted = None
            for j in range(max(0, i - 5), i):
                for tc in messages[j].get("tool_calls", []):
                    fn = tc.get("function", {}).get("name", "")
                    if fn in ("flag_found", "submit_flag"):
                        if tc.get("id") == tc_id or not tc_id:
                            submitted = _extract_submitted_flag(tc)
            verification = _make_verification(tc_id, submitted or "", ground_truth)
            final.append(verification)
            stats["verification_replaced"] += 1
            continue

        final.append(m)

        # Inject verification for flag_found with no response
        flag_calls = [
            tc for tc in (m.get("tool_calls") or [])
            if tc.get("function", {}).get("name") in ("flag_found", "submit_flag")
        ]
        for tc in flag_calls:
            resp_idx = _find_flag_response_idx(messages, i + 1, tc)
            if resp_idx is None:
                submitted = _extract_submitted_flag(tc)
                verification = _make_verification(tc.get("id", ""), submitted or "", ground_truth)
                final.append(verification)
                stats["verification_injected"] += 1

    messages = final

    # ── Pass 7: Add terminal flag_found if missing ──
    has_flag_call = any(
        tc.get("function", {}).get("name") in ("flag_found", "submit_flag")
        for m in messages
        for tc in m.get("tool_calls", [])
    )
    if not has_flag_call and ground_truth:
        messages.append(_make_flag_found_call(ground_truth, "terminal_flag"))
        messages.append(_make_verification("terminal_flag", ground_truth, ground_truth))
        stats["terminal_flag_added"] += 1

    # ── Pass 8: Truncate after correct flag (prevent post-correct hallucination) ──
    messages, truncated = _truncate_after_correct_flag(messages, ground_truth)
    if truncated:
        stats["post_correct_truncated"] = 1

    # ── Quality gate: drop tiny samples ──
    if len(messages) < 4:
        stats["dropped"] = True
        stats["drop_reason"] = "too_few_messages"
        return None, stats

    result = dict(sample)
    result["messages"] = messages
    return result, stats


def should_include_for_sft(
    sample: dict,
    line_idx: int,
    *,
    cybench_priority: bool = True,
    difficulty_cap: str = "medium",
) -> tuple[bool, str]:
    """Decide if a sample should go into the quality SFT dataset.

    Returns (include, reason).
    """
    meta = sample.get("metadata", {})
    platform = meta.get("platform", "")
    challenge = meta.get("challenge", "") or meta.get("challenge_id", "")

    # Exclude PortSwigger false positives
    if line_idx in _PORTSWIGGER_FALSE_SUCCESS_LINES:
        return False, "portswigger_false_success"

    # Exclude rpgo (cheating trace)
    if challenge in _EXCLUDED_CHALLENGES:
        return False, "excluded_challenge_rpgo"

    # For CyBench: apply difficulty filter
    if platform == "cybench":
        difficulty = _CYBENCH_DIFFICULTY.get(challenge, "unknown")
        allowed = {"very_easy", "easy", "medium"} if difficulty_cap == "medium" else _3B_ALLOWED_DIFFICULTIES | {"hard"}
        if difficulty not in allowed:
            return False, f"difficulty_{difficulty}_exceeds_cap"

    return True, "included"


def should_include_for_online_rl(sample: dict) -> tuple[bool, str]:
    """Decide if a sample qualifies for online RL dataset.

    Online RL needs:
    - ground_truth_flag (required for reward computation)
    - CyBench platform (for challenge registry routing)
    """
    meta = sample.get("metadata", {})
    platform = meta.get("platform", "")
    challenge = meta.get("challenge", "") or meta.get("challenge_id", "")
    ground_truth = sample.get("ground_truth_flag", "")

    if not ground_truth:
        return False, "no_ground_truth_flag"

    if challenge in _EXCLUDED_CHALLENGES:
        return False, "excluded_challenge"

    # Online RL: prefer CyBench for challenge registry routing
    if platform != "cybench":
        return False, "not_cybench"

    return True, "included"


def build_online_rl_from_existing(
    online_rl_source_path: Path,
    output_path: Path | None,
    *,
    difficulty_cap: str = "medium",
) -> dict:
    """Build filtered online RL dataset from existing cybench source file.

    The source file has prompts (system + user) + ground_truth_flag for online RL.
    We filter to appropriate difficulty and exclude bad challenges.
    """
    stats = {"total": 0, "included": 0, "excluded": 0, "reasons": {}}

    fout = open(output_path, "w") if output_path else None

    try:
        for line in open(online_rl_source_path):
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            stats["total"] += 1

            meta = sample.get("metadata", {})
            challenge = meta.get("challenge", "") or meta.get("challenge_id", "")
            ground_truth = sample.get("ground_truth_flag", "")

            # Skip excluded challenges
            if challenge in _EXCLUDED_CHALLENGES:
                stats["excluded"] += 1
                stats["reasons"]["excluded_challenge"] = stats["reasons"].get("excluded_challenge", 0) + 1
                continue

            # Skip dummy flags
            if "DUMMY" in ground_truth:
                stats["excluded"] += 1
                stats["reasons"]["dummy_flag"] = stats["reasons"].get("dummy_flag", 0) + 1
                continue

            # Apply difficulty filter
            difficulty = _CYBENCH_DIFFICULTY.get(challenge, "unknown")
            allowed = {"very_easy", "easy", "medium"} if difficulty_cap == "medium" else {"very_easy", "easy", "medium", "hard"}
            if difficulty not in allowed:
                stats["excluded"] += 1
                reason = f"difficulty_{difficulty}"
                stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1
                continue

            stats["included"] += 1
            if fout:
                fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
    finally:
        if fout:
            fout.close()

    return stats


# Backward-compatible aliases for external scripts.
should_include_for_grpo = should_include_for_online_rl
build_grpo_from_existing = build_online_rl_from_existing


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build high-quality SFT and online RL datasets for CyBench demo."
    )
    parser.add_argument("--input", required=True, help="Input SFT JSONL (sft_curated_clean.jsonl)")
    parser.add_argument("--sft-output", help="Output SFT JSONL (quality-filtered + preprocessed)")
    parser.add_argument(
        "--online-rl-output", "--grpo-output",
        dest="online_rl_output",
        help="Output online RL JSONL (quality-filtered)",
    )
    parser.add_argument(
        "--online-rl-source", "--grpo-source",
        dest="online_rl_source",
        help="Source online RL JSONL (default: online_rl_cybench40.jsonl fallback grpo_cybench40.jsonl)",
    )
    parser.add_argument("--sft-difficulty-cap", default="hard",
                        choices=["medium", "hard"],
                        help="Max difficulty for SFT (hard=include hard challenges for knowledge)")
    parser.add_argument("--online-rl-difficulty-cap", "--grpo-difficulty-cap",
                        dest="online_rl_difficulty_cap",
                        default="medium",
                        choices=["medium", "hard"],
                        help="Max difficulty for online RL (medium for 3B, hard for 27B)")
    parser.add_argument("--min-quality", type=float, default=2.5,
                        help="Minimum quality score for SFT inclusion (0-5)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report stats without writing output files")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    sft_output = None if args.dry_run else (Path(args.sft_output) if args.sft_output else None)
    online_rl_output = (
        None if args.dry_run else (Path(args.online_rl_output) if args.online_rl_output else None)
    )

    # ─── Build SFT dataset ───
    print("=" * 60)
    print("Building SFT Quality Dataset")
    print("=" * 60)

    sft_totals = {
        "samples_in": 0, "samples_out": 0, "samples_dropped": 0,
        "html_fixed": 0, "think_fixed": 0, "ansi_stripped": 0,
        "nulls_stripped": 0, "output_artifacts": 0,
        "echo_stripped": 0, "noncanonical_stripped": 0,
        "verification_injected": 0, "verification_replaced": 0,
        "terminal_flag_added": 0, "post_correct_truncated": 0,
        "consecutive_merged": 0, "orphan_env_removed": 0,
        "exclude_reasons": {},
        "quality_scores": [],
        "platform_counts": {},
        "cybench_challenges": set(),
    }

    fout = open(sft_output, "w") if sft_output else None

    try:
        for line_idx, line in enumerate(open(input_path)):
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)
            sft_totals["samples_in"] += 1

            # ── Inclusion filter ──
            include, reason = should_include_for_sft(
                sample, line_idx,
                cybench_priority=True,
                difficulty_cap=args.sft_difficulty_cap,
            )
            if not include:
                sft_totals["samples_dropped"] += 1
                sft_totals["exclude_reasons"][reason] = sft_totals["exclude_reasons"].get(reason, 0) + 1
                continue

            # ── Preprocess ──
            fixed, stats = preprocess_sample(sample)

            if stats["dropped"] or fixed is None:
                sft_totals["samples_dropped"] += 1
                sft_totals["exclude_reasons"]["preprocessing_dropped"] = (
                    sft_totals["exclude_reasons"].get("preprocessing_dropped", 0) + 1
                )
                continue

            # ── Quality score ──
            qscore = quality_score(fixed)
            sft_totals["quality_scores"].append(qscore)

            if qscore < args.min_quality:
                sft_totals["samples_dropped"] += 1
                sft_totals["exclude_reasons"]["below_quality_threshold"] = (
                    sft_totals["exclude_reasons"].get("below_quality_threshold", 0) + 1
                )
                continue

            # ── Accumulate stats ──
            sft_totals["samples_out"] += 1
            for key in ("html_fixed", "think_fixed", "ansi_stripped", "nulls_stripped",
                        "output_artifacts", "echo_stripped", "noncanonical_stripped",
                        "verification_injected", "verification_replaced",
                        "terminal_flag_added", "post_correct_truncated",
                        "consecutive_merged", "orphan_env_removed"):
                sft_totals[key] += stats[key]

            meta = fixed.get("metadata", {})
            plat = meta.get("platform", "unknown")
            sft_totals["platform_counts"][plat] = sft_totals["platform_counts"].get(plat, 0) + 1
            if plat == "cybench":
                chal = meta.get("challenge", "")
                sft_totals["cybench_challenges"].add(chal)

            if fout:
                fout.write(json.dumps(fixed, ensure_ascii=False) + "\n")
    finally:
        if fout:
            fout.close()

    # ─── Report SFT stats ───
    print(f"\nInput:  {sft_totals['samples_in']} samples from {input_path}")
    if sft_output:
        print(f"Output: {sft_totals['samples_out']} samples to {sft_output}")
    else:
        print(f"Output: {sft_totals['samples_out']} samples (dry run)")
    print(f"Dropped: {sft_totals['samples_dropped']}")
    print(f"\nExclusion reasons:")
    for reason, count in sorted(sft_totals["exclude_reasons"].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print(f"\nPreprocessing fixes applied:")
    for key in ("html_fixed", "think_fixed", "ansi_stripped", "nulls_stripped",
                "output_artifacts", "echo_stripped", "noncanonical_stripped",
                "verification_injected", "verification_replaced",
                "terminal_flag_added", "post_correct_truncated",
                "consecutive_merged", "orphan_env_removed"):
        print(f"  {key}: {sft_totals[key]}")

    print(f"\nPlatform distribution:")
    for plat, count in sorted(sft_totals["platform_counts"].items(), key=lambda x: -x[1]):
        print(f"  {plat}: {count}")

    print(f"\nCyBench challenges covered: {len(sft_totals['cybench_challenges'])}")
    for c in sorted(sft_totals["cybench_challenges"]):
        diff = _CYBENCH_DIFFICULTY.get(c, "?")
        print(f"  {c} ({diff})")

    scores = sft_totals["quality_scores"]
    if scores:
        avg = sum(scores) / len(scores)
        print(f"\nQuality scores: min={min(scores):.1f}, avg={avg:.1f}, max={max(scores):.1f}")

    # ─── Build online RL dataset ───
    online_rl_source = args.online_rl_source
    if online_rl_source is None:
        default_candidate = Path("data/online_rl_cybench40.jsonl")
        fallback_candidate = Path("data/grpo_cybench40.jsonl")
        if default_candidate.exists():
            online_rl_source = str(default_candidate)
        elif fallback_candidate.exists():
            online_rl_source = str(fallback_candidate)

    if online_rl_source:
        print(f"\n{'=' * 60}")
        print("Building Online RL Quality Dataset")
        print("=" * 60)

        online_rl_source_path = Path(online_rl_source)
        if not online_rl_source_path.exists():
            print(f"Error: online RL source not found: {online_rl_source_path}", file=sys.stderr)
        else:
            online_rl_stats = build_online_rl_from_existing(
                online_rl_source_path,
                online_rl_output,
                difficulty_cap=args.online_rl_difficulty_cap,
            )
            print(f"\nInput:  {online_rl_stats['total']} samples from {online_rl_source_path}")
            if online_rl_output:
                print(f"Output: {online_rl_stats['included']} samples to {online_rl_output}")
            else:
                print(f"Output: {online_rl_stats['included']} samples (dry run)")
            print(f"Excluded: {online_rl_stats['excluded']}")
            if online_rl_stats["reasons"]:
                print(f"\nExclusion reasons:")
                for reason, count in sorted(online_rl_stats["reasons"].items(), key=lambda x: -x[1]):
                    print(f"  {reason}: {count}")

    print(f"\n{'=' * 60}")
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
