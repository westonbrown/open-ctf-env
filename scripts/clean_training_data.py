#!/usr/bin/env python3
"""
Clean training data for open-ctf-env SFT and GRPO datasets.

Applies the following fixes:
  C1: List-type content on tool messages -> extract text string
  C2: Missing user message (system -> assistant) -> remove trace
  C3: Orphaned tool responses with toolu_* IDs -> reconstruct tool_call on preceding assistant
  H1: "None"-prefixed content from str(None) conversion bug -> strip leading "None"
  H2: Consecutive assistant messages each with 1 tool_call -> merge into single message
  H3: GRPO traces ending with user message -> trim trailing user messages
  M2: Empty tool_calls arrays on assistant messages -> remove the key
  M6: HTB Starting Point overrepresentation in GRPO -> cap at 2 traces per challenge

Usage:
    python scripts/clean_training_data.py

Reads:
    data/sft.jsonl
    data/grpo.jsonl

Writes:
    data/sft_clean.jsonl
    data/grpo_clean.jsonl
"""

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
SFT_IN = BASE_DIR / "data" / "sft.jsonl"
GRPO_IN = BASE_DIR / "data" / "grpo.jsonl"
SFT_OUT = BASE_DIR / "data" / "sft_clean.jsonl"
GRPO_OUT = BASE_DIR / "data" / "grpo_clean.jsonl"

# Max traces per challenge in GRPO (M6)
MAX_PER_CHALLENGE = 2


# ---------------------------------------------------------------------------
# Counters for summary
# ---------------------------------------------------------------------------
class Stats:
    def __init__(self, name: str):
        self.name = name
        self.input_traces = 0
        self.output_traces = 0
        self.c1_list_content = 0
        self.c2_missing_user = 0
        self.c3_orphaned_tool = 0
        self.h1_none_prefix = 0
        self.h2_consecutive_assistant_merges = 0
        self.h3_trailing_user = 0
        self.m2_empty_tool_calls = 0
        self.m6_deduped = 0

    def report(self) -> str:
        lines = [
            f"\n{'='*60}",
            f"  {self.name}",
            f"{'='*60}",
            f"  Input traces:                {self.input_traces}",
            f"  Output traces:               {self.output_traces}",
            f"  Removed traces:              {self.input_traces - self.output_traces}",
            f"  ---",
            f"  C1 list-type content fixed:  {self.c1_list_content}",
            f"  C2 missing user removed:     {self.c2_missing_user}",
            f"  C3 orphaned tool fixed:      {self.c3_orphaned_tool}",
            f"  H1 None-prefix stripped:     {self.h1_none_prefix}",
            f"  H2 consecutive asst merged:  {self.h2_consecutive_assistant_merges}",
            f"  H3 trailing user trimmed:    {self.h3_trailing_user}",
            f"  M2 empty tool_calls removed: {self.m2_empty_tool_calls}",
            f"  M6 challenge deduped:        {self.m6_deduped}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fix functions
# ---------------------------------------------------------------------------

def fix_c1_list_content(msgs: list, stats: Stats) -> list:
    """C1: Convert list-type content [{"type":"text","text":"..."}] to plain string."""
    for msg in msgs:
        content = msg.get("content")
        if isinstance(content, list):
            # Extract text from list of content blocks
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif isinstance(block, dict):
                    # Other block types (image, etc.) - use str repr
                    parts.append(str(block))
                elif isinstance(block, str):
                    parts.append(block)
            msg["content"] = "\n".join(parts) if parts else ""
            stats.c1_list_content += 1
    return msgs


def check_c2_missing_user(msgs: list) -> bool:
    """C2: Check if trace is missing a user message (system -> assistant directly)."""
    if len(msgs) < 2:
        return True  # Too short, remove
    # Check for system -> assistant pattern (no user prompt)
    for i in range(len(msgs) - 1):
        if msgs[i]["role"] == "system" and msgs[i + 1]["role"] == "assistant":
            return True
    # Also check if there is no user message at all
    if not any(m["role"] == "user" for m in msgs):
        return True
    return False


def fix_c3_orphaned_tool(msgs: list, stats: Stats) -> list:
    """C3: Reconstruct tool_call on preceding assistant for orphaned tool responses."""
    for j in range(len(msgs)):
        msg = msgs[j]
        if msg["role"] != "tool":
            continue
        tool_call_id = msg.get("tool_call_id", "")
        if not tool_call_id.startswith("toolu_"):
            continue

        # Find the preceding assistant message (skip over other tool messages)
        prev_idx = j - 1
        while prev_idx >= 0 and msgs[prev_idx]["role"] == "tool":
            prev_idx -= 1

        if prev_idx < 0 or msgs[prev_idx]["role"] != "assistant":
            continue

        prev_msg = msgs[prev_idx]
        existing_tc = prev_msg.get("tool_calls", [])

        # Check if this tool_call_id is already referenced
        existing_ids = {tc["id"] for tc in existing_tc}
        if tool_call_id in existing_ids:
            continue

        # This is an orphaned tool response - reconstruct the tool_call
        tool_name = msg.get("name", "unknown")
        if tool_name == "unknown":
            # Try to infer from content
            content = msg.get("content", "")
            if "Todos have been modified" in content or "todo" in content.lower():
                tool_name = "update_plan"
            else:
                tool_name = "shell_command"

        new_tc = {
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": tool_name,
                "arguments": "{}",
            },
        }

        if "tool_calls" not in prev_msg:
            prev_msg["tool_calls"] = []
        prev_msg["tool_calls"].append(new_tc)
        stats.c3_orphaned_tool += 1

    return msgs


# Pattern for H1: "None" followed by a character that indicates it was str(None) prepended.
# Matches: "NoneGood", "None The", "None  I", "None\n", "None  None", etc.
# Does NOT match: content that is just "None" by itself (handled by len check).
_NONE_BUG_RE = re.compile(r"^None(?=[A-Z \n\t])")


def fix_h1_none_prefix(msgs: list, stats: Stats) -> list:
    """H1: Strip leading 'None' from content caused by str(None) conversion bug."""
    for msg in msgs:
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        if len(content) <= 4:
            continue  # Just "None" alone -- leave it
        if _NONE_BUG_RE.match(content):
            msg["content"] = content[4:]
            stats.h1_none_prefix += 1
            # Handle double-None: "None  None  I can see..."
            # After stripping once, check again
            while msg["content"] and _NONE_BUG_RE.match(msg["content"]) and len(msg["content"]) > 4:
                msg["content"] = msg["content"][4:]
                stats.h1_none_prefix += 1
    return msgs


def fix_h2_consecutive_assistants(msgs: list, stats: Stats) -> list:
    """H2: Merge consecutive assistant messages into single message with combined tool_calls."""
    if len(msgs) < 2:
        return msgs

    merged = [msgs[0]]
    i = 1
    while i < len(msgs):
        curr = msgs[i]

        # Check if we should merge: current is assistant and previous merged is also assistant
        if curr["role"] == "assistant" and merged[-1]["role"] == "assistant":
            prev = merged[-1]

            # Merge tool_calls
            prev_tc = prev.get("tool_calls", [])
            curr_tc = curr.get("tool_calls", [])

            # Combine content
            prev_content = prev.get("content") or ""
            curr_content = curr.get("content") or ""
            if prev_content and curr_content:
                prev["content"] = prev_content.rstrip() + "\n\n" + curr_content.lstrip()
            elif curr_content:
                prev["content"] = curr_content

            # Combine tool_calls
            if curr_tc:
                if "tool_calls" not in prev:
                    prev["tool_calls"] = []
                prev["tool_calls"].extend(curr_tc)

            # Preserve reasoning_content from first message that has it
            if not prev.get("reasoning_content") and curr.get("reasoning_content"):
                prev["reasoning_content"] = curr["reasoning_content"]

            stats.h2_consecutive_assistant_merges += 1
        else:
            merged.append(curr)
        i += 1

    return merged


def fix_h3_trailing_user(msgs: list, stats: Stats) -> list:
    """H3: Trim trailing user messages from GRPO traces."""
    trimmed = False
    while msgs and msgs[-1]["role"] == "user":
        msgs.pop()
        trimmed = True
    if trimmed:
        stats.h3_trailing_user += 1
    return msgs


def fix_m2_empty_tool_calls(msgs: list, stats: Stats) -> list:
    """M2: Remove empty tool_calls arrays from assistant messages."""
    for msg in msgs:
        if msg["role"] == "assistant" and "tool_calls" in msg:
            if not msg["tool_calls"]:  # empty list
                del msg["tool_calls"]
                stats.m2_empty_tool_calls += 1
    return msgs


# ---------------------------------------------------------------------------
# M6: Challenge deduplication for GRPO
# ---------------------------------------------------------------------------

def select_grpo_traces(traces: list, stats: Stats) -> list:
    """M6: Cap at MAX_PER_CHALLENGE traces per challenge.

    Selection strategy per challenge:
    - Keep the shortest successful trace (fewest messages)
    - Keep one failure trace (shortest) if available
    - If no failures, keep the second-shortest success
    """
    # Group traces by challenge name
    by_challenge = defaultdict(list)
    for idx, trace in enumerate(traces):
        meta = trace.get("metadata", {})
        challenge = meta.get("challenge", meta.get("challenge_name", f"_unknown_{idx}"))
        by_challenge[challenge].append((idx, trace))

    selected_indices = set()

    for challenge, group in by_challenge.items():
        if len(group) <= MAX_PER_CHALLENGE:
            # Under cap, keep all
            for idx, _ in group:
                selected_indices.add(idx)
            continue

        # Separate successes and failures
        successes = []
        failures = []
        for idx, trace in group:
            meta = trace.get("metadata", {})
            if meta.get("success") is True:
                successes.append((idx, trace))
            else:
                failures.append((idx, trace))

        kept = []

        # Sort by message count (shortest first)
        successes.sort(key=lambda x: len(x[1]["messages"]))
        failures.sort(key=lambda x: len(x[1]["messages"]))

        # Pick shortest success
        if successes:
            kept.append(successes[0][0])

        # Pick one failure if available
        if failures and len(kept) < MAX_PER_CHALLENGE:
            kept.append(failures[0][0])

        # Fill remaining slots with next shortest success
        if len(kept) < MAX_PER_CHALLENGE:
            for idx, _ in successes:
                if idx not in kept:
                    kept.append(idx)
                    break

        # If still under cap (no successes), fill with failures
        while len(kept) < MAX_PER_CHALLENGE and failures:
            for idx, _ in failures:
                if idx not in kept:
                    kept.append(idx)
                    break
            else:
                break

        for idx in kept:
            selected_indices.add(idx)

        removed = len(group) - len(kept)
        if removed > 0:
            stats.m6_deduped += removed

    # Return traces in original order
    return [traces[i] for i in sorted(selected_indices)]


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def apply_fixes(trace: dict, stats: Stats, is_grpo: bool = False) -> dict | None:
    """Apply all message-level fixes to a single trace. Returns None if trace should be removed."""
    msgs = trace["messages"]

    # C1: Fix list-type content
    msgs = fix_c1_list_content(msgs, stats)

    # C2: Check for missing user message -> remove trace
    if check_c2_missing_user(msgs):
        stats.c2_missing_user += 1
        return None

    # C3: Fix orphaned tool responses
    msgs = fix_c3_orphaned_tool(msgs, stats)

    # H1: Strip None-prefix
    msgs = fix_h1_none_prefix(msgs, stats)

    # H2: Merge consecutive assistants
    msgs = fix_h2_consecutive_assistants(msgs, stats)

    # H3: Trim trailing user messages (GRPO only)
    if is_grpo:
        msgs = fix_h3_trailing_user(msgs, stats)

    # M2: Remove empty tool_calls
    msgs = fix_m2_empty_tool_calls(msgs, stats)

    # Don't return empty traces
    if len(msgs) < 2:
        stats.c2_missing_user += 1  # Count as removed
        return None

    trace["messages"] = msgs
    return trace


def process_file(
    input_path: Path,
    output_path: Path,
    stats: Stats,
    is_grpo: bool = False,
) -> list[dict]:
    """Process a single JSONL file through all fixes."""
    traces = []
    with open(input_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traces.append(json.loads(line))

    stats.input_traces = len(traces)

    # Apply per-trace fixes
    cleaned = []
    for trace in traces:
        result = apply_fixes(trace, stats, is_grpo=is_grpo)
        if result is not None:
            cleaned.append(result)

    # M6: Challenge deduplication (GRPO only)
    if is_grpo:
        cleaned = select_grpo_traces(cleaned, stats)

    stats.output_traces = len(cleaned)

    # Write output
    with open(output_path, "w") as f:
        for trace in cleaned:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    return cleaned


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(traces: list[dict], name: str) -> list[str]:
    """Run validation checks on cleaned traces. Returns list of issues found."""
    issues = []

    for i, trace in enumerate(traces):
        msgs = trace["messages"]

        for j, msg in enumerate(msgs):
            # V1: No list-type content
            if isinstance(msg.get("content"), list):
                issues.append(f"{name} trace {i} msg {j}: list-type content remains")

            # V2: No None-prefix bug
            content = msg.get("content")
            if isinstance(content, str) and len(content) > 4 and _NONE_BUG_RE.match(content):
                issues.append(f"{name} trace {i} msg {j}: None-prefix remains: {content[:50]}")

            # V3: No empty tool_calls
            if msg["role"] == "assistant" and msg.get("tool_calls") == []:
                issues.append(f"{name} trace {i} msg {j}: empty tool_calls remains")

        # V4: No system -> assistant (missing user)
        for j in range(len(msgs) - 1):
            if msgs[j]["role"] == "system" and msgs[j + 1]["role"] == "assistant":
                issues.append(f"{name} trace {i}: system->assistant (no user)")

        # V5: No consecutive assistants
        for j in range(1, len(msgs)):
            if msgs[j]["role"] == "assistant" and msgs[j - 1]["role"] == "assistant":
                issues.append(f"{name} trace {i} msgs {j-1}/{j}: consecutive assistants remain")

        # V6: No orphaned tool responses
        for j in range(1, len(msgs)):
            if msgs[j]["role"] == "tool" and msgs[j].get("tool_call_id", "").startswith("toolu_"):
                prev_idx = j - 1
                while prev_idx >= 0 and msgs[prev_idx]["role"] == "tool":
                    prev_idx -= 1
                if prev_idx >= 0 and msgs[prev_idx]["role"] == "assistant":
                    tc_ids = {tc["id"] for tc in msgs[prev_idx].get("tool_calls", [])}
                    if msgs[j]["tool_call_id"] not in tc_ids:
                        issues.append(
                            f"{name} trace {i} msg {j}: orphaned tool response "
                            f"(tool_call_id={msgs[j]['tool_call_id']})"
                        )

        # V7: GRPO traces should not end with user message
        if name == "GRPO" and msgs and msgs[-1]["role"] == "user":
            issues.append(f"{name} trace {i}: ends with user message")

    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Training Data Cleaner")
    print("=" * 60)
    print()

    # Process SFT
    print(f"Processing SFT: {SFT_IN}")
    sft_stats = Stats("SFT")
    sft_cleaned = process_file(SFT_IN, SFT_OUT, sft_stats, is_grpo=False)
    print(f"  -> Wrote {len(sft_cleaned)} traces to {SFT_OUT}")

    # Process GRPO
    print(f"Processing GRPO: {GRPO_IN}")
    grpo_stats = Stats("GRPO")
    grpo_cleaned = process_file(GRPO_IN, GRPO_OUT, grpo_stats, is_grpo=True)
    print(f"  -> Wrote {len(grpo_cleaned)} traces to {GRPO_OUT}")

    # Print summary
    print(sft_stats.report())
    print(grpo_stats.report())

    # Validate
    print(f"\n{'='*60}")
    print("  Validation")
    print(f"{'='*60}")

    sft_issues = validate(sft_cleaned, "SFT")
    grpo_issues = validate(grpo_cleaned, "GRPO")

    all_issues = sft_issues + grpo_issues
    if all_issues:
        print(f"\n  WARNING: {len(all_issues)} validation issues found:")
        for issue in all_issues[:20]:
            print(f"    - {issue}")
        if len(all_issues) > 20:
            print(f"    ... and {len(all_issues) - 20} more")
        sys.exit(1)
    else:
        print("\n  All validation checks passed.")

    # Challenge distribution for GRPO
    challenge_counts = Counter()
    for trace in grpo_cleaned:
        meta = trace.get("metadata", {})
        challenge = meta.get("challenge", meta.get("challenge_name", "unknown"))
        challenge_counts[challenge] += 1

    over_cap = {k: v for k, v in challenge_counts.items() if v > MAX_PER_CHALLENGE}
    if over_cap:
        print(f"\n  WARNING: {len(over_cap)} challenges still over cap of {MAX_PER_CHALLENGE}:")
        for name, cnt in sorted(over_cap.items(), key=lambda x: -x[1])[:10]:
            print(f"    {name}: {cnt}")
    else:
        print(f"\n  GRPO: All challenges at or below {MAX_PER_CHALLENGE} traces per challenge.")

    max_count = max(challenge_counts.values()) if challenge_counts else 0
    print(f"  GRPO: {len(challenge_counts)} unique challenges, max {max_count} traces per challenge")

    print(f"\n{'='*60}")
    print("  Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
