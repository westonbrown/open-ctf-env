#!/usr/bin/env python3
"""
Fix empty no-op assistant messages in SFT and GRPO training data.

Removes assistant messages where BOTH conditions are true:
  - content is empty string "", None, or whitespace-only
  - tool_calls is missing, empty list [], or None

After removal, merges consecutive user messages (by joining content with
newline) to maintain valid message structure.

Writes results back to the same files.
"""

import json
import sys
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

FILES = [
    DATA_DIR / "sft.jsonl",
    DATA_DIR / "grpo.jsonl",
]


def is_noop_assistant(msg: dict) -> bool:
    """Check if an assistant message is a no-op (empty content, no tool_calls)."""
    if msg.get("role") != "assistant":
        return False

    content = msg.get("content")
    is_empty_content = content is None or (isinstance(content, str) and content.strip() == "")

    tc = msg.get("tool_calls")
    has_tool_calls = tc is not None and isinstance(tc, list) and len(tc) > 0

    return is_empty_content and not has_tool_calls


def merge_consecutive_users(messages: list[dict]) -> list[dict]:
    """Merge consecutive user messages by joining content with newline."""
    if not messages:
        return messages

    merged = [messages[0]]
    merges = 0

    for msg in messages[1:]:
        prev = merged[-1]
        if prev.get("role") == "user" and msg.get("role") == "user":
            # Merge: join content
            prev_content = prev.get("content") or ""
            curr_content = msg.get("content") or ""
            merged_content = prev_content + "\n" + curr_content if prev_content and curr_content else prev_content or curr_content
            prev["content"] = merged_content
            merges += 1
        else:
            merged.append(msg)

    return merged, merges


def validate_messages(messages: list[dict]) -> list[str]:
    """Validate message structure. Returns list of issues found."""
    issues = []
    if not messages:
        issues.append("Empty message list")
        return issues

    for i in range(1, len(messages)):
        r1 = messages[i - 1].get("role")
        r2 = messages[i].get("role")
        # Consecutive same role is only valid for tool messages
        if r1 == r2 and r1 != "tool":
            issues.append(f"Consecutive {r1} messages at index {i - 1},{i}")

    return issues


def process_file(filepath: Path) -> dict:
    """Process a single JSONL file. Returns stats dict."""
    print(f"\nProcessing: {filepath.name}")
    print("=" * 60)

    traces = []
    with open(filepath) as f:
        for line in f:
            traces.append(json.loads(line.strip()))

    total_traces = len(traces)
    total_noops_removed = 0
    total_merges = 0
    traces_modified = 0
    pre_validation_issues = 0
    post_validation_issues = 0

    output_traces = []

    for trace_idx, trace in enumerate(traces):
        msgs = trace.get("messages", [])
        original_len = len(msgs)

        # Step 1: Remove no-op assistant messages
        filtered = [m for m in msgs if not is_noop_assistant(m)]
        noops_removed = original_len - len(filtered)

        # Step 2: Merge consecutive user messages
        cleaned, merges = merge_consecutive_users(filtered)

        # Step 3: Validate
        issues = validate_messages(cleaned)
        if issues:
            post_validation_issues += len(issues)

        # Update trace
        trace["messages"] = cleaned
        output_traces.append(trace)

        if noops_removed > 0 or merges > 0:
            traces_modified += 1
        total_noops_removed += noops_removed
        total_merges += merges

    # Write back
    with open(filepath, "w") as f:
        for trace in output_traces:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    stats = {
        "file": filepath.name,
        "total_traces": total_traces,
        "traces_modified": traces_modified,
        "noops_removed": total_noops_removed,
        "user_merges": total_merges,
        "post_validation_issues": post_validation_issues,
    }

    print(f"  Total traces:          {total_traces}")
    print(f"  Traces modified:       {traces_modified}")
    print(f"  No-op assistants removed: {total_noops_removed}")
    print(f"  Consecutive users merged: {total_merges}")
    print(f"  Validation issues after: {post_validation_issues}")

    return stats


def main():
    print("Fix Empty No-Op Assistant Messages")
    print("=" * 60)

    all_stats = []
    for filepath in FILES:
        if not filepath.exists():
            print(f"\nWARNING: {filepath} not found, skipping.")
            continue
        stats = process_file(filepath)
        all_stats.append(stats)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_removed = sum(s["noops_removed"] for s in all_stats)
    total_merges = sum(s["user_merges"] for s in all_stats)
    total_issues = sum(s["post_validation_issues"] for s in all_stats)

    for s in all_stats:
        print(f"  {s['file']}: {s['noops_removed']} no-ops removed, "
              f"{s['user_merges']} user merges, "
              f"{s['post_validation_issues']} remaining issues")

    print(f"\n  Total no-ops removed: {total_removed}")
    print(f"  Total user merges:    {total_merges}")
    print(f"  Remaining issues:     {total_issues}")

    if total_issues > 0:
        print("\n  WARNING: Some validation issues remain. These are pre-existing")
        print("  consecutive same-role messages not caused by this script.")

    # Verify by re-reading
    print("\n  Verification (re-read files):")
    for filepath in FILES:
        if not filepath.exists():
            continue
        remaining_noops = 0
        with open(filepath) as f:
            for line in f:
                trace = json.loads(line.strip())
                for m in trace.get("messages", []):
                    if is_noop_assistant(m):
                        remaining_noops += 1
        print(f"    {filepath.name}: {remaining_noops} no-op assistants remaining")

    return 0 if total_issues == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
