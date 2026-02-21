#!/usr/bin/env python3
"""Normalize fragmented tool names in SFT and GRPO training data.

Problem: Multiple tool names refer to the same function:
  - exec_command, execute_command → shell_command
  - Read → read_file
  - unknown (tool responses) → update_plan

This script normalizes both:
  1. tool_calls[].function.name on assistant messages
  2. name field on tool response messages (role=tool)
"""

import json
import sys
from collections import Counter
from pathlib import Path

# Normalization map: old_name → canonical_name
NORMALIZE_MAP = {
    "exec_command": "shell_command",
    "execute_command": "shell_command",
    "Read": "read_file",
    "unknown": "update_plan",
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FILES = [
    DATA_DIR / "sft.jsonl",
    DATA_DIR / "grpo.jsonl",
]


def normalize_file(filepath: Path) -> dict:
    """Normalize tool names in a single JSONL file. Returns change counts."""
    changes = {
        "tool_call_renames": Counter(),
        "tool_response_renames": Counter(),
        "lines_modified": 0,
        "lines_total": 0,
    }

    output_lines = []

    with open(filepath, "r") as f:
        for line in f:
            changes["lines_total"] += 1
            obj = json.loads(line)
            line_modified = False

            for msg in obj.get("messages", []):
                # 1. Normalize tool_calls on assistant messages
                for tc in msg.get("tool_calls", []):
                    func = tc.get("function", {})
                    old_name = func.get("name", "")
                    if old_name in NORMALIZE_MAP:
                        new_name = NORMALIZE_MAP[old_name]
                        func["name"] = new_name
                        changes["tool_call_renames"][(old_name, new_name)] += 1
                        line_modified = True

                # 2. Normalize name on tool response messages
                if msg.get("role") == "tool":
                    old_name = msg.get("name", "")
                    if old_name in NORMALIZE_MAP:
                        new_name = NORMALIZE_MAP[old_name]
                        msg["name"] = new_name
                        changes["tool_response_renames"][(old_name, new_name)] += 1
                        line_modified = True

            if line_modified:
                changes["lines_modified"] += 1

            output_lines.append(json.dumps(obj, ensure_ascii=False))

    # Write back
    with open(filepath, "w") as f:
        for i, line in enumerate(output_lines):
            f.write(line)
            if i < len(output_lines) - 1:
                f.write("\n")
            else:
                f.write("\n")  # trailing newline

    return changes


def print_summary(filepath: Path, changes: dict):
    """Print a human-readable summary of changes."""
    print(f"\n{'=' * 60}")
    print(f"  {filepath.name}")
    print(f"{'=' * 60}")
    print(f"  Lines total:    {changes['lines_total']}")
    print(f"  Lines modified: {changes['lines_modified']}")

    print(f"\n  tool_calls[].function.name renames:")
    if changes["tool_call_renames"]:
        for (old, new), count in sorted(
            changes["tool_call_renames"].items(), key=lambda x: -x[1]
        ):
            print(f"    {old:25s} → {new:20s}  ({count:,} occurrences)")
    else:
        print(f"    (none)")

    print(f"\n  tool response name renames:")
    if changes["tool_response_renames"]:
        for (old, new), count in sorted(
            changes["tool_response_renames"].items(), key=lambda x: -x[1]
        ):
            print(f"    {old:25s} → {new:20s}  ({count:,} occurrences)")
    else:
        print(f"    (none)")

    total_renames = sum(changes["tool_call_renames"].values()) + sum(
        changes["tool_response_renames"].values()
    )
    print(f"\n  Total renames: {total_renames:,}")


def main():
    print("Tool Name Normalization")
    print(f"Map: {NORMALIZE_MAP}")

    all_changes = {}
    for filepath in FILES:
        if not filepath.exists():
            print(f"\nWARNING: {filepath} not found, skipping.")
            continue
        print(f"\nProcessing {filepath.name}...")
        changes = normalize_file(filepath)
        all_changes[filepath] = changes
        print_summary(filepath, changes)

    # Grand total
    grand_tc = sum(
        sum(c["tool_call_renames"].values()) for c in all_changes.values()
    )
    grand_resp = sum(
        sum(c["tool_response_renames"].values()) for c in all_changes.values()
    )
    grand_lines = sum(c["lines_modified"] for c in all_changes.values())

    print(f"\n{'=' * 60}")
    print(f"  GRAND TOTAL")
    print(f"{'=' * 60}")
    print(f"  tool_call renames:    {grand_tc:,}")
    print(f"  tool response renames: {grand_resp:,}")
    print(f"  Lines modified:        {grand_lines:,}")
    print(f"  Total renames:         {grand_tc + grand_resp:,}")


if __name__ == "__main__":
    main()
