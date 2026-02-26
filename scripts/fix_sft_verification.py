#!/usr/bin/env python3
"""Inject verification responses after flag_found tool calls in SFT data.

Problem: Many flag_found tool calls have NO verification response, teaching the
model "call flag_found with any text = success" without seeing actual feedback.

This script scans SFT JSONL and injects appropriate tool responses:
  - Correct match  -> "Correct! Flag verified: {flag}"
  - Incorrect      -> "Incorrect submission: {submitted}. The flag was not correct."
  - No ground truth -> "Flag submitted: {submitted}"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


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


def _normalize_flag(flag: str) -> str:
    """Normalize a flag for comparison (strip whitespace)."""
    return flag.strip()


def _make_verification_message(
    tool_call_id: str,
    submitted: str | None,
    ground_truth: str | None,
) -> dict:
    """Build a tool response message for a flag_found call."""
    if submitted and ground_truth:
        if _normalize_flag(submitted) == _normalize_flag(ground_truth):
            content = f"Correct! Flag verified: {submitted}"
        else:
            content = f"Incorrect submission: {submitted}. The flag was not correct."
    elif submitted:
        content = f"Flag submitted: {submitted}"
    else:
        content = "No flag content provided."

    msg: dict = {
        "role": "tool",
        "content": content,
        "name": "flag_found",
    }
    if tool_call_id:
        msg["tool_call_id"] = tool_call_id
    return msg


def _has_response_for(messages: list[dict], start: int, tool_call: dict) -> bool:
    """Check if a tool response already exists for this tool call."""
    tc_id = tool_call.get("id", "")
    for j in range(start, min(start + 5, len(messages))):
        m = messages[j]
        if m.get("role") != "tool":
            continue
        if m.get("tool_call_id") == tc_id and tc_id:
            return True
        if m.get("name") in ("flag_found", "submit_flag"):
            return True
    return False


def inject_verification(sample: dict) -> tuple[dict, dict]:
    """Inject missing verification responses into a single sample.

    Returns (fixed_sample, stats) where stats has keys:
      injected_correct, injected_incorrect, injected_unknown, already_present
    """
    stats = {
        "injected_correct": 0,
        "injected_incorrect": 0,
        "injected_unknown": 0,
        "already_present": 0,
    }

    messages = list(sample.get("messages", []))
    ground_truth = sample.get("ground_truth_flag")

    new_messages: list[dict] = []
    i = 0
    while i < len(messages):
        msg = messages[i]
        new_messages.append(msg)

        tool_calls = msg.get("tool_calls") or []
        flag_calls = [
            tc for tc in tool_calls
            if tc.get("function", {}).get("name") in ("flag_found", "submit_flag")
        ]

        if flag_calls:
            for tc in flag_calls:
                if _has_response_for(messages, i + 1, tc):
                    stats["already_present"] += 1
                else:
                    submitted = _extract_submitted_flag(tc)
                    verification = _make_verification_message(
                        tool_call_id=tc.get("id", ""),
                        submitted=submitted,
                        ground_truth=ground_truth,
                    )
                    new_messages.append(verification)

                    if submitted and ground_truth:
                        if _normalize_flag(submitted) == _normalize_flag(ground_truth):
                            stats["injected_correct"] += 1
                        else:
                            stats["injected_incorrect"] += 1
                    else:
                        stats["injected_unknown"] += 1

        i += 1

    result = dict(sample)
    result["messages"] = new_messages
    return result, stats


def process_file(input_path: Path, output_path: Path) -> dict:
    """Process an entire JSONL file, injecting verification responses.

    Returns aggregate statistics.
    """
    totals = {
        "samples": 0,
        "samples_modified": 0,
        "injected_correct": 0,
        "injected_incorrect": 0,
        "injected_unknown": 0,
        "already_present": 0,
    }

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            fixed, stats = inject_verification(sample)
            fout.write(json.dumps(fixed, ensure_ascii=False) + "\n")

            totals["samples"] += 1
            injected = stats["injected_correct"] + stats["injected_incorrect"] + stats["injected_unknown"]
            if injected > 0:
                totals["samples_modified"] += 1
            for k in ("injected_correct", "injected_incorrect", "injected_unknown", "already_present"):
                totals[k] += stats[k]

    return totals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inject verification responses after flag_found tool calls in SFT data."
    )
    parser.add_argument("--input", required=True, help="Input SFT JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file with injected verifications")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    totals = process_file(input_path, output_path)

    print(f"Processed {totals['samples']} samples -> {output_path}")
    print(f"  Samples modified:    {totals['samples_modified']}")
    print(f"  Injected (correct):  {totals['injected_correct']}")
    print(f"  Injected (incorrect):{totals['injected_incorrect']}")
    print(f"  Injected (unknown):  {totals['injected_unknown']}")
    print(f"  Already present:     {totals['already_present']}")


if __name__ == "__main__":
    main()
