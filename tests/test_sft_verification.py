"""Tests for scripts/fix_sft_verification.py."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

# Import from scripts directory
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from fix_sft_verification import inject_verification, process_file


def _make_sample(
    messages: list[dict],
    ground_truth_flag: str | None = None,
) -> dict:
    sample: dict = {"messages": messages}
    if ground_truth_flag is not None:
        sample["ground_truth_flag"] = ground_truth_flag
    return sample


def _flag_found_call(flag: str, call_id: str = "call_1") -> dict:
    return {
        "role": "assistant",
        "content": "Found the flag!",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": "flag_found",
                    "arguments": json.dumps({"content": flag}),
                },
            }
        ],
    }


def _tool_response(content: str, call_id: str = "call_1", name: str = "flag_found") -> dict:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": name,
        "content": content,
    }


class TestInjectVerification:
    def test_injects_correct_verification(self):
        """Missing verification + matching flag -> inject 'Correct!'."""
        sample = _make_sample(
            messages=[
                {"role": "system", "content": "You are a CTF agent."},
                {"role": "user", "content": "Find the flag."},
                _flag_found_call("FLAG{abc123}"),
            ],
            ground_truth_flag="FLAG{abc123}",
        )

        fixed, stats = inject_verification(sample)

        assert len(fixed["messages"]) == 4
        injected = fixed["messages"][3]
        assert injected["role"] == "tool"
        assert injected["name"] == "flag_found"
        assert "Correct! Flag verified: FLAG{abc123}" == injected["content"]
        assert stats["injected_correct"] == 1
        assert stats["injected_incorrect"] == 0

    def test_injects_incorrect_verification(self):
        """Missing verification + non-matching flag -> inject 'Incorrect'."""
        sample = _make_sample(
            messages=[
                {"role": "user", "content": "Find the flag."},
                _flag_found_call("FLAG{wrong}"),
            ],
            ground_truth_flag="FLAG{correct}",
        )

        fixed, stats = inject_verification(sample)

        assert len(fixed["messages"]) == 3
        injected = fixed["messages"][2]
        assert "Incorrect submission: FLAG{wrong}" in injected["content"]
        assert stats["injected_incorrect"] == 1
        assert stats["injected_correct"] == 0

    def test_injects_unknown_when_no_ground_truth(self):
        """Missing verification + no ground truth -> inject neutral message."""
        sample = _make_sample(
            messages=[
                {"role": "user", "content": "Find the flag."},
                _flag_found_call("FLAG{something}"),
            ],
            ground_truth_flag=None,
        )

        fixed, stats = inject_verification(sample)

        assert len(fixed["messages"]) == 3
        injected = fixed["messages"][2]
        assert injected["content"] == "Flag submitted: FLAG{something}"
        assert stats["injected_unknown"] == 1

    def test_idempotent_when_verification_exists(self):
        """Already has verification response -> no injection."""
        sample = _make_sample(
            messages=[
                {"role": "user", "content": "Find the flag."},
                _flag_found_call("FLAG{abc}"),
                _tool_response("Flag found: FLAG{abc}"),
            ],
            ground_truth_flag="FLAG{abc}",
        )

        fixed, stats = inject_verification(sample)

        assert len(fixed["messages"]) == 3  # unchanged
        assert stats["already_present"] == 1
        assert stats["injected_correct"] == 0

    def test_preserves_non_flag_messages(self):
        """Other tool calls are not affected."""
        sample = _make_sample(
            messages=[
                {"role": "user", "content": "Explore."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_shell",
                            "type": "function",
                            "function": {
                                "name": "shell_command",
                                "arguments": '{"command": "ls"}',
                            },
                        }
                    ],
                },
                _tool_response("file1.txt\nfile2.txt", call_id="call_shell", name="shell_command"),
                _flag_found_call("FLAG{x}", call_id="call_flag"),
            ],
            ground_truth_flag="FLAG{x}",
        )

        fixed, stats = inject_verification(sample)

        assert len(fixed["messages"]) == 5  # one injection
        assert stats["injected_correct"] == 1
        # Shell tool response untouched
        assert fixed["messages"][2]["name"] == "shell_command"

    def test_handles_dict_arguments(self):
        """Arguments as dict (not JSON string) should work."""
        sample = _make_sample(
            messages=[
                {"role": "user", "content": "Go."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {
                                "name": "flag_found",
                                "arguments": {"content": "FLAG{dict_args}"},
                            },
                        }
                    ],
                },
            ],
            ground_truth_flag="FLAG{dict_args}",
        )

        fixed, stats = inject_verification(sample)
        assert stats["injected_correct"] == 1
        assert "Correct!" in fixed["messages"][2]["content"]

    def test_whitespace_normalization(self):
        """Leading/trailing whitespace in flags should not cause mismatch."""
        sample = _make_sample(
            messages=[
                {"role": "user", "content": "Go."},
                _flag_found_call("  FLAG{spaces}  "),
            ],
            ground_truth_flag="FLAG{spaces}",
        )

        fixed, stats = inject_verification(sample)
        assert stats["injected_correct"] == 1

    def test_multiple_flag_calls_in_one_sample(self):
        """Multiple flag_found calls get individual injections."""
        sample = _make_sample(
            messages=[
                {"role": "user", "content": "Go."},
                _flag_found_call("FLAG{wrong}", call_id="c1"),
                # model tries again
                {
                    "role": "assistant",
                    "content": "Trying again",
                    "tool_calls": [
                        {
                            "id": "c2",
                            "type": "function",
                            "function": {
                                "name": "flag_found",
                                "arguments": json.dumps({"content": "FLAG{right}"}),
                            },
                        }
                    ],
                },
            ],
            ground_truth_flag="FLAG{right}",
        )

        fixed, stats = inject_verification(sample)
        assert stats["injected_incorrect"] == 1
        assert stats["injected_correct"] == 1
        # Original 3 msgs + 2 injections = 5
        assert len(fixed["messages"]) == 5


class TestProcessFile:
    def test_roundtrip(self, tmp_path: Path):
        """Write JSONL, process, verify output."""
        input_file = tmp_path / "input.jsonl"
        output_file = tmp_path / "output.jsonl"

        samples = [
            _make_sample(
                [{"role": "user", "content": "Go."}, _flag_found_call("FLAG{a}")],
                ground_truth_flag="FLAG{a}",
            ),
            _make_sample(
                [{"role": "user", "content": "Go."}, _flag_found_call("FLAG{wrong}")],
                ground_truth_flag="FLAG{b}",
            ),
            _make_sample(
                [
                    {"role": "user", "content": "Go."},
                    _flag_found_call("FLAG{c}"),
                    _tool_response("Flag found: FLAG{c}"),
                ],
                ground_truth_flag="FLAG{c}",
            ),
        ]

        with open(input_file, "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")

        totals = process_file(input_file, output_file)

        assert totals["samples"] == 3
        assert totals["samples_modified"] == 2
        assert totals["injected_correct"] == 1
        assert totals["injected_incorrect"] == 1
        assert totals["already_present"] == 1

        # Verify output is valid JSONL
        with open(output_file) as f:
            output_samples = [json.loads(line) for line in f]
        assert len(output_samples) == 3
        # First sample got injection
        assert len(output_samples[0]["messages"]) == 3
        assert output_samples[0]["messages"][2]["role"] == "tool"
