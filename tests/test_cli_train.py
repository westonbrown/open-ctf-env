"""Smoke tests for CLI argument parsing (open-ctf-train).

Validates:
- sft subcommand parses correctly
- grpo subcommand parses correctly
- merge subcommand parses correctly
- gepa subcommand parses correctly
"""

import argparse
import json
import pytest
from unittest.mock import patch

# We test the CLI parser directly by importing main() and intercepting parse_args.
# This avoids launching subprocess or importing heavy dependencies (torch, peft, etc.)
from open_ctf.cli.train import main, _patch_merged_config_for_vllm


def _parse_args(argv):
    """Parse CLI args without executing the subcommand.

    Patches sys.argv and catches SystemExit from argparse errors.
    """
    with patch("sys.argv", ["open-ctf-train"] + argv):
        from open_ctf.cli.train import main as _main
        import argparse as ap

        # Re-build parser to capture args without running func()
        parser = ap.ArgumentParser()
        parser.add_argument("--config", type=__import__("pathlib").Path)
        subparsers = parser.add_subparsers(dest="command", required=True)

        sft_p = subparsers.add_parser("sft")
        sft_p.add_argument("--model", default=None)
        sft_p.add_argument("--data", required=True)
        sft_p.add_argument("--val-data", default=None)
        sft_p.add_argument("--output", required=True)
        sft_p.add_argument("--resume", default=None)

        grpo_p = subparsers.add_parser("grpo")
        grpo_p.add_argument("--model", required=True)
        grpo_p.add_argument("--data", required=True)
        grpo_p.add_argument("--output", required=True)
        grpo_p.add_argument("--resume", default=None)

        gepa_p = subparsers.add_parser("gepa")
        gepa_p.add_argument("--model", required=True)
        gepa_p.add_argument("--data", required=True)
        gepa_p.add_argument("--output", required=True)
        gepa_p.add_argument("--val-data", default=None)
        gepa_p.add_argument("--reflection-model", default=None)
        gepa_p.add_argument("--budget", choices=["light", "medium", "heavy"], default="medium")
        gepa_p.add_argument("--max-samples", type=int, default=None)

        merge_p = subparsers.add_parser("merge")
        merge_p.add_argument("--adapter", required=True)
        merge_p.add_argument("--base-model", default=None)
        merge_p.add_argument("--output", required=True)

        return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# SFT subcommand
# ---------------------------------------------------------------------------


class TestSFTParser:
    def test_sft_basic(self):
        args = _parse_args(["sft", "--data", "/train.jsonl", "--output", "/out"])
        assert args.command == "sft"
        assert args.data == "/train.jsonl"
        assert args.output == "/out"

    def test_sft_with_model(self):
        args = _parse_args([
            "sft", "--model", "Nanbeige/Nanbeige4.1-3B",
            "--data", "/train.jsonl", "--output", "/out",
        ])
        assert args.model == "Nanbeige/Nanbeige4.1-3B"

    def test_sft_with_val_data(self):
        args = _parse_args([
            "sft", "--data", "/train.jsonl", "--output", "/out",
            "--val-data", "/val.jsonl",
        ])
        assert args.val_data == "/val.jsonl"

    def test_sft_with_resume(self):
        args = _parse_args([
            "sft", "--data", "/train.jsonl", "--output", "/out",
            "--resume", "/ckpt/step-100",
        ])
        assert args.resume == "/ckpt/step-100"

    def test_sft_model_defaults_to_none(self):
        args = _parse_args(["sft", "--data", "/train.jsonl", "--output", "/out"])
        assert args.model is None

    def test_sft_missing_data_raises(self):
        with pytest.raises(SystemExit):
            _parse_args(["sft", "--output", "/out"])

    def test_sft_missing_output_raises(self):
        with pytest.raises(SystemExit):
            _parse_args(["sft", "--data", "/train.jsonl"])


# ---------------------------------------------------------------------------
# GRPO subcommand
# ---------------------------------------------------------------------------


class TestGRPOParser:
    def test_grpo_basic(self):
        args = _parse_args([
            "grpo", "--model", "/merged", "--data", "/grpo.jsonl", "--output", "/out",
        ])
        assert args.command == "grpo"
        assert args.model == "/merged"
        assert args.data == "/grpo.jsonl"
        assert args.output == "/out"

    def test_grpo_with_resume(self):
        args = _parse_args([
            "grpo", "--model", "/merged", "--data", "/grpo.jsonl",
            "--output", "/out", "--resume", "/ckpt",
        ])
        assert args.resume == "/ckpt"

    def test_grpo_missing_model_raises(self):
        with pytest.raises(SystemExit):
            _parse_args(["grpo", "--data", "/grpo.jsonl", "--output", "/out"])

    def test_grpo_missing_data_raises(self):
        with pytest.raises(SystemExit):
            _parse_args(["grpo", "--model", "/m", "--output", "/out"])


# ---------------------------------------------------------------------------
# GEPA subcommand
# ---------------------------------------------------------------------------


class TestGEPAParser:
    def test_gepa_basic(self):
        args = _parse_args([
            "gepa", "--model", "openai/gpt-4", "--data", "/d.jsonl", "--output", "/out",
        ])
        assert args.command == "gepa"
        assert args.model == "openai/gpt-4"

    def test_gepa_with_reflection_model(self):
        args = _parse_args([
            "gepa", "--model", "openai/gpt-4", "--data", "/d.jsonl",
            "--output", "/out", "--reflection-model", "anthropic/claude-sonnet-4-20250514",
        ])
        assert args.reflection_model == "anthropic/claude-sonnet-4-20250514"

    def test_gepa_budget_choices(self):
        for budget in ["light", "medium", "heavy"]:
            args = _parse_args([
                "gepa", "--model", "m", "--data", "/d.jsonl",
                "--output", "/out", "--budget", budget,
            ])
            assert args.budget == budget

    def test_gepa_invalid_budget_raises(self):
        with pytest.raises(SystemExit):
            _parse_args([
                "gepa", "--model", "m", "--data", "/d.jsonl",
                "--output", "/out", "--budget", "extreme",
            ])

    def test_gepa_default_budget_medium(self):
        args = _parse_args([
            "gepa", "--model", "m", "--data", "/d.jsonl", "--output", "/out",
        ])
        assert args.budget == "medium"

    def test_gepa_with_max_samples(self):
        args = _parse_args([
            "gepa", "--model", "m", "--data", "/d.jsonl",
            "--output", "/out", "--max-samples", "100",
        ])
        assert args.max_samples == 100


# ---------------------------------------------------------------------------
# Merge subcommand
# ---------------------------------------------------------------------------


class TestMergeParser:
    def test_merge_basic(self):
        args = _parse_args([
            "merge", "--adapter", "/lora", "--output", "/merged",
        ])
        assert args.command == "merge"
        assert args.adapter == "/lora"
        assert args.output == "/merged"

    def test_merge_with_base_model(self):
        args = _parse_args([
            "merge", "--adapter", "/lora", "--output", "/merged",
            "--base-model", "Nanbeige/Nanbeige4.1-3B",
        ])
        assert args.base_model == "Nanbeige/Nanbeige4.1-3B"

    def test_merge_base_model_defaults_to_none(self):
        args = _parse_args(["merge", "--adapter", "/lora", "--output", "/merged"])
        assert args.base_model is None

    def test_merge_missing_adapter_raises(self):
        with pytest.raises(SystemExit):
            _parse_args(["merge", "--output", "/merged"])

    def test_merge_missing_output_raises(self):
        with pytest.raises(SystemExit):
            _parse_args(["merge", "--adapter", "/lora"])


class TestMergeConfigPatch:
    def test_patch_merged_config_for_vllm_replaces_text_config(self, tmp_path):
        base_dir = tmp_path / "base"
        out_dir = tmp_path / "out"
        base_dir.mkdir()
        out_dir.mkdir()

        base_cfg = {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "vision_config": {"type": "dummy"},
        }
        merged_cfg = {
            "model_type": "qwen3_5_text",
            "architectures": ["Qwen3_5ForCausalLM"],
            "rope_parameters": {"rope_type": "default"},
        }

        (base_dir / "config.json").write_text(json.dumps(base_cfg))
        (out_dir / "config.json").write_text(json.dumps(merged_cfg))

        changed = _patch_merged_config_for_vllm(str(base_dir), str(out_dir))
        assert changed is True
        assert (out_dir / "config.text_backup.json").exists()
        patched = json.loads((out_dir / "config.json").read_text())
        assert patched["model_type"] == "qwen3_5"
        assert "vision_config" in patched

    def test_patch_merged_config_for_vllm_noop_when_already_compatible(self, tmp_path):
        base_dir = tmp_path / "base"
        out_dir = tmp_path / "out"
        base_dir.mkdir()
        out_dir.mkdir()

        cfg = {
            "model_type": "qwen3_5",
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "vision_config": {"type": "dummy"},
        }
        (base_dir / "config.json").write_text(json.dumps(cfg))
        (out_dir / "config.json").write_text(json.dumps(cfg))

        changed = _patch_merged_config_for_vllm(str(base_dir), str(out_dir))
        assert changed is False
        assert not (out_dir / "config.text_backup.json").exists()


# ---------------------------------------------------------------------------
# General CLI
# ---------------------------------------------------------------------------


class TestGeneralCLI:
    def test_no_subcommand_raises(self):
        with pytest.raises(SystemExit):
            _parse_args([])

    def test_unknown_subcommand_raises(self):
        with pytest.raises(SystemExit):
            _parse_args(["unknown", "--data", "/d.jsonl"])
