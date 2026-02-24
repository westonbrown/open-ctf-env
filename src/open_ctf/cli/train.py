#!/usr/bin/env python3
"""Open CTF training CLI.

3-stage pipeline:
  Stage 1 (SFT):  LlamaFactory — battle-tested tool format support, packing, DeepSpeed
  Stage 2 (GRPO): SkyRL — fully async Ray-based trainer, vLLM in separate process
  Stage 3 (GEPA): DSPy — prompt evolution, no weight updates

Usage:
    # Stage 1: SFT via LlamaFactory
    open-ctf-train sft \\
        --model Nanbeige/Nanbeige4.1-3B \\
        --data data/sft.jsonl \\
        --output outputs/sft

    # Stage 2: GRPO via SkyRL (requires SFT merged model)
    open-ctf-train grpo \\
        --model outputs/sft-merged \\
        --data data/grpo.jsonl \\
        --output outputs/grpo

    # Stage 3: GEPA prompt optimization (no weight updates)
    open-ctf-train gepa \\
        --model openai/ctf-agent \\
        --data data/grpo.jsonl \\
        --output outputs/gepa \\
        --reflection-model anthropic/claude-sonnet-4-20250514

    # Merge LoRA adapter into base weights
    open-ctf-train merge \\
        --adapter outputs/sft/final \\
        --base-model Nanbeige/Nanbeige4.1-3B \\
        --output outputs/merged
"""

import argparse
import logging
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "training.yaml"


def load_config(path: Path) -> dict:
    """Load YAML config, returning empty dict on failure."""
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    logger.warning("Config not found at %s, using defaults", path)
    return {}


# -----------------------------------------------------------------------
# Sub-commands
# -----------------------------------------------------------------------


def cmd_sft(args: argparse.Namespace) -> None:
    """Run SFT training via LlamaFactory."""
    from open_ctf.training.sft import train_sft

    config = load_config(args.config)
    model_id = args.model or config.get("model", {}).get("name", "Nanbeige/Nanbeige4.1-3B")

    train_sft(
        model_id=model_id,
        data_path=args.data,
        output_dir=args.output,
        config=config,
        val_data_path=args.val_data,
        resume_from=args.resume,
    )


def cmd_grpo(args: argparse.Namespace) -> None:
    """Run GRPO training via SkyRL."""
    from open_ctf.training.grpo import train_grpo

    config = load_config(args.config)

    train_grpo(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        config=config,
        resume_from=args.resume,
        challenge_registry=getattr(args, 'challenge_registry', None),
    )


def cmd_gepa(args: argparse.Namespace) -> None:
    """Run GEPA prompt optimization (Stage 3, after SFT + GRPO)."""
    from open_ctf.training.gepa import run_gepa

    config = load_config(args.config)

    run_gepa(
        model_id=args.model,
        data_path=args.data,
        output_dir=args.output,
        config=config,
        reflection_model=args.reflection_model,
        budget=args.budget,
        val_data_path=args.val_data,
        max_samples=args.max_samples,
    )


def cmd_merge(args: argparse.Namespace) -> None:
    """Merge LoRA adapter into base model weights via PEFT merge_and_unload()."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    logger.info("Merging adapter from %s", args.adapter)

    config = load_config(args.config)
    base_model_id = args.base_model or config.get("model", {}).get("name", "")
    if not base_model_id:
        raise ValueError(
            "Merge requires --base-model or model.name in config "
            "(needed to load the base model before applying adapter)"
        )

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model = model.merge_and_unload()

    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)
    logger.info("Merged via PEFT -> %s", args.output)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open CTF Training Pipeline (LlamaFactory SFT + SkyRL GRPO + GEPA)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to training YAML config (default: {DEFAULT_CONFIG})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- sft (LlamaFactory) -----------------------------------------------
    sft_parser = subparsers.add_parser("sft", help="Run SFT via LlamaFactory")
    sft_parser.add_argument("--model", default=None, help="HF model id (overrides config)")
    sft_parser.add_argument("--data", required=True, help="Path to SFT JSONL data")
    sft_parser.add_argument("--val-data", default=None, help="Path to validation JSONL")
    sft_parser.add_argument("--output", required=True, help="Output directory")
    sft_parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    sft_parser.set_defaults(func=cmd_sft)

    # -- grpo (SkyRL) -----------------------------------------------------
    grpo_parser = subparsers.add_parser("grpo", help="Run GRPO via SkyRL")
    grpo_parser.add_argument("--model", required=True, help="Path to SFT merged model")
    grpo_parser.add_argument("--data", required=True, help="Path to GRPO JSONL data")
    grpo_parser.add_argument("--output", required=True, help="Output directory")
    grpo_parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    grpo_parser.add_argument(
        "--challenge-registry", default=None,
        help="Path to challenge registry YAML for target URL resolution",
    )
    grpo_parser.set_defaults(func=cmd_grpo)

    # -- gepa (unchanged) -------------------------------------------------
    gepa_parser = subparsers.add_parser(
        "gepa",
        help="Optimize system prompt with GEPA (Stage 3, no weight updates)",
    )
    gepa_parser.add_argument("--model", required=True, help="LLM model id for dspy.LM")
    gepa_parser.add_argument("--data", required=True, help="Path to GRPO JSONL data (challenges)")
    gepa_parser.add_argument("--output", required=True, help="Output directory for optimized prompt")
    gepa_parser.add_argument("--val-data", default=None, help="Validation JSONL (separate from train)")
    gepa_parser.add_argument(
        "--reflection-model", default=None,
        help="Strong LLM for GEPA reflection (default: same as --model)",
    )
    gepa_parser.add_argument(
        "--budget", choices=["light", "medium", "heavy"], default="medium",
        help="GEPA budget preset (default: medium)",
    )
    gepa_parser.add_argument("--max-samples", type=int, default=None, help="Max training examples")
    gepa_parser.set_defaults(func=cmd_gepa)

    # -- merge (unchanged) ------------------------------------------------
    merge_parser = subparsers.add_parser("merge", help="Merge LoRA adapter into base")
    merge_parser.add_argument("--adapter", required=True, help="Path to LoRA adapter dir")
    merge_parser.add_argument("--base-model", default=None, help="Base model id (for HF merge fallback)")
    merge_parser.add_argument("--output", required=True, help="Output directory for merged model")
    merge_parser.set_defaults(func=cmd_merge)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
