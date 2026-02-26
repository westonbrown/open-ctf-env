#!/usr/bin/env python3
"""Open CTF training CLI.

3-stage pipeline:
  Stage 1 (SFT):       LlamaFactory/TRL
  Stage 2 (online RL): SkyRL (GRPO/RLOO style policy updates)
  Stage 3 (GEPA):      DSPy prompt evolution, no weight updates

Usage:
    # Stage 1: SFT via LlamaFactory
    open-ctf-train sft \\
        --model Nanbeige/Nanbeige4.1-3B \\
        --data data/sft.jsonl \\
        --output outputs/sft

    # Stage 2: online RL via SkyRL (requires SFT merged model)
    open-ctf-train rl \\
        --model outputs/sft-merged \\
        --data data/online_rl.jsonl \\
        --output outputs/online_rl

    # Stage 3: GEPA prompt optimization (no weight updates)
    # Both agent and reflection LMs default to the same model (local vLLM).
    open-ctf-train gepa \\
        --model openai/ctf-agent \\
        --data data/online_rl.jsonl \\
        --output outputs/gepa

    # Merge LoRA adapter into base weights
    open-ctf-train merge \\
        --adapter outputs/sft/final \\
        --base-model Nanbeige/Nanbeige4.1-3B \\
        --output outputs/merged
"""

import argparse
import json
import logging
import subprocess
import shutil
import sys
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


def _patch_merged_config_for_vllm(base_model_id: str, output_dir: str) -> bool:
    """Patch merged config.json when PEFT merge drops multimodal config fields.

    Returns:
        True if config.json was replaced, False otherwise.
    """
    base_cfg_path = Path(base_model_id) / "config.json"
    merged_cfg_path = Path(output_dir) / "config.json"
    if not base_cfg_path.exists() or not merged_cfg_path.exists():
        return False

    with open(base_cfg_path) as f:
        base_cfg = json.load(f)
    with open(merged_cfg_path) as f:
        merged_cfg = json.load(f)

    should_patch = (
        base_cfg.get("model_type") != merged_cfg.get("model_type")
        and "vision_config" in base_cfg
        and "vision_config" not in merged_cfg
    )
    if not should_patch:
        return False

    backup_path = Path(output_dir) / "config.text_backup.json"
    shutil.copy2(merged_cfg_path, backup_path)
    with open(merged_cfg_path, "w") as f:
        json.dump(base_cfg, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info(
        "Replaced merged config.json with base config for vLLM compatibility "
        "(backup saved to %s)",
        backup_path,
    )
    return True


def _patch_merged_tokenizer_config(base_model_id: str, output_dir: str) -> bool:
    """Patch merged tokenizer_config.json when tokenizer_class is incompatible.

    Some merge environments can emit ``tokenizer_class=TokenizersBackend`` in the
    merged artifact, which older Transformers runtimes cannot import during GRPO.

    Returns:
        True if tokenizer_config.json was replaced, False otherwise.
    """
    base_tok_cfg_path = Path(base_model_id) / "tokenizer_config.json"
    merged_tok_cfg_path = Path(output_dir) / "tokenizer_config.json"
    if not base_tok_cfg_path.exists() or not merged_tok_cfg_path.exists():
        return False

    with open(base_tok_cfg_path) as f:
        base_tok_cfg = json.load(f)
    with open(merged_tok_cfg_path) as f:
        merged_tok_cfg = json.load(f)

    merged_cls = merged_tok_cfg.get("tokenizer_class")
    base_cls = base_tok_cfg.get("tokenizer_class")

    # Keep this narrowly targeted to avoid overriding valid tokenizer classes.
    if merged_cls != "TokenizersBackend":
        return False
    if not base_cls:
        return False

    backup_path = Path(output_dir) / "tokenizer_config.text_backup.json"
    shutil.copy2(merged_tok_cfg_path, backup_path)
    with open(merged_tok_cfg_path, "w") as f:
        json.dump(base_tok_cfg, f, indent=2, sort_keys=True)
        f.write("\n")
    logger.info(
        "Replaced merged tokenizer_config.json with base tokenizer config "
        "(backup saved to %s)",
        backup_path,
    )
    return True


# -----------------------------------------------------------------------
# Sub-commands
# -----------------------------------------------------------------------


def _auto_detect_backend(model_id: str) -> str:
    """Auto-detect the best SFT backend for a given model.

    Defaults to 'trl' — uses the model's native tokenizer.apply_chat_template()
    which guarantees correct tool call formatting, thinking block handling, and
    special token rendering for any HuggingFace model.

    Use 'llamafactory' only when its extra features (DeepSpeed ZeRO, neat_packing,
    built-in tool format adapters) are specifically needed.
    """
    return "trl"


def cmd_sft(args: argparse.Namespace) -> None:
    """Run SFT training via LlamaFactory or TRL (selectable backend)."""
    config = load_config(args.config)
    model_id = args.model or config.get("model", {}).get("name", "Nanbeige/Nanbeige4.1-3B")

    # Resolve backend: explicit flag > config > auto-detect
    backend = (
        args.backend
        or config.get("sft", {}).get("backend")
        or _auto_detect_backend(model_id)
    )
    logger.info("SFT backend: %s", backend)

    if backend == "trl":
        from open_ctf.training.sft import train_sft_trl as train_sft
    else:
        from open_ctf.training.sft import train_sft

    train_sft(
        model_id=model_id,
        data_path=args.data,
        output_dir=args.output,
        config=config,
        val_data_path=args.val_data,
        resume_from=args.resume,
    )


def _add_online_rl_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, help="Path to SFT merged model")
    parser.add_argument("--data", required=True, help="Path to online RL JSONL data")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--challenge-registry", default=None,
        help="Path to challenge registry YAML for target URL resolution",
    )
    parser.add_argument(
        "--agent", default=None,
        help="Dotted path to a StepAgent class for tool execution (e.g. my_module.MyAgent)",
    )
    parser.add_argument(
        "--target-map",
        default=None,
        help="Optional challenge target map JSON used by preflight checks.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip online RL preflight validation gate before launch.",
    )
    parser.add_argument(
        "--allow-missing-manifest",
        action="store_true",
        help="Allow dataset launch without <data>.manifest.json provenance file.",
    )
    parser.add_argument(
        "--require-target-map-coverage",
        action="store_true",
        help="Require dataset to include all challenge IDs present in --target-map.",
    )


def _run_online_rl_preflight(args: argparse.Namespace) -> None:
    """Run preflight gate and fail fast before stage-2 training starts."""
    if args.skip_preflight:
        logger.warning("Skipping online RL preflight (--skip-preflight).")
        return

    cmd = [
        sys.executable,
        "-m",
        "open_ctf.cli.validate_pipeline",
        "--mode",
        "grpo-preflight",
        "--online-rl-data",
        args.data,
    ]
    if getattr(args, "challenge_registry", None):
        cmd.extend(["--challenge-registry", args.challenge_registry])
    if getattr(args, "target_map", None):
        cmd.extend(["--target-map", args.target_map])
    if not getattr(args, "allow_missing_manifest", False):
        cmd.append("--require-manifest")
    if getattr(args, "require_target_map_coverage", False):
        cmd.append("--require-target-map-coverage")

    logger.info("Running online RL preflight gate...")
    subprocess.run(cmd, check=True)


def cmd_rl(args: argparse.Namespace) -> None:
    """Run stage-2 online RL training (``grpo`` remains a CLI alias)."""
    from open_ctf.training.online_rl import train_online_rl

    config = load_config(args.config)
    _run_online_rl_preflight(args)

    train_online_rl(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        config=config,
        resume_from=args.resume,
        challenge_registry=getattr(args, 'challenge_registry', None),
        agent_class=getattr(args, 'agent', None),
    )


def cmd_gepa(args: argparse.Namespace) -> None:
    """Run GEPA prompt optimization (Stage 3, after SFT + online RL)."""
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
        challenge_registry=getattr(args, 'challenge_registry', None),
        agent_class=getattr(args, 'agent', None),
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

    # Qwen3.5 compatibility: PEFT merge can emit qwen3_5_text config, but vLLM
    # expects qwen3_5 + vision_config for the renderer path.
    try:
        _patch_merged_config_for_vllm(base_model_id, args.output)
    except Exception as exc:
        logger.warning("Could not apply merged-config compatibility fix: %s", exc)
    try:
        _patch_merged_tokenizer_config(base_model_id, args.output)
    except Exception as exc:
        logger.warning("Could not apply merged-tokenizer compatibility fix: %s", exc)

    logger.info("Merged via PEFT -> %s", args.output)


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Open CTF Training Pipeline (SFT + online RL + GEPA)",
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
    sft_parser = subparsers.add_parser(
        "sft",
        help="Run SFT (--backend trl for Qwen3.5+, llamafactory for others)",
    )
    sft_parser.add_argument("--model", default=None, help="HF model id (overrides config)")
    sft_parser.add_argument("--data", required=True, help="Path to SFT JSONL data")
    sft_parser.add_argument("--val-data", default=None, help="Path to validation JSONL")
    sft_parser.add_argument("--output", required=True, help="Output directory")
    sft_parser.add_argument("--resume", default=None, help="Resume from checkpoint")
    sft_parser.add_argument(
        "--backend",
        choices=["trl", "llamafactory"],
        default=None,
        help="SFT backend (default: auto-detect from model — trl for Qwen3.5+, llamafactory for others)",
    )
    sft_parser.set_defaults(func=cmd_sft)

    # -- stage-2 online RL (SkyRL) ---------------------------------------
    rl_parser = subparsers.add_parser(
        "rl",
        help="Run stage-2 online RL via SkyRL (preferred command)",
    )
    _add_online_rl_args(rl_parser)
    rl_parser.set_defaults(func=cmd_rl)

    # Backward-compatible alias for existing scripts/users.
    grpo_parser = subparsers.add_parser(
        "grpo",
        help="Legacy alias for 'rl' (online RL via SkyRL)",
    )
    _add_online_rl_args(grpo_parser)
    grpo_parser.set_defaults(func=cmd_rl)

    # -- gepa (unchanged) -------------------------------------------------
    gepa_parser = subparsers.add_parser(
        "gepa",
        help="Optimize system prompt with GEPA (Stage 3, no weight updates)",
    )
    gepa_parser.add_argument("--model", required=True, help="LLM model id for dspy.LM")
    gepa_parser.add_argument("--data", required=True, help="Path to online RL JSONL data (challenges)")
    gepa_parser.add_argument("--output", required=True, help="Output directory for optimized prompt")
    gepa_parser.add_argument("--val-data", default=None, help="Validation JSONL (separate from train)")
    gepa_parser.add_argument(
        "--reflection-model", default=None,
        help="LLM for GEPA reflection (default: same as --model). "
             "For stronger mutations, point at a larger local model.",
    )
    gepa_parser.add_argument(
        "--budget", choices=["light", "medium", "heavy"], default="medium",
        help="GEPA budget preset (default: medium)",
    )
    gepa_parser.add_argument("--max-samples", type=int, default=None, help="Max training examples")
    gepa_parser.add_argument(
        "--challenge-registry", default=None,
        help="Path to challenge registry YAML for target URL resolution",
    )
    gepa_parser.add_argument(
        "--agent", default=None,
        help="Dotted path to a CTFAgent class (e.g. my_module.MyAgent). "
             "When set, wraps the agent in a DSPy Module for GEPA optimization.",
    )
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
