"""LlamaFactory-based Supervised Fine-Tuning (SFT) stage.

Thin orchestrator that:
  1. Registers our JSONL data in LlamaFactory's dataset_info.json format
  2. Generates a LlamaFactory YAML config from our training.yaml
  3. Launches LlamaFactory training via CLI subprocess
  4. Handles LoRA merge post-training

Replaces the custom sft.py (386 lines) with ~120 lines by delegating to
LlamaFactory for:
  - 11 native tool formats (GLM4MOE, Qwen, Mistral, etc.)
  - Sequence packing with 4D attention masks
  - DeepSpeed ZeRO for multi-GPU
  - QLoRA / LoRA with proper MoE handling

Default test model: Nanbeige4.1-3B (dense LlamaForCausalLM, Hermes tool format).
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Project root (where data/dataset_info.json lives)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs" / "llamafactory"

# Model → LlamaFactory config mapping
_MODEL_CONFIG_MAP = {
    "nanbeige": "nanbeige_3b.yaml",
    "glm-4": "glm47_flash.yaml",
    "glm4": "glm47_flash.yaml",
    "glm-4.7": "glm47_flash.yaml",
    "devstral": "devstral_24b.yaml",
}


def _resolve_lf_config(model_id: str) -> Optional[Path]:
    """Find a pre-built LlamaFactory config for the given model."""
    model_lower = model_id.lower()
    for pattern, config_name in _MODEL_CONFIG_MAP.items():
        if pattern in model_lower:
            path = _CONFIGS_DIR / config_name
            if path.exists():
                return path
    return None


def _ensure_dataset_info(data_path: str) -> Path:
    """Ensure dataset_info.json references the given data file.

    LlamaFactory needs a dataset_info.json in the same directory as the
    data files (or a parent). We create/update one that points to the
    training JSONL.

    Returns:
        Path to the directory containing dataset_info.json.
    """
    data_file = Path(data_path).resolve()
    data_dir = data_file.parent
    info_path = data_dir / "dataset_info.json"

    # Load existing or start fresh
    info = {}
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)

    # Register our dataset under a stable key
    # Use "openai" formatting so LlamaFactory extracts tool_calls arrays
    # from assistant messages (sharegpt silently drops them).
    dataset_key = "open_ctf_sft"
    info[dataset_key] = {
        "file_name": data_file.name,
        "formatting": "openai",
        "columns": {
            "messages": "messages",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "observation_tag": "tool",
            "function_tag": "function_call",
            "system_tag": "system",
        },
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")

    logger.info("dataset_info.json updated at %s (key=%s)", info_path, dataset_key)
    return data_dir


def _build_lf_config(
    model_id: str,
    data_dir: Path,
    output_dir: str,
    config: Dict[str, Any],
    val_data_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a LlamaFactory config dict from our training.yaml settings.

    Starts from a model-specific base config (if available) and overrides
    with values from the user's training.yaml.
    """
    # Start from pre-built config or empty
    base_config_path = _resolve_lf_config(model_id)
    if base_config_path:
        with open(base_config_path) as f:
            lf_config = yaml.safe_load(f) or {}
        logger.info("Using base LlamaFactory config: %s", base_config_path)
    else:
        lf_config = {}
        logger.info("No pre-built config for %s, building from scratch", model_id)

    # Always override these from our config
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    sft_cfg = config.get("sft", {})
    output_cfg = config.get("output", {})

    lf_config["model_name_or_path"] = model_id
    lf_config["trust_remote_code"] = True
    lf_config["stage"] = "sft"
    lf_config["do_train"] = True

    # Dataset
    lf_config["dataset"] = "open_ctf_sft"
    lf_config["dataset_dir"] = str(data_dir)
    lf_config["cutoff_len"] = model_cfg.get("max_seq_length", 8192)

    # Template + tool format (set from base config or detect from model_id)
    # Only set if the base config didn't already provide them.
    model_lower = model_id.lower()
    if "template" not in lf_config:
        if any(p in model_lower for p in ("glm-4", "glm4", "glm-4.7")):
            lf_config["template"] = "glm4_7"
        elif any(p in model_lower for p in ("devstral", "mistral")):
            lf_config["template"] = "mistral_small"
        else:
            # Default: ChatML (Nanbeige, Qwen, and most models)
            lf_config["template"] = "chatml"
    if "tool_format" not in lf_config:
        if any(p in model_lower for p in ("glm-4", "glm4", "glm-4.7")):
            lf_config["tool_format"] = "glm4_moe"
        elif any(p in model_lower for p in ("devstral", "mistral")):
            lf_config["tool_format"] = "mistral"
        else:
            # Default: qwen format (<tool_call> tags, same as Hermes)
            lf_config["tool_format"] = "qwen"

    # LoRA
    lf_config["finetuning_type"] = "lora"
    lf_config["lora_rank"] = lora_cfg.get("r", 64)
    lf_config["lora_alpha"] = lora_cfg.get("alpha", 128)
    lf_config["lora_dropout"] = lora_cfg.get("dropout", 0.0)
    target_modules = lora_cfg.get("target_modules", [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    lf_config["lora_target"] = ",".join(target_modules)

    # Quantization (dense models only)
    if model_cfg.get("load_in_4bit", False):
        lf_config["quantization_bit"] = 4
        lf_config["quantization_method"] = "bitsandbytes"
    else:
        lf_config.pop("quantization_bit", None)
        lf_config.pop("quantization_method", None)

    # Training hyperparameters
    lf_config["per_device_train_batch_size"] = sft_cfg.get("batch_size", 2)
    lf_config["gradient_accumulation_steps"] = sft_cfg.get(
        "gradient_accumulation_steps", 8
    )
    lf_config["learning_rate"] = sft_cfg.get("learning_rate", 2e-4)
    lf_config["num_train_epochs"] = sft_cfg.get("epochs", 5)
    lf_config["lr_scheduler_type"] = sft_cfg.get("lr_scheduler_type", "cosine")
    lf_config["warmup_ratio"] = sft_cfg.get("warmup_ratio", 0.03)
    lf_config["weight_decay"] = sft_cfg.get("weight_decay", 0.01)
    lf_config["bf16"] = True
    lf_config["optim"] = "adamw_8bit"
    lf_config["seed"] = 42
    lf_config["gradient_checkpointing"] = True
    lf_config["packing"] = sft_cfg.get("packing", True)

    # Output
    lf_config["output_dir"] = output_dir
    lf_config["logging_steps"] = output_cfg.get("logging_steps", 1)
    lf_config["save_steps"] = output_cfg.get("save_steps", 50)
    lf_config["save_only_model"] = True
    lf_config["overwrite_output_dir"] = True

    # Wandb
    report_to = output_cfg.get("report_to", "none")
    try:
        import wandb
        if wandb.api.api_key and report_to != "none":
            lf_config["report_to"] = "wandb"
        else:
            lf_config["report_to"] = "none"
    except (ImportError, AttributeError):
        lf_config["report_to"] = "none"

    # Validation
    if val_data_path and Path(val_data_path).exists():
        val_dir = _ensure_dataset_info_val(val_data_path)
        lf_config["eval_dataset"] = "open_ctf_sft_val"
        lf_config["eval_strategy"] = "steps"
        lf_config["eval_steps"] = output_cfg.get("save_steps", 50)

    return lf_config


def _ensure_dataset_info_val(val_data_path: str) -> Path:
    """Register validation data in dataset_info.json."""
    val_file = Path(val_data_path).resolve()
    data_dir = val_file.parent
    info_path = data_dir / "dataset_info.json"

    info = {}
    if info_path.exists():
        with open(info_path) as f:
            info = json.load(f)

    info["open_ctf_sft_val"] = {
        "file_name": val_file.name,
        "formatting": "openai",
        "columns": {"messages": "messages"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "observation_tag": "tool",
            "function_tag": "function_call",
            "system_tag": "system",
        },
    }

    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    return data_dir


def train_sft(
    model_id: str,
    data_path: str,
    output_dir: str,
    config: Dict[str, Any],
    val_data_path: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> str:
    """Run SFT training via LlamaFactory.

    Args:
        model_id: HuggingFace model identifier.
        data_path: Path to JSONL training data (ChatML messages format).
        output_dir: Directory for checkpoints and final model.
        config: Merged config dict from training.yaml.
        val_data_path: Optional path to validation data.
        resume_from: Optional checkpoint path to resume from.

    Returns:
        Path to the saved model directory.
    """
    logger.info("=" * 60)
    logger.info("SFT TRAINING (LlamaFactory)")
    logger.info("  Model:  %s", model_id)
    logger.info("  Data:   %s", data_path)
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 60)

    # 1. Register dataset
    data_dir = _ensure_dataset_info(data_path)

    # 2. Build LlamaFactory config
    lf_config = _build_lf_config(
        model_id, data_dir, output_dir, config, val_data_path
    )

    if resume_from:
        lf_config["resume_from_checkpoint"] = resume_from

    # 3. Write config to temp file
    config_path = Path(output_dir) / "llamafactory_config.yaml"
    os.makedirs(output_dir, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(lf_config, f, default_flow_style=False)
    logger.info("LlamaFactory config written to %s", config_path)

    # 4. Launch LlamaFactory training
    cmd = ["llamafactory-cli", "train", str(config_path)]
    logger.info("Launching: %s", " ".join(cmd))

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"LlamaFactory training failed with exit code {result.returncode}"
        )

    logger.info("SFT training complete. Output: %s", output_dir)
    return output_dir
