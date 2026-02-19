"""Supervised Fine-Tuning (SFT) stage.

Uses Unsloth + TRL SFTTrainer with:
  - LoRA (r=64, alpha=128, use_rslora=False) via FastLanguageModel or PEFT
  - Sequence packing for ~3x throughput (Unsloth only)
  - Gradient checkpointing for VRAM efficiency
  - Native ChatML message handling (no dataset_text_field)

Falls back to HuggingFace transformers + PEFT when:
  - OPEN_CTF_NO_UNSLOTH=1 env var is set (for GB10 Triton OOM issues)
  - Unsloth import fails

HF fallback uses SDPA (PyTorch Scaled Dot-Product Attention) instead of
FlashAttention 2. On Blackwell GB10 (sm_121), PyTorch SDPA with cuDNN 9.13
is faster than FlashAttention 2.

MoE model notes (GLM-4.7-Flash):
  - 4-bit QLoRA NOT supported for MoE models (BitsAndBytes limitation)
  - Use BF16 LoRA instead (~60GB VRAM for GLM-4.7-Flash)
  - On GB10 (Blackwell sm_121): set UNSLOTH_MOE_BACKEND=grouped_mm to avoid
    Triton shared memory OOM (99KB limit vs 104KB+ required by MoE kernels)
  - Router layers NOT targeted by LoRA (per Unsloth recommendation)

Compatible with TRL >= 0.26 (SFTConfig, processing_class, max_length).
"""

import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


def _normalize_messages(messages: List[Dict]) -> List[Dict]:
    """Normalize messages for compatibility with model chat templates.

    Fixes common format mismatches:
    - Converts tool_calls[].function.arguments from JSON string to dict
      (required by GLM-4, Qwen3, and other models whose Jinja templates
      iterate over arguments with .items())
    - Removes empty assistant content fields when tool_calls are present
    """
    normalized = []
    for msg in messages:
        msg = dict(msg)  # shallow copy
        if "tool_calls" in msg and msg["tool_calls"]:
            tool_calls = []
            for tc in msg["tool_calls"]:
                tc = dict(tc)
                if "function" in tc:
                    func = dict(tc["function"])
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            func["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            func["arguments"] = {"raw": args}
                    tc["function"] = func
                tool_calls.append(tc)
            msg["tool_calls"] = tool_calls
        normalized.append(msg)
    return normalized


def _set_moe_backend():
    """Set UNSLOTH_MOE_BACKEND for GB10 compatibility if not already set.

    On Blackwell GB10 (sm_121), Triton MoE kernels need >99KB shared memory
    per thread block, exceeding the 99KB hardware limit. The 'grouped_mm'
    backend uses torch._grouped_mm instead of Triton kernels, avoiding the OOM.
    """
    if "UNSLOTH_MOE_BACKEND" not in os.environ:
        os.environ["UNSLOTH_MOE_BACKEND"] = "grouped_mm"
        logger.info("Set UNSLOTH_MOE_BACKEND=grouped_mm (GB10 safe default)")


def _load_model_unsloth(model_id, max_seq_length, load_in_4bit, lora_cfg):
    """Load model via Unsloth FastLanguageModel (faster, optimized kernels)."""
    _set_moe_backend()
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        lora_dropout=lora_cfg.get("dropout", 0),
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        use_rslora=lora_cfg.get("use_rslora", False),
        use_gradient_checkpointing="unsloth",
    )
    return model, tokenizer


def _load_model_hf(model_id, max_seq_length, load_in_4bit, lora_cfg):
    """Load model via standard HuggingFace transformers + PEFT (slower, no Unsloth kernels)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_cfg.get("r", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        lora_dropout=lora_cfg.get("dropout", 0),
        target_modules=lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.gradient_checkpointing_enable()

    return model, tokenizer


def train_sft(
    model_id: str,
    data_path: str,
    output_dir: str,
    config: Dict[str, Any],
    val_data_path: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> str:
    """Run SFT training with Unsloth.

    Args:
        model_id: HuggingFace model identifier.
        data_path: Path to JSONL training data (ChatML messages format).
        output_dir: Directory for checkpoints and final model.
        config: Merged config dict with keys: model, lora, sft, output.
        val_data_path: Optional path to validation data.
        resume_from: Optional checkpoint path to resume from.

    Returns:
        Path to the saved final model directory.
    """
    logger.info("=" * 60)
    logger.info("SFT TRAINING")
    logger.info("  Model:  %s", model_id)
    logger.info("  Data:   %s", data_path)
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 60)

    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    sft_cfg = config.get("sft", {})
    output_cfg = config.get("output", {})

    max_seq_length = model_cfg.get("max_seq_length", 8192)
    load_in_4bit = model_cfg.get("load_in_4bit", True)

    # --- Model + LoRA (Unsloth or HF fallback) --------------------------
    use_unsloth = os.environ.get("OPEN_CTF_NO_UNSLOTH", "").lower() not in ("1", "true")
    if use_unsloth:
        try:
            model, tokenizer = _load_model_unsloth(
                model_id, max_seq_length, load_in_4bit, lora_cfg
            )
            logger.info("Loaded model via Unsloth (optimized kernels)")
        except (ImportError, RuntimeError, OSError) as e:
            logger.warning("Unsloth loading failed (%s), falling back to HF", e)
            use_unsloth = False

    if not use_unsloth:
        model, tokenizer = _load_model_hf(
            model_id, max_seq_length, load_in_4bit, lora_cfg
        )
        logger.info("Loaded model via HuggingFace transformers + PEFT (no Unsloth)")

    # --- Dataset ---------------------------------------------------------
    dataset = load_dataset("json", data_files=data_path, split="train")
    eval_dataset = None
    if val_data_path and Path(val_data_path).exists():
        eval_dataset = load_dataset("json", data_files=val_data_path, split="train")

    # Drop columns SFTTrainer doesn't need (avoids PyArrow type conflicts
    # from mixed types in metadata/optimal_steps columns)
    keep_cols = {"messages"}
    for ds_name, ds in [("train", dataset), ("val", eval_dataset)]:
        if ds is None:
            continue
        drop = [c for c in ds.column_names if c not in keep_cols]
        if drop:
            if ds_name == "train":
                dataset = dataset.remove_columns(drop)
            else:
                eval_dataset = eval_dataset.remove_columns(drop)
            logger.info("Dropped columns from %s dataset: %s", ds_name, drop)

    # --- Tool schemas for chat template ---------------------------------
    # Inject formal JSON tool schemas into the chat template so the model
    # learns native tool-call format (parameter names, types, descriptions)
    # during SFT — not just the textual description in the system prompt.
    # This matches what TRL's GRPOTrainer injects during GRPO (tools= param).
    from open_ctf.training.tools import get_all_tools

    _tool_fns = get_all_tools()  # Safe without init_env(): only metadata used
    tools_for_template = None
    try:
        _test_msgs = [{"role": "user", "content": "test"}]
        tokenizer.apply_chat_template(
            _test_msgs, tools=_tool_fns, tokenize=False,
        )
        tools_for_template = _tool_fns
        logger.info(
            "Chat template supports tools= (%d tool schemas will be injected)",
            len(_tool_fns),
        )
    except Exception as e:
        logger.warning(
            "Chat template doesn't support tools= (%s), "
            "falling back to text-only tool descriptions",
            e,
        )

    # Build formatting_func to normalize messages at training time.
    # We do NOT use dataset.map() because _normalize_messages converts
    # tool_call arguments from JSON strings to dicts, and PyArrow can't
    # handle the resulting mixed types across rows (e.g., timeout: 300 vs "300").
    # Instead, normalize on-the-fly via formatting_func which returns
    # tokenizer.apply_chat_template output directly.
    #
    # TRL calls formatting_func in two ways:
    #   1. Test call: formatting_func(single_example) where single_example is
    #      a dict like {"messages": [list of message dicts]}
    #   2. Batch call: formatting_func(batch) where batch["messages"] is
    #      a list of message lists
    def _formatting_func(examples):
        """Format messages with normalization applied at template time."""
        raw_messages = examples["messages"]

        # Detect single vs batch: single example has messages as a list of dicts,
        # batch has messages as a list of lists of dicts.
        if raw_messages and isinstance(raw_messages[0], dict):
            # Single example: raw_messages IS the message list
            messages_list = [raw_messages]
        else:
            # Batch: raw_messages is a list of message lists
            messages_list = raw_messages

        texts = []
        for messages in messages_list:
            normalized = _normalize_messages(messages)
            text = tokenizer.apply_chat_template(
                normalized,
                tools=tools_for_template,
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return texts

    # --- Determine wandb availability -----------------------------------
    from open_ctf.training import check_wandb_available

    report_to = check_wandb_available(output_cfg.get("report_to", "wandb"))

    # --- Convert warmup_ratio to warmup_steps ----------------------------
    # TRL >= 0.26 deprecated warmup_ratio in favor of warmup_steps.
    warmup_ratio = sft_cfg.get("warmup_ratio", 0.03)
    num_epochs = sft_cfg.get("epochs", 3)
    batch_size = sft_cfg.get("batch_size", 2)
    grad_accum = sft_cfg.get("gradient_accumulation_steps", 8)
    total_samples = len(dataset)
    steps_per_epoch = max(1, total_samples // (batch_size * grad_accum))
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(0, int(math.ceil(warmup_ratio * total_steps)))

    # --- SFT config (TRL >= 0.26 uses SFTConfig) -------------------------
    # Packing is an Unsloth optimization - disable if using HF fallback
    enable_packing = sft_cfg.get("packing", True) and use_unsloth

    # When using formatting_func, SFTTrainer expects a "text" dataset_text_field
    # or we must set dataset_text_field=None and use formatting_func directly.
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=sft_cfg.get("learning_rate", 2e-4),
        warmup_steps=warmup_steps,
        weight_decay=sft_cfg.get("weight_decay", 0.01),
        lr_scheduler_type=sft_cfg.get("lr_scheduler_type", "cosine"),
        logging_steps=output_cfg.get("logging_steps", 1),
        save_steps=output_cfg.get("save_steps", 50),
        bf16=True,
        optim="adamw_8bit",
        seed=42,
        report_to=report_to,
        gradient_checkpointing=True,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=output_cfg.get("save_steps", 50) if eval_dataset else None,
        # SFT-specific (moved from SFTTrainer init to SFTConfig in TRL 0.26)
        max_length=max_seq_length,
        packing=enable_packing,
    )

    # --- Trainer ---------------------------------------------------------
    # Use formatting_func to normalize messages on-the-fly during training.
    # This avoids PyArrow serialization issues from mixed types in tool_call
    # arguments (e.g., timeout: 300 vs "300" across different rows).
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        formatting_func=_formatting_func,
    )

    trainer.train(resume_from_checkpoint=resume_from)

    # --- Save final model ------------------------------------------------
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("SFT model saved to %s", final_dir)
    return final_dir
