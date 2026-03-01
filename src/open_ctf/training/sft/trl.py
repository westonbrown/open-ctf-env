"""TRL-based Supervised Fine-Tuning (SFT) stage.

Drop-in alternative to the LlamaFactory SFT backend for models that require
``transformers >= 5.2.0`` (e.g. Qwen3.5-27B with Gated DeltaNet attention).

Uses vanilla HuggingFace ``SFTTrainer`` (TRL) + ``peft`` LoRA directly,
bypassing LlamaFactory entirely.  Same ``train_sft()`` signature so the CLI
can route between backends with ``--backend trl|llamafactory``.

Advantages:
  - No pinned ``transformers`` ceiling — works with any HF-supported model
  - Native ``tokenizer.apply_chat_template()`` formatting (model-specific)
  - Simpler dependency surface (trl + peft + transformers + torch)

Limitations vs LlamaFactory:
  - No built-in tool format adapters (relies on the tokenizer's own template)
  - No built-in DeepSpeed ZeRO integration (single-GPU focus for B200)
  - No built-in ``neat_packing`` (uses TRL's packing instead)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _load_model_config(model_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve model-specific training config.

    Reads from config dict (loaded from training_*.yaml) or falls back to
    sensible defaults matching the B200 VRAM budget.
    """
    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    sft_cfg = config.get("sft", {})
    output_cfg = config.get("output", {})

    return {
        "model_name": model_cfg.get("name", model_id),
        "max_seq_length": model_cfg.get("max_seq_length", 8192),
        "load_in_4bit": model_cfg.get("load_in_4bit", False),
        "load_in_8bit": model_cfg.get("load_in_8bit", False),
        # LoRA
        "lora_r": lora_cfg.get("r", 64),
        "lora_alpha": lora_cfg.get("alpha", 128),
        "lora_dropout": lora_cfg.get("dropout", 0.0),
        "lora_target_modules": lora_cfg.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        "use_rslora": lora_cfg.get("use_rslora", False),
        # SFT hyperparameters
        "epochs": sft_cfg.get("epochs", 3),
        "batch_size": sft_cfg.get("batch_size", 1),
        "gradient_accumulation_steps": sft_cfg.get("gradient_accumulation_steps", 8),
        "learning_rate": sft_cfg.get("learning_rate", 2e-4),
        "warmup_ratio": sft_cfg.get("warmup_ratio", 0.03),
        "weight_decay": sft_cfg.get("weight_decay", 0.01),
        "lr_scheduler_type": sft_cfg.get("lr_scheduler_type", "cosine"),
        "packing": sft_cfg.get("packing", True),
        "flash_attn": sft_cfg.get("flash_attn", True),
        "gradient_checkpointing": sft_cfg.get("gradient_checkpointing", True),
        "tf32": sft_cfg.get("tf32", False),
        # Output
        "save_steps": output_cfg.get("save_steps", 25),
        "logging_steps": output_cfg.get("logging_steps", 1),
        "report_to": output_cfg.get("report_to", "none"),
    }


def _load_jsonl_messages(data_path: str) -> List[Dict[str, Any]]:
    """Load JSONL data and return list of samples with 'messages' key."""
    samples = []
    with open(data_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if "messages" in sample:
                samples.append(sample)
    logger.info("Loaded %d samples from %s", len(samples), data_path)
    return samples


def _format_dataset(
    samples: List[Dict[str, Any]],
    tokenizer,
    max_seq_length: int,
    think_kwargs: Optional[Dict[str, Any]] = None,
) -> "Dataset":
    """Format samples using the tokenizer's native chat template.

    Uses ``tokenizer.apply_chat_template()`` to produce the model's native
    format (ChatML for Qwen3, Mistral format for Devstral, etc.).  This
    ensures tool calls, thinking blocks, and special tokens are rendered
    correctly without manual template logic.

    Args:
        think_kwargs: Dict of template kwargs for thinking preservation
            (from ``_detect_thinking_support``). Empty dict or None to disable.

    Returns a HuggingFace Dataset with a ``text`` column.
    """
    from datasets import Dataset

    texts = []
    skipped = 0
    for sample in samples:
        messages = sample["messages"]

        # Normalize tool_calls arguments: some traces have JSON strings
        # instead of dicts (tokenizer.apply_chat_template expects dicts).
        messages = _normalize_messages(messages)

        try:
            # apply_chat_template with tokenize=False → raw text string
            # including all special tokens. SFTTrainer will tokenize.
            chat_kwargs = {"tokenize": False, "add_generation_prompt": False}
            if think_kwargs:
                chat_kwargs.update(think_kwargs)

            text = tokenizer.apply_chat_template(messages, **chat_kwargs)
            texts.append(text)
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                logger.warning("Skipping sample (template error): %s", e)

    if skipped:
        logger.warning("Skipped %d/%d samples due to template errors", skipped, len(samples))

    logger.info("Formatted %d samples (skipped %d)", len(texts), skipped)
    return Dataset.from_dict({"text": texts})


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize messages for tokenizer.apply_chat_template().

    Fixes:
    - tool_calls[].function.arguments: JSON string → dict
    - Missing content fields: None → ""
    - tool role messages: ensure tool_call_id present
    """
    normalized = []
    for msg in messages:
        msg = dict(msg)  # shallow copy

        # Ensure content is a string (some traces have None)
        if msg.get("content") is None:
            msg["content"] = ""

        # Normalize tool_calls arguments from JSON strings to dicts
        if "tool_calls" in msg and msg["tool_calls"]:
            new_tool_calls = []
            for tc in msg["tool_calls"]:
                tc = dict(tc)
                if "function" in tc:
                    fn = dict(tc["function"])
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            fn["arguments"] = json.loads(args)
                        except (json.JSONDecodeError, TypeError):
                            fn["arguments"] = {"raw": args}
                    elif args is None:
                        fn["arguments"] = {}
                    tc["function"] = fn
                new_tool_calls.append(tc)
            msg["tool_calls"] = new_tool_calls

        normalized.append(msg)
    return normalized


def train_sft(
    model_id: str,
    data_path: str,
    output_dir: str,
    config: Dict[str, Any],
    val_data_path: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> str:
    """Run SFT training via TRL SFTTrainer + peft LoRA.

    Drop-in replacement for the LlamaFactory ``train_sft()`` — same
    signature, same config format, different backend.

    Args:
        model_id: HuggingFace model identifier or local path.
        data_path: Path to JSONL training data (OpenAI messages format).
        output_dir: Directory for checkpoints and final adapter.
        config: Merged config dict from training.yaml.
        val_data_path: Optional path to validation JSONL.
        resume_from: Optional checkpoint path to resume from.

    Returns:
        Path to the saved LoRA adapter directory.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, TaskType
    from trl import SFTConfig, SFTTrainer

    logger.info("=" * 60)
    logger.info("SFT TRAINING (TRL)")
    logger.info("  Model:  %s", model_id)
    logger.info("  Data:   %s", data_path)
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 60)

    # 1. Resolve config
    cfg = _load_model_config(model_id, config)
    os.makedirs(output_dir, exist_ok=True)

    # Save resolved config for reproducibility
    config_path = os.path.join(output_dir, "trl_sft_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    logger.info("Resolved config written to %s", config_path)

    # 2. Load tokenizer
    logger.info("Loading tokenizer: %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token = eos_token (%s)", tokenizer.eos_token)

    # 3. Load model
    logger.info("Loading model: %s", model_id)
    # Resolve attention backend: try flash_attention_2, fall back to sdpa/eager.
    # B200 vLLM containers often have a flash_attn stub (for vLLM) that doesn't
    # export flash_attn_func needed by transformers. SDPA is the safe default.
    attn_impl = "eager"
    if cfg["flash_attn"]:
        try:
            from flash_attn import flash_attn_func  # noqa: F401
            attn_impl = "flash_attention_2"
        except (ImportError, AttributeError):
            attn_impl = "sdpa"
            logger.info("flash_attn not available, using SDPA attention")

    model_kwargs = {
        "trust_remote_code": True,
        # HF AutoModelForCausalLM expects torch_dtype (not dtype).
        "torch_dtype": torch.bfloat16,
        "attn_implementation": attn_impl,
    }

    # Quantization (optional — not needed on B200 but supported)
    if cfg["load_in_4bit"] or cfg["load_in_8bit"]:
        try:
            # Verify bitsandbytes can actually quantize by testing the native lib.
            # Import may succeed but CUDA binary may be missing/incompatible.
            import bitsandbytes as bnb
            # Create a small test tensor and try to quantize it
            test_tensor = torch.randn(16, 16, dtype=torch.float32, device="cuda")
            bnb.functional.quantize_4bit(test_tensor)
            del test_tensor
            torch.cuda.empty_cache()
        except Exception as e:
            logger.warning(
                "bitsandbytes 4-bit quantization not functional (%s), "
                "falling back to bf16. Model will use more memory but "
                "training quality is unaffected.",
                e,
            )
            cfg["load_in_4bit"] = False
            cfg["load_in_8bit"] = False

    if cfg["load_in_4bit"]:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    elif cfg["load_in_8bit"]:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)

    # 4. Configure LoRA (SFTTrainer applies it via peft_config)
    logger.info(
        "LoRA config: r=%d, alpha=%d, modules=%s",
        cfg["lora_r"], cfg["lora_alpha"], cfg["lora_target_modules"],
    )
    lora_config = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        task_type=TaskType.CAUSAL_LM,
        use_rslora=cfg["use_rslora"],
        bias="none",
    )

    # 5. Format data
    logger.info("Formatting training data with tokenizer chat template...")
    train_samples = _load_jsonl_messages(data_path)

    # Detect thinking mode support and get template kwargs.
    # Returns dict of kwargs like {enable_thinking: True, keep_all_think: True}
    # or empty dict if model doesn't support <think> blocks.
    think_kwargs = _detect_thinking_support(tokenizer, model_id)
    logger.info("Thinking mode: %s", "enabled" if think_kwargs else "disabled")
    if think_kwargs:
        logger.info("Thinking kwargs: %s", think_kwargs)

    train_dataset = _format_dataset(
        train_samples, tokenizer, cfg["max_seq_length"],
        think_kwargs=think_kwargs,
    )

    eval_dataset = None
    if val_data_path and Path(val_data_path).exists():
        val_samples = _load_jsonl_messages(val_data_path)
        eval_dataset = _format_dataset(
            val_samples, tokenizer, cfg["max_seq_length"],
            think_kwargs=think_kwargs,
        )
        logger.info("Validation dataset: %d samples", len(eval_dataset))

    # 6. Configure SFTTrainer
    # Detect wandb availability
    from open_ctf.training import check_wandb_available
    report_to = check_wandb_available(cfg["report_to"])

    sft_config = SFTConfig(
        output_dir=output_dir,
        # Training
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg["weight_decay"],
        bf16=True,
        tf32=cfg.get("tf32", False),
        # Use 8-bit optimizer if bitsandbytes works, else standard AdamW
        optim="adamw_8bit" if cfg.get("load_in_4bit") or cfg.get("load_in_8bit") else "adamw_torch",
        # Context — TRL 0.28+ uses max_length (not max_seq_length)
        max_length=cfg["max_seq_length"],
        packing=cfg["packing"],
        dataset_text_field="text",
        # Checkpointing
        save_steps=cfg["save_steps"],
        save_total_limit=5,
        save_only_model=True,
        # Logging
        logging_steps=cfg["logging_steps"],
        report_to=report_to,
        run_name="sft-trl",
        # Misc
        seed=42,
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.get("gradient_checkpointing", True) else None,
        dataloader_num_workers=0,  # GB10 unified memory: fork overhead from workers causes OOM
        remove_unused_columns=True,
        # Evaluation
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=cfg["save_steps"] if eval_dataset else None,
        # Liger Kernel: fused cross-entropy avoids materializing full
        # vocab_size×seq_len float32 logits tensor. Critical for large-vocab
        # models (Nanbeige 166K vocab at 131K ctx = 82GB float32 logits OOM).
        use_liger_kernel=cfg.get("use_liger_kernel", True),
    )

    if resume_from:
        sft_config.resume_from_checkpoint = resume_from

    # 7. Train
    logger.info("Starting SFT training...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )
    trainer.train(resume_from_checkpoint=resume_from)

    # 8. Save final adapter
    final_dir = os.path.join(output_dir, "final")
    logger.info("Saving final LoRA adapter to %s", final_dir)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    logger.info("SFT training complete (TRL backend). Output: %s", final_dir)
    return final_dir


def _detect_thinking_support(tokenizer, model_id: str) -> Dict[str, Any]:
    """Detect thinking mode support and return kwargs to preserve ``<think>``.

    Different models use different template kwargs for thinking:
      - Qwen3/3.5: ``enable_thinking=True``
      - Nanbeige4.1: ``keep_all_think=True`` (preserves thinking in ALL turns)

    Returns a dict of kwargs to pass to ``apply_chat_template()`` for SFT.
    Empty dict means no thinking support detected.
    """
    vocab = tokenizer.get_vocab()
    if "<think>" not in vocab:
        return {}

    # Probe which thinking kwargs the template accepts.
    # For SFT we want ALL turns to preserve <think> so the model learns
    # reasoning patterns throughout multi-turn conversations.
    think_kwargs: Dict[str, Any] = {}
    test_msgs = [{"role": "user", "content": "test"}]

    for kwarg in ("enable_thinking", "keep_all_think"):
        try:
            tokenizer.apply_chat_template(
                test_msgs, tokenize=False,
                add_generation_prompt=True,
                **{kwarg: True},
            )
            think_kwargs[kwarg] = True
        except (TypeError, Exception):
            pass

    if think_kwargs:
        logger.info("Thinking kwargs detected: %s", list(think_kwargs.keys()))
    else:
        logger.info("<think> in vocab but no template kwargs accepted; "
                     "inline <think> tags will be passed through as-is")

    return think_kwargs
