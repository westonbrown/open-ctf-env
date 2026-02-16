"""Group Relative Policy Optimization (GRPO) stage.

Uses TRL GRPOTrainer with:
  - DAPO loss with asymmetric clipping (epsilon_high=0.28)
  - beta=0.0 (no KL penalty, pure DAPO)
  - num_generations=8 for better group reward estimation
  - max_completion_length=4096 for full CTF trajectories
  - FP16 precision (better for RL per Unsloth docs)
  - reward_funcs=[fn] (list, not bare function)
  - GRPOConfig passed via ``args=`` (not ``config=``)
  - Unsloth vLLM fast inference when available (UNSLOTH_VLLM_STANDBY=1)

Uses Unsloth for model loading when available, falls back to standard
HuggingFace transformers + PEFT otherwise. The fallback is also used when
Unsloth's GRPO kernels have dtype issues (e.g. on Blackwell GB10).

Compatible with TRL >= 0.26 (processing_class, warmup_steps).
"""

import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional

# Pre-set vLLM standby mode before any Unsloth/vLLM imports (~30% memory savings)
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

import torch
from datasets import load_dataset
from trl import GRPOConfig, GRPOTrainer

logger = logging.getLogger(__name__)


def _set_moe_backend():
    """Set UNSLOTH_MOE_BACKEND for GB10 compatibility if not already set."""
    if "UNSLOTH_MOE_BACKEND" not in os.environ:
        os.environ["UNSLOTH_MOE_BACKEND"] = "grouped_mm"
        logger.info("Set UNSLOTH_MOE_BACKEND=grouped_mm (GB10 safe default)")


def _load_model_unsloth(model_path, max_seq_length, load_in_4bit, lora_cfg):
    """Load model via Unsloth FastLanguageModel (faster, optimized kernels)."""
    _set_moe_backend()
    from unsloth import FastLanguageModel
    from peft import PeftModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        gpu_memory_utilization=0.6,
    )

    if isinstance(model, PeftModel):
        logger.info("Model already has LoRA adapters from SFT, skipping get_peft_model")
        FastLanguageModel.for_training(model)
    else:
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


def _load_model_hf(model_path, max_seq_length, load_in_4bit, lora_cfg):
    """Load model via standard HuggingFace transformers + PEFT."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # Check if this is already a PEFT model (adapter checkpoint)
    if not isinstance(model, PeftModel):
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

    # Unsloth patches TRL's GRPOTrainer at import time in containers that
    # have Unsloth pre-installed. The patched trainer calls model.for_training()
    # and model.for_inference() which are Unsloth-specific. Add no-op stubs.
    if not hasattr(model, "for_training"):
        model.for_training = lambda **kw: None
    if not hasattr(model, "for_inference"):
        model.for_inference = lambda **kw: None
    # Also patch the base model if it's a PEFT wrapper
    base = getattr(model, "model", None)
    if base is not None:
        if not hasattr(base, "for_training"):
            base.for_training = lambda **kw: None
        if not hasattr(base, "for_inference"):
            base.for_inference = lambda **kw: None
    # And the underlying pretrained model
    pretrained = getattr(base, "model", None) if base else None
    if pretrained is not None:
        if not hasattr(pretrained, "for_training"):
            pretrained.for_training = lambda **kw: None
        if not hasattr(pretrained, "for_inference"):
            pretrained.for_inference = lambda **kw: None

    return model, tokenizer


def train_grpo(
    model_path: str,
    data_path: str,
    output_dir: str,
    config: Dict[str, Any],
    reward_fn: Callable[..., List[float]],
    resume_from: Optional[str] = None,
) -> str:
    """Run GRPO training with TRL.

    Tries Unsloth first for speed, falls back to standard HuggingFace
    if Unsloth's GRPO kernels fail (e.g. dtype issues on some GPUs).

    Args:
        model_path: Path to the SFT model (merged or adapter directory).
        data_path: Path to JSONL GRPO data with ``ground_truth_flag`` and
            ``optimal_steps`` columns.
        output_dir: Directory for checkpoints and final model.
        config: Merged config dict with keys: model, lora, grpo, output.
        reward_fn: A callable that scores completions. Must accept
            ``(completions, prompts=None, **kwargs)`` and return ``list[float]``.
        resume_from: Optional checkpoint path to resume from.

    Returns:
        Path to the saved final model directory.
    """
    logger.info("=" * 60)
    logger.info("GRPO TRAINING")
    logger.info("  Model:  %s", model_path)
    logger.info("  Data:   %s", data_path)
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 60)

    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    grpo_cfg = config.get("grpo", {})
    output_cfg = config.get("output", {})

    max_seq_length = model_cfg.get("max_seq_length", 8192)
    load_in_4bit = model_cfg.get("load_in_4bit", True)

    # --- Model + LoRA ---------------------------------------------------
    use_unsloth = os.environ.get("OPEN_CTF_NO_UNSLOTH", "").lower() not in ("1", "true")
    if use_unsloth:
        try:
            model, tokenizer = _load_model_unsloth(
                model_path, max_seq_length, load_in_4bit, lora_cfg
            )
            logger.info("Loaded model via Unsloth")
        except (ImportError, RuntimeError, ValueError, OSError) as e:
            logger.warning("Unsloth loading failed (%s), falling back to HF", e)
            use_unsloth = False

    if not use_unsloth:
        model, tokenizer = _load_model_hf(
            model_path, max_seq_length, load_in_4bit, lora_cfg
        )
        logger.info("Loaded model via HuggingFace transformers + PEFT")

    # --- Dataset ---------------------------------------------------------
    dataset = load_dataset("json", data_files=data_path, split="train")

    # GRPOTrainer requires a "prompt" column. Extract the system + user
    # messages from the full trajectory as the prompt.
    def _extract_prompt(example):
        messages = example["messages"]
        prompt_msgs = []
        for msg in messages:
            role = msg.get("role", "")
            if role in ("system", "user"):
                clean_msg = {"role": role, "content": msg.get("content", "")}
                prompt_msgs.append(clean_msg)
            else:
                break  # Stop at first assistant/tool message
        example["prompt"] = prompt_msgs
        return example

    dataset = dataset.map(_extract_prompt)
    if "messages" in dataset.column_names:
        dataset = dataset.remove_columns(["messages"])
    if "metadata" in dataset.column_names:
        dataset = dataset.remove_columns(["metadata"])

    # --- Determine wandb availability -----------------------------------
    from open_ctf.training import check_wandb_available

    report_to = check_wandb_available(output_cfg.get("report_to", "wandb"))

    # --- Convert warmup_ratio to warmup_steps ----------------------------
    warmup_ratio = grpo_cfg.get("warmup_ratio", 0.10)
    num_epochs = grpo_cfg.get("epochs", 1)
    batch_size = grpo_cfg.get("batch_size", 1)
    grad_accum = grpo_cfg.get("gradient_accumulation_steps", 8)
    total_samples = len(dataset)
    steps_per_epoch = max(1, total_samples // (batch_size * grad_accum))
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(0, int(math.ceil(warmup_ratio * total_steps)))

    # --- GRPO config (passed as args=, NOT config=) ----------------------
    grpo_training_config = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=grpo_cfg.get("learning_rate", 5e-6),
        warmup_steps=warmup_steps,
        logging_steps=output_cfg.get("logging_steps", 1),
        save_steps=output_cfg.get("save_steps", 50),
        fp16=True,
        optim="adamw_8bit",
        seed=42,
        report_to=report_to,
        gradient_checkpointing=True,
        max_grad_norm=grpo_cfg.get("max_grad_norm", 0.1),
        weight_decay=grpo_cfg.get("weight_decay", 0.1),
        # GRPO-specific
        num_generations=grpo_cfg.get("num_generations", 8),
        max_completion_length=grpo_cfg.get("max_completion_length", 4096),
        beta=grpo_cfg.get("beta", 0.0),
        loss_type=grpo_cfg.get("loss_type", "dapo"),
        epsilon_high=grpo_cfg.get("epsilon_high", 0.28),
        scale_rewards=grpo_cfg.get("scale_rewards", "group"),
    )

    # --- Trainer ---------------------------------------------------------
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        args=grpo_training_config,
    )

    trainer.train(resume_from_checkpoint=resume_from)

    # --- Save final model ------------------------------------------------
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("GRPO model saved to %s", final_dir)
    return final_dir
