#!/usr/bin/env python3
"""Run Qwen3.5-27B SFT across 2x B200 GPUs using TRL + PEFT.

This script is tuned for long-context SFT on RunPod dual-B200 nodes.
It shards the base model across both GPUs via ``device_map=auto`` and
supports optional QLoRA (4-bit base model + BF16 compute).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

LOGGER = logging.getLogger("run_sft_qwen35_dual_b200")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="/workspace/models/qwen35-27b")
    parser.add_argument("--data", default="/workspace/open-ctf-env/data/sft.jsonl")
    parser.add_argument("--output", default="/workspace/outputs/sft_qwen35_27b_dualgpu")
    parser.add_argument("--max-length", type=int, default=49152)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--save-steps", type=int, default=25)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--max-memory-gpu0", default="175GiB")
    parser.add_argument("--max-memory-gpu1", default="175GiB")
    parser.add_argument("--attn-impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    parser.add_argument("--packing", action="store_true")
    parser.add_argument("--no-packing", dest="packing", action="store_false")
    parser.set_defaults(packing=False)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    parser.set_defaults(enable_thinking=True)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.set_defaults(load_in_4bit=False)
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-liger-kernel", action="store_true")
    parser.add_argument("--require-both-gpus", action="store_true")
    parser.set_defaults(require_both_gpus=True)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="If > 0, stop training after this many optimizer steps.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="If > 0, use only the first N samples from the input JSONL.",
    )
    return parser.parse_args()


def _normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for msg in messages:
        out = dict(msg)
        if out.get("content") is None:
            out["content"] = ""
        tool_calls = out.get("tool_calls")
        if tool_calls:
            patched = []
            for tc in tool_calls:
                tc_out = dict(tc)
                fn = dict(tc_out.get("function", {}))
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        fn["arguments"] = {"raw": args}
                elif args is None:
                    fn["arguments"] = {}
                tc_out["function"] = fn
                patched.append(tc_out)
            out["tool_calls"] = patched
        normalized.append(out)
    return normalized


def _messages_to_text(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, Any]],
    enable_thinking: bool,
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": False,
    }
    if enable_thinking:
        kwargs["enable_thinking"] = True
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        lines = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            lines.append(f"<|{role}|>\n{content}")
        return "\n".join(lines)


def _load_dataset(
    path: str,
    tokenizer: AutoTokenizer,
    enable_thinking: bool,
    max_samples: int = 0,
) -> Dataset:
    texts: list[str] = []
    skipped = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            messages = _normalize_messages(sample.get("messages", []))
            text = _messages_to_text(tokenizer, messages, enable_thinking=enable_thinking)
            if text.strip():
                texts.append(text)
            else:
                skipped += 1
            if max_samples > 0 and len(texts) >= max_samples:
                break
    LOGGER.info("Loaded %d text samples (skipped %d empty)", len(texts), skipped)
    return Dataset.from_dict({"text": texts})


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    os.makedirs(args.output, exist_ok=True)
    LOGGER.info("=" * 72)
    LOGGER.info("QWEN3.5-27B SFT (Dual B200)")
    LOGGER.info("model=%s", args.model)
    LOGGER.info("data=%s", args.data)
    LOGGER.info("output=%s", args.output)
    LOGGER.info("max_length=%d | qlora_4bit=%s | attn=%s", args.max_length, args.load_in_4bit, args.attn_impl)
    LOGGER.info("=" * 72)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": "auto",
        "attn_implementation": args.attn_impl,
        "device_map": "auto",
        "max_memory": {0: args.max_memory_gpu0, 1: args.max_memory_gpu1},
        "low_cpu_mem_usage": True,
    }
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16

    LOGGER.info("Loading base model with device_map=auto ...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False

    device_map = getattr(model, "hf_device_map", {})
    devices = sorted({d for d in device_map.values() if isinstance(d, int)})
    LOGGER.info("hf_device_map devices=%s (entries=%d)", devices, len(device_map))
    if args.require_both_gpus and len(devices) < 2:
        raise RuntimeError(
            f"Model did not shard across both GPUs (got devices={devices}). "
            "Adjust max_memory or device_map settings."
        )

    dataset = _load_dataset(
        args.data,
        tokenizer,
        enable_thinking=args.enable_thinking,
        max_samples=args.max_samples,
    )

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    sft_config = SFTConfig(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        max_length=args.max_length,
        bf16=True,
        optim="adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_text_field="text",
        packing=args.packing,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_only_model=True,
        save_total_limit=4,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=True,
        use_liger_kernel=args.use_liger_kernel,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    LOGGER.info("Starting SFT train() ...")
    trainer.train()

    final_dir = os.path.join(args.output, "final")
    LOGGER.info("Saving final adapter to %s", final_dir)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    LOGGER.info("SFT complete.")


if __name__ == "__main__":
    main()
