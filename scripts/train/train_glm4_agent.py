#!/usr/bin/env python3
"""
Fine-tune GLM-4.7-Flash for CTF agent using Unsloth QLoRA.

Usage:
    python train_glm4_agent.py --config train_config.yaml
    python train_glm4_agent.py --config train_config.yaml --data custom_data.jsonl
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments


def load_config(config_path: str) -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_jsonl_dataset(path: str) -> Dataset:
    """Load JSONL dataset with messages format."""
    data = []
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            if "messages" in item:
                data.append(item)
            elif "conversations" in item:
                # Convert ShareGPT to messages format
                messages = []
                for conv in item["conversations"]:
                    role_map = {"system": "system", "human": "user", "gpt": "assistant"}
                    role = role_map.get(conv["from"], conv["from"])
                    messages.append({"role": role, "content": conv["value"]})
                data.append({"messages": messages})
    return Dataset.from_list(data)


def train(config: dict, data_override: str = None):
    """Fine-tune GLM-4.7-Flash using Unsloth QLoRA."""

    model_cfg = config["model"]
    lora_cfg = config["lora"]
    train_cfg = config["training"]
    output_cfg = config["output"]
    data_cfg = config["data"]

    data_path = data_override or data_cfg["path"]

    print("=" * 60)
    print("GLM-4.7-Flash Fine-tuning")
    print("=" * 60)
    print(f"Model: {model_cfg['name']}")
    print(f"Dataset: {data_path}")
    print(f"Output: {output_cfg['dir']}")
    print(f"LoRA: r={lora_cfg['r']}, alpha={lora_cfg['alpha']}")
    print(f"Batch: {train_cfg['batch_size']} x {train_cfg['gradient_accumulation_steps']}")
    print("=" * 60)

    # 1. Load Model
    print("\n[1/6] Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_cfg["name"],
        max_seq_length=model_cfg["max_seq_length"],
        dtype=None,
        load_in_4bit=model_cfg.get("load_in_4bit", True),
    )

    # 2. Add LoRA Adapters
    print("[2/6] Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_cfg["r"],
        target_modules=lora_cfg["target_modules"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg.get("dropout", 0),
        bias=lora_cfg.get("bias", "none"),
        use_gradient_checkpointing="unsloth",
        random_state=train_cfg.get("seed", 3407),
        use_rslora=lora_cfg.get("use_rslora", False),
        loftq_config=None,
    )

    # 3. Load Dataset
    print(f"[3/6] Loading dataset from {data_path}...")
    dataset = load_jsonl_dataset(data_path)
    print(f"       Loaded {len(dataset)} examples")

    # 4. Setup Training
    print("[4/6] Configuring trainer...")
    training_args = TrainingArguments(
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        warmup_steps=train_cfg.get("warmup_steps", 10),
        max_steps=train_cfg["max_steps"],
        learning_rate=train_cfg["learning_rate"],
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=output_cfg.get("logging_steps", 1),
        optim=train_cfg.get("optim", "adamw_8bit"),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        seed=train_cfg.get("seed", 3407),
        output_dir=output_cfg["dir"],
        save_steps=output_cfg.get("save_steps", 25),
        save_total_limit=output_cfg.get("save_total_limit", 3),
        report_to=output_cfg.get("report_to", "none"),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field=data_cfg.get("text_field", "messages"),
        max_seq_length=model_cfg["max_seq_length"],
        dataset_num_proc=data_cfg.get("num_proc", 2),
        packing=data_cfg.get("packing", False),
        args=training_args,
    )

    # 5. Train
    print("[5/6] Training...")
    gpu_stats = torch.cuda.get_device_properties(0)
    print(f"       GPU: {gpu_stats.name} ({gpu_stats.total_memory // 1024**3} GB)")

    trainer_stats = trainer.train()

    # 6. Save
    print("[6/6] Saving model...")
    lora_path = f"{output_cfg['dir']}/lora_model"
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)

    # Report
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  Time: {trainer_stats.metrics['train_runtime']:.1f}s")
    print(f"  Loss: {trainer_stats.metrics.get('train_loss', 'N/A')}")
    print(f"  Saved to: {lora_path}")
    print("=" * 60)

    return trainer_stats


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GLM-4.7-Flash")
    parser.add_argument("--config", "-c", default="scripts/train/train_config.yaml")
    parser.add_argument("--data", "-d", help="Override data path from config")
    args = parser.parse_args()

    config = load_config(args.config)
    Path(config["output"]["dir"]).mkdir(parents=True, exist_ok=True)
    train(config, args.data)


if __name__ == "__main__":
    main()
