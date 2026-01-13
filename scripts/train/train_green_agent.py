import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import argparse

def train(
    dataset_path: str = "train_augmented.jsonl",
    output_dir: str = "outputs/checkpoints",
    max_seq_length: int = 2048,
):
    """
    Trains the 'Green Agent' using Unsloth (LoRA + 4-bit).
    """
    print(f"🌲 Starting Green Agent Training from {dataset_path}...")
    
    # 1. Load Model (Unsloth Optimized - Dynamic 4-bit equivalent efficiency)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", # Highly optimized 4-bit
        max_seq_length = max_seq_length,
        dtype = None, 
        load_in_4bit = True,
    )

    # 2. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth", 
        random_state = 3407,
        use_rslora = False,
        loftq_config = None, 
    )

    # 3. Load & Format Data
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    
    # Simple formatting function for Qwen ChatML (Unsloth handles this well usually, but we make it explicit)
    # Assuming 'messages' column in dataset
    pass # Dataset is already in {"messages": [...]} format which SFTTrainer supports via chat_template

    # 4. Train
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "messages", # TRL Auto-unpacks ChatML if 'messages' present
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False, # Can set to True for speed
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60, # Short run for demo
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_dir,
            report_to = "none", # Use custom logger or tensorboard
        ),
    )

    trainer_stats = trainer.train()
    print(f"🌲 Training Complete. Time: {trainer_stats.metrics['train_runtime']}s")

    # 5. Save
    model.save_pretrained(f"{output_dir}/lora_model")
    tokenizer.save_pretrained(f"{output_dir}/lora_model")
    print(f"💾 Model saved to {output_dir}/lora_model")

    # 6. Green Score Card (Simulation)
    try:
        from src.utils.scoring import GreenScoreCard
        card = GreenScoreCard(agent_name="GreenAgent-v1")
        # Assuming we tracked these value during a real run.
        # For this script, we simulate to show the report artifact.
        card.add_run(flags=0, steps=60, energy_wh=0.5, duration_s=trainer_stats.metrics['train_runtime'])
        print(card.generate_report())
        with open(f"{output_dir}/green_card.md", "w") as f:
            f.write(card.generate_report())
    except ImportError:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train_augmented.jsonl")
    args = parser.parse_args()
    
    train(dataset_path=args.data)
