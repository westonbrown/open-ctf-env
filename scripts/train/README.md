# GLM-4.7-Flash Training for OpenCTF

Train a CTF-solving agent using GLM-4.7-Flash with QLoRA fine-tuning.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  1. Baseline    →   2. Fine-tune   →   3. Evaluate              │
│  ───────────        ───────────        ─────────                │
│  Serve base         Convert data       Compare base             │
│  GLM-4.7-Flash      Train QLoRA        vs fine-tuned            │
│  Run on OpenEnv     Save adapters      on same challenges       │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv venv && source .venv/bin/activate && uv pip sync pyproject.toml
UV_PROJECT_ENVIRONMENT=.venv
uv add zmq
python -m ipykernel install --user --name=.venv --display-name="Python (uv env)"

# Core dependencies
uv pip install unsloth
uv pip install -U vllm --pre --index-url https://pypi.org/simple --extra-index-url https://wheels.vllm.ai/nightly
uv pip install git+https://github.com/huggingface/transformers.git
```

**Hardware**: A100 80GB or H100 recommended. A100 40GB may work with reduced batch size.

---

## Step 1: Serve Base Model (No Fine-tuning)

### Option A: vLLM

```bash
vllm serve zai-org/GLM-4.7-Flash \
     --tensor-parallel-size 1 \
     --speculative-config.method mtp \
     --speculative-config.num_speculative_tokens 1 \
     --tool-call-parser glm47 \
     --reasoning-parser glm45 \
     --enable-auto-tool-choice \
     --served-model-name glm-4.7-flash
```

### Run Baseline with run_agent.py

Configure `.env` to point to your served model:

```bash
# .env
CYBER_AGENT_PROVIDER=litellm
CYBER_AGENT_LLM_MODEL=openai/glm-4.7-flash
OPENAI_API_BASE=http://localhost:8001/v1
OPENAI_API_KEY=sk-no-key-required
```

Then run:

```bash
# Single challenge
python scripts/run_agent.py --challenge XBEN-001-24

# Or direct target
python scripts/run_agent.py --target http://localhost:8080 --model openai/glm-4.7-flash
```

Record the results (success/fail, steps taken) for comparison later.

---

## Step 2: Prepare Training Data

Convert Cyber-AutoAgent session data to training format.

```bash
# Convert sessions to OpenAI messages format
python scripts/data/convert_to_sharegpt.py \
    --input scripts/train/training_data/sessions \
    --output scripts/train/train_sharegpt.jsonl \
    --format openai_tools \
    --success-only  # Only successful flag captures

# (Optional) Add failure recovery examples
python scripts/data/augment_xbow.py \
    --input scripts/train/train_sharegpt.jsonl \
    --output scripts/train/train_augmented.jsonl
```

---

## Step 3: Fine-tune with QLoRA

### Configure

Edit `train_config.yaml`:

```yaml
model:
  name: "unsloth/GLM-4.7-Flash-bnb-4bit"
  max_seq_length: 8192

lora:
  r: 16
  alpha: 32

training:
  batch_size: 1
  gradient_accumulation_steps: 8
  max_steps: 100
  learning_rate: 2.0e-4
```

### Train

```bash
python scripts/train/train_glm4_agent.py --config scripts/train/train_config.yaml
```

Output:
```
============================================================
GLM-4.7-Flash Fine-tuning
============================================================
[1/6] Loading model...
[2/6] Adding LoRA adapters...
[3/6] Loading dataset...
       Loaded 10 examples
[4/6] Configuring trainer...
[5/6] Training...
[6/6] Saving model...

Training Complete!
  Saved to: outputs/glm4-ctf/lora_model
============================================================
```

---

## Step 4: Serve Fine-tuned Model

### Export to GGUF

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("outputs/glm4-ctf/lora_model")
model.save_pretrained_gguf("outputs/glm4-ctf/gguf", tokenizer, quantization_method="q4_k_m")
```

### Serve

```bash
./llama.cpp/llama-server \
    --model outputs/glm4-ctf/gguf/model-Q4_K_M.gguf \
    --alias "glm-4.7-flash-ctf" \
    --port 8002 \
    --jinja
```

---

## Step 5: Evaluate Fine-tuned Model

Update `.env` to point to fine-tuned model:

```bash
OPENAI_API_BASE=http://localhost:8002/v1
CYBER_AGENT_LLM_MODEL=openai/glm-4.7-flash-ctf
```

Run same challenges:

```bash
python scripts/run_agent.py --challenge XBEN-001-24
```

Compare results with baseline.

---

## Quick Reference

```bash
# Full pipeline
# 1. Convert data
python scripts/data/convert_to_sharegpt.py \
    -i scripts/train/training_data/sessions \
    -o scripts/train/train_sharegpt.jsonl --success-only

# 2. Serve base model (terminal 1)
./llama.cpp/llama-server --model ./models/GLM-4.7-Flash-UD-Q4_K_XL.gguf --port 8001 --jinja

# 3. Baseline run
python scripts/run_agent.py -c XBEN-001-24 -m openai/glm-4.7-flash

# 4. Fine-tune
python scripts/train/train_glm4_agent.py -c scripts/train/train_config.yaml

# 5. Export & serve fine-tuned (terminal 2)
python -c "from unsloth import FastLanguageModel; m,t=FastLanguageModel.from_pretrained('outputs/glm4-ctf/lora_model'); m.save_pretrained_gguf('outputs/glm4-ctf/gguf',t,quantization_method='q4_k_m')"
./llama.cpp/llama-server --model outputs/glm4-ctf/gguf/model-Q4_K_M.gguf --port 8002 --jinja

# 6. Fine-tuned run
python scripts/run_agent.py -c XBEN-001-24 -m openai/glm-4.7-flash-ctf
```

---

## Files

```
scripts/train/
├── README.md              # This file
├── train_config.yaml      # Hyperparameters
├── train_glm4_agent.py    # Training script
└── training_data/sessions # Raw session data

scripts/data/
├── convert_to_sharegpt.py # Data conversion
└── augment_xbow.py        # Failure injection

scripts/
└── run_agent.py           # Agent runner (uses CAA)
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| OOM | Reduce `batch_size`, increase `gradient_accumulation_steps` |
| Slow training | Enable `packing: true` in config |
| Model not learning | Increase `max_steps`, check data quality |
| vLLM errors | Need transformers >= 4.49.0 |
