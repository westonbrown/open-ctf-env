# GLM-4.7-Flash: Deploy → SFT → QLoRA (OpenCTF/OpenEnv)

This guide shows a full path to:
1) deploy a base GLM-4.7-Flash model,
2) run it against OpenCTF/OpenEnv to collect data,
3) run SFT + QLoRA using the provided training script.

Everything below is based on the files in this repo.

---

## 0) Prereqs (local)

- Python 3.10+
- GPU with enough VRAM for GLM-4.7-Flash (A100/H100 recommended)
- Docker + Docker Compose (for XBow challenges)
- `uv` (recommended) or another Python env manager

Activate your env and install dependencies (example):

```bash
uv venv && source .venv/bin/activate
uv pip sync pyproject.toml
```

---

## 1) Deploy (serve) base GLM-4.7-Flash

Pick **one** serving option and keep it running.

### Option A: vLLM (matches scripts/train/README.md)

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

Then set `.env` in repo root (copy from `env.example`):

```bash
cp env.example .env
```

Edit `.env` for a local OpenAI-compatible endpoint:

```
CYBER_AGENT_PROVIDER=litellm
CYBER_AGENT_LLM_MODEL=openai/glm-4.7-flash
OPENAI_API_BASE=http://localhost:8001/v1
OPENAI_API_KEY=sk-no-key-required
```

### Option B: llama.cpp server (GGUF)

If you already have a GGUF build of GLM-4.7-Flash:

```bash
./llama.cpp/llama-server \
  --model /path/to/glm-4.7-flash.gguf \
  --alias "glm-4.7-flash" \
  --port 8001 \
  --jinja
```

Then set `.env` the same way as Option A (OpenAI-compatible).

---

## 2) Run OpenCTF/OpenEnv to collect data

You can drive the environment via the provided runner to produce agent sessions.

### 2.1 Start a challenge and run the agent

```bash
python scripts/run_agent.py --challenge XBEN-001-24 --iterations 50
```

This will:
- start the XBow challenge with Docker Compose,
- run Cyber-AutoAgent against it,
- optionally stop the challenge when done.

### 2.2 Save sessions to training data

The data conversion script expects session logs in a directory. Point it at your logs:

```bash
python scripts/data/convert_to_sharegpt.py \
  --input /path/to/your/session/logs \
  --output scripts/train/train_sharegpt.jsonl \
  --format openai_tools \
  --success-only
```

(Optional) add failure-recovery augmentation:

```bash
python scripts/data/augment_xbow.py \
  --input scripts/train/train_sharegpt.jsonl \
  --output scripts/train/train_augmented.jsonl
```

If you generate augmented data, update `scripts/train/train_config.yaml` to point to it:

```yaml
# scripts/train/train_config.yaml
data:
  path: "scripts/train/train_augmented.jsonl"
```

---

## 3) SFT + QLoRA training (Unsloth)

Training is done by `scripts/train/train_glm4_agent.py` using the config in
`scripts/train/train_config.yaml`.

```bash
python scripts/train/train_glm4_agent.py --config scripts/train/train_config.yaml
```

Outputs are written to:

```
outputs/glm4-ctf/lora_model
```

---

## 4) Serve the fine‑tuned model

If you want to export and serve the LoRA model, use the same serving approach
as step (1) after converting/exporting as needed.

Example (from README) using Unsloth export to GGUF:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained("outputs/glm4-ctf/lora_model")
model.save_pretrained_gguf("outputs/glm4-ctf/gguf", tokenizer, quantization_method="q4_k_m")
```

Then serve with llama.cpp (example):

```bash
./llama.cpp/llama-server \
  --model outputs/glm4-ctf/gguf/model-Q4_K_M.gguf \
  --alias "glm-4.7-flash-ctf" \
  --port 8002 \
  --jinja
```

Update `.env` to point to the fine‑tuned model:

```
OPENAI_API_BASE=http://localhost:8002/v1
CYBER_AGENT_LLM_MODEL=openai/glm-4.7-flash-ctf
```

---

## 5) “Train through OpenEnv” (how it fits here)

In this repo, OpenEnv is used to **generate interaction data** (agent steps
against the environment). Training happens by converting those interactions
into ShareGPT/openai-tools format and running SFT/QLoRA.

High‑level loop:

1) Run the environment (OpenCTF / XBow) with an agent.
2) Collect logs of the agent’s actions + observations.
3) Convert logs to a training dataset.
4) Run `train_glm4_agent.py` (SFT + QLoRA).

If you want a custom data collection loop (no CAA), you can build one directly
on the Gym wrapper (`OpenCTFEnv`) or the OpenEnv server.

Minimal example using the Gym wrapper:

```python
from src.envs.open_ctf import OpenCTFEnv

env = OpenCTFEnv(challenge_id="sqli-login-1")
obs, info = env.reset()

# Example manual loop
log = []
for step in range(10):
    action = "whoami"  # replace with your policy output
    obs, reward, done, truncated, _ = env.step(action)
    log.append({"action": action, "obs": obs, "reward": reward})
    if done or truncated:
        break

env.close()
```

You would then serialize `log` into the expected `messages` format and run the
conversion + SFT steps above.

---

## Quick reference

```bash
# 1) Serve base model (vLLM)
vllm serve zai-org/GLM-4.7-Flash --served-model-name glm-4.7-flash

# 2) Run a challenge and collect sessions
python scripts/run_agent.py --challenge XBEN-001-24 --iterations 50

# 3) Convert logs to dataset
python scripts/data/convert_to_sharegpt.py -i /path/to/logs -o scripts/train/train_sharegpt.jsonl --format openai_tools --success-only

# 4) Train (SFT + QLoRA)
python scripts/train/train_glm4_agent.py --config scripts/train/train_config.yaml
```
