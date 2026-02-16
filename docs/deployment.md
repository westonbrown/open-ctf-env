# Deployment Guide

Instructions for deploying a trained Open CTF model for inference.

## Prerequisites

- A trained LoRA adapter (from `open-ctf-train sft` or `open-ctf-train grpo`)
- One of: Ollama, llama.cpp, or vLLM installed

## Export Pipeline

### 1. Merge LoRA + Convert to GGUF

```bash
# Merge adapter, convert to GGUF, quantize to Q4_K_M (one command)
open-ctf-export \
    --adapter outputs/sft/final \
    --base-model unsloth/GLM-4.7-Flash \
    --output models/ctf-agent-Q4_K_M.gguf \
    --quant Q4_K_M

# Export without quantization (F16)
open-ctf-export \
    --adapter outputs/sft/final \
    --base-model unsloth/GLM-4.7-Flash \
    --output models/ctf-agent-f16.gguf \
    --quant none
```

### 2. Merge Only (for vLLM serving)

```bash
open-ctf-train merge \
    --adapter outputs/sft/final \
    --output outputs/merged
```

## Deployment Options

### Option A: DGX Spark with Ollama

Best for: local development, small models (8B-30B).

```bash
# 1. Create Ollama model from GGUF
echo 'FROM ./models/ctf-agent-Q4_K_M.gguf
PARAMETER num_ctx 32768' > Modelfile
ollama create ctf-agent -f Modelfile

# 2. Test the model
ollama run ctf-agent "What tools do you have available?"

# 3. Run the agent
open-ctf-agent \
    --platform cybench \
    --target "[Very Easy] Dynastic" \
    --model ollama/ctf-agent \
    --max-turns 30
```

**DGX Spark specs:**
- GPU: Grace Blackwell GB10, 128GB unified memory
- Fits: BF16 models up to ~60GB, Q4_K_M up to ~120B params
- Ollama Docker: `docker run -d --gpus all -p 11434:11434 ollama/ollama:latest`

### Option B: DGX Spark with llama.cpp

Best for: maximum context, reasoning models, GGUF with `--jinja`.

```bash
# 1. Serve the model
llama-server \
    -m models/ctf-agent-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8080 \
    --jinja \
    -c 32768

# 2. Run the agent (from another terminal)
open-ctf-agent \
    --platform cybench \
    --target "[Very Easy] Dynastic" \
    --model openai/ctf-agent \
    --max-turns 30
# Set OPENAI_API_BASE=http://localhost:8080/v1
```

### Option C: RunPod H200 with vLLM

Best for: large models (100B+), high throughput, production inference.

```bash
# 1. SSH to RunPod instance
ssh root@<RUNPOD_IP>

# 2. Serve the merged model
vllm serve outputs/merged \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# 3. Run the agent locally (with SSH tunnel)
ssh -L 8000:localhost:8000 root@<RUNPOD_IP>

open-ctf-agent \
    --platform cybench \
    --target "[Easy] TimeKORP" \
    --model openai/ctf-agent \
    --max-turns 30
# Set OPENAI_API_BASE=http://localhost:8000/v1
```

**RunPod H200 specs:**
- GPU: NVIDIA H200 SXM, 141GB VRAM
- Cost: ~$3.59/hr
- Fits: BF16 models up to ~70B, or Q4 up to ~200B

## Quantization Options

| Quant | Bits/Weight | Quality | Speed | Use Case |
|-------|-------------|---------|-------|----------|
| F16 | 16 | Best | Slow | Evaluation |
| Q8_0 | 8 | Excellent | Medium | Development |
| Q5_K_M | 5 | Very Good | Fast | Balanced |
| **Q4_K_M** | **4** | **Good** | **Fast** | **Recommended** |
| Q3_K_M | 3 | Acceptable | Fastest | VRAM-limited |
| IQ2_M | 2 | Degraded | Fastest | Extreme compression |

## Model Size Estimates

| Base Model | Params | F16 | Q4_K_M | VRAM (Q4) |
|-----------|--------|-----|--------|-----------|
| Qwen3-8B | 8B | 16GB | 5GB | ~8GB |
| Devstral Small 2 | 24B | 48GB | 14GB | ~18GB |
| GLM-4.7 Flash | 8B | 16GB | 5GB | ~8GB |
| GPT-OSS 120B | 120B | 240GB | 67GB | ~80GB |

## Troubleshooting

**llama.cpp not found:**
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && make -j$(nproc)
export LLAMA_CPP_DIR=$(pwd)
```

**vLLM tool call errors:**
Add `--tool-call-parser hermes` for Qwen-family models,
or `--tool-call-parser mistral` for Mistral-family models.

**Ollama context too short:**
Always create a Modelfile with explicit `num_ctx`:
```
FROM ./model.gguf
PARAMETER num_ctx 32768
```

**GGUF conversion fails:**
Ensure the merged model has all required files:
`config.json`, `model*.safetensors`, `tokenizer.json`, `tokenizer_config.json`.
