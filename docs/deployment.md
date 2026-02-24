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
    --adapter outputs/grpo/final \
    --base-model Nanbeige/Nanbeige4.1-3B \
    --output models/ctf-agent-Q4_K_M.gguf \
    --quant Q4_K_M

# Export without quantization (F16)
open-ctf-export \
    --adapter outputs/grpo/final \
    --base-model Nanbeige/Nanbeige4.1-3B \
    --output models/ctf-agent-f16.gguf \
    --quant none
```

### 2. Merge Only (for vLLM serving)

```bash
open-ctf-train merge \
    --adapter outputs/grpo/final \
    --base-model Nanbeige/Nanbeige4.1-3B \
    --output outputs/merged
```

## Deployment Options

### Option A: Ollama (Recommended for Local)

Best for: local development, small-to-medium models (3B-30B).

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

### Option B: llama.cpp

Best for: maximum context, reasoning models, GGUF with `--jinja` template support.

```bash
# 1. Serve the model
llama-server \
    -m models/ctf-agent-Q4_K_M.gguf \
    --host 0.0.0.0 --port 8080 \
    --jinja \
    -c 32768

# 2. Run the agent (from another terminal)
OPENAI_API_BASE=http://localhost:8080/v1 \
open-ctf-agent \
    --platform cybench \
    --target "[Very Easy] Dynastic" \
    --model openai/ctf-agent \
    --max-turns 30
```

### Option C: vLLM (Production)

Best for: large models (24B+), high throughput, production inference.

```bash
# 1. Serve the merged model
vllm serve outputs/merged \
    --host 0.0.0.0 --port 8000 \
    --max-model-len 32768 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes

# 2. Run the agent
OPENAI_API_BASE=http://localhost:8000/v1 \
open-ctf-agent \
    --platform cybench \
    --target "[Easy] TimeKORP" \
    --model openai/ctf-agent \
    --max-turns 30
```

**Tool call parser selection:**

| Model Family | vLLM Parser | Notes |
|-------------|-------------|-------|
| Nanbeige4.1-3B, Qwen3 | `hermes` | Hermes tool format (ChatML) |
| GLM-4.7-Flash | `glm47` | GLM4 XML tool format |
| Devstral/Mistral | `mistral` | Mistral tool format |

### Option D: RunPod H200 (Cloud)

Best for: large models on cloud GPU.

```bash
# 1. SSH to RunPod instance
ssh root@<RUNPOD_IP>

# 2. Serve the model
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

OPENAI_API_BASE=http://localhost:8000/v1 \
open-ctf-agent \
    --platform cybench \
    --target "[Easy] TimeKORP" \
    --model openai/ctf-agent \
    --max-turns 30
```

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
| Nanbeige4.1-3B | 3B | 6GB | 2GB | ~4GB |
| Devstral Small 2 | 24B | 48GB | 14GB | ~18GB |
| GLM-4.7-Flash | 30B MoE | 60GB | 18GB | ~22GB |

## Docker Deployment

```bash
# Export via Docker
docker compose run --rm export

# Or serve the merged model directly
docker run --gpus all -p 8000:8000 \
    -v ./outputs/merged:/model \
    vllm/vllm-openai:latest \
    --model /model \
    --max-model-len 32768 \
    --dtype bfloat16
```

## Troubleshooting

**llama.cpp not found:**
```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp && make -j$(nproc)
export LLAMA_CPP_DIR=$(pwd)
```

**vLLM tool call errors:**
Add the correct `--tool-call-parser` for your model family (see table above).

**Ollama context too short:**
Always create a Modelfile with explicit `num_ctx`:
```
FROM ./model.gguf
PARAMETER num_ctx 32768
```

**GGUF conversion fails:**
Ensure the merged model has all required files:
`config.json`, `model*.safetensors`, `tokenizer.json`, `tokenizer_config.json`.

**DGX Spark (GB10) vLLM issues:**
GB10's sm_121a compute capability may not be supported by all vLLM versions. Use llama.cpp or Ollama as alternatives.
