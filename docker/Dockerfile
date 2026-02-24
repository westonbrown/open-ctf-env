# Open CTF Environment - Unified Dockerfile
#
# Multi-stage build with targets for each pipeline stage:
#   - base:     Core package (data tools, reward, formatters, ToolExecutor)
#   - sft:      LlamaFactory SFT (Stage 1) + merge + validate + export
#   - grpo:     SkyRL GRPO (Stage 2) + Ray + vLLM
#
# Build:
#   docker build -t open-ctf:sft  --target sft  -f docker/Dockerfile .
#   docker build -t open-ctf:grpo --target grpo -f docker/Dockerfile .
#
# Run:
#   docker compose run --rm sft
#   docker compose run --rm grpo

# ============================================================
# Base stage -- shared across all targets
# ============================================================
FROM nvcr.io/nvidia/pytorch:25.11-py3 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1
WORKDIR /workspace/open-ctf-env

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

# Project source
COPY . /workspace/open-ctf-env/
RUN pip install --no-cache-dir -e ".[dev]"

ENV PYTHONPATH=/workspace/open-ctf-env/src:$PYTHONPATH

# ============================================================
# SFT stage — LlamaFactory + merge + validate + export
# ============================================================
FROM base AS sft

RUN pip install --no-cache-dir \
    "llamafactory>=0.9.0" \
    "transformers>=5.0.0,!=5.1.0" \
    "peft>=0.14.0" \
    "accelerate>=1.4.0" \
    "bitsandbytes>=0.48.0" \
    "datasets>=3.0.0" \
    "deepspeed>=0.15.0" \
    "wandb"

CMD ["open-ctf-train", "sft", "--help"]

# ============================================================
# GRPO stage — SkyRL + Ray + vLLM
# ============================================================
FROM base AS grpo

RUN pip install --no-cache-dir \
    "skyrl-gym>=0.1.0" \
    "ray[default]>=2.40.0" \
    "peft>=0.14.0" \
    "accelerate>=1.4.0" \
    "bitsandbytes>=0.48.0" \
    "datasets>=3.0.0" \
    "transformers>=5.0.0,!=5.1.0" \
    "omegaconf" "hydra-core" \
    "wandb" "colorama"

ENV BNB_CUDA_VERSION=130 \
    TORCHINDUCTOR_MAX_AUTOTUNE_PRUNE_CHOICES_BASED_ON_SHARED_MEM=1

CMD ["open-ctf-train", "grpo", "--help"]
