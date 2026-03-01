# Open CTF Environment - Unified Dockerfile
#
# Multi-stage build with targets for each pipeline stage:
#   - base:      Core package (data tools, reward, formatters, ToolExecutor)
#   - sft:       TRL-based SFT (transformers>=5.2.0 for Qwen3.5)
#   - grpo:      SkyRL GRPO + Ray + vLLM nightly (Qwen3.5 support)
#
# Required versions (Qwen3.5 hybrid linear-attention):
#   - PyTorch:        >=2.10.0 (cu128)
#   - transformers:   >=5.2.0  (Qwen3_5ForConditionalGeneration)
#   - vLLM:           nightly  (>=0.16.1rc1 from wheels.vllm.ai/nightly)
#   - causal-conv1d:  ==1.6.0  (ABI must match PyTorch version)
#   - flash-linear-attention: ==0.4.1
#
# Build:
#   docker build -t open-ctf:sft      --target sft      -f docker/Dockerfile .
#   docker build -t open-ctf:grpo     --target grpo     -f docker/Dockerfile .
#
# Run:
#   docker compose run --rm sft
#   docker compose run --rm grpo

# ============================================================
# Base stage -- shared across all targets
# NGC PyTorch 25.11 provides CUDA 12.8, torch 2.9.1+cu128, NCCL
# ============================================================
FROM nvcr.io/nvidia/pytorch:25.11-py3 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1
WORKDIR /workspace/open-ctf-env

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    nmap net-tools libimage-exiftool-perl \
    gobuster nikto sqlmap \
    dnsutils netcat-openbsd ncat \
    jq whois traceroute iputils-ping \
    ffuf tcpdump smbclient \
    && rm -rf /var/lib/apt/lists/*

# Project source
COPY . /workspace/open-ctf-env/
RUN pip install --no-cache-dir -e ".[dev]"

ENV PYTHONPATH=/workspace/open-ctf-env/src:$PYTHONPATH

# ============================================================
# SFT stage — TRL backend (transformers>=5.2.0 for Qwen3.5)
# ============================================================
FROM base AS sft

RUN pip install --no-cache-dir -e ".[sft-trl]"

CMD ["open-ctf-train", "sft", "--help"]

# ============================================================
# GRPO stage — SkyRL + Ray + vLLM
# ============================================================
FROM base AS grpo

# skyrl-train is source-installed from GitHub (not published on PyPI).
# Pinned to the commit our 19 patches are validated against.
RUN pip install --no-cache-dir \
    "git+https://github.com/SkyRL-Team/SkyRL-Train.git@19b5746a9ad973715ad9b171158f3564b9e3bc85" \
    -e ".[grpo]" \
    "skyrl-gym>=0.1.0" \
    "ray[default]>=2.40.0" \
    "colorama"

# Apply deterministic SkyRL/vLLM/Ray compatibility patches for this image.
RUN bash /workspace/open-ctf-env/docker/patches/apply_all_patches.sh

ENV BNB_CUDA_VERSION=130 \
    TORCHINDUCTOR_MAX_AUTOTUNE_PRUNE_CHOICES_BASED_ON_SHARED_MEM=1

CMD ["open-ctf-train", "rl", "--help"]
