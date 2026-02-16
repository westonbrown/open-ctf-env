# Open CTF Environment - Training Container
# Base: NGC PyTorch (ARM aarch64 compatible for DGX Spark GB10)
#
# Supports both SFT (with Unsloth) and GRPO (with OPEN_CTF_NO_UNSLOTH=1 fallback).
#
# Build:
#   docker build -t open-ctf-env .
#
# Run SFT:
#   docker run --gpus all -v ./data:/workspace/data -v ./outputs:/workspace/outputs \
#       open-ctf-env open-ctf-train sft --model unsloth/GLM-4.7-Flash \
#       --data /workspace/data/sft.jsonl --output /workspace/outputs/sft
#
# Run GRPO (no Unsloth):
#   docker run --gpus all -e OPEN_CTF_NO_UNSLOTH=1 \
#       -v ./data:/workspace/data -v ./outputs:/workspace/outputs \
#       open-ctf-env open-ctf-train grpo --model /workspace/outputs/sft/final \
#       --data /workspace/data/grpo.jsonl --output /workspace/outputs/grpo

FROM nvcr.io/nvidia/pytorch:25.11-py3

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /workspace

# Install Python dependencies in a single layer
# Pinned versions for reproducibility
RUN pip install --no-cache-dir \
    "trl>=0.28.0" \
    "transformers>=5.1.0" \
    "peft" \
    "accelerate" \
    "bitsandbytes" \
    "datasets" \
    "pydantic>=2.0.0" \
    "gymnasium" \
    "pyyaml" \
    "jsonlines" \
    "rich" \
    "wandb" \
    "docker"

# Install Unsloth (optional - GRPO may use OPEN_CTF_NO_UNSLOTH=1 fallback)
# This is a separate layer so GRPO-only images can skip it
RUN pip install --no-cache-dir "unsloth>=2026.2.1" || \
    echo "WARNING: Unsloth installation failed. Use OPEN_CTF_NO_UNSLOTH=1 for GRPO."

# Copy source code (configs are inside src/open_ctf/configs/)
COPY src/ /workspace/src/
COPY pyproject.toml /workspace/

# Install the package in editable mode
RUN pip install --no-cache-dir -e /workspace

# Environment configuration
ENV PYTHONPATH=/workspace/src:$PYTHONPATH \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PYTHONUNBUFFERED=1

# Default command shows help
CMD ["open-ctf-train", "--help"]
