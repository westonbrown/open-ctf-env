#!/bin/bash
# Open CTF Full Training Pipeline for DGX Spark (GB10)
# ====================================================
# Runs: SFT -> Merge -> GRPO in Docker containers.
#
# SFT + Merge: Uses unsloth-blackwell:v3 (Unsloth SFT works on GB10)
# GRPO: Uses nvcr.io/nvidia/pytorch:25.11-py3 + OPEN_CTF_NO_UNSLOTH=1
#       (workaround for Unsloth GRPO dtype bug on Blackwell GB10)
#
# Usage:
#   nohup bash scripts/run_full_pipeline.sh \
#     > logs/pipeline.log 2>&1 &
#
# To resume from a specific stage:
#   SKIP_SFT=1 bash scripts/run_full_pipeline.sh    # Skip SFT, run merge+GRPO
#   SKIP_MERGE=1 bash scripts/run_full_pipeline.sh   # Skip merge (use existing)
#   GRPO_ONLY=1 bash scripts/run_full_pipeline.sh    # Only run GRPO

set -euo pipefail

# -----------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------
WORKSPACE="${WORKSPACE:-$(pwd)}"
MODEL="unsloth/GLM-4.7-Flash"
SFT_DATA="data/sft.jsonl"
GRPO_DATA="data/grpo.jsonl"

# Use runs/ directory (user-writable) instead of outputs/ (root-owned from prev Docker runs)
RUN_ID="${RUN_ID:-run_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="runs/${RUN_ID}"
SFT_OUTPUT="${OUTPUT_DIR}/sft"
MERGE_OUTPUT="${OUTPUT_DIR}/merged"
GRPO_OUTPUT="${OUTPUT_DIR}/grpo"

# Container images
UNSLOTH_IMAGE="unsloth-blackwell:v3"
PYTORCH_IMAGE="nvcr.io/nvidia/pytorch:25.11-py3"

# Stage control (set env vars to skip stages)
SKIP_SFT="${SKIP_SFT:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"
GRPO_ONLY="${GRPO_ONLY:-0}"

# -----------------------------------------------------------------------
# Common Docker flags
# -----------------------------------------------------------------------
DOCKER_BASE_FLAGS=(
    --gpus all
    --rm
    --shm-size=32g
    --ulimit memlock=-1
    --ulimit stack=67108864
    -v "${WORKSPACE}:/workspace/open-ctf-env"
    -v "${HOME}/.cache/huggingface:/root/.cache/huggingface"
    -w /workspace/open-ctf-env
)

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
timestamp() { date "+%Y-%m-%d %H:%M:%S"; }

log() {
    echo "$(timestamp) | $*"
}

run_in_docker() {
    local image="$1"
    shift
    local env_flags=()
    while [[ "$1" == -e ]]; do
        env_flags+=("$1" "$2")
        shift 2
    done
    local cmd="$1"

    log "Docker image: ${image}"
    log "Command: ${cmd}"
    docker run "${DOCKER_BASE_FLAGS[@]}" "${env_flags[@]}" "${image}" bash -c "${cmd}"
}

# -----------------------------------------------------------------------
# Pre-flight checks
# -----------------------------------------------------------------------
log "================================================================"
log "OPEN CTF TRAINING PIPELINE - DGX Spark (GB10)"
log "================================================================"
log "Model:       ${MODEL}"
log "SFT data:    ${WORKSPACE}/${SFT_DATA}"
log "GRPO data:   ${WORKSPACE}/${GRPO_DATA}"
log "Output:      ${WORKSPACE}/${OUTPUT_DIR}"
log "SFT image:   ${UNSLOTH_IMAGE}"
log "GRPO image:  ${PYTORCH_IMAGE}"
log "================================================================"

# Create output directory
mkdir -p "${WORKSPACE}/${OUTPUT_DIR}"

# Verify data files exist
if [[ ! -f "${WORKSPACE}/${SFT_DATA}" ]]; then
    log "ERROR: SFT data not found at ${WORKSPACE}/${SFT_DATA}"
    exit 1
fi
if [[ ! -f "${WORKSPACE}/${GRPO_DATA}" ]]; then
    log "ERROR: GRPO data not found at ${WORKSPACE}/${GRPO_DATA}"
    exit 1
fi

# Check GPU availability
log "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.used,temperature.gpu --format=csv,noheader 2>/dev/null || {
    log "WARNING: nvidia-smi failed. GPU may not be available."
}

# -----------------------------------------------------------------------
# Stage 1: SFT
# -----------------------------------------------------------------------
if [[ "${GRPO_ONLY}" != "1" && "${SKIP_SFT}" != "1" ]]; then
    log ""
    log "================================================================"
    log "[Stage 1/3] SUPERVISED FINE-TUNING (SFT)"
    log "================================================================"

    SFT_CMD="pip install -e . 2>&1 | tail -1 && \
open-ctf-train sft \
  --model ${MODEL} \
  --data ${SFT_DATA} \
  --output ${SFT_OUTPUT}"

    run_in_docker "${UNSLOTH_IMAGE}" \
        -e UNSLOTH_MOE_BACKEND=grouped_mm \
        -e PYTHONPATH=/workspace/open-ctf-env/src \
        -e WANDB_MODE=disabled \
        -e TORCHDYNAMO_DISABLE=1 \
        "${SFT_CMD}"

    log "[Stage 1/3] SFT COMPLETE. Adapter at: ${SFT_OUTPUT}/final"
else
    log "Skipping SFT stage (SKIP_SFT=${SKIP_SFT}, GRPO_ONLY=${GRPO_ONLY})"
fi

# -----------------------------------------------------------------------
# Stage 2: Merge LoRA adapter into base weights
# -----------------------------------------------------------------------
if [[ "${GRPO_ONLY}" != "1" && "${SKIP_MERGE}" != "1" ]]; then
    log ""
    log "================================================================"
    log "[Stage 2/3] MERGE LoRA ADAPTER"
    log "================================================================"

    # Verify SFT output exists
    if [[ ! -d "${WORKSPACE}/${SFT_OUTPUT}/final" ]]; then
        log "ERROR: SFT adapter not found at ${WORKSPACE}/${SFT_OUTPUT}/final"
        log "Run SFT first or set SKIP_SFT=0"
        exit 1
    fi

    MERGE_CMD="pip install -e . 2>&1 | tail -1 && \
open-ctf-train merge \
  --adapter ${SFT_OUTPUT}/final \
  --base-model ${MODEL} \
  --output ${MERGE_OUTPUT}"

    run_in_docker "${UNSLOTH_IMAGE}" \
        -e UNSLOTH_MOE_BACKEND=grouped_mm \
        -e PYTHONPATH=/workspace/open-ctf-env/src \
        -e WANDB_MODE=disabled \
        -e TORCHDYNAMO_DISABLE=1 \
        "${MERGE_CMD}"

    log "[Stage 2/3] MERGE COMPLETE. Merged model at: ${MERGE_OUTPUT}"
else
    log "Skipping merge stage (SKIP_MERGE=${SKIP_MERGE}, GRPO_ONLY=${GRPO_ONLY})"
fi

# -----------------------------------------------------------------------
# Stage 3: GRPO
# -----------------------------------------------------------------------
if [[ "${SKIP_SFT}" != "1" || "${GRPO_ONLY}" == "1" || "${SKIP_MERGE}" != "1" ]]; then
    log ""
    log "================================================================"
    log "[Stage 3/3] GRPO TRAINING"
    log "================================================================"
    log "NOTE: Using PyTorch container + OPEN_CTF_NO_UNSLOTH=1"
    log "      (workaround for Unsloth GRPO dtype bug on GB10)"

    # Determine the model path for GRPO
    GRPO_MODEL="${MERGE_OUTPUT}"
    if [[ ! -d "${WORKSPACE}/${MERGE_OUTPUT}" ]]; then
        # If no merged model, try the SFT adapter directly
        GRPO_MODEL="${SFT_OUTPUT}/final"
        if [[ ! -d "${WORKSPACE}/${GRPO_MODEL}" ]]; then
            log "ERROR: No model found for GRPO. Need either merged model or SFT adapter."
            exit 1
        fi
        log "WARNING: Using SFT adapter directly (no merged model found)"
    fi

    # GRPO with HF fallback (no Unsloth) to avoid dtype bug
    GRPO_CMD="pip install trl>=0.28.0 peft>=0.14.0 accelerate>=1.0.0 bitsandbytes>=0.44.0 \
datasets>=2.0.0 pyyaml rich jsonlines transformers>=5.1.0 2>&1 | tail -5 && \
pip install -e . --no-deps 2>&1 | tail -1 && \
open-ctf-train grpo \
  --model ${GRPO_MODEL} \
  --data ${GRPO_DATA} \
  --output ${GRPO_OUTPUT}"

    run_in_docker "${PYTORCH_IMAGE}" \
        -e OPEN_CTF_NO_UNSLOTH=1 \
        -e UNSLOTH_MOE_BACKEND=grouped_mm \
        -e PYTHONPATH=/workspace/open-ctf-env/src \
        -e WANDB_MODE=disabled \
        -e TORCHDYNAMO_DISABLE=1 \
        "${GRPO_CMD}"

    log "[Stage 3/3] GRPO COMPLETE. Model at: ${GRPO_OUTPUT}/final"
fi

# -----------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------
log ""
log "================================================================"
log "PIPELINE COMPLETE"
log "================================================================"
log "SFT adapter:   ${WORKSPACE}/${SFT_OUTPUT}/final"
log "Merged model:  ${WORKSPACE}/${MERGE_OUTPUT}"
log "GRPO model:    ${WORKSPACE}/${GRPO_OUTPUT}/final"
log "================================================================"
log "Next steps:"
log "  1. Evaluate: open-ctf-eval --model ${GRPO_OUTPUT}/final"
log "  2. Export GGUF: open-ctf-export --model ${MERGE_OUTPUT}"
log "================================================================"
