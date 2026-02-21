#!/usr/bin/env bash
# =============================================================================
# Nanbeige4.1-3B Training Pipeline on DGX Spark
# =============================================================================
# Full pipeline: baseline → SFT → merge → GRPO → eval
#
# Usage:
#   # Run full pipeline
#   bash scripts/dgx_nanbeige.sh all
#
#   # Run individual stages
#   bash scripts/dgx_nanbeige.sh baseline   # BoxPwnr CyBench 40 challenges
#   bash scripts/dgx_nanbeige.sh sft        # SFT training
#   bash scripts/dgx_nanbeige.sh merge      # Merge LoRA into base
#   bash scripts/dgx_nanbeige.sh grpo       # GRPO training
#   bash scripts/dgx_nanbeige.sh eval       # Post-training CyBench eval
#
# Prerequisites:
#   - DGX Spark accessible via SSH (abrown@100.91.175.48)
#   - Docker with GPU support
#   - open-ctf-env synced to /home/abrown/open-ctf-env/
#
# This script is generalizable: change MODEL_ID and CONFIG to train
# other models (e.g. GLM-4.7-Flash with training_dgx.yaml).
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (edit these for different models)
# ---------------------------------------------------------------------------
MODEL_ID="${MODEL_ID:-Nanbeige/Nanbeige4.1-3B}"
MODEL_SHORT="${MODEL_SHORT:-nanbeige3b}"
CONFIG="${CONFIG:-src/open_ctf/configs/training_nanbeige.yaml}"
DGX_HOST="${DGX_HOST:-abrown@100.91.175.48}"
DGX_DIR="${DGX_DIR:-/home/abrown/open-ctf-env}"
LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# Training paths (on DGX)
SFT_DATA="${SFT_DATA:-data/sft.jsonl}"
GRPO_DATA="${GRPO_DATA:-data/grpo.jsonl}"
SFT_OUTPUT="outputs/sft-${MODEL_SHORT}"
SFT_MERGED="outputs/sft-${MODEL_SHORT}-merged"
GRPO_OUTPUT="outputs/grpo-${MODEL_SHORT}"

# Baseline config
CYBENCH_DIR="${CYBENCH_DIR:-/home/abrown/cybench-benchmark}"
BOXPWNR_DIR="${BOXPWNR_DIR:-/home/abrown/BoxPwnr}"
BASELINE_CTX="${BASELINE_CTX:-131072}"
BASELINE_MAX_TIME="${BASELINE_MAX_TIME:-1800}"
BASELINE_MAX_ROUNDS="${BASELINE_MAX_ROUNDS:-15}"
OLLAMA_MODEL="${OLLAMA_MODEL:-hf.co/mradermacher/Nanbeige4.1-3B-GGUF:Q8_0}"

# Docker image
DOCKER_IMAGE="${DOCKER_IMAGE:-open-ctf-env:latest}"
CONTAINER_PREFIX="nanbeige"

# Logging
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }
ssh_dgx() { ssh -o ConnectTimeout=10 "$DGX_HOST" "$@"; }

check_dgx() {
    log "Checking DGX connectivity..."
    if ! ssh_dgx 'echo OK' >/dev/null 2>&1; then
        echo "ERROR: Cannot reach DGX at $DGX_HOST"
        echo "Try: ssh -o StrictHostKeyChecking=accept-new $DGX_HOST"
        exit 1
    fi
    # Ensure log directory exists (needed for individual stage runs)
    ssh_dgx "mkdir -p ${DGX_DIR}/${LOG_DIR}"
    log "DGX reachable"
}

# ---------------------------------------------------------------------------
# Stage 0: Sync code + data to DGX
# ---------------------------------------------------------------------------
cmd_sync() {
    log "=== SYNC: $LOCAL_DIR → $DGX_HOST:$DGX_DIR ==="
    rsync -avz --progress \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.git' \
        --exclude 'outputs/' \
        --exclude '*.bak' \
        --exclude 'references/' \
        "$LOCAL_DIR/" "$DGX_HOST:$DGX_DIR/"
    log "Sync complete"
}

# ---------------------------------------------------------------------------
# Stage 1: Baseline with BoxPwnr (40 CyBench challenges)
# ---------------------------------------------------------------------------
cmd_baseline() {
    log "=== BASELINE: $MODEL_ID via Ollama (${BASELINE_CTX} ctx) ==="

    # Step 1: Pull model via Ollama on DGX
    log "Pulling model via Ollama..."
    ssh_dgx "docker exec ollama ollama pull ${OLLAMA_MODEL} || true"

    # Step 2: Create Modelfile with extended context and import into Ollama
    log "Creating Modelfile with ${BASELINE_CTX} context..."
    ssh_dgx "cat > /tmp/Modelfile-${MODEL_SHORT} << MEOF
FROM ${OLLAMA_MODEL}
PARAMETER num_ctx ${BASELINE_CTX}
PARAMETER temperature 0.6
MEOF
docker cp /tmp/Modelfile-${MODEL_SHORT} ollama:/tmp/Modelfile-${MODEL_SHORT} && \
docker exec ollama ollama create ${MODEL_SHORT}-128k -f /tmp/Modelfile-${MODEL_SHORT}"

    # Step 3: Run CyBench baseline via BoxPwnr
    log "Running CyBench baseline (40 challenges)..."
    ssh_dgx "mkdir -p ${CYBENCH_DIR}/traces/${MODEL_SHORT}-baseline"

    ssh_dgx "cd $BOXPWNR_DIR && \
        export MODEL_PROVIDER=ollama && \
        export MODEL_ID=${MODEL_SHORT}-128k && \
        export OLLAMA_HOST=http://localhost:11434 && \
        python3 -m boxpwnr.cli run \
            --benchmark cybench \
            --max-time ${BASELINE_MAX_TIME} \
            --max-rounds ${BASELINE_MAX_ROUNDS} \
            --output-dir ${CYBENCH_DIR}/traces/${MODEL_SHORT}-baseline \
            2>&1 | tee ${CYBENCH_DIR}/traces/${MODEL_SHORT}-baseline/run_${TIMESTAMP}.log"

    log "Baseline complete. Results in ${CYBENCH_DIR}/traces/${MODEL_SHORT}-baseline/"
}

# ---------------------------------------------------------------------------
# Stage 2: SFT Training
# ---------------------------------------------------------------------------
cmd_sft() {
    log "=== SFT: $MODEL_ID ==="
    log "Config: $CONFIG"
    log "Data: $SFT_DATA"
    log "Output: $SFT_OUTPUT"

    ssh_dgx "cd $DGX_DIR && \
        docker run --rm --gpus all \
            --name ${CONTAINER_PREFIX}-sft \
            --shm-size=16g \
            -v ${DGX_DIR}/data:/workspace/data \
            -v ${DGX_DIR}/outputs:/workspace/outputs \
            -v ${DGX_DIR}/src:/workspace/open-ctf-env/src \
            -e PYTHONPATH=/workspace/open-ctf-env/src \
            ${DOCKER_IMAGE} \
            python3 -m open_ctf.cli.train \
                --config /workspace/open-ctf-env/${CONFIG} \
                sft \
                --model '${MODEL_ID}' \
                --data /workspace/${SFT_DATA} \
                --output /workspace/${SFT_OUTPUT} \
        2>&1 | tee ${DGX_DIR}/${LOG_DIR}/sft_${MODEL_SHORT}_${TIMESTAMP}.log"

    log "SFT complete. Adapter at ${SFT_OUTPUT}/final/"
}

# ---------------------------------------------------------------------------
# Stage 3: Merge LoRA
# ---------------------------------------------------------------------------
cmd_merge() {
    log "=== MERGE: ${SFT_OUTPUT}/final → ${SFT_MERGED} ==="

    ssh_dgx "cd $DGX_DIR && \
        docker run --rm --gpus all \
            --name ${CONTAINER_PREFIX}-merge \
            --shm-size=16g \
            -v ${DGX_DIR}/outputs:/workspace/outputs \
            -v ${DGX_DIR}/src:/workspace/open-ctf-env/src \
            -e PYTHONPATH=/workspace/open-ctf-env/src \
            ${DOCKER_IMAGE} \
            python3 -m open_ctf.cli.train \
                --config /workspace/open-ctf-env/${CONFIG} \
                merge \
                --adapter /workspace/${SFT_OUTPUT}/final \
                --base-model '${MODEL_ID}' \
                --output /workspace/${SFT_MERGED} \
        2>&1 | tee ${DGX_DIR}/${LOG_DIR}/merge_${MODEL_SHORT}_${TIMESTAMP}.log"

    log "Merge complete. Model at ${SFT_MERGED}/"
}

# ---------------------------------------------------------------------------
# Stage 4: GRPO Training
# ---------------------------------------------------------------------------
cmd_grpo() {
    log "=== GRPO: ${SFT_MERGED} ==="
    log "Config: $CONFIG"
    log "Data: $GRPO_DATA"
    log "Output: $GRPO_OUTPUT"

    ssh_dgx "cd $DGX_DIR && \
        docker run --rm --gpus all \
            --name ${CONTAINER_PREFIX}-grpo \
            --shm-size=16g \
            -v ${DGX_DIR}/data:/workspace/data \
            -v ${DGX_DIR}/outputs:/workspace/outputs \
            -v ${DGX_DIR}/src:/workspace/open-ctf-env/src \
            -e PYTHONPATH=/workspace/open-ctf-env/src \
            ${DOCKER_IMAGE} \
            python3 -m open_ctf.cli.train \
                --config /workspace/open-ctf-env/${CONFIG} \
                grpo \
                --model /workspace/${SFT_MERGED} \
                --data /workspace/${GRPO_DATA} \
                --output /workspace/${GRPO_OUTPUT} \
        2>&1 | tee ${DGX_DIR}/${LOG_DIR}/grpo_${MODEL_SHORT}_${TIMESTAMP}.log"

    log "GRPO complete. Model at ${GRPO_OUTPUT}/final/"
}

# ---------------------------------------------------------------------------
# Stage 5: Post-Training Evaluation
# ---------------------------------------------------------------------------
cmd_eval() {
    log "=== EVAL: Post-training CyBench benchmark ==="

    # Export GRPO model to GGUF via llama.cpp
    log "Converting to GGUF..."
    ssh_dgx "cd $DGX_DIR && \
        python3 -c \"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    '${DGX_DIR}/${GRPO_OUTPUT}/final',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map='cpu',
)
tokenizer = AutoTokenizer.from_pretrained(
    '${DGX_DIR}/${GRPO_OUTPUT}/final',
    trust_remote_code=True,
)
# Save in HF format for Ollama import
model.save_pretrained('${DGX_DIR}/${GRPO_OUTPUT}/hf_export', safe_serialization=True)
tokenizer.save_pretrained('${DGX_DIR}/${GRPO_OUTPUT}/hf_export')
print('Exported to HF format for Ollama import')
\""

    # Import into Ollama
    log "Importing fine-tuned model into Ollama..."
    ssh_dgx "cat > /tmp/Modelfile-${MODEL_SHORT}-ft << MEOF
FROM ${DGX_DIR}/${GRPO_OUTPUT}/hf_export
PARAMETER num_ctx ${BASELINE_CTX}
PARAMETER temperature 0.6
MEOF
docker cp /tmp/Modelfile-${MODEL_SHORT}-ft ollama:/tmp/Modelfile-${MODEL_SHORT}-ft && \
docker exec ollama ollama create ${MODEL_SHORT}-ft-128k -f /tmp/Modelfile-${MODEL_SHORT}-ft"

    # Run CyBench with fine-tuned model
    log "Running CyBench eval (40 challenges) with fine-tuned model..."
    ssh_dgx "cd $BOXPWNR_DIR && \
        export MODEL_PROVIDER=ollama && \
        export MODEL_ID=${MODEL_SHORT}-ft-128k && \
        export OLLAMA_HOST=http://localhost:11434 && \
        python3 -m boxpwnr.cli run \
            --benchmark cybench \
            --max-time ${BASELINE_MAX_TIME} \
            --max-rounds ${BASELINE_MAX_ROUNDS} \
            --output-dir ${CYBENCH_DIR}/traces/${MODEL_SHORT}-ft \
            2>&1 | tee ${CYBENCH_DIR}/traces/${MODEL_SHORT}-ft/run_${TIMESTAMP}.log"

    log "Eval complete. Compare:"
    log "  Baseline: ${CYBENCH_DIR}/traces/${MODEL_SHORT}-baseline/"
    log "  Fine-tuned: ${CYBENCH_DIR}/traces/${MODEL_SHORT}-ft/"
}

# ---------------------------------------------------------------------------
# Stage: All (full pipeline)
# ---------------------------------------------------------------------------
cmd_all() {
    log "========================================="
    log "FULL PIPELINE: Nanbeige4.1-3B"
    log "========================================="

    cmd_sync
    cmd_baseline &   # Run baseline in background while SFT trains
    BASELINE_PID=$!

    cmd_sft
    cmd_merge

    # Wait for baseline to finish
    wait $BASELINE_PID || log "WARNING: Baseline may have failed (continuing anyway)"

    cmd_grpo
    cmd_eval

    log "========================================="
    log "PIPELINE COMPLETE"
    log "========================================="
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
case "${1:-help}" in
    sync)     check_dgx; cmd_sync ;;
    baseline) check_dgx; cmd_baseline ;;
    sft)      check_dgx; cmd_sft ;;
    merge)    check_dgx; cmd_merge ;;
    grpo)     check_dgx; cmd_grpo ;;
    eval)     check_dgx; cmd_eval ;;
    all)      check_dgx; cmd_all ;;
    *)
        echo "Usage: $0 {sync|baseline|sft|merge|grpo|eval|all}"
        echo ""
        echo "Stages:"
        echo "  sync      Rsync code + data to DGX"
        echo "  baseline  Run CyBench 40 with base model via Ollama"
        echo "  sft       SFT training (LoRA)"
        echo "  merge     Merge LoRA into base weights"
        echo "  grpo      GRPO training (vLLM colocate)"
        echo "  eval      Post-training CyBench evaluation"
        echo "  all       Run full pipeline"
        echo ""
        echo "Environment overrides:"
        echo "  MODEL_ID       HuggingFace model (default: Nanbeige/Nanbeige4.1-3B)"
        echo "  CONFIG         YAML config path (default: training_nanbeige.yaml)"
        echo "  DGX_HOST       SSH target (default: abrown@100.91.175.48)"
        echo "  BASELINE_CTX   Context window for baseline (default: 131072)"
        echo "  OLLAMA_MODEL   GGUF model for Ollama baseline (default: mradermacher Q8_0)"
        echo "  BASELINE_MAX_ROUNDS  Max tool-calling rounds per challenge (default: 15)"
        echo ""
        echo "Example with GLM-4.7-Flash:"
        echo "  MODEL_ID=unsloth/GLM-4.7-Flash CONFIG=src/open_ctf/configs/training_dgx.yaml \\"
        echo "    MODEL_SHORT=glm47 bash $0 sft"
        exit 1
        ;;
esac
