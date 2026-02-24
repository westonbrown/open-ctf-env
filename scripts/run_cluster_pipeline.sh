#!/usr/bin/env bash
# =============================================================================
# Nanbeige4.1-3B Training Pipeline on DGX Spark
# =============================================================================
# Full pipeline: baseline → SFT → merge → GRPO → eval
#
# Usage:
#   # Run full pipeline
#   bash scripts/run_cluster_pipeline.sh all
#
# Selective execution:
#   bash scripts/run_cluster_pipeline.sh baseline   # BoxPwnr CyBench 40 challenges
#   bash scripts/run_cluster_pipeline.sh sft        # SFT training
#   bash scripts/run_cluster_pipeline.sh merge      # Merge LoRA into base
#   bash scripts/run_cluster_pipeline.sh grpo       # GRPO training
#   bash scripts/run_cluster_pipeline.sh eval       # Post-training CyBench eval
#
# Prerequisites:
#   - Compute Node accessible via SSH (user@node_ip)
#   - Docker with GPU support
#   - open-ctf-env synced to /home/user/open-ctf-env/
#
# This script is generalizable: change MODEL_ID and CONFIG to train
# other models (e.g. GLM-4.7-Flash with training_120gb_moe.yaml).
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration (edit these for different models)
# ---------------------------------------------------------------------------
MODEL_ID="${MODEL_ID:-Nanbeige/Nanbeige4.1-3B}"
MODEL_SHORT="${MODEL_SHORT:-nanbeige3b}"
CONFIG="${CONFIG:-src/open_ctf/configs/training_120gb_dense.yaml}"
CLUSTER_HOST="${CLUSTER_HOST:-user@node_ip}"
CLUSTER_DIR="${CLUSTER_DIR:-/home/user/open-ctf-env}"
LOCAL_DIR="${LOCAL_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

# Training paths (on DGX)
SFT_DATA="${SFT_DATA:-data/sft.jsonl}"
GRPO_DATA="${GRPO_DATA:-data/grpo.jsonl}"
SFT_OUTPUT="outputs/sft-${MODEL_SHORT}"
SFT_MERGED="outputs/sft-${MODEL_SHORT}-merged"
GRPO_OUTPUT="outputs/grpo-${MODEL_SHORT}"

# Baseline config
CYBENCH_DIR="${CYBENCH_DIR:-/home/user/cybench-benchmark}"
BOXPWNR_DIR="${BOXPWNR_DIR:-/home/user/BoxPwnr}"
BASELINE_CTX="${BASELINE_CTX:-131072}"
BASELINE_MAX_TIME="${BASELINE_MAX_TIME:-1800}"
BASELINE_MAX_TURNS="${BASELINE_MAX_TURNS:-15}"
OLLAMA_MODEL="${OLLAMA_MODEL:-hf.co/mradermacher/Nanbeige4.1-3B-GGUF:Q8_0}"

# Docker compose is used for SFT/merge/GRPO stages (see docker-compose.yaml).
# DOCKER_IMAGE is no longer needed — each stage uses its own Dockerfile
# (docker/Dockerfile.sft for SFT+merge, docker/Dockerfile.grpo for GRPO).

# Logging
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }
ssh_cluster() { ssh -o ConnectTimeout=10 "$CLUSTER_HOST" "$@"; }

check_cluster() {
    log "Checking DGX connectivity..."
    if ! ssh_cluster 'echo OK' >/dev/null 2>&1; then
        echo "ERROR: Cannot reach DGX at $CLUSTER_HOST"
        echo "Try: ssh -o StrictHostKeyChecking=accept-new $CLUSTER_HOST"
        exit 1
    fi
    # Ensure log directory exists (needed for individual stage runs)
    ssh_cluster "mkdir -p ${CLUSTER_DIR}/${LOG_DIR}"
    log "DGX reachable"
}

# ---------------------------------------------------------------------------
# Stage 0: Sync code + data to DGX
# ---------------------------------------------------------------------------
cmd_sync() {
    log "=== SYNC: $LOCAL_DIR → $CLUSTER_HOST:$CLUSTER_DIR ==="
    rsync -avz --progress \
        --exclude '__pycache__' \
        --exclude '*.pyc' \
        --exclude '.git' \
        --exclude 'outputs/' \
        --exclude '*.bak' \
        --exclude 'references/' \
        "$LOCAL_DIR/" "$CLUSTER_HOST:$CLUSTER_DIR/"
    log "Sync complete"
}

# ---------------------------------------------------------------------------
# Stage 1: Baseline with BoxPwnr (40 CyBench challenges)
# ---------------------------------------------------------------------------
cmd_baseline() {
    log "=== BASELINE: $MODEL_ID via Ollama (${BASELINE_CTX} ctx) ==="

    # Step 1: Pull model via Ollama on DGX
    log "Pulling model via Ollama..."
    ssh_cluster "docker exec ollama ollama pull ${OLLAMA_MODEL} || true"

    # Step 2: Create Modelfile with extended context and import into Ollama
    log "Creating Modelfile with ${BASELINE_CTX} context..."
    ssh_cluster "cat > /tmp/Modelfile-${MODEL_SHORT} << MEOF
FROM ${OLLAMA_MODEL}
PARAMETER num_ctx ${BASELINE_CTX}
PARAMETER temperature 0.6
MEOF
docker cp /tmp/Modelfile-${MODEL_SHORT} ollama:/tmp/Modelfile-${MODEL_SHORT} && \
docker exec ollama ollama create ${MODEL_SHORT}-128k -f /tmp/Modelfile-${MODEL_SHORT}"

    # Step 3: Run CyBench baseline via BoxPwnr
    # BoxPwnr CLI: uv run boxpwnr --platform cybench --target <name> --max-turns N
    # No batch mode — loop over challenges. Use --list to discover targets.
    log "Running CyBench baseline (all challenges)..."
    ssh_cluster "mkdir -p ${CYBENCH_DIR}/traces/${MODEL_SHORT}-baseline"

    ssh_cluster "cd $BOXPWNR_DIR && \
        export OLLAMA_HOST=http://localhost:11434 && \
        for target in \$(uv run boxpwnr --platform cybench --list 2>/dev/null | grep '^\[' | head -40); do
            log \"Running: \$target\"
            uv run boxpwnr \
                --platform cybench \
                --target \"\$target\" \
                --model ollama/${MODEL_SHORT}-128k \
                --max-turns ${BASELINE_MAX_TURNS} \
                --max-cost 0 \
                2>&1 | tee -a ${CYBENCH_DIR}/traces/${MODEL_SHORT}-baseline/run_${TIMESTAMP}.log
        done"

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

    ssh_cluster "cd $CLUSTER_DIR && \
        MODEL='${MODEL_ID}' \
        SFT_DATA='${SFT_DATA}' \
        docker compose run --rm sft \
        2>&1 | tee ${CLUSTER_DIR}/${LOG_DIR}/sft_${MODEL_SHORT}_${TIMESTAMP}.log"

    log "SFT complete. Adapter at ${SFT_OUTPUT}/final/"
}

# ---------------------------------------------------------------------------
# Stage 3: Merge LoRA
# ---------------------------------------------------------------------------
cmd_merge() {
    log "=== MERGE: ${SFT_OUTPUT}/final → ${SFT_MERGED} ==="

    ssh_cluster "cd $CLUSTER_DIR && \
        ADAPTER=/workspace/${SFT_OUTPUT}/final \
        MODEL='${MODEL_ID}' \
        docker compose run --rm merge \
        2>&1 | tee ${CLUSTER_DIR}/${LOG_DIR}/merge_${MODEL_SHORT}_${TIMESTAMP}.log"

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

    ssh_cluster "cd $CLUSTER_DIR && \
        GRPO_MODEL=/workspace/${SFT_MERGED} \
        GRPO_DATA='${GRPO_DATA}' \
        docker compose run --rm grpo \
        2>&1 | tee ${CLUSTER_DIR}/${LOG_DIR}/grpo_${MODEL_SHORT}_${TIMESTAMP}.log"

    log "GRPO complete. Model at ${GRPO_OUTPUT}/final/"
}

# ---------------------------------------------------------------------------
# Stage 5: Post-Training Evaluation
# ---------------------------------------------------------------------------
cmd_eval() {
    log "=== EVAL: Post-training CyBench benchmark ==="

    # Export GRPO model to GGUF via llama.cpp
    log "Converting to GGUF..."
    ssh_cluster "cd $CLUSTER_DIR && \
        python3 -c \"
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model = AutoModelForCausalLM.from_pretrained(
    '${CLUSTER_DIR}/${GRPO_OUTPUT}/final',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map='cpu',
)
tokenizer = AutoTokenizer.from_pretrained(
    '${CLUSTER_DIR}/${GRPO_OUTPUT}/final',
    trust_remote_code=True,
)
# Save in HF format for Ollama import
model.save_pretrained('${CLUSTER_DIR}/${GRPO_OUTPUT}/hf_export', safe_serialization=True)
tokenizer.save_pretrained('${CLUSTER_DIR}/${GRPO_OUTPUT}/hf_export')
print('Exported to HF format for Ollama import')
\""

    # Import into Ollama
    log "Importing fine-tuned model into Ollama..."
    ssh_cluster "cat > /tmp/Modelfile-${MODEL_SHORT}-ft << MEOF
FROM ${CLUSTER_DIR}/${GRPO_OUTPUT}/hf_export
PARAMETER num_ctx ${BASELINE_CTX}
PARAMETER temperature 0.6
MEOF
docker cp /tmp/Modelfile-${MODEL_SHORT}-ft ollama:/tmp/Modelfile-${MODEL_SHORT}-ft && \
docker exec ollama ollama create ${MODEL_SHORT}-ft-128k -f /tmp/Modelfile-${MODEL_SHORT}-ft"

    # Run CyBench with fine-tuned model
    log "Running CyBench eval (all challenges) with fine-tuned model..."
    ssh_cluster "cd $BOXPWNR_DIR && \
        export OLLAMA_HOST=http://localhost:11434 && \
        for target in \$(uv run boxpwnr --platform cybench --list 2>/dev/null | grep '^\[' | head -40); do
            log \"Running: \$target\"
            uv run boxpwnr \
                --platform cybench \
                --target \"\$target\" \
                --model ollama/${MODEL_SHORT}-ft-128k \
                --max-turns ${BASELINE_MAX_TURNS} \
                --max-cost 0 \
                2>&1 | tee -a ${CYBENCH_DIR}/traces/${MODEL_SHORT}-ft/run_${TIMESTAMP}.log
        done"

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
    sync)     check_cluster; cmd_sync ;;
    baseline) check_cluster; cmd_baseline ;;
    sft)      check_cluster; cmd_sft ;;
    merge)    check_cluster; cmd_merge ;;
    grpo)     check_cluster; cmd_grpo ;;
    eval)     check_cluster; cmd_eval ;;
    all)      check_cluster; cmd_all ;;
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
        echo "  CONFIG         YAML config path (default: training_120gb_dense.yaml)"
        echo "  CLUSTER_HOST       SSH target (default: user@node_ip)"
        echo "  BASELINE_CTX   Context window for baseline (default: 131072)"
        echo "  OLLAMA_MODEL   GGUF model for Ollama baseline (default: mradermacher Q8_0)"
        echo "  BASELINE_MAX_TURNS   Max tool-calling turns per challenge (default: 15)"
        echo ""
        echo "Example with GLM-4.7-Flash:"
        echo "  MODEL_ID=unsloth/GLM-4.7-Flash CONFIG=src/open_ctf/configs/training_120gb_moe.yaml \\"
        echo "    MODEL_SHORT=glm47 bash $0 sft"
        exit 1
        ;;
esac
