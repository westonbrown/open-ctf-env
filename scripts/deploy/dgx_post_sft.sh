#!/bin/bash
# =============================================================================
# Post-SFT Pipeline: Merge → CyBench Setup → GRPO → GEPA
# =============================================================================
# Run inside the DGX container after SFT completes.
#
# Usage:
#   bash scripts/deploy/dgx_post_sft.sh            # full post-SFT pipeline
#   bash scripts/deploy/dgx_post_sft.sh merge       # merge only
#   bash scripts/deploy/dgx_post_sft.sh grpo        # GRPO only (assumes merged)
#   bash scripts/deploy/dgx_post_sft.sh gepa        # GEPA only
#
# Environment:
#   PROJECT_ROOT    Default: /workspace/open-ctf-env
#   SFT_OUTPUT      SFT adapter directory (must contain adapter_config.json)
#   MERGED_OUTPUT   Merged model output directory
#   GRPO_OUTPUT     GRPO output directory
#   GEPA_OUTPUT     GEPA output directory
#   VLLM_PORT       vLLM server port (default: 8001)
#   VLLM_GPU_UTIL   vLLM GPU utilization (default: 0.15)
# =============================================================================

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/workspace/open-ctf-env}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/outputs/pipeline_logs"
mkdir -p "${LOG_DIR}"

MODEL="Qwen/Qwen3-8B"
CONFIG="${PROJECT_ROOT}/src/open_ctf/configs/training_qwen3_8b.yaml"
SFT_OUTPUT="${SFT_OUTPUT:-${PROJECT_ROOT}/outputs/sft_qwen3}"
MERGED_OUTPUT="${MERGED_OUTPUT:-${PROJECT_ROOT}/outputs/sft_qwen3_merged}"
GRPO_OUTPUT="${GRPO_OUTPUT:-${PROJECT_ROOT}/outputs/grpo_qwen3_${TIMESTAMP}}"
GRPO_DATA="${PROJECT_ROOT}/data/grpo_cybench40.jsonl"
GEPA_OUTPUT="${GEPA_OUTPUT:-${PROJECT_ROOT}/outputs/gepa_qwen3_${TIMESTAMP}}"
CHALLENGE_REGISTRY="${PROJECT_ROOT}/configs/challenges/cybench.yaml"

VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.20}"
VLLM_PID=""

log() { echo "[$(date '+%H:%M:%S')] $*"; }

cleanup_vllm() {
    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        log "Stopping vLLM server (PID ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
        VLLM_PID=""
    fi
}
trap cleanup_vllm EXIT

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
setup_env() {
    export RAY_memory_monitor_refresh_ms=0
    export VLLM_USE_V1=0
    export VLLM_ENABLE_V1_MULTIPROCESSING=0
    log "Environment variables set"
}

# ---------------------------------------------------------------------------
# Stage 1b: Merge LoRA adapter
# ---------------------------------------------------------------------------
run_merge() {
    log "========================================="
    log "MERGE LoRA ADAPTER"
    log "  SFT Output: ${SFT_OUTPUT}"
    log "  Base Model: ${MODEL}"
    log "  Merged To:  ${MERGED_OUTPUT}"
    log "========================================="

    # Find adapter_config.json
    local adapter_dir="${SFT_OUTPUT}"
    if [[ ! -f "${adapter_dir}/adapter_config.json" ]]; then
        adapter_dir=$(find "${SFT_OUTPUT}" -name "adapter_config.json" -exec dirname {} \; 2>/dev/null | head -1)
        if [[ -z "${adapter_dir}" ]]; then
            log "ERROR: No adapter_config.json found in ${SFT_OUTPUT}"
            exit 1
        fi
        log "Found adapter at: ${adapter_dir}"
    fi

    cd "${PROJECT_ROOT}"
    open-ctf-train --config "${CONFIG}" merge \
        --adapter "${adapter_dir}" \
        --base-model "${MODEL}" \
        --output "${MERGED_OUTPUT}" \
        2>&1 | tee "${LOG_DIR}/merge_${TIMESTAMP}.log"

    if [[ -f "${MERGED_OUTPUT}/config.json" ]]; then
        local size
        size=$(du -sh "${MERGED_OUTPUT}" | awk '{print $1}')
        log "Merge complete: ${MERGED_OUTPUT} (${size})"
    else
        log "ERROR: Merged model not found"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Stage 2: GRPO with external vLLM
# ---------------------------------------------------------------------------
start_vllm() {
    log "Starting vLLM server on port ${VLLM_PORT}..."

    python3 -m open_ctf.training.skyrl_vllm_server \
        --model "${MERGED_OUTPUT}" \
        --host 0.0.0.0 --port "${VLLM_PORT}" \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization "${VLLM_GPU_UTIL}" \
        --max-num-seqs 2 \
        --enforce-eager \
        --trust-remote-code \
        2>&1 | tee "${LOG_DIR}/vllm_${TIMESTAMP}.log" &

    VLLM_PID=$!
    log "vLLM PID: ${VLLM_PID}"

    local max_wait=300
    local waited=0
    while [[ ${waited} -lt ${max_wait} ]]; do
        if curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
            log "vLLM ready after ${waited}s"
            return 0
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            log "ERROR: vLLM died during startup"
            VLLM_PID=""
            exit 1
        fi
        sleep 5
        waited=$((waited + 5))
    done

    log "ERROR: vLLM failed to start within ${max_wait}s"
    cleanup_vllm
    exit 1
}

run_grpo() {
    log "========================================="
    log "GRPO TRAINING (SkyRL)"
    log "  Model:    ${MERGED_OUTPUT}"
    log "  Data:     ${GRPO_DATA}"
    log "  Output:   ${GRPO_OUTPUT}"
    log "  Registry: ${CHALLENGE_REGISTRY}"
    log "========================================="

    if [[ ! -d "${MERGED_OUTPUT}" ]]; then
        log "ERROR: Merged model not found at ${MERGED_OUTPUT}"
        exit 1
    fi

    # Apply SkyRL patches
    if [[ -f "${PROJECT_ROOT}/docker/patches/apply_all_patches.sh" ]]; then
        log "Applying SkyRL patches..."
        bash "${PROJECT_ROOT}/docker/patches/apply_all_patches.sh" 2>&1 | tail -5
    fi

    # Start vLLM
    start_vllm

    # Run GRPO
    cd "${PROJECT_ROOT}"
    open-ctf-train --config "${CONFIG}" grpo \
        --model "${MERGED_OUTPUT}" \
        --data "${GRPO_DATA}" \
        --output "${GRPO_OUTPUT}" \
        --challenge-registry "${CHALLENGE_REGISTRY}" \
        2>&1 | tee "${LOG_DIR}/grpo_${TIMESTAMP}.log"

    cleanup_vllm
    log "GRPO complete: ${GRPO_OUTPUT}"
}

# ---------------------------------------------------------------------------
# Stage 3: GEPA
# ---------------------------------------------------------------------------
run_gepa() {
    log "========================================="
    log "GEPA PROMPT OPTIMIZATION"
    log "  Output: ${GEPA_OUTPUT}"
    log "========================================="

    local has_api_key=false
    if [[ -n "${OPENAI_API_KEY:-}" ]] || [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
        has_api_key=true
    fi

    if [[ "${has_api_key}" == "false" ]]; then
        log "WARNING: No API key set. Skipping GEPA."
        log "  Set OPENAI_API_KEY or ANTHROPIC_API_KEY to enable."
        return 0
    fi

    cd "${PROJECT_ROOT}"
    open-ctf-train --config "${CONFIG}" gepa \
        --model "openai/ctf-agent" \
        --data "${GRPO_DATA}" \
        --output "${GEPA_OUTPUT}" \
        --budget medium \
        2>&1 | tee "${LOG_DIR}/gepa_${TIMESTAMP}.log"

    log "GEPA complete: ${GEPA_OUTPUT}"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print_summary() {
    log "========================================="
    log "PIPELINE COMPLETE"
    log "========================================="
    echo ""
    echo "  Outputs:"
    [[ -d "${SFT_OUTPUT}" ]]    && echo "    SFT adapter:   ${SFT_OUTPUT}"
    [[ -d "${MERGED_OUTPUT}" ]] && echo "    Merged model:  ${MERGED_OUTPUT}"
    [[ -d "${GRPO_OUTPUT}" ]]   && echo "    GRPO model:    ${GRPO_OUTPUT}"
    [[ -d "${GEPA_OUTPUT}" ]]   && echo "    GEPA prompts:  ${GEPA_OUTPUT}"
    echo ""
    echo "  Logs: ${LOG_DIR}/"
    echo ""
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
setup_env

case "${1:-all}" in
    merge) run_merge ;;
    grpo)  run_grpo ;;
    gepa)  run_gepa ;;
    all)
        run_merge
        run_grpo
        run_gepa
        print_summary
        ;;
    *)
        echo "Usage: $0 {merge|grpo|gepa|all}"
        exit 1
        ;;
esac
