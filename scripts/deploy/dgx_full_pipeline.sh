#!/bin/bash
# =============================================================================
# Open CTF Full Training Pipeline -- DGX Spark (GB10)
# =============================================================================
# Runs the full 3-stage training pipeline inside the DGX container:
#   Stage 1:  SFT via LlamaFactory (LoRA)
#   Stage 1b: Merge LoRA adapter into base model
#   Stage 2:  GRPO via SkyRL (external vLLM)
#   Stage 3:  GEPA prompt optimization (optional, requires API key)
#
# Usage:
#   bash scripts/deploy/dgx_full_pipeline.sh            # full pipeline
#   bash scripts/deploy/dgx_full_pipeline.sh sft         # SFT only
#   bash scripts/deploy/dgx_full_pipeline.sh merge       # merge only
#   bash scripts/deploy/dgx_full_pipeline.sh grpo        # GRPO only
#   bash scripts/deploy/dgx_full_pipeline.sh gepa        # GEPA only
#
# Environment overrides:
#   MODEL        HuggingFace model ID (default: Qwen/Qwen3-8B)
#   CONFIG       Training YAML config path
#   SFT_DATA     SFT training data (default: data/sft.jsonl)
#   GRPO_DATA    GRPO training data (default: data/grpo_cybench40.jsonl)
#   SFT_OUTPUT   SFT adapter output dir
#   MERGED_OUTPUT Merged model output dir
#   GRPO_OUTPUT  GRPO output dir
#   GEPA_OUTPUT  GEPA output dir
#   VLLM_PORT    vLLM server port (default: 8001)
#   VLLM_GPU_UTIL vLLM GPU memory utilization (default: 0.15)
#   SKIP_PATCHES Set to 1 to skip SkyRL patch application
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT="${PROJECT_ROOT:-/workspace/open-ctf-env}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${PROJECT_ROOT}/outputs/pipeline_logs"
mkdir -p "${LOG_DIR}"

# Model & data
MODEL="${MODEL:-Qwen/Qwen3-8B}"
CONFIG="${CONFIG:-${PROJECT_ROOT}/src/open_ctf/configs/training_qwen3_8b.yaml}"
SFT_DATA="${SFT_DATA:-${PROJECT_ROOT}/data/sft.jsonl}"
GRPO_DATA="${GRPO_DATA:-${PROJECT_ROOT}/data/grpo_cybench40.jsonl}"

# Output paths
SFT_OUTPUT="${SFT_OUTPUT:-${PROJECT_ROOT}/outputs/sft_qwen3_${TIMESTAMP}}"
MERGED_OUTPUT="${MERGED_OUTPUT:-${PROJECT_ROOT}/outputs/sft_qwen3_merged_${TIMESTAMP}}"
GRPO_OUTPUT="${GRPO_OUTPUT:-${PROJECT_ROOT}/outputs/grpo_qwen3_${TIMESTAMP}}"
GEPA_OUTPUT="${GEPA_OUTPUT:-${PROJECT_ROOT}/outputs/gepa_qwen3_${TIMESTAMP}}"

# vLLM server config
VLLM_PORT="${VLLM_PORT:-8001}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.15}"
VLLM_PID=""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log() { echo "[$(date '+%H:%M:%S')] $*"; }

cleanup_vllm() {
    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        log "Stopping vLLM server (PID ${VLLM_PID})..."
        kill "${VLLM_PID}" 2>/dev/null || true
        wait "${VLLM_PID}" 2>/dev/null || true
        VLLM_PID=""
        log "vLLM server stopped"
    fi
}

trap cleanup_vllm EXIT

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
preflight() {
    log "========================================="
    log "PRE-FLIGHT CHECKS"
    log "========================================="

    # GPU check
    if command -v nvidia-smi &>/dev/null; then
        log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
    else
        log "WARNING: nvidia-smi not found -- GPU may not be available"
    fi

    # Python check
    log "Python: $(python3 --version 2>&1)"

    # Check project root
    if [[ ! -d "${PROJECT_ROOT}/src/open_ctf" ]]; then
        log "ERROR: Project not found at ${PROJECT_ROOT}"
        log "  Expected: ${PROJECT_ROOT}/src/open_ctf/"
        exit 1
    fi

    # Check data files
    for f in "${SFT_DATA}" "${GRPO_DATA}"; do
        if [[ -f "${f}" ]]; then
            local lines
            lines=$(wc -l < "${f}" | tr -d ' ')
            log "Data: ${f} (${lines} lines)"
        else
            log "WARNING: Data file not found: ${f}"
        fi
    done

    # Check config
    if [[ -f "${CONFIG}" ]]; then
        log "Config: ${CONFIG}"
    else
        log "WARNING: Config not found at ${CONFIG}, will use defaults"
    fi

    # Install package if needed
    cd "${PROJECT_ROOT}"
    if ! python3 -c "import open_ctf" 2>/dev/null; then
        log "Installing open-ctf-env..."
        pip install -e ".[sft,grpo,dev]" 2>&1 | tail -3
    fi
    log "open-ctf-env: installed"

    # Apply SkyRL patches (unless skipped)
    if [[ "${SKIP_PATCHES:-0}" != "1" ]]; then
        if [[ -f "${PROJECT_ROOT}/docker/patches/apply_all_patches.sh" ]]; then
            log "Applying SkyRL patches..."
            bash "${PROJECT_ROOT}/docker/patches/apply_all_patches.sh" 2>&1 | tail -5
        else
            log "WARNING: SkyRL patches not found at ${PROJECT_ROOT}/docker/patches/"
        fi
    else
        log "Skipping SkyRL patches (SKIP_PATCHES=1)"
    fi

    # Set required env vars
    export RAY_memory_monitor_refresh_ms=0
    export VLLM_USE_V1=0
    export VLLM_ENABLE_V1_MULTIPROCESSING=0
    log "Environment: RAY_memory_monitor_refresh_ms=0, VLLM_USE_V1=0"

    log "Pre-flight checks complete"
    echo ""
}

# ---------------------------------------------------------------------------
# Stage 1: SFT (LlamaFactory)
# ---------------------------------------------------------------------------
run_sft() {
    log "========================================="
    log "STAGE 1: SFT (LlamaFactory)"
    log "  Model:  ${MODEL}"
    log "  Data:   ${SFT_DATA}"
    log "  Output: ${SFT_OUTPUT}"
    log "  Config: ${CONFIG}"
    log "========================================="

    if [[ ! -f "${SFT_DATA}" ]]; then
        log "ERROR: SFT data not found: ${SFT_DATA}"
        exit 1
    fi

    open-ctf-train --config "${CONFIG}" sft \
        --model "${MODEL}" \
        --data "${SFT_DATA}" \
        --output "${SFT_OUTPUT}" \
        2>&1 | tee "${LOG_DIR}/sft_${TIMESTAMP}.log"

    # Verify adapter was saved
    if find "${SFT_OUTPUT}" -name "adapter_config.json" 2>/dev/null | head -1 | grep -q .; then
        log "SFT complete. Adapter saved to ${SFT_OUTPUT}"
    else
        log "ERROR: No LoRA adapter found in ${SFT_OUTPUT}"
        exit 1
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Stage 1b: Merge LoRA adapter
# ---------------------------------------------------------------------------
run_merge() {
    log "========================================="
    log "STAGE 1b: MERGE LoRA"
    log "  Adapter:    ${SFT_OUTPUT}"
    log "  Base model: ${MODEL}"
    log "  Output:     ${MERGED_OUTPUT}"
    log "========================================="

    # Find the adapter directory (may be in a subdirectory)
    local adapter_dir="${SFT_OUTPUT}"
    if [[ ! -f "${adapter_dir}/adapter_config.json" ]]; then
        adapter_dir=$(find "${SFT_OUTPUT}" -name "adapter_config.json" -exec dirname {} \; 2>/dev/null | head -1)
        if [[ -z "${adapter_dir}" ]]; then
            log "ERROR: No adapter_config.json found in ${SFT_OUTPUT}"
            exit 1
        fi
    fi

    open-ctf-train --config "${CONFIG}" merge \
        --adapter "${adapter_dir}" \
        --base-model "${MODEL}" \
        --output "${MERGED_OUTPUT}" \
        2>&1 | tee "${LOG_DIR}/merge_${TIMESTAMP}.log"

    # Verify merged model
    if [[ -f "${MERGED_OUTPUT}/config.json" ]]; then
        local size
        size=$(du -sh "${MERGED_OUTPUT}" | awk '{print $1}')
        log "Merge complete. Model at ${MERGED_OUTPUT} (${size})"
    else
        log "ERROR: Merged model config.json not found in ${MERGED_OUTPUT}"
        exit 1
    fi
    echo ""
}

# ---------------------------------------------------------------------------
# Stage 2: GRPO (SkyRL + external vLLM)
# ---------------------------------------------------------------------------
start_vllm() {
    log "Starting vLLM server on port ${VLLM_PORT}..."

    export VLLM_USE_V1=0
    export VLLM_ENABLE_V1_MULTIPROCESSING=0

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
    log "vLLM server started (PID ${VLLM_PID})"

    # Wait for vLLM to be ready (health check loop)
    local max_wait=300
    local waited=0
    log "Waiting for vLLM to be ready (max ${max_wait}s)..."
    while [[ ${waited} -lt ${max_wait} ]]; do
        if curl -s "http://127.0.0.1:${VLLM_PORT}/health" >/dev/null 2>&1; then
            log "vLLM server ready after ${waited}s"
            return 0
        fi
        # Check if vLLM process died
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            log "ERROR: vLLM server process died during startup"
            log "Check log: ${LOG_DIR}/vllm_${TIMESTAMP}.log"
            VLLM_PID=""
            exit 1
        fi
        sleep 5
        waited=$((waited + 5))
    done

    log "ERROR: vLLM server failed to start within ${max_wait}s"
    cleanup_vllm
    exit 1
}

run_grpo() {
    log "========================================="
    log "STAGE 2: GRPO (SkyRL)"
    log "  Model:  ${MERGED_OUTPUT}"
    log "  Data:   ${GRPO_DATA}"
    log "  Output: ${GRPO_OUTPUT}"
    log "  Config: ${CONFIG}"
    log "========================================="

    if [[ ! -d "${MERGED_OUTPUT}" ]] || [[ ! -f "${MERGED_OUTPUT}/config.json" ]]; then
        log "ERROR: Merged model not found at ${MERGED_OUTPUT}"
        log "  Run 'merge' stage first"
        exit 1
    fi

    if [[ ! -f "${GRPO_DATA}" ]]; then
        log "ERROR: GRPO data not found: ${GRPO_DATA}"
        exit 1
    fi

    # Start vLLM server
    start_vllm

    # Run GRPO training
    open-ctf-train --config "${CONFIG}" grpo \
        --model "${MERGED_OUTPUT}" \
        --data "${GRPO_DATA}" \
        --output "${GRPO_OUTPUT}" \
        2>&1 | tee "${LOG_DIR}/grpo_${TIMESTAMP}.log"

    # Stop vLLM
    cleanup_vllm

    log "GRPO complete. Output at ${GRPO_OUTPUT}"
    echo ""
}

# ---------------------------------------------------------------------------
# Stage 3: GEPA (DSPy prompt optimization)
# ---------------------------------------------------------------------------
run_gepa() {
    log "========================================="
    log "STAGE 3: GEPA (DSPy)"
    log "  Output: ${GEPA_OUTPUT}"
    log "========================================="

    # GEPA needs an LLM backend (OpenAI, Anthropic, etc.)
    local has_api_key=false
    if [[ -n "${OPENAI_API_KEY:-}" ]] || [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
        has_api_key=true
    fi

    if [[ "${has_api_key}" == "false" ]]; then
        log "WARNING: No OPENAI_API_KEY or ANTHROPIC_API_KEY set"
        log "  GEPA requires an external LLM API for prompt optimization."
        log "  Skipping GEPA stage."
        log "  To enable: export OPENAI_API_KEY=sk-... or ANTHROPIC_API_KEY=sk-ant-..."
        echo ""
        return 0
    fi

    # Use the GRPO model if available, otherwise the merged SFT model
    local gepa_model="${GRPO_OUTPUT:-${MERGED_OUTPUT}}"
    if [[ ! -d "${gepa_model}" ]]; then
        gepa_model="${MERGED_OUTPUT}"
    fi

    open-ctf-train --config "${CONFIG}" gepa \
        --model "openai/ctf-agent" \
        --data "${GRPO_DATA}" \
        --output "${GEPA_OUTPUT}" \
        --budget medium \
        2>&1 | tee "${LOG_DIR}/gepa_${TIMESTAMP}.log"

    log "GEPA complete. Output at ${GEPA_OUTPUT}"
    echo ""
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print_summary() {
    log "========================================="
    log "PIPELINE COMPLETE"
    log "========================================="
    echo ""
    echo "  Model:         ${MODEL}"
    echo "  Config:        ${CONFIG}"
    echo ""
    echo "  Outputs:"
    [[ -d "${SFT_OUTPUT}" ]]    && echo "    SFT adapter:   ${SFT_OUTPUT}"
    [[ -d "${MERGED_OUTPUT}" ]] && echo "    Merged model:  ${MERGED_OUTPUT}"
    [[ -d "${GRPO_OUTPUT}" ]]   && echo "    GRPO model:    ${GRPO_OUTPUT}"
    [[ -d "${GEPA_OUTPUT}" ]]   && echo "    GEPA prompts:  ${GEPA_OUTPUT}"
    echo ""
    echo "  Logs:          ${LOG_DIR}/"
    echo ""
    echo "  Next steps:"
    echo "    1. Evaluate: open-ctf-eval --model ${GRPO_OUTPUT:-${MERGED_OUTPUT}}"
    echo "    2. Export:   open-ctf-export --model ${GRPO_OUTPUT:-${MERGED_OUTPUT}} --quant Q4_K_M"
    echo ""
}

# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------
run_all() {
    preflight
    run_sft
    run_merge
    run_grpo
    run_gepa
    print_summary
}

case "${1:-all}" in
    preflight) preflight ;;
    sft)       preflight; run_sft ;;
    merge)     preflight; run_merge ;;
    grpo)      preflight; run_grpo ;;
    gepa)      preflight; run_gepa ;;
    all)       run_all ;;
    *)
        echo "Usage: $0 {preflight|sft|merge|grpo|gepa|all}"
        echo ""
        echo "Stages:"
        echo "  preflight  Run pre-flight checks only"
        echo "  sft        Stage 1: SFT via LlamaFactory"
        echo "  merge      Stage 1b: Merge LoRA adapter"
        echo "  grpo       Stage 2: GRPO via SkyRL (starts vLLM)"
        echo "  gepa       Stage 3: GEPA prompt optimization"
        echo "  all        Run full pipeline (default)"
        echo ""
        echo "Environment overrides:"
        echo "  MODEL          HuggingFace model (default: Qwen/Qwen3-8B)"
        echo "  CONFIG         YAML config path"
        echo "  SFT_DATA       SFT training data"
        echo "  GRPO_DATA      GRPO training data"
        echo "  VLLM_PORT      vLLM server port (default: 8001)"
        echo "  VLLM_GPU_UTIL  vLLM GPU utilization (default: 0.15)"
        echo "  SKIP_PATCHES   Set to 1 to skip SkyRL patch application"
        exit 1
        ;;
esac
