#!/bin/bash
# =============================================================================
# Run full pipeline inside DGX container (remote wrapper)
# =============================================================================
# SSHes to the DGX Spark, copies the pipeline script into the container,
# and executes it. Captures all output to a local log file.
#
# Usage:
#   bash scripts/deploy/dgx_run_in_container.sh              # full pipeline
#   bash scripts/deploy/dgx_run_in_container.sh sft           # SFT only
#   bash scripts/deploy/dgx_run_in_container.sh grpo          # GRPO only
#
# Environment overrides:
#   DGX_HOST     SSH target (default: abrown@100.91.175.48)
#   CONTAINER    Docker container name (default: open-ctf-grpo)
#   DGX_PATH     Host path for open-ctf-env (default: /home/abrown/open-ctf-env)
#   MODEL        HuggingFace model ID (passed through to pipeline)
#   CONFIG       Config path inside container (passed through to pipeline)
# =============================================================================

set -euo pipefail

DGX_HOST="${DGX_HOST:-abrown@100.91.175.48}"
CONTAINER="${CONTAINER:-open-ctf-grpo}"
DGX_PATH="${DGX_PATH:-/home/abrown/open-ctf-env}"
STAGE="${1:-all}"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOCAL_LOG="dgx_pipeline_${TIMESTAMP}.log"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# 1. Verify connectivity
# ---------------------------------------------------------------------------
log "Connecting to DGX at ${DGX_HOST}..."
if ! ssh -o ConnectTimeout=10 "${DGX_HOST}" 'echo OK' >/dev/null 2>&1; then
    echo "ERROR: Cannot reach DGX at ${DGX_HOST}"
    echo "Try: ssh -o StrictHostKeyChecking=accept-new ${DGX_HOST}"
    exit 1
fi
log "DGX reachable"

# ---------------------------------------------------------------------------
# 2. Verify container is running
# ---------------------------------------------------------------------------
log "Checking container '${CONTAINER}'..."
CONTAINER_STATUS=$(ssh "${DGX_HOST}" "docker inspect -f '{{.State.Status}}' ${CONTAINER} 2>/dev/null" || echo "not_found")

if [[ "${CONTAINER_STATUS}" == "not_found" ]]; then
    log "ERROR: Container '${CONTAINER}' not found on DGX"
    log "  Available containers:"
    ssh "${DGX_HOST}" "docker ps -a --format '  {{.Names}} ({{.Status}})'" || true
    exit 1
elif [[ "${CONTAINER_STATUS}" != "running" ]]; then
    log "Container '${CONTAINER}' exists but is ${CONTAINER_STATUS}. Starting..."
    ssh "${DGX_HOST}" "docker start ${CONTAINER}"
    sleep 3
fi
log "Container '${CONTAINER}' is running"

# ---------------------------------------------------------------------------
# 3. Copy pipeline script into container
# ---------------------------------------------------------------------------
log "Copying pipeline script into container..."
ssh "${DGX_HOST}" "\
    docker cp ${DGX_PATH}/scripts/deploy/dgx_full_pipeline.sh \
        ${CONTAINER}:/workspace/open-ctf-env/scripts/deploy/dgx_full_pipeline.sh && \
    docker exec ${CONTAINER} chmod +x /workspace/open-ctf-env/scripts/deploy/dgx_full_pipeline.sh"

log "Pipeline script copied"

# ---------------------------------------------------------------------------
# 4. Build env vars to pass through
# ---------------------------------------------------------------------------
ENV_ARGS=""
[[ -n "${MODEL:-}" ]]  && ENV_ARGS="${ENV_ARGS} -e MODEL=${MODEL}"
[[ -n "${CONFIG:-}" ]] && ENV_ARGS="${ENV_ARGS} -e CONFIG=${CONFIG}"
[[ -n "${SFT_DATA:-}" ]] && ENV_ARGS="${ENV_ARGS} -e SFT_DATA=${SFT_DATA}"
[[ -n "${GRPO_DATA:-}" ]] && ENV_ARGS="${ENV_ARGS} -e GRPO_DATA=${GRPO_DATA}"
[[ -n "${VLLM_PORT:-}" ]] && ENV_ARGS="${ENV_ARGS} -e VLLM_PORT=${VLLM_PORT}"
[[ -n "${VLLM_GPU_UTIL:-}" ]] && ENV_ARGS="${ENV_ARGS} -e VLLM_GPU_UTIL=${VLLM_GPU_UTIL}"
[[ -n "${SKIP_PATCHES:-}" ]] && ENV_ARGS="${ENV_ARGS} -e SKIP_PATCHES=${SKIP_PATCHES}"
[[ -n "${HF_TOKEN:-}" ]] && ENV_ARGS="${ENV_ARGS} -e HF_TOKEN=${HF_TOKEN}"
[[ -n "${WANDB_API_KEY:-}" ]] && ENV_ARGS="${ENV_ARGS} -e WANDB_API_KEY=${WANDB_API_KEY}"
[[ -n "${OPENAI_API_KEY:-}" ]] && ENV_ARGS="${ENV_ARGS} -e OPENAI_API_KEY=${OPENAI_API_KEY}"
[[ -n "${ANTHROPIC_API_KEY:-}" ]] && ENV_ARGS="${ENV_ARGS} -e ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}"

# ---------------------------------------------------------------------------
# 5. Run pipeline inside container
# ---------------------------------------------------------------------------
log "Running pipeline stage '${STAGE}' in container '${CONTAINER}'..."
log "Log file: ${LOCAL_LOG}"
echo ""

# Use docker exec to run the pipeline. The env vars are set inside the
# exec command since docker exec doesn't support -e on all versions.
ssh "${DGX_HOST}" "\
    docker exec \
        -e RAY_memory_monitor_refresh_ms=0 \
        -e VLLM_USE_V1=0 \
        -e VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        ${ENV_ARGS} \
        ${CONTAINER} \
        bash /workspace/open-ctf-env/scripts/deploy/dgx_full_pipeline.sh ${STAGE} \
    2>&1" | tee "${LOCAL_LOG}"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [[ ${EXIT_CODE} -eq 0 ]]; then
    log "Pipeline completed successfully"
else
    log "Pipeline failed with exit code ${EXIT_CODE}"
fi
log "Full log: ${LOCAL_LOG}"

exit "${EXIT_CODE}"
