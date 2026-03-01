#!/usr/bin/env bash
# =============================================================================
# GEPA Prompt Evolution — Flag Command Example
# =============================================================================
#
# Runs GEPA (Stage 3) against a single HackTheBox "[Very Easy] Flag Command"
# challenge to evolve a system prompt that helps the model solve:
#
#   HTML page  ->  discover JS import  ->  GET /api/options  ->
#   find secret command  ->  submit flag
#
# GEPA creates two dspy.LM objects pointing at the same vLLM endpoint:
#   - Agent LM:      temperature=0.7, max_tokens=4096
#   - Reflection LM: temperature=1.0, max_tokens=32000
# No second server needed.
#
# Prerequisites:
#   1. vLLM serving any supported model on port 8001
#   2. Flag Command container running on port 32810
#   3. pip install -e ".[gepa]"  (installs dspy>=3.1.0, gepa>=0.0.26)
#
# Usage (from repo root):
#   bash examples/gepa_flag_command/run.sh --model openai/<your-model-id>
#
# Options:
#   --model MODEL_ID    Model identifier for dspy.LM (required)
#   --port PORT         vLLM server port (default: 8001)
#   --target-port PORT  Challenge container port (default: 32810)
#   --budget BUDGET     GEPA budget: light|medium|heavy (default: light)
#   --output DIR        Output directory (default: outputs/gepa_flag_command)
#
# Output:
#   outputs/gepa_flag_command/
#     optimized_prompt.txt   - The evolved system prompt
#     gepa_results.json      - Scores per candidate
#     gepa_logs/             - Optimizer traces
#
# =============================================================================

set -euo pipefail

# ── Defaults ─────────────────────────────────────────────────────────────────

MODEL_ID=""
VLLM_PORT=8001
TARGET_PORT=32810
BUDGET="light"
OUTPUT_DIR="outputs/gepa_flag_command"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Parse args ───────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case $1 in
        --model)       MODEL_ID="$2";     shift 2 ;;
        --port)        VLLM_PORT="$2";    shift 2 ;;
        --target-port) TARGET_PORT="$2";  shift 2 ;;
        --budget)      BUDGET="$2";       shift 2 ;;
        --output)      OUTPUT_DIR="$2";   shift 2 ;;
        -h|--help)
            head -n 38 "${BASH_SOURCE[0]}" | tail -n +2 | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "${MODEL_ID}" ]]; then
    echo "Error: --model is required"
    echo "Usage: bash examples/gepa_flag_command/run.sh --model openai/<your-model-id>"
    exit 1
fi

# ── Preflight checks ────────────────────────────────────────────────────────

echo "============================================================"
echo "GEPA Flag Command Example"
echo "============================================================"
echo "  Model:       ${MODEL_ID}"
echo "  vLLM:        http://localhost:${VLLM_PORT}/v1"
echo "  Target:      http://localhost:${TARGET_PORT}"
echo "  Budget:      ${BUDGET}"
echo "  Output:      ${OUTPUT_DIR}"
echo "============================================================"

# Check vLLM server
echo ""
echo "[1/3] Checking vLLM server on port ${VLLM_PORT}..."
if curl -sf "http://localhost:${VLLM_PORT}/v1/models" > /dev/null 2>&1; then
    echo "  OK — vLLM is serving"
else
    echo "  FAIL — vLLM not reachable on port ${VLLM_PORT}"
    echo "  Start it with:"
    echo "    vllm serve <model_path> --port ${VLLM_PORT} --dtype bfloat16 \\"
    echo "      --gpu-memory-utilization 0.50 --trust-remote-code"
    exit 1
fi

# Check challenge container
echo "[2/3] Checking Flag Command on port ${TARGET_PORT}..."
if curl -sf "http://localhost:${TARGET_PORT}" > /dev/null 2>&1; then
    echo "  OK — Flag Command is running"
else
    echo "  FAIL — Flag Command not reachable on port ${TARGET_PORT}"
    echo "  Start it with:"
    echo "    open-ctf-challenges setup --challenge '[Very Easy] Flag Command'"
    exit 1
fi

# Check dependencies
echo "[3/3] Checking GEPA dependencies..."
if python3 -c "import dspy; import gepa" 2>/dev/null; then
    echo "  OK — dspy + gepa installed"
else
    echo "  FAIL — missing dependencies"
    echo "  Install with: pip install -e '.[gepa]'"
    exit 1
fi

# ── Patch target port in challenge data ──────────────────────────────────────

DATA_FILE="${SCRIPT_DIR}/challenge.jsonl"

if [[ "${TARGET_PORT}" != "32810" ]]; then
    echo ""
    echo "Patching target port: 32810 -> ${TARGET_PORT}"
    TEMP_DATA=$(mktemp)
    sed "s|localhost:32810|localhost:${TARGET_PORT}|g" "${DATA_FILE}" > "${TEMP_DATA}"
    DATA_FILE="${TEMP_DATA}"
    trap "rm -f ${TEMP_DATA}" EXIT
fi

# ── Run GEPA ─────────────────────────────────────────────────────────────────

echo ""
echo "Starting GEPA prompt evolution..."
echo ""

cd "${REPO_ROOT}"

OPENAI_API_BASE="http://localhost:${VLLM_PORT}/v1" \
OPENAI_API_KEY="dummy" \
open-ctf-train gepa \
    --model "${MODEL_ID}" \
    --data "${DATA_FILE}" \
    --output "${OUTPUT_DIR}" \
    --budget "${BUDGET}" \
    --challenge-registry configs/challenges/cybench.yaml

# ── Report ───────────────────────────────────────────────────────────────────

echo ""
echo "============================================================"
echo "GEPA complete"
echo "============================================================"

if [[ -f "${OUTPUT_DIR}/optimized_prompt.txt" ]]; then
    echo ""
    echo "Evolved prompt:"
    echo "------------------------------------------------------------"
    cat "${OUTPUT_DIR}/optimized_prompt.txt"
    echo ""
    echo "------------------------------------------------------------"
fi

if [[ -f "${OUTPUT_DIR}/gepa_results.json" ]]; then
    echo ""
    echo "Results: ${OUTPUT_DIR}/gepa_results.json"
fi

if [[ -d "${OUTPUT_DIR}/gepa_logs" ]]; then
    echo "Logs:    ${OUTPUT_DIR}/gepa_logs/"
fi

echo ""
echo "Done."
