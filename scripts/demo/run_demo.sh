#!/usr/bin/env bash
# =============================================================================
# Open CTF Environment - Live Demo
# =============================================================================
#
# One-command demo: start a challenge, run the agent, show results.
#
# Usage:
#   ./scripts/demo/run_demo.sh                    # Use defaults
#   ./scripts/demo/run_demo.sh --model ollama/qwen3:8b
#   ./scripts/demo/run_demo.sh --challenge XBEN-003-24
#   ./scripts/demo/run_demo.sh --skip-setup       # Skip dependency checks
#
# Prerequisites:
#   - Docker running
#   - Python 3.10+ with uv
#   - An LLM backend (Ollama, vLLM, or any OpenAI-compatible API)
#   - XBow benchmarks at benchmarks/xbow/
#
# =============================================================================

set -euo pipefail

# Defaults
CHALLENGE="${CHALLENGE:-XBEN-003-24}"
MODEL="${MODEL:-openrouter/openai/gpt-oss-120b}"
PLATFORM="${PLATFORM:-xbow}"
STRATEGY="${STRATEGY:-chat_tools}"
MAX_TURNS="${MAX_TURNS:-30}"
MAX_TIME="${MAX_TIME:-15}"
SKIP_SETUP="${SKIP_SETUP:-false}"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --challenge) CHALLENGE="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --platform) PLATFORM="$2"; shift 2 ;;
        --strategy) STRATEGY="$2"; shift 2 ;;
        --max-turns) MAX_TURNS="$2"; shift 2 ;;
        --max-time) MAX_TIME="$2"; shift 2 ;;
        --skip-setup) SKIP_SETUP="true"; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BOLD}${BLUE}"
echo "  ___                    ____ _____ _____"
echo " / _ \ _ __   ___ _ __ / ___|_   _|  ___|"
echo "| | | | '_ \ / _ \ '_ \| |     | | | |_"
echo "| |_| | |_) |  __/ | | | |___  | | |  _|"
echo " \___/| .__/ \___|_| |_|\____| |_| |_|"
echo "      |_|"
echo -e "${NC}"
echo -e "${BOLD}Open CTF Environment - Live Demo${NC}"
echo ""
echo -e "  Challenge:  ${GREEN}${CHALLENGE}${NC}"
echo -e "  Model:      ${GREEN}${MODEL}${NC}"
echo -e "  Platform:   ${GREEN}${PLATFORM}${NC}"
echo -e "  Strategy:   ${GREEN}${STRATEGY}${NC}"
echo -e "  Max turns:  ${GREEN}${MAX_TURNS}${NC}"
echo -e "  Max time:   ${GREEN}${MAX_TIME} min${NC}"
echo ""

# -----------------------------------------------------------------------
# Step 0: Environment checks
# -----------------------------------------------------------------------

if [[ "$SKIP_SETUP" != "true" ]]; then
    echo -e "${YELLOW}[0/4] Checking prerequisites...${NC}"

    # Docker
    if ! docker info > /dev/null 2>&1; then
        echo -e "${RED}  Docker is not running. Start Docker Desktop and retry.${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}Docker:    OK${NC}"

    # Python
    if ! python3 --version > /dev/null 2>&1; then
        echo -e "${RED}  Python 3 not found.${NC}"
        exit 1
    fi
    echo -e "  ${GREEN}Python:    $(python3 --version)${NC}"

    # BoxPwnr reference
    if [[ ! -d "references/boxpwnr/src/boxpwnr" ]]; then
        echo -e "${RED}  BoxPwnr reference not found at references/boxpwnr/${NC}"
        echo -e "  Run: git clone https://github.com/0ca/BoxPwnr.git references/boxpwnr"
        exit 1
    fi
    echo -e "  ${GREEN}BoxPwnr:   OK${NC}"

    # Agent runner check
    if python3 -m open_ctf.cli.run_agent --check > /dev/null 2>&1; then
        echo -e "  ${GREEN}Agent:     OK${NC}"
    else
        echo -e "${YELLOW}  Agent:     Import check failed (may still work)${NC}"
    fi

    echo ""
fi

# -----------------------------------------------------------------------
# Step 1: Start the challenge
# -----------------------------------------------------------------------

echo -e "${YELLOW}[1/4] Starting challenge ${CHALLENGE}...${NC}"

CHALLENGE_DIR="benchmarks/xbow/benchmarks/${CHALLENGE}"
if [[ -d "$CHALLENGE_DIR" ]]; then
    cd "$CHALLENGE_DIR"
    docker compose up -d 2>/dev/null || docker-compose up -d 2>/dev/null
    cd "$PROJECT_ROOT"
    echo -e "  ${GREEN}Challenge running${NC}"
else
    echo -e "  ${YELLOW}Challenge dir not found, using platform's built-in provisioning${NC}"
fi

# Wait for services to be ready
echo -e "  Waiting for services to start..."
sleep 5

# -----------------------------------------------------------------------
# Step 2: Run the agent
# -----------------------------------------------------------------------

echo -e "${YELLOW}[2/4] Running agent...${NC}"
echo ""

python3 -m open_ctf.cli.run_agent \
    --platform "$PLATFORM" \
    --target "$CHALLENGE" \
    --model "$MODEL" \
    --strategy "$STRATEGY" \
    --max-turns "$MAX_TURNS" \
    --max-time "$MAX_TIME" \
    --traces-dir "./targets" \
    2>&1 | tee "/tmp/octf_demo_${CHALLENGE}.log"

AGENT_EXIT=$?

echo ""

# -----------------------------------------------------------------------
# Step 3: Show results
# -----------------------------------------------------------------------

echo -e "${YELLOW}[3/4] Results${NC}"

if [[ $AGENT_EXIT -eq 0 ]]; then
    echo -e "  ${GREEN}Agent completed successfully${NC}"
else
    echo -e "  ${RED}Agent exited with code ${AGENT_EXIT}${NC}"
fi

# Check for traces
LATEST_TRACE=$(find ./targets -name "conversation.json" -newer /tmp/octf_demo_${CHALLENGE}.log 2>/dev/null | head -1)
if [[ -n "$LATEST_TRACE" ]]; then
    TRACE_DIR=$(dirname "$LATEST_TRACE")
    echo -e "  Trace:  ${BLUE}${TRACE_DIR}${NC}"

    # Check stats
    if [[ -f "${TRACE_DIR}/stats.json" ]]; then
        STATUS=$(python3 -c "import json; print(json.load(open('${TRACE_DIR}/stats.json')).get('status', 'unknown'))" 2>/dev/null || echo "unknown")
        TURNS=$(python3 -c "import json; print(json.load(open('${TRACE_DIR}/stats.json')).get('total_turns', '?'))" 2>/dev/null || echo "?")
        echo -e "  Status: ${GREEN}${STATUS}${NC}"
        echo -e "  Turns:  ${TURNS}"
    fi
fi

echo -e "  Log:    ${BLUE}/tmp/octf_demo_${CHALLENGE}.log${NC}"

# -----------------------------------------------------------------------
# Step 4: Cleanup
# -----------------------------------------------------------------------

echo ""
echo -e "${YELLOW}[4/4] Cleanup${NC}"

if [[ -d "$CHALLENGE_DIR" ]]; then
    cd "$CHALLENGE_DIR"
    docker compose down 2>/dev/null || docker-compose down 2>/dev/null
    cd "$PROJECT_ROOT"
    echo -e "  ${GREEN}Challenge containers stopped${NC}"
fi

echo ""
echo -e "${BOLD}${GREEN}Demo complete.${NC}"
echo ""
echo -e "Next steps:"
echo -e "  # Convert trace to training data:"
echo -e "  open-ctf-convert --input targets/ --output data/traces.jsonl --success-only"
echo ""
echo -e "  # Train a model on the traces:"
echo -e "  open-ctf-train sft --model unsloth/GLM-4.7-Flash --data data/traces.jsonl --output outputs/sft"
echo ""
echo -e "  # Export to GGUF for local deployment:"
echo -e "  open-ctf-export --adapter outputs/sft/final --base-model unsloth/GLM-4.7-Flash --output models/ctf-agent.gguf"
