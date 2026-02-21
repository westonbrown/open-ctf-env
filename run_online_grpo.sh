#!/bin/bash
# Online GRPO Training - Real tool execution via OpenEnv server
#
# Uses open-ctf-env:latest container which includes:
#   - TRL 0.28 with tools= parameter (multi-turn tool calling)
#   - vLLM 0.16 compiled for Blackwell GB10 (colocate mode, 3-6x faster)
#   - Unsloth (used for SFT only; OPEN_CTF_NO_UNSLOTH=1 bypasses for GRPO)
#   - OpenEnv server (13 BoxPwnr tool handlers)
#   - CTFReward (6 signals + hallucination penalty)
#
# Architecture:
#   [TRL GRPOTrainer] --tools=--> [_tool_call_loop]
#        |                              |
#        v                              v
#   [vLLM colocate]              [OpenEnv HTTP server]
#   (fast generation)             (shell/python/flag execution)
#
# Usage:
#   bash run_online_grpo.sh                                        # Full dataset
#   GRPO_DATA=data/grpo_test2.jsonl bash run_online_grpo.sh        # Test (2 samples)
#   GRPO_DATA=data/sample/grpo_sample.jsonl bash run_online_grpo.sh  # Sample (16)

set -euo pipefail

WORKSPACE="${WORKSPACE:-$(cd "$(dirname "$0")" && pwd)}"
cd "${WORKSPACE}"

# Configuration
GRPO_MODEL="${GRPO_MODEL:-outputs/sft-merged}"
GRPO_DATA="${GRPO_DATA:-data/grpo.jsonl}"
CONFIG=src/open_ctf/configs/training_dgx.yaml
RUN_ID="grpo_online_$(date +%Y%m%d_%H%M%S)"
GRPO_OUTPUT="runs/${RUN_ID}/grpo"
LOG_FILE="logs/grpo_online_${RUN_ID}.log"

echo "$(date) | Online GRPO Training (vLLM colocate)"
echo "$(date) | Model: ${GRPO_MODEL}"
echo "$(date) | Data: ${GRPO_DATA} ($(wc -l < ${GRPO_DATA}) samples)"
echo "$(date) | Output: ${GRPO_OUTPUT}"
echo "$(date) | Log: ${LOG_FILE}"

mkdir -p "runs/${RUN_ID}" logs

# Remove any previous grpo-train container
docker rm -f grpo-train 2>/dev/null || true

docker run \
  --gpus all --name grpo-train --shm-size=64g --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${WORKSPACE}:/workspace/open-ctf-env" \
  -v "${HOME}/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace/open-ctf-env \
  -e OPEN_CTF_NO_UNSLOTH=1 \
  -e OPEN_CTF_ENV_URL=http://localhost:8100 \
  -e PYTHONPATH=/workspace/open-ctf-env/src \
  -e WANDB_MODE=disabled \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  open-ctf-env:latest \
  bash -c '
    set -e
    echo "$(date) | Verifying installation..."
    python3 -c "
import trl; print(f\"  TRL: {trl.__version__}\")
import transformers; print(f\"  transformers: {transformers.__version__}\")
import vllm; print(f\"  vLLM: {vllm.__version__}\")
from open_ctf.training.grpo import train_grpo; print(\"  grpo module: OK\")
from open_ctf.rewards import CTFReward; print(\"  reward module: OK\")
from open_ctf.training.tools import get_core_tools; print(f\"  tools: {len(get_core_tools())} core tools\")
from trl import GRPOConfig
import inspect; src = inspect.getsource(GRPOConfig)
print(f\"  vLLM colocate support: {chr(39)}use_vllm{chr(39)} in src = {\"use_vllm\" in src}\")
print(f\"  tools= support: {chr(39)}max_tool_calling{chr(39)} in src = {\"max_tool_calling\" in src}\")
"

    # Start OpenEnv server in background
    echo "$(date) | Starting OpenEnv server on localhost:8100..."
    python3 -m open_ctf.envs.openenv.server &
    ENV_PID=$!
    sleep 3

    # Verify env server is up
    if ! curl -sf http://localhost:8100/health; then
      echo "$(date) | ERROR: OpenEnv server failed to start"
      kill $ENV_PID 2>/dev/null || true
      exit 1
    fi
    echo "$(date) | OpenEnv server ready"

    # Run GRPO training
    echo "$(date) | Starting GRPO training..."
    EXIT_CODE=0
    open-ctf-train --config '"${CONFIG}"' grpo \
      --model '"${GRPO_MODEL}"' \
      --data '"${GRPO_DATA}"' \
      --output '"${GRPO_OUTPUT}"' || EXIT_CODE=$?

    # Cleanup
    kill $ENV_PID 2>/dev/null || true
    wait $ENV_PID 2>/dev/null || true

    echo "$(date) | Training finished (exit: $EXIT_CODE)"
    exit $EXIT_CODE
  ' 2>&1 | tee "${LOG_FILE}"

echo "$(date) | DONE (exit: ${PIPESTATUS[0]})"
