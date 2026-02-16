#!/bin/bash
# Open CTF Training Pipeline Launcher
# Runs 2-stage training: SFT -> GRPO with optional merge.
#
# Usage:
#   bash scripts/launch_training.sh                   # full pipeline
#   bash scripts/launch_training.sh --sft-only        # SFT only
#   bash scripts/launch_training.sh --grpo-only       # GRPO only (requires SFT model)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# -----------------------------------------------------------------------
# Defaults (override via environment variables)
# -----------------------------------------------------------------------
MODEL="${MODEL:-unsloth/GLM-4.7-Flash}"
SFT_DATA="${SFT_DATA:-$ROOT_DIR/data/sft.jsonl}"
GRPO_DATA="${GRPO_DATA:-$ROOT_DIR/data/grpo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs}"
CONFIG="${CONFIG:-$ROOT_DIR/src/open_ctf/configs/training.yaml}"

SFT_ONLY=false
GRPO_ONLY=false

for arg in "$@"; do
    case $arg in
        --sft-only) SFT_ONLY=true ;;
        --grpo-only) GRPO_ONLY=true ;;
    esac
done

echo "================================================================"
echo "OPEN CTF TRAINING PIPELINE"
echo "================================================================"
echo "Model:      $MODEL"
echo "SFT Data:   $SFT_DATA"
echo "GRPO Data:  $GRPO_DATA"
echo "Output:     $OUTPUT_DIR"
echo "Config:     $CONFIG"
echo "================================================================"

mkdir -p "$OUTPUT_DIR"

# -----------------------------------------------------------------------
# Stage 1: SFT
# -----------------------------------------------------------------------
if [ "$GRPO_ONLY" = false ]; then
    echo ""
    echo "[Stage 1/2] Supervised Fine-Tuning..."
    python3 -m open_ctf.cli.train sft \
        --model "$MODEL" \
        --data "$SFT_DATA" \
        --output "$OUTPUT_DIR/sft" \
        --config "$CONFIG"
    echo "[Stage 1/2] SFT complete. Model at: $OUTPUT_DIR/sft/final"
fi

# -----------------------------------------------------------------------
# Stage 2: GRPO
# -----------------------------------------------------------------------
SFT_MODEL="${OUTPUT_DIR}/sft/final"

if [ "$SFT_ONLY" = false ]; then
    if [ ! -d "$SFT_MODEL" ] && [ "$GRPO_ONLY" = true ]; then
        echo "ERROR: SFT model not found at $SFT_MODEL"
        echo "Run SFT first or set SFT_MODEL to an existing model path."
        exit 1
    fi

    echo ""
    echo "[Stage 2/2] GRPO Training..."
    python3 -m open_ctf.cli.train grpo \
        --model "$SFT_MODEL" \
        --data "$GRPO_DATA" \
        --output "$OUTPUT_DIR/grpo" \
        --config "$CONFIG"
    echo "[Stage 2/2] GRPO complete. Model at: $OUTPUT_DIR/grpo/final"

    echo ""
    echo "[Merge] Merging LoRA adapter..."
    python3 -m open_ctf.cli.train merge \
        --adapter "$OUTPUT_DIR/grpo/final" \
        --base-model "$MODEL" \
        --output "$OUTPUT_DIR/merged" \
        --config "$CONFIG"
    echo "[Merge] Merged model at: $OUTPUT_DIR/merged"
fi

echo ""
echo "================================================================"
echo "TRAINING COMPLETE"
echo "================================================================"
if [ "$SFT_ONLY" = true ]; then
    echo "SFT Model: $OUTPUT_DIR/sft/final"
elif [ "$GRPO_ONLY" = true ]; then
    echo "GRPO Model: $OUTPUT_DIR/grpo/final"
    echo "Merged:     $OUTPUT_DIR/merged"
else
    echo "SFT Model:  $OUTPUT_DIR/sft/final"
    echo "GRPO Model: $OUTPUT_DIR/grpo/final"
    echo "Merged:     $OUTPUT_DIR/merged"
fi
echo "================================================================"
