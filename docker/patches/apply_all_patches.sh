#!/bin/bash
# Apply all patches for GB10 training (SkyRL + LlamaFactory).
# Run this inside the container after installing dependencies.
#
# Usage: bash docker/patches/apply_all_patches.sh

set -e
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Applying patches from $PATCH_DIR..."
echo ""

# --- SkyRL patches (for GRPO) ---

# 1. Version comparison fix (string "2.10" < "2.6" breaks torch 2.10+)
python3 "$PATCH_DIR/patch_skyrl_version_comparison.py"

# 2. bf16 policy init (prevents OOM on GB10)
python3 "$PATCH_DIR/patch_skyrl_bf16_policy_init.py"

# 3. NCCL weight sync skip for LoRA + remote engines (prevents deadlock)
python3 "$PATCH_DIR/patch_skyrl_weight_sync.py"

# 4. BatchEncoding-safe token normalization (fixes token_ids corruption)
python3 "$PATCH_DIR/patch_skyrl_batchencoding.py"

# 5. MixedPrecisionPolicy export (fsdp_utils.py missing re-export)
python3 "$PATCH_DIR/patch_skyrl_fsdp_mixed_precision.py"

# --- vLLM compatibility shims ---

# 6. vLLM 0.16+ compat shims (SkyRL expects 0.13-0.15 import paths)
python3 "$PATCH_DIR/patch_vllm_compat_shims.py"

# 7. vLLM 0.16 serving API changes (model_config removed from constructors)
python3 "$PATCH_DIR/patch_vllm_serving_api.py"

# --- Ray compatibility ---

# 8. Ray 2.54+ compat (ray.experimental.collective.util removed)
python3 "$PATCH_DIR/patch_ray_collective_compat.py"

# --- Stub packages ---

# 9. flash_attn stub (real flash_attn not compiled for GB10 sm_121a)
python3 "$PATCH_DIR/patch_flash_attn_stub.py"

# 10. torchaudio stub (NGC PyTorch ABI incompatibility)
python3 "$PATCH_DIR/patch_torchaudio_stub.py"

# --- LlamaFactory patches (for SFT) ---

# 11. tool_calls None guard (HuggingFace datasets schema normalization)
python3 "$PATCH_DIR/patch_llamafactory_tool_calls.py"

# --- Cleanup ---
echo ""
echo "Clearing __pycache__..."
find /usr/local/lib/python3.12/dist-packages/skyrl_train -name "*.pyc" -delete 2>/dev/null || true
find /usr/local/lib/python3.12/dist-packages/llamafactory -name "*.pyc" -delete 2>/dev/null || true
echo "   Done"

echo ""
echo "All patches applied successfully."
