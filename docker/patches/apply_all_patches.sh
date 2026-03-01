#!/bin/bash
# Apply all patches for GB10 training (SkyRL).
# Run this inside the container after installing dependencies.
#
# Usage: bash docker/patches/apply_all_patches.sh

# No set -e: individual patch scripts handle their own errors and exit codes.
# A cosmetic failure (e.g. vLLM serving API syntax warning) should not abort
# subsequent critical patches.
PATCH_DIR="$(cd "$(dirname "$0")" && pwd)"

PASS=0
FAIL=0
CRITICAL_FAIL=0

run_patch() {
    local label="$1"
    local critical="$2"  # "critical" or "optional"
    shift 2
    if "$@"; then
        PASS=$((PASS + 1))
    else
        FAIL=$((FAIL + 1))
        echo "   *** FAILED: $label ***"
        if [ "$critical" = "critical" ]; then
            CRITICAL_FAIL=$((CRITICAL_FAIL + 1))
        fi
    fi
}

echo "Applying patches from $PATCH_DIR..."
echo ""

# --- SkyRL patches (for GRPO) ---

# 1. Version comparison fix (string "2.10" < "2.6" breaks torch 2.10+)
run_patch "skyrl_version_comparison" critical python3 "$PATCH_DIR/patch_skyrl_version_comparison.py"

# 2. bf16 policy init (prevents OOM on GB10)
run_patch "skyrl_bf16_policy_init" critical python3 "$PATCH_DIR/patch_skyrl_bf16_policy_init.py"

# 3. NCCL weight sync skip for LoRA + remote engines (prevents deadlock)
run_patch "skyrl_weight_sync" critical python3 "$PATCH_DIR/patch_skyrl_weight_sync.py"

# 4. BatchEncoding-safe token normalization (fixes token_ids corruption)
run_patch "skyrl_batchencoding" critical python3 "$PATCH_DIR/patch_skyrl_batchencoding.py"

# 5. MixedPrecisionPolicy export (fsdp_utils.py missing re-export)
run_patch "skyrl_fsdp_mixed_precision" critical python3 "$PATCH_DIR/patch_skyrl_fsdp_mixed_precision.py"

# --- vLLM compatibility shims ---

# 6. vLLM 0.16+ compat shims (SkyRL expects 0.13-0.15 import paths)
run_patch "vllm_compat_shims" critical python3 "$PATCH_DIR/patch_vllm_compat_shims.py"

# 7. vLLM 0.16 serving API changes (model_config removed from constructors)
run_patch "vllm_serving_api" critical python3 "$PATCH_DIR/patch_vllm_serving_api.py"

# --- Ray compatibility ---

# 8. Ray 2.54+ compat (ray.experimental.collective.util removed)
run_patch "ray_collective_compat" optional python3 "$PATCH_DIR/patch_ray_collective_compat.py"

# --- Stub packages ---

# 9. flash_attn stub (real flash_attn not compiled for GB10 sm_121a)
run_patch "flash_attn_stub" optional python3 "$PATCH_DIR/patch_flash_attn_stub.py"


# --- SkyRL step-wise trajectory fixes ---

# 12. filter_generator_output drops is_last_step/trajectory_ids (KeyError in postprocess)
run_patch "skyrl_filter_stepwise" optional python3 "$PATCH_DIR/patch_skyrl_filter_stepwise.py"

# 13. collect_lora_params FSDP2 fix (summon_full_params is FSDP1-only, blocks on FSDP2)
run_patch "skyrl_collect_lora_fsdp2" optional python3 "$PATCH_DIR/patch_skyrl_collect_lora_fsdp2.py"

# 14. step_wise_trajectories truncation fix (max_trajectories truncates steps, not trajectories)
run_patch "skyrl_stepwise_truncation" optional python3 "$PATCH_DIR/patch_skyrl_stepwise_truncation.py"

# 15. step_wise reward index guard (skip out-of-range resp_end_idx writes)
run_patch "skyrl_stepwise_index_guard" optional python3 "$PATCH_DIR/patch_skyrl_stepwise_index_guard.py"

# 16. Empty per_token_reward fallback (prevents ValueError when step has 0 tokens)
run_patch "skyrl_empty_reward_fallback" optional python3 "$PATCH_DIR/patch_skyrl_empty_reward_fallback.py"

# --- vLLM GB10 memory profiling fix ---

# 17. vLLM memory profiling assertion (GB10 unified memory: other processes
#     release memory during profiling, causing false AssertionError)
run_patch "vllm_memory_assert" critical python3 "$PATCH_DIR/patch_vllm_memory_assert.py"

# --- Loss diagnostics + fix ---

# 18. loss_mask diagnostics + all-zero fallback (root-causes policy_loss=0.0;
#     falls back to action_mask when loss_mask is all zeros)
run_patch "skyrl_loss_diagnostics" optional python3 "$PATCH_DIR/patch_skyrl_loss_diagnostics.py"

# --- OmegaConf → plain Python for chat_template_kwargs ---

# 19. Convert OmegaConf DictConfig/ListConfig in chat_template_kwargs to plain
#     Python before apply_chat_template() calls (transformers isinstance(tool, dict)
#     validation rejects DictConfig; blocks native_tool_schemas=true)
run_patch "skyrl_chat_template_kwargs" optional python3 "$PATCH_DIR/patch_skyrl_chat_template_kwargs.py"

# --- Cleanup ---
echo ""
echo "Clearing __pycache__..."
SITE_PACKAGES="$(python3 -c 'import sysconfig; print(sysconfig.get_path("purelib"))' 2>/dev/null || echo '/usr/local/lib/python3.12/dist-packages')"
for pkg in skyrl_train vllm ray flash_attn; do
    find "$SITE_PACKAGES/$pkg" -name "*.pyc" -delete 2>/dev/null || true
done
echo "   Done"

# --- Summary ---
echo ""
TOTAL=$((PASS + FAIL))
echo "Patch summary: $PASS/$TOTAL passed, $FAIL failed ($CRITICAL_FAIL critical)"
if [ "$CRITICAL_FAIL" -gt 0 ]; then
    echo "ERROR: $CRITICAL_FAIL critical patch(es) failed — training may not work."
    exit 1
elif [ "$FAIL" -gt 0 ]; then
    echo "WARNING: $FAIL optional patch(es) failed — check logs above."
else
    echo "All patches applied successfully."
fi
