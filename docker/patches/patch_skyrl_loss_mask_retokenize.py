#!/usr/bin/env python3
"""REVERTED: Patch #21 — loss_mask retokenize fix.

Original hypothesis was WRONG:
We believed `return_assistant_tokens_mask` returned all-zeros with custom
templates, causing policy_loss=0.  Investigation by the data-science team
proved this was incorrect — `return_assistant_tokens_mask` works correctly
(assistant_masks sum=230 in v8 diagnostics).

ACTUAL ROOT CAUSE (Patch #22):
`apply_overlong_filtering()` in skyrl_gym_generator.py zeros the ENTIRE
loss_mask when `response[-1] != eos_token_id`.  The custom template
`qwen3_without_thinking` adds a trailing `\n` after `{% endgeneration %}`,
so `response[-1]` is `\n` (token 198) instead of `<|im_end|>` (token 248046).
This triggers the overlong filter, which zeros all masks.

FIX: Set `apply_overlong_filtering: false` in the training config YAML.
This is safe for retokenized multi-turn because the retokenize path already
handles sequence length correctly.

This patch file is kept for documentation.  It is no longer applied by
apply_all_patches.sh.

Timeline:
- v8: 4/4 flags, avg_reward=0.736, policy_loss=1.97e-08 (no-op)
- LOSS_DIAG: loss_mask sum=0.0, action_mask_sum=3149
- Patch #21 (this file): replaced return_assistant_tokens_mask with
  get_response_ids_and_loss_mask_from_messages() — WRONG FIX
- v10: crashed with TypeError (chat_template= kwarg not supported)
- Root cause found: apply_overlong_filtering() zeros loss_mask
- Patch #22: apply_overlong_filtering: false in config — CORRECT FIX
"""
import sys

print("patch_skyrl_loss_mask_retokenize: REVERTED (no-op)")
print("  Real fix: apply_overlong_filtering: false in training config (Patch #22)")
print("  See patch file docstring for full root cause analysis.")
