"""Group Relative Policy Optimization (GRPO) stage.

Uses TRL GRPOTrainer with:
  - DAPO loss with asymmetric clipping (epsilon_high=0.28)
  - beta=0.0 (no KL penalty, pure DAPO)
  - num_generations for group reward estimation
  - BF16 precision (avoids Half/BFloat16 dtype mismatch on Blackwell GB10)
  - reward_funcs=[fn] (list, not bare function)
  - GRPOConfig passed via ``args=`` (not ``config=``)
  - Unsloth vLLM fast inference when available (UNSLOTH_VLLM_STANDBY=1)

Uses Unsloth for model loading when available, falls back to standard
HuggingFace transformers + PEFT otherwise. The fallback is also used when
Unsloth's GRPO kernels have dtype issues (e.g. on Blackwell GB10).

Supports two training modes (selected automatically):

  **Mode 1 — online (tools=)**: ``OPEN_CTF_ENV_URL`` is set. Uses
  ``OnlineGRPOTrainer`` (subclass of TRL's ``GRPOTrainer``) with TRL's
  ``tools=`` parameter and ``max_tool_calling_iterations``. The trainer
  resets the environment before each batch and tracks episode completion
  so that ``num_generations > 1`` works safely. vLLM colocate mode
  accelerates generation while the tool-call loop handles multi-turn
  execution against the live environment.

  **Mode 2 — offline**: No ``OPEN_CTF_ENV_URL``. Uses vanilla
  ``GRPOTrainer`` with the provided ``reward_fn`` for offline scoring.

Compatible with TRL >= 0.28 (tools=, max_tool_calling_iterations).
"""

import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional

# Pre-set vLLM standby mode before any Unsloth/vLLM imports (~30% memory savings)
os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

import torch
from datasets import load_dataset

logger = logging.getLogger(__name__)


from trl import GRPOConfig, GRPOTrainer


def _patch_trl_prefix_check():
    """Patch TRL's prefix-preserving checks to be lenient at runtime.

    GLM-4.7-Flash's chat template is not prefix-preserving (``<think>`` /
    ``<|observation|>`` tags cause tokenized prefixes to differ). TRL checks
    this in TWO places:

    1. ``GRPOTrainer.__init__`` via ``get_training_chat_template()`` — we
       patch that function to swallow the ValueError.
    2. ``GRPOTrainer._tool_call_loop`` at line ~107 — after each tool-call
       iteration, TRL re-tokenizes the prompt+completion+tool and compares
       the prompt prefix tokens. If they differ, it raises ValueError.
       We monkey-patch ``_tool_call_loop`` to turn this into a warning.

    Both patches are applied in-memory (safe for read-only pip installs).
    Called lazily from ``train_grpo()`` before tools= mode.
    """
    try:
        import inspect
        import trl.trainer.grpo_trainer as _grpo_mod

        # --- Patch 1: get_training_chat_template (init-time check) --------
        import trl.chat_template_utils as _chat_utils

        _orig = _chat_utils.get_training_chat_template

        def _safe_get_training_chat_template(tok):
            try:
                return _orig(tok)
            except ValueError:
                logger.warning(
                    "Chat template is not prefix-preserving (patched to continue). "
                    "Tool calling will still work correctly."
                )
                return None

        _chat_utils.get_training_chat_template = _safe_get_training_chat_template
        _grpo_mod.get_training_chat_template = _safe_get_training_chat_template

        # --- Patch 2: _tool_call_loop (runtime prefix comparison) ---------
        # TRL 0.28 line ~107: raises ValueError if re-tokenized prompt IDs
        # don't match original prompt_ids. For GLM-4.7-Flash, the
        # <|observation|> tag causes a ~2 token mismatch. We replace the
        # ValueError raise with a warning + prompt_ids fixup so the rest
        # of the tool loop continues normally.
        _orig_src = inspect.getsource(_grpo_mod.GRPOTrainer._tool_call_loop)

        # Remove the raise ValueError block and replace with a warning +
        # prompt_ids fixup (update prompt_ids to match re-tokenized prefix).
        _old_check = (
            '                if prompt_ids[idx_with_tool] != pct[: len(prompt_ids[idx_with_tool])]:\n'
            '                    raise ValueError(\n'
            '                        "The chat template is not prefix-preserving. '
            'Please update it to use a prefix-preserving "\n'
            '                        "format."\n'
            '                    )'
        )
        _new_check = (
            '                if prompt_ids[idx_with_tool] != pct[: len(prompt_ids[idx_with_tool])]:\n'
            '                    import logging as _log\n'
            '                    _log.getLogger("open_ctf.training.grpo").warning(\n'
            '                        "Prefix mismatch in _tool_call_loop (expected for GLM-4.7-Flash). "\n'
            '                        "Fixing up prompt_ids to match re-tokenized prefix."\n'
            '                    )\n'
            '                    # Fix: update prompt_ids to match the re-tokenized prefix\n'
            '                    prompt_ids[idx_with_tool] = pct[: len(prompt_ids[idx_with_tool])]'
        )

        if _old_check in _orig_src:
            _patched_src = _orig_src.replace(_old_check, _new_check)
            # Compile and replace the method
            _ns = {}
            # Need to dedent from class method to function level
            import textwrap
            _patched_src = textwrap.dedent(_patched_src)
            exec(compile(_patched_src, "<prefix-patch>", "exec"), _grpo_mod.__dict__, _ns)
            _grpo_mod.GRPOTrainer._tool_call_loop = _ns["_tool_call_loop"]
            logger.info("Patched _tool_call_loop: prefix check replaced with fixup")
        else:
            logger.warning(
                "Could not find prefix check pattern in _tool_call_loop source. "
                "The check may have changed in this TRL version."
            )

        # --- Patch 3: _compute_loss tool_mask shape mismatch ------------------
        # TRL 0.28 bug: when tool calling extends the completion beyond
        # max_completion_length, tool_mask (actual seq length) doesn't match
        # completion_mask (padded to max_completion_length). The completion_mask
        # is a local var in _compute_loss, so we truncate tool_mask to
        # max_completion_length before the original method runs.
        _orig_compute_loss = _grpo_mod.GRPOTrainer._compute_loss

        def _patched_compute_loss(self, model, inputs):
            if "tool_mask" in inputs:
                tool_mask = inputs["tool_mask"]
                max_clen = getattr(self.args, "max_completion_length", None)
                if max_clen and tool_mask.shape[-1] != max_clen:
                    if tool_mask.shape[-1] > max_clen:
                        inputs["tool_mask"] = tool_mask[:, :max_clen]
                    else:
                        pad = torch.zeros(
                            tool_mask.shape[0],
                            max_clen - tool_mask.shape[-1],
                            dtype=tool_mask.dtype,
                            device=tool_mask.device,
                        )
                        inputs["tool_mask"] = torch.cat([tool_mask, pad], dim=-1)
            return _orig_compute_loss(self, model, inputs)

        _grpo_mod.GRPOTrainer._compute_loss = _patched_compute_loss
        logger.info("Patched _compute_loss: tool_mask shape alignment")

        logger.info("Patched TRL prefix-preserving checks (init + runtime)")
    except Exception as e:
        logger.warning("Could not patch TRL prefix check: %s", e)



def _set_moe_backend():
    """Set UNSLOTH_MOE_BACKEND for GB10 compatibility if not already set."""
    from open_ctf.training.quantize import set_moe_backend
    set_moe_backend()


def _patch_grouped_mm_dtype():
    """Patch Unsloth's grouped_mm to cast inputs to weight dtype.

    Fixes: RuntimeError: expected mat1 and mat2 to have the same dtype,
    but got: float != c10::BFloat16

    Root cause: During GRPO generation, hidden states can arrive as float32
    (from router softmax or autocast context) but MoE base weights are bfloat16.
    Unsloth's ``_grouped_mm_with_backward_fix`` passes both directly to
    ``torch._grouped_mm`` with no dtype check.  The LoRA weight path in the
    same file *does* cast (``.to(permuted_input.dtype)``), but the base weight
    path does not.  This patch adds the missing cast.

    See: unsloth_zoo/temporary_patches/moe_utils.py line ~73
    See: https://github.com/unslothai/unsloth/issues/3506
    """
    try:
        import unsloth_zoo.temporary_patches.moe_utils as moe_utils

        _original = moe_utils._grouped_mm_with_backward_fix

        def _dtype_safe_grouped_mm(inputs, weight, offsets):
            if inputs.dtype != weight.dtype:
                inputs = inputs.to(weight.dtype)
            return _original(inputs, weight, offsets)

        moe_utils._grouped_mm_with_backward_fix = _dtype_safe_grouped_mm
        logger.info("Patched _grouped_mm_with_backward_fix for dtype safety")
    except (ImportError, AttributeError) as e:
        logger.debug("Could not patch grouped_mm (Unsloth not loaded): %s", e)


def _patch_vllm_sync_weights_for_parametrized(trainer):
    """Patch vLLM sync_weights to skip parametrized (4-bit quantized) params.

    After ``replace_parameter_4bit``, MoE expert param names change from
    ``mlp.experts.gate_up_proj`` to
    ``mlp.experts.parametrizations.gate_up_proj.original``.
    vLLM's model doesn't recognize these names, causing KeyError during
    ``sync_weights()``.

    Since these expert weights are frozen (no LoRA), they never change between
    training and generation steps — skipping them in sync is safe.

    Approach: replace ``sync_weights`` with a wrapper that reimplements the
    non-FSDP PEFT branch with parametrized params filtered out. This avoids
    monkey-patching ``named_parameters`` on the model class.
    """
    vllm_gen = getattr(trainer, "vllm_generation", None)
    if vllm_gen is None:
        return

    from peft import PeftModel

    def _patched_sync():
        model = vllm_gen.model
        accelerator = vllm_gen.accelerator

        # Only handle our specific case: PEFT, no FSDP, no DeepSpeed ZeRO-3,
        # colocate mode. Fall back to original for anything else.
        deepspeed_plugin = accelerator.state.deepspeed_plugin
        zero3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero3 or vllm_gen.is_fsdp_enabled or vllm_gen.mode != "colocate":
            logger.warning(
                "vLLM sync patch: unsupported config (FSDP=%s, ZeRO3=%s, mode=%s), "
                "falling back to original sync_weights",
                vllm_gen.is_fsdp_enabled, zero3, vllm_gen.mode,
            )
            return _original_sync()

        if not isinstance(model, PeftModel):
            return _original_sync()

        # Merge LoRA adapters before syncing
        model.merge_adapter()

        llm_model = (
            vllm_gen.llm.llm_engine.model_executor.driver_worker
            .model_runner.model
        )
        skipped = 0
        synced = 0
        for name, param in model.named_parameters():
            # Skip parametrized (4-bit quantized) expert weights
            if "parametrizations" in name:
                skipped += 1
                continue

            # Strip PEFT prefix (same logic as TRL's sync_weights)
            name = name.removeprefix("base_model.model.").replace(
                ".base_layer", ""
            )
            # Skip PEFT-only layers (already merged)
            if model.prefix in name:
                continue
            if "original_module" in name:
                continue
            name = vllm_gen._fix_param_name_to_vllm(
                name, extra_prefixes=["modules_to_save.default."]
            )

            llm_model.load_weights([(name, param.data)])
            synced += 1

        # Unmerge adapters for continued training
        model.unmerge_adapter()

        # Reset vLLM KV cache
        vllm_gen.llm.reset_prefix_cache()

        logger.debug(
            "vLLM sync: synced=%d params, skipped=%d parametrized", synced, skipped
        )

    _original_sync = vllm_gen.sync_weights
    vllm_gen.sync_weights = _patched_sync
    logger.info(
        "Patched vLLM sync_weights: skipping parametrized (4-bit MoE) params"
    )


def _load_model_unsloth(model_path, max_seq_length, load_in_4bit, lora_cfg,
                        grpo_cfg=None):
    """Load model via Unsloth FastLanguageModel (faster, optimized kernels).

    Uses BF16 dtype to avoid the Half/BFloat16 mismatch bug that affects
    Unsloth GRPO on Blackwell GB10 (previously seen with FP16).  The
    gpu_memory_utilization is read from grpo_cfg to allow DGX-safe values.
    """
    _set_moe_backend()
    from unsloth import FastLanguageModel
    from peft import PeftModel

    gpu_mem_util = (grpo_cfg or {}).get("gpu_memory_utilization", 0.6)
    logger.info("Unsloth fast_inference gpu_memory_utilization=%.2f", gpu_mem_util)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=load_in_4bit,
        fast_inference=True,
        gpu_memory_utilization=gpu_mem_util,
    )

    if isinstance(model, PeftModel):
        logger.info("Model already has LoRA adapters from SFT, skipping get_peft_model")
        FastLanguageModel.for_training(model)
    else:
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_cfg.get("r", 64),
            lora_alpha=lora_cfg.get("alpha", 128),
            lora_dropout=lora_cfg.get("dropout", 0),
            target_modules=lora_cfg.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            use_rslora=lora_cfg.get("use_rslora", False),
            use_gradient_checkpointing="unsloth",
        )
    return model, tokenizer


def _find_moe_expert_param_names(model) -> list:
    """Detect 3D+ nn.Parameter tensors that BnB skips. Delegates to quantize.py."""
    from open_ctf.training.quantize import find_moe_expert_param_names
    return find_moe_expert_param_names(model)


def _quantize_moe_expert_params(model, quant_type=None,
                                 compress_statistics=None):
    """Quantize 3D MoE expert nn.Parameter tensors. Delegates to quantize.py."""
    from open_ctf.training.quantize import quantize_moe_expert_params
    return quantize_moe_expert_params(model, quant_type, compress_statistics)


def _load_model_hf(model_path, max_seq_length, load_in_4bit, lora_cfg,
                    load_in_8bit=False):
    """Load model via standard HuggingFace transformers + PEFT.

    Supports three quantization modes for MoE models:

    - **BF16** (default): Full precision, ~60 GB for GLM-4.7-Flash.
    - **8-bit** (``load_in_8bit=True``): ~30 GB. Skips
      ``prepare_model_for_kbit_training`` to avoid OOM from fp32 cast.
    - **4-bit** (``load_in_4bit=True``): ~18 GB. BnB quantizes
      ``nn.Linear`` layers normally, then ``_quantize_moe_expert_params``
      handles the fused 3D expert tensors that BnB skips.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
    }
    if load_in_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        logger.info("Loading model in 8-bit (saves ~30GB VRAM vs BF16)")
    elif load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    # --- Post-load 4-bit quantization for MoE 3D expert tensors -----------
    # BnB only quantizes nn.Linear (2D). Transformers v5 MoE models store
    # experts as fused 3D nn.Parameter which BnB skips, leaving them in
    # BF16 (~52 GB for GLM-4.7-Flash). Quantize them post-load.
    moe_experts_quantized = False
    moe_expert_param_names = []
    if load_in_4bit:
        moe_expert_param_names = _find_moe_expert_param_names(model)
        if moe_expert_param_names:
            moe_experts_quantized = _quantize_moe_expert_params(model)

    # --- Prepare for k-bit training ----------------------------------------
    # All MoE quantized paths skip prepare_model_for_kbit_training because
    # it casts non-quantized params to fp32, causing OOM (~115 GB peak).
    # Instead, manually enable input_require_grads (gradient checkpointing
    # is handled by GRPOConfig).
    if load_in_8bit or (load_in_4bit and moe_experts_quantized):
        model.enable_input_require_grads()
        quant_label = "8-bit" if load_in_8bit else "4-bit MoE"
        logger.info(
            "%s: skipped prepare_model_for_kbit_training (OOMs for MoE), "
            "enabled input_require_grads manually", quant_label
        )
    elif load_in_4bit:
        # Non-MoE 4-bit: standard path
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    # --- LoRA configuration ------------------------------------------------
    if not isinstance(model, PeftModel):
        lora_kwargs = dict(
            r=lora_cfg.get("r", 64),
            lora_alpha=lora_cfg.get("alpha", 128),
            lora_dropout=lora_cfg.get("dropout", 0),
            target_modules=lora_cfg.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            task_type="CAUSAL_LM",
        )

        # For quantized MoE: LoRA targets attention + shared expert
        # via target_modules only.  The routed expert 3D tensors are
        # quantized for memory savings but NOT LoRA'd — this follows
        # Unsloth's recommendation for MoE models and avoids the
        # parametrization name mismatch (replace_parameter_4bit wraps
        # params, changing their names in named_parameters()).
        #
        # Exclude BnB parametrization sub-modules from target_modules
        # scan to avoid accidental matches on parametrization wrappers.
        if moe_experts_quantized:
            lora_kwargs["exclude_modules"] = r".*\.parametrizations\..*"
            logger.info(
                "MoE 4-bit LoRA: target_modules=%s (attention + shared expert only, "
                "routed experts quantized but not LoRA'd)",
                lora_kwargs["target_modules"],
            )

        lora_config = LoraConfig(**lora_kwargs)
        model = get_peft_model(model, lora_config)
    # Don't call gradient_checkpointing_enable() here — GRPOConfig handles it.
    # Double-enable can cause issues with MoE models on some hardware.

    # Unsloth patches TRL's GRPOTrainer at import time in containers that
    # have Unsloth pre-installed. The patched trainer calls model.for_training()
    # and model.for_inference() which are Unsloth-specific. Add no-op stubs.
    if not hasattr(model, "for_training"):
        model.for_training = lambda **kw: None
    if not hasattr(model, "for_inference"):
        model.for_inference = lambda **kw: None
    # Also patch the base model if it's a PEFT wrapper
    base = getattr(model, "model", None)
    if base is not None:
        if not hasattr(base, "for_training"):
            base.for_training = lambda **kw: None
        if not hasattr(base, "for_inference"):
            base.for_inference = lambda **kw: None
    # And the underlying pretrained model
    pretrained = getattr(base, "model", None) if base else None
    if pretrained is not None:
        if not hasattr(pretrained, "for_training"):
            pretrained.for_training = lambda **kw: None
        if not hasattr(pretrained, "for_inference"):
            pretrained.for_inference = lambda **kw: None

    return model, tokenizer


def _add_generic_parse_response(tokenizer) -> None:
    """Add a basic ``parse_response`` to a tokenizer that lacks one.

    TRL's GRPOTrainer requires ``tokenizer.parse_response`` when ``tools=``
    is set. If TRL's ``add_response_schema`` can't recognise the chat
    template (e.g. GLM-4.7-Flash), we fall back to a simple JSON-based
    parser that extracts tool calls from the generated text.
    """
    import json as _json
    import re as _re

    # GLM-4.7-Flash XML tool call pattern:
    #   <tool_call>func_name<arg_key>k1</arg_key><arg_value>v1</arg_value>...</tool_call>
    _glm_tc_pat = _re.compile(
        r"<tool_call>(\S+?)((?:<arg_key>.*?</arg_key><arg_value>.*?</arg_value>)*)\s*</tool_call>",
        _re.DOTALL,
    )
    _glm_arg_pat = _re.compile(
        r"<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>", _re.DOTALL,
    )

    def _parse_response(text_or_ids) -> dict:
        """Extract tool calls from generated text or token IDs.

        TRL calls this with a list of token IDs, not a string.
        We decode first if needed.
        """
        if isinstance(text_or_ids, list):
            text = tokenizer.decode(text_or_ids, skip_special_tokens=True)
        else:
            text = str(text_or_ids)

        tool_calls = []

        # GLM-4.7-Flash XML format (non-JSON)
        for m in _glm_tc_pat.finditer(text):
            name = m.group(1).strip()
            args = {}
            for am in _glm_arg_pat.finditer(m.group(2)):
                key = am.group(1).strip()
                val = am.group(2).strip()
                # Try to parse value as JSON for proper typing
                try:
                    val = _json.loads(val)
                except (ValueError, _json.JSONDecodeError):
                    pass
                args[key] = val
            if name:
                tool_calls.append({
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                })

        # JSON-based tool call formats
        if not tool_calls:
            patterns = [
                _re.compile(r"<\|tool_call\|>\s*(\{.*?\})\s*<\|/tool_call\|>", _re.DOTALL),
                _re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", _re.DOTALL),
                _re.compile(r"```json\s*(\{.*?\})\s*```", _re.DOTALL),
            ]
            for pat in patterns:
                for m in pat.finditer(text):
                    try:
                        d = _json.loads(m.group(1))
                        name = d.get("name", "")
                        args = d.get("arguments", {})
                        if isinstance(args, str):
                            try:
                                args = _json.loads(args)
                            except _json.JSONDecodeError:
                                args = {}
                        if name:
                            tool_calls.append({
                                "type": "function",
                                "function": {"name": name, "arguments": args},
                            })
                    except _json.JSONDecodeError:
                        continue
                if tool_calls:
                    break

        # Fallback: look for bare JSON with "name" key
        if not tool_calls:
            for m in _re.finditer(r'\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*\}', text):
                try:
                    d = _json.loads(m.group(0))
                    name = d.get("name", "")
                    args = d.get("arguments", {})
                    if isinstance(args, str):
                        try:
                            args = _json.loads(args)
                        except _json.JSONDecodeError:
                            args = {}
                    if name:
                        tool_calls.append({
                            "type": "function",
                            "function": {"name": name, "arguments": args},
                        })
                except _json.JSONDecodeError:
                    continue

        if tool_calls:
            return {"role": "assistant", "content": "", "tool_calls": tool_calls}
        return {"role": "assistant", "content": text}

    tokenizer.parse_response = _parse_response
    # TRL's GRPOTrainer checks response_schema (not parse_response) to decide
    # whether to call add_response_schema. Set a sentinel so TRL skips its check.
    tokenizer.response_schema = {"type": "object", "properties": {"role": {"const": "assistant"}}}
    logger.info("Added generic parse_response + response_schema to tokenizer")


def _detect_env_mode(grpo_cfg: Dict[str, Any]) -> str:
    """Detect which OpenEnv integration mode to use.

    Returns one of: ``"tools"``, ``"offline"``.

    When ``OPEN_CTF_ENV_URL`` is set, always uses TRL's ``tools=`` parameter.
    vLLM (if available) is used for fast generation *alongside* tools= --
    they are orthogonal: vLLM accelerates ``_generate_single_turn`` while
    ``_tool_call_loop`` handles multi-turn tool execution.
    """
    env_url = os.environ.get("OPEN_CTF_ENV_URL", "").strip()
    if not env_url:
        return "offline"
    return "tools"


class OnlineGRPOTrainer(GRPOTrainer):
    """GRPOTrainer with per-batch environment resets for online RL.

    Solves the episode management gap in TRL's ``tools=`` feature:

    1. **Per-batch reset** — ``_tool_call_loop`` is wrapped to call
       ``mark_step_begin()`` before processing, giving all
       ``num_generations`` completions a clean environment.
    2. **Done protection** — once any generation triggers ``done`` (e.g.
       flag submission, task completion), ``tools.py`` short-circuits
       subsequent tool calls so later generations don't corrupt state.
    3. **Logging** — emits tool-call and episode statistics per step.

    With ``num_generations > 1``, generations within a batch still share
    the same server process.  Read-only operations are safe; stateful
    operations may interfere, but the done flag prevents the worst case.
    For full isolation, use a multi-episode server (future work).
    """

    def __init__(self, *args, challenge_id: Optional[str] = None,
                 kv_cache_dtype: Optional[str] = None, **kwargs):
        self._challenge_id = challenge_id
        # Inject FP8 KV cache dtype into vLLM engine BEFORE parent creates it.
        # GRPOConfig doesn't expose kv_cache_dtype; we monkey-patch vllm.LLM
        # to inject the kwarg during colocate engine creation. This halves KV
        # cache memory with negligible quality impact for tool-calling.
        _unpatch = None
        if kv_cache_dtype:
            _unpatch = self._patch_vllm_kv_dtype(kv_cache_dtype)
        super().__init__(*args, **kwargs)
        if _unpatch:
            _unpatch()

    @staticmethod
    def _patch_vllm_kv_dtype(dtype: str):
        """Monkey-patch vllm.LLM.__init__ to inject kv_cache_dtype.

        Returns an unpatch callable (or None if vLLM not available).
        """
        try:
            import vllm
            _orig_init = vllm.LLM.__init__

            def _patched_init(self_llm, *a, **kw):
                kw.setdefault("kv_cache_dtype", dtype)
                return _orig_init(self_llm, *a, **kw)

            vllm.LLM.__init__ = _patched_init
            logger.info("Patched vLLM LLM to use kv_cache_dtype=%s", dtype)

            def _unpatch():
                vllm.LLM.__init__ = _orig_init

            return _unpatch
        except ImportError:
            return None

    def _tool_call_loop(self, *args, **kwargs):
        """Wrap parent's tool-call loop with environment reset.

        Uses ``*args, **kwargs`` to stay compatible across TRL versions
        (the internal signature of ``_tool_call_loop`` changed between
        0.26 → 0.27 → 0.28).
        """
        from open_ctf.training.tools import mark_step_begin

        mark_step_begin(self._challenge_id)
        result = super()._tool_call_loop(*args, **kwargs)

        # Log tool call stats from this batch
        # result is a tuple; last two elements are tool_call_count and
        # tool_failure_count (TRL 0.28+).
        if isinstance(result, tuple) and len(result) >= 6:
            tc_count, tc_fail = result[-2], result[-1]
            logger.info(
                "Tool loop done: %d calls, %d failures", tc_count, tc_fail,
            )
        return result


def train_grpo(
    model_path: str,
    data_path: str,
    output_dir: str,
    config: Dict[str, Any],
    reward_fn: Callable[..., List[float]],
    resume_from: Optional[str] = None,
) -> str:
    """Run GRPO training with TRL.

    Tries Unsloth first for speed, falls back to standard HuggingFace
    if Unsloth's GRPO kernels fail (e.g. dtype issues on some GPUs).

    Supports two OpenEnv integration modes (selected automatically):

    - ``OPEN_CTF_ENV_URL`` set: **tools=** mode (TRL tool calling loop)
    - ``OPEN_CTF_ENV_URL`` not set: **offline** mode (existing behavior)

    Args:
        model_path: Path to the SFT model (merged or adapter directory).
        data_path: Path to JSONL GRPO data with ``ground_truth_flag`` and
            ``optimal_steps`` columns.
        output_dir: Directory for checkpoints and final model.
        config: Merged config dict with keys: model, lora, grpo, output.
        reward_fn: A callable that scores completions. Must accept
            ``(completions, prompts=None, **kwargs)`` and return ``list[float]``.
        resume_from: Optional checkpoint path to resume from.

    Returns:
        Path to the saved final model directory.
    """
    logger.info("=" * 60)
    logger.info("GRPO TRAINING")
    logger.info("  Model:  %s", model_path)
    logger.info("  Data:   %s", data_path)
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 60)

    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    grpo_cfg = config.get("grpo", {})
    output_cfg = config.get("output", {})

    max_seq_length = model_cfg.get("max_seq_length", 8192)
    load_in_4bit = model_cfg.get("load_in_4bit", True)
    load_in_8bit = model_cfg.get("load_in_8bit", False)

    # --- Detect OpenEnv mode ---------------------------------------------
    env_mode = _detect_env_mode(grpo_cfg)
    env_url = os.environ.get("OPEN_CTF_ENV_URL", "").strip()
    logger.info("OpenEnv mode: %s", env_mode)
    if env_url:
        logger.info("  Env URL: %s", env_url)

    # --- Patch Unsloth MoE dtype bug (must run before any forward pass) --
    # Unsloth patches TRL at import time in containers that have it installed.
    # The MoE kernel patches are active even if we fall back to HF model loading.
    # Apply our dtype fix whenever the Unsloth MoE module is importable.
    _patch_grouped_mm_dtype()

    # --- Model + LoRA ---------------------------------------------------
    use_unsloth = os.environ.get("OPEN_CTF_NO_UNSLOTH", "").lower() not in ("1", "true")
    if use_unsloth:
        try:
            model, tokenizer = _load_model_unsloth(
                model_path, max_seq_length, load_in_4bit, lora_cfg,
                grpo_cfg=grpo_cfg,
            )
            logger.info("Loaded model via Unsloth")
        except (ImportError, RuntimeError, ValueError, OSError) as e:
            logger.warning("Unsloth loading failed (%s), falling back to HF", e)
            use_unsloth = False

    if not use_unsloth:
        model, tokenizer = _load_model_hf(
            model_path, max_seq_length, load_in_4bit, lora_cfg,
            load_in_8bit=load_in_8bit,
        )
        logger.info("Loaded model via HuggingFace transformers + PEFT")

    # --- Dataset ---------------------------------------------------------
    dataset = load_dataset("json", data_files=data_path, split="train")

    # GRPOTrainer requires a "prompt" column. Extract the system + user
    # messages from the full trajectory as the prompt.
    # Some BoxPwnr traces lack a "user" message (challenge is in system prompt).
    # TRL requires the prompt to end with role "user", so inject a fallback.
    def _extract_prompt(example):
        messages = example["messages"]
        prompt_msgs = []
        for msg in messages:
            role = msg.get("role", "")
            if role in ("system", "user"):
                clean_msg = {"role": role, "content": msg.get("content", "")}
                prompt_msgs.append(clean_msg)
            else:
                break  # Stop at first assistant/tool message
        # TRL's apply_chat_template requires last message to be "user"
        if not prompt_msgs or prompt_msgs[-1]["role"] != "user":
            challenge = example.get("metadata", {}).get("challenge", "")
            user_content = (
                f"Solve the CTF challenge{f': {challenge}' if challenge else ''}. "
                "Find and capture the flag."
            )
            prompt_msgs.append({"role": "user", "content": user_content})
        example["prompt"] = prompt_msgs
        return example

    dataset = dataset.map(_extract_prompt)
    if "messages" in dataset.column_names:
        dataset = dataset.remove_columns(["messages"])
    # Extract metadata.success into a top-level column before dropping metadata.
    # CTFReward uses this as the authoritative signal for flag capture scoring.
    if "metadata" in dataset.column_names:
        def _extract_success(example):
            meta = example.get("metadata")
            if isinstance(meta, dict):
                example["success"] = meta.get("success")
            else:
                example["success"] = None
            return example
        dataset = dataset.map(_extract_success)
        dataset = dataset.remove_columns(["metadata"])

    # --- Determine wandb availability -----------------------------------
    from open_ctf.training import check_wandb_available

    report_to = check_wandb_available(output_cfg.get("report_to", "wandb"))

    # --- Convert warmup_ratio to warmup_steps ----------------------------
    warmup_ratio = grpo_cfg.get("warmup_ratio", 0.10)
    num_epochs = grpo_cfg.get("epochs", 1)
    batch_size = grpo_cfg.get("batch_size", 1)
    grad_accum = grpo_cfg.get("gradient_accumulation_steps", 8)
    total_samples = len(dataset)
    steps_per_epoch = max(1, total_samples // (batch_size * grad_accum))
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(0, int(math.ceil(warmup_ratio * total_steps)))

    # --- GRPO config (passed as args=, NOT config=) ----------------------
    grpo_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=grpo_cfg.get("learning_rate", 5e-6),
        warmup_steps=warmup_steps,
        logging_steps=output_cfg.get("logging_steps", 1),
        save_steps=output_cfg.get("save_steps", 50),
        bf16=True,
        optim="adamw_8bit",
        seed=42,
        report_to=report_to,
        gradient_checkpointing=True,
        max_grad_norm=grpo_cfg.get("max_grad_norm", 0.1),
        weight_decay=grpo_cfg.get("weight_decay", 0.1),
        # GRPO-specific
        num_generations=grpo_cfg.get("num_generations", 8),
        max_completion_length=grpo_cfg.get("max_completion_length", 4096),
        beta=grpo_cfg.get("beta", 0.0),
        loss_type=grpo_cfg.get("loss_type", "dapo"),
        epsilon_high=grpo_cfg.get("epsilon_high", 0.28),
        scale_rewards=grpo_cfg.get("scale_rewards", "group"),
    )

    # --- vLLM for fast generation (3-6x faster than HF generate) ----------
    vllm_available = False
    if grpo_cfg.get("use_vllm", False):
        try:
            import vllm  # noqa: F401
            vllm_available = True
            grpo_kwargs["use_vllm"] = True
            grpo_kwargs["vllm_mode"] = grpo_cfg.get("vllm_mode", "colocate")
            grpo_kwargs["vllm_gpu_memory_utilization"] = grpo_cfg.get(
                "vllm_gpu_memory_utilization",
                grpo_cfg.get("gpu_memory_utilization", 0.3),
            )
            logger.info(
                "vLLM enabled: mode=%s, gpu_mem=%.2f",
                grpo_kwargs["vllm_mode"],
                grpo_kwargs["vllm_gpu_memory_utilization"],
            )
        except ImportError:
            logger.warning("use_vllm=True but vllm not installed, falling back to HF generate")

    # --- OpenEnv integration: tools / offline reward ----------------------
    trainer_extra_kwargs: Dict[str, Any] = {}
    reward_funcs = [reward_fn]

    if env_mode == "tools":
        # Mode 1: tools= parameter with full BoxPwnr tool set
        from open_ctf.training.tools import get_all_tools, get_core_tools, init_env

        init_env(env_url)

        # Use full tool set by default, or core-only if configured
        use_core_only = grpo_cfg.get("core_tools_only", False)
        if use_core_only:
            tools = get_core_tools()
        else:
            tools = get_all_tools()

        trainer_extra_kwargs["tools"] = tools
        max_tool_iters = grpo_cfg.get("max_tool_calling_iterations", 15)
        grpo_kwargs["max_tool_calling_iterations"] = max_tool_iters
        logger.info(
            "OpenEnv tools= mode: %d tools, max_tool_calling_iterations=%d",
            len(tools),
            max_tool_iters,
        )

    else:
        # Mode 2: offline (no changes to existing behavior)
        logger.info("Offline mode: no live environment, using offline reward only")

    # --- Ensure tokenizer has response_schema for tools= mode ----------------
    # TRL's GRPOTrainer.__init__ calls add_response_schema(tokenizer) when
    # tools= is set. It also calls get_training_chat_template() to verify the
    # chat template is prefix-preserving. Both fail for models with
    # unrecognized chat templates (e.g. GLM-4.7-Flash). Pre-set response_schema
    # and patch get_training_chat_template to return None so TRL's checks pass.
    if env_mode == "tools":
        # Apply prefix-preserving patch (in-memory, safe for read-only installs)
        _patch_trl_prefix_check()

        if not getattr(tokenizer, "response_schema", None):
            try:
                from trl.chat_template_utils import add_response_schema
                tokenizer = add_response_schema(tokenizer)
                logger.info("Response schema added via TRL auto-detection")
            except (ValueError, Exception) as e:
                logger.warning("TRL add_response_schema failed (%s), adding generic parser", e)
                _add_generic_parse_response(tokenizer)

    grpo_training_config = GRPOConfig(**grpo_kwargs)

    # --- Trainer ---------------------------------------------------------
    # Use OnlineGRPOTrainer for tools= mode (handles episode resets + done
    # tracking); fall back to vanilla GRPOTrainer for offline mode.
    TrainerCls = OnlineGRPOTrainer if env_mode == "tools" else GRPOTrainer
    trainer_init_kwargs = dict(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        args=grpo_training_config,
        **trainer_extra_kwargs,
    )
    if TrainerCls is OnlineGRPOTrainer:
        # Pass challenge_id for per-batch resets (extracted from data if available)
        trainer_init_kwargs["challenge_id"] = grpo_cfg.get("challenge_id")
        # FP8 KV cache dtype (applied after vLLM engine creation)
        kv_dtype = grpo_cfg.get("vllm_kv_cache_dtype")
        if kv_dtype:
            trainer_init_kwargs["kv_cache_dtype"] = kv_dtype
    trainer = TrainerCls(**trainer_init_kwargs)

    # Patch vLLM sync_weights to skip parametrized (4-bit MoE) params.
    # Must be called AFTER trainer init (vllm_generation is created in __init__).
    if vllm_available:
        _patch_vllm_sync_weights_for_parametrized(trainer)

    trainer.train(resume_from_checkpoint=resume_from)

    # --- Cleanup OpenEnv --------------------------------------------------
    if env_mode == "tools":
        try:
            from open_ctf.training.tools import close_env
            close_env()
        except Exception as e:
            logger.warning("Failed to close OpenEnv client: %s", e)

    # --- Save final model ------------------------------------------------
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("GRPO model saved to %s", final_dir)
    return final_dir
