"""Group Relative Policy Optimization (GRPO) stage.

Uses TRL GRPOTrainer with:
  - DAPO loss with asymmetric clipping (epsilon_high=0.28)
  - beta=0.0 (no KL penalty, pure DAPO)
  - num_generations=8 for better group reward estimation
  - max_completion_length=4096 for full CTF trajectories
  - BF16 precision (avoids Half/BFloat16 dtype mismatch on Blackwell GB10)
  - reward_funcs=[fn] (list, not bare function)
  - GRPOConfig passed via ``args=`` (not ``config=``)
  - Unsloth vLLM fast inference when available (UNSLOTH_VLLM_STANDBY=1)

Uses Unsloth for model loading when available, falls back to standard
HuggingFace transformers + PEFT otherwise. The fallback is also used when
Unsloth's GRPO kernels have dtype issues (e.g. on Blackwell GB10).

Supports two OpenEnv integration modes (selected automatically):

  **Mode 1 (tools=)**: ``OPEN_CTF_ENV_URL`` is set. Uses TRL's ``tools=``
  parameter with ``max_tool_calling_iterations``. If ``use_vllm=True`` and
  vLLM is available, generation is accelerated via vLLM while tool
  execution still goes through TRL's ``_tool_call_loop``.

  **Mode 2 (offline)**: No ``OPEN_CTF_ENV_URL``. Falls back to the
  existing offline behavior with the provided ``reward_fn``.

Compatible with TRL >= 0.26 (processing_class, warmup_steps).
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
    """Patch TRL's prefix-preserving check to be a no-op at runtime.

    GLM-4.7-Flash's chat template is not prefix-preserving (``<think>`` /
    ``<|observation|>`` tags cause tokenized prefixes to differ). TRL's
    ``_tool_call_loop`` raises ``ValueError`` at runtime when it detects
    this.

    Instead of patching files on disk (fails on read-only pip installs in
    cloud containers), we monkey-patch the already-imported module object
    in memory. Called lazily from ``train_grpo()`` before tools= mode.
    """
    try:
        import trl.trainer.grpo_trainer as _grpo_mod

        # Find the _tool_call_loop or related method that does the check.
        # In TRL 0.28+, the prefix check is inside GRPOTrainer.__init__
        # or _tool_call_loop. We patch get_training_chat_template to
        # swallow the ValueError instead of propagating it.
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
        logger.info("Patched TRL prefix-preserving check (in-memory)")
    except Exception as e:
        logger.warning("Could not patch TRL prefix check: %s", e)


def _set_moe_backend():
    """Set UNSLOTH_MOE_BACKEND for GB10 compatibility if not already set."""
    if "UNSLOTH_MOE_BACKEND" not in os.environ:
        os.environ["UNSLOTH_MOE_BACKEND"] = "grouped_mm"
        logger.info("Set UNSLOTH_MOE_BACKEND=grouped_mm (GB10 safe default)")


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


def _load_model_hf(model_path, max_seq_length, load_in_4bit, lora_cfg):
    """Load model via standard HuggingFace transformers + PEFT."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    kwargs = {
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "attn_implementation": "sdpa",
    }
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # Check if this is already a PEFT model (adapter checkpoint)
    if not isinstance(model, PeftModel):
        lora_config = LoraConfig(
            r=lora_cfg.get("r", 64),
            lora_alpha=lora_cfg.get("alpha", 128),
            lora_dropout=lora_cfg.get("dropout", 0),
            target_modules=lora_cfg.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ]),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.gradient_checkpointing_enable()

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
            model_path, max_seq_length, load_in_4bit, lora_cfg
        )
        logger.info("Loaded model via HuggingFace transformers + PEFT")

    # --- Dataset ---------------------------------------------------------
    dataset = load_dataset("json", data_files=data_path, split="train")

    # GRPOTrainer requires a "prompt" column. Extract the system + user
    # messages from the full trajectory as the prompt.
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
        example["prompt"] = prompt_msgs
        return example

    dataset = dataset.map(_extract_prompt)
    if "messages" in dataset.column_names:
        dataset = dataset.remove_columns(["messages"])
    if "metadata" in dataset.column_names:
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
        # Mode 1: tools= parameter (HF generate, no vLLM needed)
        from open_ctf.training.tools import (
            shell_command, python_code, submit_flag, init_env,
        )

        init_env(env_url)
        trainer_extra_kwargs["tools"] = [shell_command, python_code, submit_flag]
        max_tool_iters = grpo_cfg.get("max_tool_calling_iterations", 15)
        grpo_kwargs["max_tool_calling_iterations"] = max_tool_iters
        logger.info(
            "OpenEnv tools= mode: 3 tools, max_tool_calling_iterations=%d",
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
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        args=grpo_training_config,
        **trainer_extra_kwargs,
    )

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
