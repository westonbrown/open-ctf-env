"""SkyRL-based Group Relative Policy Optimization (GRPO) stage.

Thin orchestrator that:
  1. Converts our GRPO JSONL to SkyRL dataset format (messages -> prompt + extras)
  2. Registers OpenCTFTextEnv with SkyRL-Gym
  3. Launches SkyRL training via BasePPOExp

Replaces the custom grpo.py (1305 lines, 6 monkey-patches) with ~200 lines
by delegating to SkyRL for:
  - Fully async Ray-based trainer (eliminates NCCL segfault patch)
  - vLLM in separate process (eliminates dtype/weight sync patches)
  - Module-grouped weight sync (eliminates 3 HF->vLLM translation patches)
  - TIS off-policy correction + dynamic DAPO sampling
  - Process isolation per env (eliminates thread-local episode management)

Uses OpenCTFTextEnv (SkyRL-Gym BaseTextEnv subclass) which executes
tools directly via ToolExecutor (no HTTP server needed).

Default test model: Nanbeige4.1-3B (dense LlamaForCausalLM).
"""

import json
import importlib.util
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import yaml

logger = logging.getLogger(__name__)

# Canonical difficulty ordering for curriculum filtering.
_DIFFICULTY_ORDER: List[str] = ["very_easy", "easy", "medium", "hard", "expert", "master"]
_DIFFICULTY_RANK: Dict[str, int] = {d: i for i, d in enumerate(_DIFFICULTY_ORDER)}

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs" / "skyrl"


def _as_positive_int(name: str, raw_value: Any, default: int) -> int:
    """Parse a positive int config value with warning-backed fallback."""
    if raw_value is None:
        return default
    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        logger.warning("Invalid %s=%r; defaulting to %d.", name, raw_value, default)
        return default
    if value <= 0:
        logger.warning("Invalid %s=%r (must be >0); defaulting to %d.", name, raw_value, default)
        return default
    return value


def _detect_visible_gpu_count() -> Optional[int]:
    """Best-effort count of GPUs visible to the current process."""
    visible = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        stripped = visible.strip()
        if not stripped or stripped == "-1":
            return 0
        devices = [chunk.strip() for chunk in stripped.split(",") if chunk.strip()]
        if devices:
            return len(devices)

    try:
        import torch
    except Exception:
        return None

    try:
        if not torch.cuda.is_available():
            return 0
        return int(torch.cuda.device_count())
    except Exception:
        return None


def _flash_attn_available() -> bool:
    """Return True if flash-attn exports the symbols Transformers expects."""
    try:
        import flash_attn  # type: ignore
    except Exception:
        return False
    return all(
        hasattr(flash_attn, attr)
        for attr in ("flash_attn_func", "flash_attn_varlen_func")
    )


def _is_qwen3_5_config(hf_cfg: Any) -> bool:
    """Return True if a HF config appears to be Qwen3.5."""
    model_type = str(getattr(hf_cfg, "model_type", "")).lower()
    cfg_cls_name = str(hf_cfg.__class__.__name__).lower()
    architectures = [
        str(arch).lower()
        for arch in (getattr(hf_cfg, "architectures", None) or [])
    ]
    return (
        "qwen3_5" in model_type
        or "qwen3_5" in cfg_cls_name
        or any("qwen3_5" in arch for arch in architectures)
    )


def _missing_qwen3_5_fast_path_deps() -> list[str]:
    """Return missing Qwen3.5 linear-attention fast-path dependencies.

    Qwen3.5 uses flash linear attention (`fla`) and causal-conv1d for
    optimized linear-attention blocks. If missing, Transformers falls back
    to a slow torch path (`torch_chunk_gated_delta_rule`) that is unstable
    for long-horizon online RL workloads.
    """
    missing = []
    try:
        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )

        if not is_flash_linear_attention_available():
            missing.append("flash-linear-attention (module: fla)")
        if not is_causal_conv1d_available():
            missing.append("causal-conv1d")
        return missing
    except Exception:
        # Fallback for older Transformers that may not expose these helpers.
        if importlib.util.find_spec("fla") is None:
            missing.append("flash-linear-attention (module: fla)")
        if importlib.util.find_spec("causal_conv1d") is None:
            missing.append("causal-conv1d")
        return missing


def _validate_qwen3_5_runtime_dependencies(
    hf_cfg: Any,
    grpo_cfg: Dict[str, Any],
) -> None:
    """Fail fast when Qwen3.5 runtime dependencies are missing."""
    if not _is_qwen3_5_config(hf_cfg):
        return

    missing = _missing_qwen3_5_fast_path_deps()
    if not missing:
        return

    missing_str = ", ".join(missing)
    msg = (
        "Qwen3.5 detected but required linear-attention runtime deps are "
        f"missing: {missing_str}. Install with: "
        "`uv sync --extra grpo --frozen` (preferred) or "
        "`pip install --no-deps flash-linear-attention==0.4.1 causal-conv1d==1.6.0`. "
        "Without these libs, Transformers falls back to torch linear-attention "
        "kernels that are unstable for long-context GRPO."
    )
    strict = bool(grpo_cfg.get("require_fast_linear_attention", True))
    if strict:
        raise RuntimeError(
            msg
            + " To bypass temporarily, set grpo.require_fast_linear_attention=false "
            "(not recommended for production)."
        )
    logger.warning(
        "%s Proceeding because grpo.require_fast_linear_attention=false.",
        msg,
    )


def _resolve_reward_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize reward config and enforce a dict payload."""
    reward_cfg = config.get("reward")
    if reward_cfg is None:
        logger.info("No reward config provided; using default CTFReward weights.")
        return {}
    if not isinstance(reward_cfg, dict):
        raise TypeError(
            f"config['reward'] must be a dict if provided, got {type(reward_cfg).__name__}."
        )
    return reward_cfg


def _should_force_legacy_inference(
    model_path: str,
    *,
    allow_qwen35_new_inference: bool = False,
) -> bool:
    """Return True when SkyRL new inference should be disabled for model config.

    SkyRL's new inference server path currently initializes vLLM renderers in a
    way that can mis-handle some HuggingFace text-wrapper configs (for example
    Qwen3_5TextConfig). When this happens, vLLM raises a config type mismatch in
    multimodal processor setup before training starts.
    """
    try:
        from transformers import AutoConfig
        hf_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        logger.warning(
            "Could not inspect model config for inference backend selection: %s",
            exc,
        )
        return False

    cfg_cls_name = hf_cfg.__class__.__name__
    if cfg_cls_name.endswith("TextConfig"):
        logger.warning(
            "Model config class %s detected at %s. Forcing SkyRL legacy "
            "inference path (new inference is incompatible with text-wrapper configs).",
            cfg_cls_name,
            model_path,
        )
        return True
    if _is_qwen3_5_config(hf_cfg) and not allow_qwen35_new_inference:
        logger.warning(
            "Qwen3.5 config detected at %s. Forcing SkyRL legacy inference path "
            "(new inference shows intermittent /inference/v1/generate 400s and "
            "engine-core exits on this stack). Set "
            "grpo.allow_new_inference_for_qwen35=true to override.",
            model_path,
        )
        return True
    return False


def _resolve_vllm_ready_model_path(model_path: str) -> str:
    """Resolve a vLLM-compatible model handoff path for GRPO.

    Qwen3.5 merged checkpoints used for HF training can expose a text-wrapper
    config (for example ``model_type=qwen3_5_text``). SkyRL+vLLM runtime paths
    expect the vLLM-ready variant (for example sibling ``*_vllm`` directory).
    """
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception as exc:
        logger.warning("Could not inspect model config at %s: %s", model_path, exc)
        return model_path

    cfg_cls_name = cfg.__class__.__name__
    cfg_model_type = str(getattr(cfg, "model_type", ""))
    looks_text_wrapper = cfg_cls_name.endswith("TextConfig") or cfg_model_type.endswith("_text")
    if not looks_text_wrapper:
        return model_path

    candidate = f"{model_path.rstrip('/')}_vllm"
    if os.path.isdir(candidate):
        try:
            from transformers import AutoConfig
            cand_cfg = AutoConfig.from_pretrained(candidate, trust_remote_code=True)
            cand_cls_name = cand_cfg.__class__.__name__
            cand_model_type = str(getattr(cand_cfg, "model_type", ""))
            cand_is_text_wrapper = (
                cand_cls_name.endswith("TextConfig") or cand_model_type.endswith("_text")
            )
            if not cand_is_text_wrapper:
                logger.warning(
                    "Model path %s uses text-wrapper config (%s/%s). "
                    "Auto-switching GRPO runtime model to sibling vLLM-ready path: %s",
                    model_path,
                    cfg_model_type or "<unknown_model_type>",
                    cfg_cls_name,
                    candidate,
                )
                return candidate
        except Exception as exc:
            logger.warning("Failed to validate sibling vLLM model path %s: %s", candidate, exc)

    logger.warning(
        "Model path %s appears to use text-wrapper config (%s/%s) and no validated "
        "sibling '*_vllm' path was found. GRPO runtime may fail in vLLM.",
        model_path,
        cfg_model_type or "<unknown_model_type>",
        cfg_cls_name,
    )
    return model_path


def _parse_lora_rank(lora_cfg: Dict[str, Any]) -> int:
    """Parse LoRA rank from config with a defensive fallback."""
    raw_rank = lora_cfg.get("r", 64)
    try:
        return int(raw_rank)
    except (TypeError, ValueError):
        logger.warning("Invalid lora.r=%r; defaulting to rank 64.", raw_rank)
        return 64


def _normalize_module_filter(raw_modules: Any, *, default: Optional[str]) -> Any:
    """Normalize LoRA target/exclude module filters for SkyRL + PEFT.

    Important: SkyRL's FSDP worker serializes PEFT target_modules via
    ``list(peft_config["target_modules"])`` before LoRA sync. If a plain
    string is used (for example "q_proj,k_proj"), that becomes a character list
    and breaks vLLM adapter loading. We therefore pass structured lists for
    explicit module selections, and reserve a string only for the special
    ``all-linear`` selector.
    """
    if raw_modules is None:
        return default

    if isinstance(raw_modules, str):
        value = raw_modules.strip()
        if not value:
            return default
        if value == "all-linear":
            return value
        if "," in value:
            modules = [part.strip() for part in value.split(",") if part.strip()]
            return modules or default
        return [value]

    if isinstance(raw_modules, (list, tuple, set)):
        modules = [str(module).strip() for module in raw_modules if str(module).strip()]
        return modules or default

    value = str(raw_modules).strip()
    if not value:
        return default
    if value == "all-linear":
        return value
    return [value]


def _parse_lora_modules(lora_cfg: Dict[str, Any]) -> tuple[Any, Any]:
    """Parse LoRA target/exclude module filters from config."""
    target_modules = _normalize_module_filter(
        lora_cfg.get("target_modules"),
        default="all-linear",
    )
    exclude_modules = _normalize_module_filter(
        lora_cfg.get("exclude_modules"),
        default=None,
    )
    return target_modules, exclude_modules


def _normalize_remote_url(url: str) -> str:
    """Normalize remote vLLM URL to SkyRL host:port format."""
    return re.sub(r"^https?://", "", str(url).strip()).rstrip("/")


def _resolve_generator_topology(grpo_cfg: Dict[str, Any], lora_rank: int) -> Dict[str, Any]:
    """Resolve SkyRL generator topology from config.

    SkyRL currently supports LoRA weight sync only when engines are local
    (``run_engines_locally=true``). Remote engine mode does not support
    ``LoraLoadRequest`` in upstream SkyRL.
    """
    vllm_mode = str(grpo_cfg.get("vllm_mode", "colocate")).strip().lower()
    requested_remote_url = grpo_cfg.get("vllm_server_url")
    remote_requested = bool(requested_remote_url)

    local_disagg_modes = {
        "server",
        "disagg",
        "disaggregated",
        "non_colocate",
        "non-colocate",
    }
    valid_modes = {"colocate", "local"} | local_disagg_modes
    if vllm_mode not in valid_modes:
        logger.warning(
            "Unknown grpo.vllm_mode=%r; defaulting to 'colocate'.", vllm_mode
        )
        vllm_mode = "colocate"

    # Upstream SkyRL remote inference mode does not support NCCL LoRA sync.
    # With our weight_sync patch (patch_skyrl_weight_sync.py), file-based
    # LoRA sync (save adapter → HTTP /v1/load_lora_adapter) works instead.
    # On constrained GPUs (e.g. GB10 unified memory), local non-colocated
    # engines crash due to vLLM V1 subprocess issues — remote is the only
    # working topology.  Set ``allow_remote_lora: true`` to keep remote mode.
    allow_remote_lora = bool(grpo_cfg.get("allow_remote_lora", False))
    if remote_requested and lora_rank > 0 and not allow_remote_lora:
        logger.warning(
            "grpo.vllm_server_url=%r requested with LoRA rank=%d. "
            "SkyRL remote engines do not support NCCL LoRA weight sync; "
            "falling back to local non-colocated vLLM engines. "
            "Set grpo.allow_remote_lora=true to keep remote mode "
            "(requires patch_skyrl_weight_sync.py for file-based sync).",
            requested_remote_url,
            lora_rank,
        )
        remote_requested = False
        vllm_mode = "server"
    elif remote_requested and lora_rank > 0 and allow_remote_lora:
        logger.info(
            "Remote vLLM + LoRA: using file-based weight sync "
            "(grpo.allow_remote_lora=true). Ensure patch_skyrl_weight_sync.py "
            "is applied and vLLM server supports /v1/load_lora_adapter."
        )

    run_engines_locally = not remote_requested
    colocate_all = run_engines_locally and vllm_mode in {"colocate", "local"}
    remote_urls = (
        [_normalize_remote_url(requested_remote_url)]
        if remote_requested
        else ["127.0.0.1:8001"]
    )

    return {
        "remote_vllm": remote_requested,
        "run_engines_locally": run_engines_locally,
        "colocate_all": colocate_all,
        "weight_sync_backend": "broadcast" if remote_requested else "nccl",
        "remote_inference_engine_urls": remote_urls,
    }


def _convert_grpo_data(
    data_path: str,
    output_dir: str,
    registry=None,
    drop_unresolved_registry_samples: bool = False,
    drop_static_challenges: bool = False,
    max_samples: Optional[int] = None,
    max_samples_per_challenge: Optional[int] = None,
    target_port_offset: int = 0,
    target_host_override: Optional[str] = None,
    fail_on_target_collisions: bool = False,
    prefer_registry_target: bool = False,
    difficulty_min: Optional[str] = None,
    difficulty_max: Optional[str] = None,
) -> str:
    """Convert our GRPO JSONL to SkyRL dataset format.

    SkyRL expects each sample to have:
      - prompt: list of message dicts (system + user)
      - Per-sample extras as flat top-level keys (ground_truth_flag, etc.)

    Our GRPO JSONL has:
      - messages: full trajectory (system, user, assistant, tool, ...)
      - ground_truth_flag: str
      - metadata: dict with optimal_steps, task_type, etc.

    We extract the prompt (system + user messages before first assistant)
    and flatten metadata as top-level keys for SkyRL extras.

    Args:
        data_path: Source GRPO JSONL path.
        output_dir: Output directory for converted JSONL.
        registry: Optional ChallengeRegistry for challenge ID normalization.
        drop_unresolved_registry_samples: If True and registry is provided,
            samples whose challenge ID cannot be resolved are dropped.
        drop_static_challenges: If True and registry is provided, samples
            whose resolved challenge has infra_type="static" are dropped.
            Static challenges have no running Docker service, so they waste
            compute during online GRPO training.
        max_samples: Optional cap on converted samples (after filtering).
        max_samples_per_challenge: Optional per-challenge cap for balancing.
        target_port_offset: Optional port offset applied to parsed target URLs.
            Useful for SSH-forwarded challenge ranges (e.g., 328xx -> 430xx).
        target_host_override: Optional host override for parsed target URLs.
        fail_on_target_collisions: If True, raise when multiple challenge IDs
            resolve to the same target URL.
        prefer_registry_target: If True, use registry-resolved target URL when
            available, even when a user message already contains a URL.
        difficulty_min: Optional minimum difficulty (inclusive). Requires registry.
            Samples below this difficulty are skipped. One of:
            very_easy, easy, medium, hard, expert, master.
        difficulty_max: Optional maximum difficulty (inclusive). Requires registry.
            Samples above this difficulty are skipped.

    Returns:
        Path to the converted JSONL file.
    """
    import jsonlines

    output_path = os.path.join(output_dir, "skyrl_grpo_data.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    # Validate difficulty bounds.
    min_rank: Optional[int] = None
    max_rank: Optional[int] = None
    if difficulty_min is not None:
        if difficulty_min not in _DIFFICULTY_RANK:
            raise ValueError(
                f"Invalid difficulty_min={difficulty_min!r}. "
                f"Must be one of: {_DIFFICULTY_ORDER}"
            )
        min_rank = _DIFFICULTY_RANK[difficulty_min]
    if difficulty_max is not None:
        if difficulty_max not in _DIFFICULTY_RANK:
            raise ValueError(
                f"Invalid difficulty_max={difficulty_max!r}. "
                f"Must be one of: {_DIFFICULTY_ORDER}"
            )
        max_rank = _DIFFICULTY_RANK[difficulty_max]
    if min_rank is not None and max_rank is not None and min_rank > max_rank:
        raise ValueError(
            f"difficulty_min={difficulty_min!r} is harder than "
            f"difficulty_max={difficulty_max!r}."
        )

    converted = 0
    skipped = 0
    skipped_static = 0
    skipped_difficulty = 0
    unresolved_counts: Dict[str, int] = {}
    missing_challenge_id = 0
    per_challenge_counts: Dict[str, int] = {}
    target_to_challenges: Dict[str, set[str]] = {}

    def _rewrite_target(raw_url: str) -> str:
        """Apply host/port overrides to a target URL."""
        try:
            parsed = urlparse(raw_url)
        except Exception:
            return raw_url
        if not parsed.scheme or not parsed.netloc:
            return raw_url

        host = target_host_override or parsed.hostname or ""
        port = parsed.port
        if port is not None and target_port_offset:
            port = port + int(target_port_offset)

        netloc = host
        if parsed.username:
            auth = parsed.username
            if parsed.password:
                auth += f":{parsed.password}"
            netloc = f"{auth}@{netloc}"
        if port is not None:
            netloc = f"{netloc}:{port}"

        return urlunparse(
            (
                parsed.scheme,
                netloc,
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )

    with jsonlines.open(data_path) as reader, jsonlines.open(output_path, "w") as writer:
        for sample in reader:
            if max_samples and converted >= int(max_samples):
                break
            messages = sample.get("messages", [])

            # Extract prompt: system + user messages before first assistant/tool
            prompt = []
            for msg in messages:
                role = msg.get("role", "")
                if role in ("system", "user"):
                    prompt.append({"role": role, "content": msg.get("content", "")})
                else:
                    break

            # Ensure prompt ends with user message (SkyRL requirement)
            if not prompt or prompt[-1]["role"] != "user":
                challenge = sample.get("metadata", {}).get("challenge", "")
                prompt.append({
                    "role": "user",
                    "content": (
                        f"Solve the CTF challenge{f': {challenge}' if challenge else ''}. "
                        "Find and capture the flag."
                    ),
                })

            # Flatten extras as top-level keys (SkyRL reads them as extras).
            # env_class is required — SkyRL dataset pops it to find the registered env.
            metadata = sample.get("metadata", {})

            # Extract target URL from user messages
            target = None
            for msg in messages:
                if msg.get("role") == "user":
                    urls = re.findall(r"https?://[^\s)]+", msg.get("content", ""))
                    if urls:
                        target = urls[0]
                        break
            if not target:
                target = metadata.get("target")

            # Resolve challenge ID against registry when available.
            challenge_id = metadata.get("challenge_id") or metadata.get("challenge")
            resolved_challenge_id = challenge_id
            if registry:
                if challenge_id:
                    resolved = registry.resolve_id(str(challenge_id))
                    if resolved is not None:
                        resolved_challenge_id = resolved
                    elif drop_unresolved_registry_samples:
                        skipped += 1
                        key = str(challenge_id)
                        unresolved_counts[key] = unresolved_counts.get(key, 0) + 1
                        continue
                elif drop_unresolved_registry_samples:
                    skipped += 1
                    missing_challenge_id += 1
                    continue

            # Drop static challenges (no Docker service to attack during online GRPO).
            if drop_static_challenges and registry and resolved_challenge_id:
                try:
                    _static_info = registry.get(str(resolved_challenge_id))
                    if _static_info.infra_type == "static":
                        skipped += 1
                        skipped_static += 1
                        continue
                except KeyError:
                    pass

            # Difficulty curriculum filter: skip challenges outside the allowed range.
            if (min_rank is not None or max_rank is not None) and registry and resolved_challenge_id:
                try:
                    _diff_info = registry.get(str(resolved_challenge_id))
                    diff_rank = _DIFFICULTY_RANK.get(_diff_info.difficulty)
                    if diff_rank is not None:
                        if min_rank is not None and diff_rank < min_rank:
                            skipped += 1
                            skipped_difficulty += 1
                            continue
                        if max_rank is not None and diff_rank > max_rank:
                            skipped += 1
                            skipped_difficulty += 1
                            continue
                except KeyError:
                    pass

            registry_target = None
            registry_category = None
            if registry and resolved_challenge_id:
                try:
                    info = registry.get(resolved_challenge_id)
                    registry_target = registry.get_target_url(resolved_challenge_id)
                    registry_category = info.category or None
                except KeyError:
                    registry_target = None

            # Prefer canonical registry target when configured (useful for
            # remote/tunneled runs where prompts may contain stale localhost URLs).
            if prefer_registry_target and registry_target:
                target = registry_target
            # Otherwise, only use registry target as fallback when no URL was parsed.
            elif not target and registry_target:
                target = registry_target
            if target:
                target = _rewrite_target(str(target))

            if max_samples_per_challenge and resolved_challenge_id:
                key = str(resolved_challenge_id)
                current = per_challenge_counts.get(key, 0)
                if current >= int(max_samples_per_challenge):
                    skipped += 1
                    continue

            # Category from registry (e.g. "crypto", "rev", "forensics", "web")
            # falls back to metadata.category if no registry match.
            category = registry_category or metadata.get("category")

            row = {
                "prompt": prompt,
                "env_class": "openctf",
                "ground_truth_flag": sample.get("ground_truth_flag"),
                "optimal_steps": sample.get("optimal_steps") or metadata.get("optimal_steps"),
                "challenge_id": resolved_challenge_id,
                "task_type": metadata.get("task_type", "ctf"),
                "target": target,
                "category": category,
            }

            writer.write(row)
            converted += 1
            if resolved_challenge_id:
                key = str(resolved_challenge_id)
                per_challenge_counts[key] = per_challenge_counts.get(key, 0) + 1
                if target:
                    target_to_challenges.setdefault(str(target), set()).add(key)

    if skipped:
        logger.warning(
            "Skipped %d/%d GRPO samples during conversion (registry filtering enabled=%s)",
            skipped,
            skipped + converted,
            bool(registry and drop_unresolved_registry_samples),
        )
    if unresolved_counts:
        top = sorted(unresolved_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        logger.warning("Top unresolved challenge IDs (sample count): %s", top)
    if missing_challenge_id:
        logger.warning(
            "Skipped %d samples with missing challenge_id/challenge metadata.",
            missing_challenge_id,
        )
    if skipped_static:
        logger.info(
            "Dropped %d static challenge samples (infra_type='static', no Docker service).",
            skipped_static,
        )
    if skipped_difficulty:
        logger.info(
            "Dropped %d samples by difficulty filter (min=%s, max=%s).",
            skipped_difficulty,
            difficulty_min,
            difficulty_max,
        )
    if converted == 0:
        raise ValueError(
            "No GRPO samples remained after conversion. "
            "Check challenge registry mappings or disable drop_unresolved_registry_samples."
        )

    if max_samples_per_challenge:
        logger.info(
            "Per-challenge cap active: max_samples_per_challenge=%s (kept %d challenges)",
            max_samples_per_challenge,
            len(per_challenge_counts),
        )
    collisions = {
        tgt: sorted(ids)
        for tgt, ids in target_to_challenges.items()
        if len(ids) > 1
    }
    if collisions:
        top = sorted(collisions.items(), key=lambda x: len(x[1]), reverse=True)[:10]
        logger.warning(
            "Detected %d target URL collisions (multiple challenge IDs share one target). "
            "This often indicates stale tunnel/port mapping. Top collisions: %s",
            len(collisions),
            top,
        )
        if fail_on_target_collisions:
            raise ValueError(
                "Target URL collisions detected during GRPO data conversion; "
                "provide a challenge target map (OPEN_CTF_TARGET_MAP_PATH / "
                "grpo.target_map_path) or disable fail_on_target_collisions."
            )

    logger.info("Converted %d GRPO samples → %s", converted, output_path)
    return output_path


def _resolve_skyrl_logger(report_to: str, output_dir: str) -> str:
    """Map our ``report_to`` config value to a SkyRL logger backend name.

    SkyRL tracking backends: wandb, mlflow, swanlab, tensorboard, console.
    ``"none"`` is not supported and is mapped to ``"console"``.
    ``"tensorboard"`` is supported natively; we also set the env var
    ``TENSORBOARD_LOGDIR`` so SkyRL writes to the correct directory.
    """
    value = str(report_to).strip().lower()

    _VALID_SKYRL_LOGGERS = {"wandb", "mlflow", "swanlab", "tensorboard", "console"}

    if value in ("none", "", "null"):
        return "console"

    if value == "tensorboard":
        tb_dir = os.path.join(output_dir, "tensorboard")
        try:
            os.makedirs(tb_dir, exist_ok=True)
        except OSError:
            # Output dir may not exist yet (unit tests use fake paths).
            # The directory will be created later by train_grpo().
            pass
        # SkyRL's TensorBoardLogger reads TENSORBOARD_LOGDIR from env.
        os.environ["TENSORBOARD_LOGDIR"] = tb_dir
        return "tensorboard"

    if value in _VALID_SKYRL_LOGGERS:
        return value

    logger.warning(
        "Unrecognized report_to=%r; falling back to 'console'. "
        "Valid options: %s",
        report_to,
        sorted(_VALID_SKYRL_LOGGERS),
    )
    return "console"


def _setup_persistent_logging(output_dir: str) -> None:
    """Configure file-based logging to ``{output_dir}/training.log``.

    Adds a FileHandler to the root logger so that all log output
    (including SkyRL, vLLM, and our own modules) is captured in a
    persistent file alongside the usual console output.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, "training.log")
    handler = logging.FileHandler(log_path, mode="a")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logging.getLogger().addHandler(handler)
    logger.info("Persistent training log: %s", log_path)


def _build_skyrl_config(
    model_path: str,
    output_dir: str,
    config: Dict[str, Any],
    data_path: str,
) -> Dict[str, Any]:
    """Build a SkyRL config dict matching SkyRLConfig dataclass schema.

    Uses SkyRLConfig dataclass defaults as the base, then overrides with
    our training-specific values. This ensures all required keys exist
    regardless of SkyRL version.
    """
    # Load SkyRL's default config as a base. SkyRL 0.3.1 uses a Hydra
    # YAML config (ppo_base_config.yaml) rather than a Python dataclass.
    skyrl_defaults = {}
    try:
        from dataclasses import asdict
        from skyrl_train.config.config import SkyRLConfig
        skyrl_defaults = asdict(SkyRLConfig())
    except (ImportError, ModuleNotFoundError):
        pass
    if not skyrl_defaults:
        try:
            import importlib.resources as pkg_resources
            from omegaconf import OmegaConf as _OC
            # Load the YAML base config from skyrl_train package
            cfg_dir = Path(
                pkg_resources.files("skyrl_train") / "config"
            )
            base_yaml = cfg_dir / "ppo_base_config.yaml"
            if base_yaml.exists():
                raw = _OC.load(base_yaml)
                skyrl_defaults = _OC.to_container(raw, resolve=False)
                logger.info("Loaded SkyRL base config from %s", base_yaml)
        except Exception as exc:
            logger.warning("Could not load SkyRL base config: %s", exc)

    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    grpo_cfg = config.get("grpo", {})
    output_cfg = config.get("output", {})
    lora_rank = _parse_lora_rank(lora_cfg)
    lora_target_modules, lora_exclude_modules = _parse_lora_modules(lora_cfg)
    topology = _resolve_generator_topology(grpo_cfg, lora_rank=lora_rank)
    remote_vllm = topology["remote_vllm"]
    requested_flash_attn = bool(grpo_cfg.get("flash_attn", False))
    enable_flash_attn = requested_flash_attn and _flash_attn_available()
    if requested_flash_attn and not enable_flash_attn:
        logger.warning(
            "flash_attn requested but unavailable in this environment; falling back to SDPA."
        )
    use_sample_packing = bool(grpo_cfg.get("use_sample_packing", False))
    if use_sample_packing and not enable_flash_attn:
        logger.warning(
            "use_sample_packing requested without flash_attn support; disabling sample packing."
        )
        use_sample_packing = False
    model_max_seq_length = _as_positive_int(
        "model.max_seq_length",
        model_cfg.get("max_seq_length"),
        8192,
    )
    max_completion_length = _as_positive_int(
        "grpo.max_completion_length",
        grpo_cfg.get("max_completion_length"),
        8192,
    )
    max_prompt_length = _as_positive_int(
        "grpo.max_prompt_length",
        grpo_cfg.get("max_prompt_length"),
        model_max_seq_length,
    )
    # Keep vLLM's max_model_len sized for actual rollout windows instead of the
    # model's full context by default (for example 262K Qwen max pos emb), which
    # can allocate excessive KV cache and OOM on otherwise-valid settings.
    vllm_headroom_tokens = _as_positive_int(
        "grpo.vllm_context_headroom_tokens",
        grpo_cfg.get("vllm_context_headroom_tokens"),
        1024,
    )
    min_required_vllm_len = max_prompt_length + max_completion_length
    default_vllm_max_model_len = min(
        model_max_seq_length,
        min_required_vllm_len + vllm_headroom_tokens,
    )
    if default_vllm_max_model_len < min_required_vllm_len:
        default_vllm_max_model_len = min_required_vllm_len
    vllm_max_model_len = _as_positive_int(
        "grpo.vllm_max_model_len",
        grpo_cfg.get("vllm_max_model_len"),
        default_vllm_max_model_len,
    )
    if vllm_max_model_len < min_required_vllm_len:
        logger.warning(
            "grpo.vllm_max_model_len=%d is smaller than max_prompt_length + "
            "max_completion_length (%d + %d = %d); overriding to %d.",
            vllm_max_model_len,
            max_prompt_length,
            max_completion_length,
            min_required_vllm_len,
            min_required_vllm_len,
        )
        vllm_max_model_len = min_required_vllm_len
    vllm_language_model_only = bool(grpo_cfg.get("vllm_language_model_only", False))
    num_generations = _as_positive_int(
        "grpo.num_generations",
        grpo_cfg.get("num_generations"),
        8,
    )
    max_num_seqs = _as_positive_int(
        "grpo.max_num_seqs",
        grpo_cfg.get("max_num_seqs"),
        max(8, num_generations * 2),
    )
    if max_num_seqs < num_generations:
        logger.warning(
            "grpo.max_num_seqs=%d is smaller than num_generations=%d; "
            "overriding max_num_seqs to %d.",
            max_num_seqs,
            num_generations,
            num_generations,
        )
        max_num_seqs = num_generations
    default_batched_tokens = min(
        32768,
        max(max_prompt_length, num_generations * max(1024, max_completion_length // 2)),
    )
    max_num_batched_tokens = _as_positive_int(
        "grpo.max_num_batched_tokens",
        grpo_cfg.get("max_num_batched_tokens"),
        default_batched_tokens,
    )
    if max_num_batched_tokens < max_prompt_length:
        logger.warning(
            "grpo.max_num_batched_tokens=%d is smaller than max_prompt_length=%d; "
            "overriding to %d.",
            max_num_batched_tokens,
            max_prompt_length,
            max_prompt_length,
        )
        max_num_batched_tokens = max_prompt_length
    max_prefill_capacity = max_prompt_length * max_num_seqs
    if max_num_batched_tokens > max_prefill_capacity:
        logger.warning(
            "grpo.max_num_batched_tokens=%d exceeds max_prefill_capacity=%d "
            "(max_prompt_length * max_num_seqs); clamping.",
            max_num_batched_tokens,
            max_prefill_capacity,
        )
        max_num_batched_tokens = max_prefill_capacity

    num_inference_engines = _as_positive_int(
        "grpo.num_inference_engines",
        grpo_cfg.get("num_inference_engines"),
        1,
    )
    inference_engine_tensor_parallel_size = _as_positive_int(
        "grpo.inference_engine_tensor_parallel_size",
        grpo_cfg.get("inference_engine_tensor_parallel_size"),
        1,
    )
    inference_engine_pipeline_parallel_size = _as_positive_int(
        "grpo.inference_engine_pipeline_parallel_size",
        grpo_cfg.get("inference_engine_pipeline_parallel_size"),
        1,
    )
    inference_engine_data_parallel_size = _as_positive_int(
        "grpo.inference_engine_data_parallel_size",
        grpo_cfg.get("inference_engine_data_parallel_size"),
        1,
    )
    max_tool_calling_iterations = _as_positive_int(
        "grpo.max_tool_calling_iterations",
        grpo_cfg.get("max_tool_calling_iterations"),
        15,
    )
    max_env_workers = _as_positive_int(
        "grpo.max_env_workers",
        grpo_cfg.get("max_env_workers"),
        32,
    )
    use_ref_model = bool(grpo_cfg.get("beta", 0.0) > 0.0 or grpo_cfg.get("use_kl_in_reward", False))
    policy_num_gpus_per_node = _as_positive_int(
        "grpo.policy_num_gpus_per_node",
        grpo_cfg.get("policy_num_gpus_per_node"),
        1,
    )
    policy_num_nodes = _as_positive_int(
        "grpo.policy_num_nodes",
        grpo_cfg.get("policy_num_nodes"),
        1,
    )
    critic_model_path = grpo_cfg.get("critic_model_path")
    if critic_model_path:
        critic_num_gpus_per_node = _as_positive_int(
            "grpo.critic_num_gpus_per_node",
            grpo_cfg.get("critic_num_gpus_per_node"),
            1,
        )
    else:
        critic_num_gpus_per_node = 0
    critic_num_nodes = _as_positive_int(
        "grpo.critic_num_nodes",
        grpo_cfg.get("critic_num_nodes"),
        1,
    )
    ref_num_nodes = _as_positive_int(
        "grpo.ref_num_nodes",
        grpo_cfg.get("ref_num_nodes"),
        policy_num_nodes,
    )
    ref_num_gpus_per_node = _as_positive_int(
        "grpo.ref_num_gpus_per_node",
        grpo_cfg.get("ref_num_gpus_per_node"),
        policy_num_gpus_per_node,
    )
    if not use_ref_model:
        ref_num_gpus_per_node = 0
    colocate_policy_ref = bool(grpo_cfg.get("colocate_policy_ref", True)) and use_ref_model

    visible_gpu_count = _detect_visible_gpu_count()
    if (
        visible_gpu_count is not None
        and visible_gpu_count > 0
        and topology["run_engines_locally"]
        and not topology["colocate_all"]
    ):
        policy_gpus = policy_num_nodes * policy_num_gpus_per_node
        ref_gpus = ref_num_nodes * ref_num_gpus_per_node if use_ref_model else 0
        critic_gpus = critic_num_nodes * critic_num_gpus_per_node if critic_model_path else 0
        gpus_per_engine = (
            inference_engine_tensor_parallel_size
            * inference_engine_pipeline_parallel_size
            * inference_engine_data_parallel_size
        )
        required_gpus = policy_gpus + ref_gpus + critic_gpus + (num_inference_engines * gpus_per_engine)
        if required_gpus > visible_gpu_count:
            explicit_num_engines = "num_inference_engines" in grpo_cfg
            available_for_inference = max(0, visible_gpu_count - policy_gpus - ref_gpus - critic_gpus)
            max_auto_engines = available_for_inference // max(1, gpus_per_engine)
            if not explicit_num_engines and max_auto_engines > 0:
                logger.warning(
                    "Local non-colocated topology requests %d GPUs but only %d are visible. "
                    "Auto-adjusting num_inference_engines from %d to %d.",
                    required_gpus,
                    visible_gpu_count,
                    num_inference_engines,
                    max_auto_engines,
                )
                num_inference_engines = max_auto_engines
            else:
                logger.warning(
                    "Local non-colocated topology requests %d GPUs (%d policy + %d ref + %d critic + "
                    "%d inference) but only %d are visible. Training may stall or OOM.",
                    required_gpus,
                    policy_gpus,
                    ref_gpus,
                    critic_gpus,
                    num_inference_engines * gpus_per_engine,
                    visible_gpu_count,
                )
    chat_template_name = grpo_cfg.get("chat_template", None)
    chat_template_kwargs = grpo_cfg.get("chat_template_kwargs", {})
    step_wise_trajectories = bool(grpo_cfg.get("step_wise_trajectories", False))
    allow_step_wise_with_custom_template = bool(
        grpo_cfg.get("allow_step_wise_with_custom_chat_template", False)
    )
    default_logprobs = None if remote_vllm else 0
    train_logprobs = grpo_cfg.get("logprobs", default_logprobs)
    eval_logprobs = grpo_cfg.get("eval_logprobs", train_logprobs)

    # SkyRL's multi-turn generator does not support response-logprob bookkeeping
    # when a custom chat template is used. Enforce this at config build time
    # so every benchmark/model path gets the same safe behavior.
    if chat_template_name and train_logprobs is not None:
        logger.warning(
            "Custom chat_template=%r set with logprobs=%r; forcing "
            "generator.sampling_params.logprobs=None for SkyRL compatibility.",
            chat_template_name,
            train_logprobs,
        )
        train_logprobs = None
    if chat_template_name and eval_logprobs is not None:
        logger.warning(
            "Custom chat_template=%r set with eval_logprobs=%r; forcing "
            "generator.eval_sampling_params.logprobs=None for SkyRL compatibility.",
            chat_template_name,
            eval_logprobs,
        )
        eval_logprobs = None

    # SkyRL currently rejects step-wise trajectories with custom chat templates.
    # Apply one centralized compatibility policy for all model/config combinations.
    if (
        chat_template_name
        and step_wise_trajectories
        and not allow_step_wise_with_custom_template
    ):
        if bool(grpo_cfg.get("step_wise_strict_compat", False)):
            raise ValueError(
                "grpo.step_wise_trajectories=true is incompatible with "
                f"grpo.chat_template={chat_template_name!r} in current SkyRL. "
                "Set grpo.step_wise_trajectories=false, remove grpo.chat_template, "
                "or set grpo.step_wise_strict_compat=false to auto-disable."
            )
        logger.warning(
            "grpo.step_wise_trajectories=true is incompatible with custom "
            "chat_template=%r in current SkyRL; auto-disabling step-wise "
            "trajectories for this run.",
            chat_template_name,
        )
        step_wise_trajectories = False
    elif (
        chat_template_name
        and step_wise_trajectories
        and allow_step_wise_with_custom_template
    ):
        logger.warning(
            "Using grpo.allow_step_wise_with_custom_chat_template=true with "
            "chat_template=%r. Ensure SkyRL includes the step-wise + custom "
            "chat-template compatibility fix.",
            chat_template_name,
        )

    # Detect transformer layer class for FSDP wrapping.
    # model._no_split_modules returns a set on some architectures (e.g. Llama),
    # which triggers a set-indexing bug in SkyRL's apply_fsdp2.
    # We auto-detect and pass the class name as a string to avoid this.
    _ARCH_TO_LAYER_CLS = {
        "LlamaForCausalLM": "LlamaDecoderLayer",
        "Qwen2ForCausalLM": "Qwen2DecoderLayer",
        "Qwen3ForCausalLM": "Qwen3DecoderLayer",
        "Qwen3_5ForConditionalGeneration": "Qwen3_5DecoderLayer",
        "MistralForCausalLM": "MistralDecoderLayer",
        "GptOssForCausalLM": "GptOssDecoderLayer",
    }
    auto_cfg = None
    transformer_layer_cls = None
    try:
        from transformers import AutoConfig
        auto_cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        arch = getattr(auto_cfg, "architectures", [None])[0]
        transformer_layer_cls = _ARCH_TO_LAYER_CLS.get(arch)
        if not transformer_layer_cls and arch:
            # Fallback: guess from architecture name
            base = arch.replace("ForCausalLM", "")
            transformer_layer_cls = f"{base}DecoderLayer"
    except Exception:
        pass
    if auto_cfg is not None:
        _validate_qwen3_5_runtime_dependencies(auto_cfg, grpo_cfg)

    # Reference model path for KL divergence. Defaults to the policy model
    # (standard GRPO), but can be overridden for distillation.
    ref_model_path = config.get("ref_model_path", model_path)
    eps_clip_low = grpo_cfg.get("epsilon_low", 0.2)
    eps_clip_high = grpo_cfg.get("epsilon_high", eps_clip_low)

    skyrl_config = {
        # Data
        "data": {
            "train_data": [data_path],
            "val_data": [],
        },

        # Trainer — includes all fields required by SkyRL 0.3.1 validate_cfg
        "trainer": {
            "strategy": "fsdp2",
            "bf16": True,
            "gradient_checkpointing": True,
            "gradient_checkpointing_use_reentrant": False,
            "seed": 42,
            "sequence_parallel_backend": "ulysses",
            "epochs": grpo_cfg.get("epochs", 1),
            "update_epochs_per_batch": 1,
            "train_batch_size": grpo_cfg.get("batch_size", 1),
            "policy_mini_batch_size": grpo_cfg.get("batch_size", 1),
            "critic_mini_batch_size": grpo_cfg.get("batch_size", 1),
            "micro_train_batch_size_per_gpu": 1,
            "micro_forward_batch_size_per_gpu": 1,
            "max_prompt_length": max_prompt_length,
            "use_sample_packing": use_sample_packing,
            "eval_batch_size": 1,
            "eval_interval": -1,
            "flash_attn": enable_flash_attn,
            "disable_fast_tokenizer": False,
            "update_ref_every_epoch": False,
            "resume_mode": None,
            "resume_path": None,
            "ckpt_path": output_dir,
            "max_ckpts_to_keep": -1,
            "hf_save_interval": -1,
            "ckpt_interval": output_cfg.get("save_steps", 50),
            "export_path": os.path.join(output_dir, "final"),
            "eval_before_train": False,
            "project_name": "open-ctf",
            "run_name": "grpo",
            "logger": _resolve_skyrl_logger(
                output_cfg.get("report_to", "tensorboard"), output_dir
            ),
            "dump_data_batch": False,
            "dump_eval_results": False,
            "target_modules": None,
            "exclude_modules": None,
            "rope_scaling": None,
            "rope_theta": None,

            "placement": {
                # Non-colocated mode allows trainer and local engines to use
                # separate GPUs while staying on SkyRL's supported LoRA path.
                "colocate_all": topology["colocate_all"],
                "colocate_policy_ref": colocate_policy_ref,
                "policy_num_nodes": policy_num_nodes,
                "policy_num_gpus_per_node": policy_num_gpus_per_node,
                "critic_num_nodes": critic_num_nodes,
                "critic_num_gpus_per_node": critic_num_gpus_per_node,
                "ref_num_nodes": ref_num_nodes,
                "ref_num_gpus_per_node": ref_num_gpus_per_node,
            },

            "fully_async": {
                "max_staleness_steps": 4,
                "num_parallel_generation_workers": 1,
            },

            "policy": {
                "model": {
                    "path": model_path,
                    "lora": {
                        "rank": lora_rank,
                        "alpha": lora_cfg.get("alpha", 128),
                        "dropout": lora_cfg.get("dropout", 0.0),
                        "target_modules": lora_target_modules,
                        "exclude_modules": lora_exclude_modules,
                        "lora_sync_path": os.path.join(output_dir, "lora_sync"),
                        "init_method": "kaiming",
                    },
                    "config_kwargs": {},
                },
                "model_config_kwargs": {},
                "fsdp_config": {
                    "cpu_offload": False,
                    "reshard_after_forward": True,
                    "fsdp_size": -1,
                    "wrap_policy": (
                        {"transformer_layer_cls_to_wrap": [transformer_layer_cls]}
                        if transformer_layer_cls
                        else {}
                    ),
                },
                "sequence_parallel_size": 1,
                "use_torch_compile": False,
                "record_memory": False,
                "optimizer_config": {
                    "lr": grpo_cfg.get("learning_rate", 5e-6),
                    "adam_betas": [0.9, 0.999],
                    "weight_decay": grpo_cfg.get("weight_decay", 0.0),
                    "max_grad_norm": grpo_cfg.get("max_grad_norm", 5.0),
                    "offload_after_step": True,
                    "num_warmup_steps": 0,
                    "scheduler": "constant_with_warmup",
                },
            },

            "ref": {
                "model": {
                    "path": ref_model_path,
                    "config_kwargs": {},
                },
                "model_config_kwargs": {},
                "sequence_parallel_size": 1,
                "fsdp_config": {
                    # CPU offload ref model when KL beta is 0 (ref logprobs unused).
                    # Saves ~16-32GB GPU memory for 8B+ models on single-GPU setups.
                    "cpu_offload": grpo_cfg.get("beta", 0.0) == 0.0,
                    "reshard_after_forward": True,
                    "fsdp_size": -1,
                },
            },

            "critic": {
                "model": {
                    "path": critic_model_path,
                    "lora": {
                        "rank": 0,
                        "alpha": 16,
                        "dropout": 0,
                        "target_modules": lora_target_modules,
                        "exclude_modules": lora_exclude_modules,
                        "init_method": "kaiming",
                    },
                },
                "model_config_kwargs": {},
                "sequence_parallel_size": 1,
                "fsdp_config": {
                    "cpu_offload": False,
                    "reshard_after_forward": True,
                    "fsdp_size": -1,
                },
                "optimizer_config": {
                    "lr": 5e-6,
                    "adam_betas": [0.9, 0.999],
                    "weight_decay": 0.01,
                    "max_grad_norm": 1.0,
                    "offload_after_step": True,
                    "num_warmup_steps": 0,
                    "scheduler": "constant_with_warmup",
                },
            },

            "algorithm": {
                "advantage_estimator": grpo_cfg.get("advantage_estimator", "rloo_n"),
                "policy_loss_type": "regular",
                "kl_loss_coef": grpo_cfg.get("beta", 0.0),
                "use_kl_loss": grpo_cfg.get("beta", 0.0) > 0,
                "use_kl_in_reward": False,
                "kl_ctrl": {
                    "type": "fixed",
                    "kl_target": 0.1,
                    "horizon": 10000,
                },
                "kl_estimator_type": "k3",
                "use_kl_estimator_k3": False,
                "use_abs_kl": False,
                "use_entropy_loss": False,
                "entropy_loss_coef": 0.01,
                "advantage_batch_normalize": False,
                "value_head_prefix": "value_head",
                "loss_reduction": "token_mean",
                "grpo_norm_by_std": True,
                "zero_variance_filter": bool(grpo_cfg.get("zero_variance_filter", False)),
                "lambd": 1.0,
                "gamma": 1.0,
                "eps_clip_low": eps_clip_low,
                "eps_clip_high": eps_clip_high,
                "clip_ratio_c": 3.0,
                "tis_imp_ratio_cap": -1.0,
                "use_tis": False,
                "sapo": {"tau_pos": 1.0, "tau_neg": 1.05},
                "value_clip": 0.2,
                "dynamic_sampling": {
                    "type": grpo_cfg.get("dynamic_sampling", {}).get("type", None),
                    "max_sample_batches": _as_positive_int(
                        "grpo.dynamic_sampling.max_sample_batches",
                        grpo_cfg.get("dynamic_sampling", {}).get("max_sample_batches"),
                        30,
                    ),
                    "min_replace_ratio": 0.3,
                },
                "clip_cov": {
                    "clip_ratio": 0.0002,
                    "clip_cov_lb": 1.0,
                    "clip_cov_ub": 5.0,
                },
                "kl_cov": {
                    "kl_cov_frac": 0.2,
                    "ppo_kl_coef": 1.0,
                },
                "cispo": {
                    "cispo_eps_clip_low": 0,
                    "cispo_eps_clip_high": 5,
                },
            },
        },

        # Generator (vLLM inference) — all fields from SkyRL 0.3.1
        "generator": {
            "model_name": model_path,
            "model_dtype": "bfloat16",
            "run_engines_locally": topology["run_engines_locally"],
            "num_inference_engines": num_inference_engines,
            "backend": "vllm",
            "weight_sync_backend": topology["weight_sync_backend"],
            "weight_transfer_threshold_cuda_ipc_GB": 1.0,
            "inference_engine_tensor_parallel_size": inference_engine_tensor_parallel_size,
            "inference_engine_pipeline_parallel_size": inference_engine_pipeline_parallel_size,
            "inference_engine_expert_parallel_size": 1,
            "inference_engine_data_parallel_size": inference_engine_data_parallel_size,
            "n_samples_per_prompt": num_generations,
            "async_engine": True,
            "batched": False,
            "max_input_length": max_prompt_length,
            "vllm_v1_disable_multiproc": True,
            "enable_prefix_caching": bool(grpo_cfg.get("enable_prefix_caching", True)),
            "enable_chunked_prefill": bool(grpo_cfg.get("enable_chunked_prefill", True)),
            "max_num_batched_tokens": max_num_batched_tokens,
            "enforce_eager": bool(grpo_cfg.get("enforce_eager", True)),
            "fully_sharded_loras": False,
            "enable_ray_prometheus_stats": False,
            "gpu_memory_utilization": grpo_cfg.get("gpu_memory_utilization", 0.4),
            "max_num_seqs": max_num_seqs,
            "remote_inference_engine_urls": topology["remote_inference_engine_urls"],
            "enable_http_endpoint": False,
            "http_endpoint_host": "127.0.0.1",
            "http_endpoint_port": 8000,
            "served_model_name": None,
            "max_turns": max_tool_calling_iterations,
            "chat_template": {
                "source": "name",
                "name_or_path": chat_template_name,
            },
            # Pass enable_thinking, reasoning_effort, etc. to tokenizer.
            # SkyRL unpacks these in every apply_chat_template() call.
            # For Qwen3/Qwen3.5: {"enable_thinking": true} activates
            # <think>...</think> generation and correct loss masking.
            "chat_template_kwargs": chat_template_kwargs,
            # max_model_len limits vLLM's KV cache allocation.  Without this,
            # vLLM uses the model's max_position_embeddings (e.g. 262144) which
            # can exceed available GPU memory.  Set to max_input_length + headroom.
            "engine_init_kwargs": {
                "max_model_len": vllm_max_model_len,
                "language_model_only": vllm_language_model_only,
            },
            "override_existing_update_group": "auto",
            "sampling_params": {
                "max_generate_length": max_completion_length,
                "repetition_penalty": grpo_cfg.get("repetition_penalty", 1.0),
                "temperature": 1.0,
                "top_p": 0.95,
                "min_p": 0.0,
                "top_k": -1,
                # Local default is 0, remote default is None. If a custom
                # chat template is configured we force None above.
                "logprobs": train_logprobs,
                "stop": None,
            },
            "use_conversation_multi_turn": True,
            "append_eos_token_after_stop_str_in_multi_turn": True,
            "eval_sampling_params": {
                "max_generate_length": max_completion_length,
                "repetition_penalty": grpo_cfg.get("repetition_penalty", 1.0),
                "temperature": 0.6,
                "top_p": 0.95,
                "min_p": 0.0,
                "top_k": -1,
                "logprobs": eval_logprobs,
                "stop": None,
            },
            "eval_n_samples_per_prompt": 1,
            "zero_reward_on_non_stop": False,
            "apply_overlong_filtering": False,
            "rope_scaling": None,
            "rope_theta": None,
            "step_wise_trajectories": step_wise_trajectories,
        },

        # Environment
        "environment": {
            "env_class": "openctf",
            "skyrl_gym": {
                "max_env_workers": max_env_workers,
            },
        },
    }

    # Merge with SkyRL defaults to ensure all required keys exist.
    # Our overrides take precedence over defaults.
    if skyrl_defaults:
        # Strip Hydra-specific keys that contain OmegaConf interpolations
        # (e.g. ${deepspeed_config.train}) — we don't use DeepSpeed/Megatron.
        _HYDRA_KEYS = {"defaults", "deepspeed_config", "megatron_config"}
        for k in _HYDRA_KEYS:
            skyrl_defaults.pop(k, None)

        # Strip any values containing OmegaConf interpolation syntax
        def _strip_interpolations(d):
            """Remove dict values that are OmegaConf interpolation strings."""
            if not isinstance(d, dict):
                return d
            cleaned = {}
            for k, v in d.items():
                if isinstance(v, str) and "${" in v:
                    continue  # Skip interpolation references
                elif isinstance(v, dict):
                    cleaned[k] = _strip_interpolations(v)
                else:
                    cleaned[k] = v
            return cleaned
        skyrl_defaults = _strip_interpolations(skyrl_defaults)

        def _deep_merge(base, override):
            """Recursively merge override into base dict."""
            result = dict(base)
            for k, v in override.items():
                if k in result and isinstance(result[k], dict) and isinstance(v, dict):
                    result[k] = _deep_merge(result[k], v)
                else:
                    result[k] = v
            return result
        skyrl_config = _deep_merge(skyrl_defaults, skyrl_config)

    # Remove any remaining Hydra/OmegaConf interpolation keys from final config
    for k in ("defaults", "deepspeed_config", "megatron_config"):
        skyrl_config.pop(k, None)

    # vLLM 0.16 rejects SamplingParams.additional_kwargs; older SkyRL defaults
    # can reintroduce it during deep-merge even when our overrides omit it.
    generator_cfg = skyrl_config.get("generator", {})
    for key in ("sampling_params", "eval_sampling_params"):
        sampling = generator_cfg.get(key)
        if isinstance(sampling, dict):
            sampling.pop("additional_kwargs", None)

    return skyrl_config


def train_grpo(
    model_path: str,
    data_path: str,
    output_dir: str,
    config: Dict[str, Any],
    resume_from: Optional[str] = None,
    challenge_registry: Optional[str] = None,
    agent_class: Optional[str] = None,
) -> str:
    """Run online GRPO training via SkyRL.

    Uses OpenCTFTextEnv (SkyRL-Gym BaseTextEnv subclass) with ToolExecutor
    for direct tool execution (no HTTP server needed).

    The reward function is reconstructed inside each SkyRL env instance
    from ``config["reward"]`` (a serializable dict). This avoids passing
    non-serializable callables through Ray.

    Args:
        model_path: Path to the SFT model (merged directory).
        data_path: Path to JSONL GRPO data with ground_truth_flag.
        output_dir: Directory for checkpoints and final model.
        config: Merged config dict from training.yaml.
        resume_from: Optional checkpoint path to resume from.

    Returns:
        Path to the saved final model directory.
    """
    model_path = _resolve_vllm_ready_model_path(model_path)

    # Set up persistent file logging before any other output.
    os.makedirs(output_dir, exist_ok=True)
    _setup_persistent_logging(output_dir)

    logger.info("=" * 60)
    logger.info("GRPO TRAINING (SkyRL)")
    logger.info("  Model:  %s", model_path)
    logger.info("  Data:   %s", data_path)
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 60)

    # 1. Convert data to SkyRL format
    grpo_cfg = config.get("grpo", {})
    registry = None
    target_map_path = (
        os.getenv("OPEN_CTF_TARGET_MAP_PATH")
        or grpo_cfg.get("target_map_path")
        or None
    )
    if challenge_registry:
        from open_ctf.challenges.registry import ChallengeRegistry
        registry = ChallengeRegistry(challenge_registry)
        if target_map_path:
            strict_map = bool(
                int(os.getenv("OPEN_CTF_TARGET_MAP_STRICT", "0"))
            ) or bool(grpo_cfg.get("target_map_strict", False))
            registry.load_target_overrides(str(target_map_path), strict=strict_map)
    drop_unresolved = bool(
        grpo_cfg.get("drop_unresolved_registry_samples", True)
    )
    port_offset = int(
        os.getenv(
            "OPEN_CTF_TARGET_PORT_OFFSET",
            str(grpo_cfg.get("target_port_offset", 0)),
        )
    )
    host_override = (
        os.getenv("OPEN_CTF_TARGET_HOST_OVERRIDE")
        or grpo_cfg.get("target_host_override")
        or None
    )
    prefer_registry_target = bool(
        grpo_cfg.get("prefer_registry_target", bool(target_map_path))
    )
    converted_data = _convert_grpo_data(
        data_path,
        output_dir,
        registry=registry,
        drop_unresolved_registry_samples=drop_unresolved,
        drop_static_challenges=bool(grpo_cfg.get("drop_static_challenges", True)),
        max_samples=grpo_cfg.get("max_samples"),
        max_samples_per_challenge=grpo_cfg.get("max_samples_per_challenge"),
        target_port_offset=port_offset,
        target_host_override=host_override,
        fail_on_target_collisions=bool(grpo_cfg.get("fail_on_target_collisions", False)),
        prefer_registry_target=prefer_registry_target,
        difficulty_min=grpo_cfg.get("difficulty_min"),
        difficulty_max=grpo_cfg.get("difficulty_max"),
    )

    # 2. Build SkyRL config
    skyrl_config = _build_skyrl_config(model_path, output_dir, config, converted_data)

    if resume_from:
        skyrl_config["trainer"]["resume_path"] = resume_from
        skyrl_config["trainer"]["resume_mode"] = "from_path"

    # 3. Write config for reference
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "skyrl_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(skyrl_config, f, default_flow_style=False)
    logger.info("SkyRL config written to %s", config_path)

    # 4. Launch SkyRL training
    reward_config = _resolve_reward_config(config)
    use_new_inference = bool(grpo_cfg.get("use_new_inference", False))
    allow_qwen35_new_inference = bool(
        grpo_cfg.get("allow_new_inference_for_qwen35", False)
    )
    if use_new_inference and _should_force_legacy_inference(
        model_path,
        allow_qwen35_new_inference=allow_qwen35_new_inference,
    ):
        use_new_inference = False

    # Trajectory logging: pass output_dir through env kwargs (Ray-serializable string).
    logging_cfg = config.get("grpo_logging", {})
    enable_trajectory_logging = bool(
        logging_cfg.get("enable_trajectory_logging", True)
    )
    trajectory_output_dir = output_dir if enable_trajectory_logging else None

    # Agent class from CLI flag > config file > None (DefaultStepAgent)
    resolved_agent_class = agent_class or grpo_cfg.get("agent_class")
    resolved_agent_kwargs = grpo_cfg.get("agent_kwargs", {})
    pytorch_cuda_alloc_conf = grpo_cfg.get("pytorch_cuda_alloc_conf")
    try:
        _run_skyrl_training(
            skyrl_config, reward_config,
            agent_class=resolved_agent_class,
            agent_kwargs=resolved_agent_kwargs,
            use_new_inference=use_new_inference,
            trajectory_output_dir=trajectory_output_dir,
            pytorch_cuda_alloc_conf=pytorch_cuda_alloc_conf,
        )
    except ImportError as e:
        logger.error(
            "SkyRL not installed. Install with: pip install skyrl-train skyrl-gym ray[default] vllm"
        )
        raise

    # Save challenge scoreboard after training completes.
    if trajectory_output_dir:
        try:
            from open_ctf.training.trajectory_logger import TrajectoryLogger
            tl = TrajectoryLogger(trajectory_output_dir)
            tl.save_scoreboard()
        except Exception as exc:
            logger.warning("Failed to save final scoreboard: %s", exc)

    logger.info("GRPO training complete. Output: %s", output_dir)

    final_dir = os.path.join(output_dir, "final")
    if os.path.exists(final_dir):
        return final_dir
    return output_dir


def _run_skyrl_training(
    config: Dict[str, Any],
    reward_config: Dict[str, Any],
    agent_class: Optional[str] = None,
    agent_kwargs: Optional[Dict[str, Any]] = None,
    use_new_inference: bool = False,
    trajectory_output_dir: Optional[str] = None,
    pytorch_cuda_alloc_conf: Optional[str] = None,
) -> None:
    """Launch SkyRL training with the given config.

    Uses SkyRL's Python API (BasePPOExp). The environment is registered
    inside a Ray remote task so Ray workers can access it.

    Args:
        config: SkyRL config dict.
        reward_config: Serializable reward weight dict (reconstructed
            into CTFReward inside each env instance).
        agent_class: Dotted path to a StepAgent class (Ray-safe string).
        agent_kwargs: Dict of primitives for the StepAgent constructor.
        trajectory_output_dir: Output directory for trajectory JSONL logs.
            Passed through as a string (Ray-serializable) to each env instance.

    Key: exp.run() already calls asyncio.run() internally -- do NOT wrap
    in another asyncio.run().
    """
    import ray
    from omegaconf import OmegaConf

    # Convert dict to OmegaConf DictConfig (SkyRL expects this)
    cfg = OmegaConf.create(config)

    # Import SkyRL utilities
    from skyrl_train.entrypoints.main_base import validate_cfg
    from skyrl_train.utils import initialize_ray

    # Validate config against SkyRLConfig schema
    validate_cfg(cfg)

    # SkyRL's Ray actors should run vLLM V1 with in-process worker handling.
    # Keep multiprocess disabled to avoid Ray actor engine bootstrap issues.
    os.environ.setdefault("VLLM_USE_V1", "1")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if pytorch_cuda_alloc_conf:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(pytorch_cuda_alloc_conf)
        logger.info("Set PYTORCH_CUDA_ALLOC_CONF=%s", pytorch_cuda_alloc_conf)
    # Use SkyRL's new HTTP inference layer when requested.
    # This avoids legacy Ray-wrapped vLLM LoRA startup issues on Qwen3.5.
    os.environ["_SKYRL_USE_NEW_INFERENCE"] = "1" if use_new_inference else "0"
    if use_new_inference:
        logger.info("Enabled SkyRL new inference layer (_SKYRL_USE_NEW_INFERENCE=1)")

    # Initialize Ray cluster
    initialize_ray(cfg)

    # Register env and run training inside a Ray remote task.
    # This ensures the env registration is visible to Ray workers.
    @ray.remote(num_cpus=1)
    def _skyrl_entrypoint(
        cfg_dict,
        reward_config,
        agent_class,
        agent_kwargs,
        use_new_inference,
        trajectory_output_dir,
        pytorch_cuda_alloc_conf,
    ):
        import os as _os
        _os.environ["VLLM_USE_V1"] = "1"
        _os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        if pytorch_cuda_alloc_conf:
            _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = str(pytorch_cuda_alloc_conf)
        _os.environ["_SKYRL_USE_NEW_INFERENCE"] = "1" if use_new_inference else "0"

        from omegaconf import OmegaConf
        from skyrl_gym.envs import register
        from open_ctf.envs.skyrl.openctf_env import OpenCTFTextEnv
        from skyrl_train.entrypoints.main_base import BasePPOExp

        cfg = OmegaConf.create(cfg_dict)

        # Register OpenCTFTextEnv with serializable kwargs only.
        # reward_config is a plain dict of floats -- JSON-safe for SkyRL's
        # EnvSpec._check_can_jsonify(). The env reconstructs CTFReward
        # from this config in __init__().
        # agent_class is a dotted path string (Ray-safe).
        # agent_kwargs is a dict of primitives (Ray-safe).
        env_kwargs = {
            "reward_config": reward_config,
            # Keep env-side guard aligned with generator max_turns, even though
            # SkyRL also injects max_turns per-sample via env_extras.
            "max_turns": int(getattr(cfg.generator, "max_turns", 15)),
            # Step-wise trajectory rewards: when enabled, per-step rewards
            # include small format-compliance and phase-progression signals.
            "step_wise_trajectories": bool(
                getattr(cfg.generator, "step_wise_trajectories", False)
            ),
        }
        if agent_class:
            env_kwargs["agent_class"] = agent_class
        if agent_kwargs:
            env_kwargs["agent_kwargs"] = agent_kwargs
        if trajectory_output_dir:
            env_kwargs["trajectory_output_dir"] = trajectory_output_dir

        register(
            id="openctf",
            entry_point=OpenCTFTextEnv,
            kwargs=env_kwargs,
        )

        exp = BasePPOExp(cfg)
        exp.run()  # Already calls asyncio.run() internally

    # Convert back to dict for serialization through Ray
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    ray.get(
        _skyrl_entrypoint.remote(
            cfg_dict,
            reward_config,
            agent_class,
            agent_kwargs or {},
            use_new_inference,
            trajectory_output_dir,
            pytorch_cuda_alloc_conf,
        )
    )
