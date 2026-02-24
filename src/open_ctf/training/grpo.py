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
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_CONFIGS_DIR = _PROJECT_ROOT / "configs" / "skyrl"


def _convert_grpo_data(data_path: str, output_dir: str, registry=None) -> str:
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

    Returns:
        Path to the converted JSONL file.
    """
    import jsonlines

    output_path = os.path.join(output_dir, "skyrl_grpo_data.jsonl")
    os.makedirs(output_dir, exist_ok=True)

    converted = 0
    with jsonlines.open(data_path) as reader, jsonlines.open(output_path, "w") as writer:
        for sample in reader:
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
                    urls = re.findall(r'http://localhost:\d+', msg.get("content", ""))
                    if urls:
                        target = urls[0]
                        break
            if not target:
                target = metadata.get("target")

            # Registry fallback: if no target in prompt/metadata, check registry
            challenge_id = metadata.get("challenge_id") or metadata.get("challenge")
            if not target and registry and challenge_id:
                try:
                    target = registry.get_target_url(challenge_id)
                except KeyError:
                    pass

            row = {
                "prompt": prompt,
                "env_class": "openctf",
                "ground_truth_flag": sample.get("ground_truth_flag"),
                "optimal_steps": metadata.get("optimal_steps"),
                "challenge_id": challenge_id,
                "task_type": metadata.get("task_type", "ctf"),
                "target": target,
            }

            writer.write(row)
            converted += 1

    logger.info("Converted %d GRPO samples → %s", converted, output_path)
    return output_path


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

    # Detect transformer layer class for FSDP wrapping.
    # model._no_split_modules returns a set on some architectures (e.g. Llama),
    # which triggers a set-indexing bug in SkyRL's apply_fsdp2.
    # We auto-detect and pass the class name as a string to avoid this.
    _ARCH_TO_LAYER_CLS = {
        "LlamaForCausalLM": "LlamaDecoderLayer",
        "Qwen2ForCausalLM": "Qwen2DecoderLayer",
        "Qwen3ForCausalLM": "Qwen3DecoderLayer",
        "MistralForCausalLM": "MistralDecoderLayer",
        "GptOssForCausalLM": "GptOssDecoderLayer",
    }
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

    # Reference model path for KL divergence. Defaults to the policy model
    # (standard GRPO), but can be overridden for distillation.
    ref_model_path = config.get("ref_model_path", model_path)

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
            "max_prompt_length": model_cfg.get("max_seq_length", 8192),
            "use_sample_packing": grpo_cfg.get("use_sample_packing", False),  # Requires flash_attention_2 which has issues on GB10
            "eval_batch_size": 1,
            "eval_interval": -1,
            "flash_attn": grpo_cfg.get("flash_attn", False),  # GB10 sm_121a: "Cannot access data pointer" with flash_attn
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
            "logger": output_cfg.get("report_to", "console"),
            "dump_data_batch": False,
            "dump_eval_results": False,
            "target_modules": None,
            "exclude_modules": None,
            "rope_scaling": None,
            "rope_theta": None,

            "placement": {
                # colocate_all must be False when using external vLLM server
                "colocate_all": not bool(grpo_cfg.get("vllm_server_url")),
                "colocate_policy_ref": True,
                "policy_num_nodes": 1,
                "policy_num_gpus_per_node": 1,
                "critic_num_nodes": 1,
                "critic_num_gpus_per_node": 1,
                "ref_num_nodes": 1,
                "ref_num_gpus_per_node": 1,
            },

            "fully_async": {
                "max_staleness_steps": 4,
                "num_parallel_generation_workers": 1,
            },

            "policy": {
                "model": {
                    "path": model_path,
                    "lora": {
                        "rank": lora_cfg.get("r", 64),
                        "alpha": lora_cfg.get("alpha", 128),
                        "dropout": lora_cfg.get("dropout", 0.0),
                        "target_modules": "all-linear",
                        "exclude_modules": None,
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
                    "cpu_offload": False,
                    "reshard_after_forward": True,
                    "fsdp_size": -1,
                },
            },

            "critic": {
                "model": {
                    "path": None,
                    "lora": {
                        "rank": 0,
                        "alpha": 16,
                        "dropout": 0,
                        "target_modules": "all-linear",
                        "exclude_modules": None,
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
                "advantage_estimator": grpo_cfg.get("advantage_estimator", "grpo"),
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
                "zero_variance_filter": False,
                "lambd": 1.0,
                "gamma": 1.0,
                "eps_clip_low": 0.2,
                "eps_clip_high": 0.2,
                "clip_ratio_c": 3.0,
                "tis_imp_ratio_cap": -1.0,
                "use_tis": False,
                "sapo": {"tau_pos": 1.0, "tau_neg": 1.05},
                "value_clip": 0.2,
                "dynamic_sampling": {
                    "type": None,
                    "max_sample_batches": 30,
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
        #
        # When vllm_server_url is set, run_engines_locally=False causes
        # SkyRL to create RemoteInferenceEngine objects that connect to
        # the external vLLM server via standard /v1/completions.
        "generator": {
            "model_name": model_path,
            "model_dtype": "bfloat16",
            "run_engines_locally": not bool(grpo_cfg.get("vllm_server_url")),
            "num_inference_engines": 1,
            "backend": "vllm",
            "weight_sync_backend": "nccl",
            "weight_transfer_threshold_cuda_ipc_GB": 1.0,
            "inference_engine_tensor_parallel_size": 1,
            "inference_engine_pipeline_parallel_size": 1,
            "inference_engine_expert_parallel_size": 1,
            "inference_engine_data_parallel_size": 1,
            "n_samples_per_prompt": grpo_cfg.get("num_generations", 8),
            "async_engine": True,
            "batched": False,
            "max_input_length": model_cfg.get("max_seq_length", 8192),
            "vllm_v1_disable_multiproc": True,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 32768,
            "enforce_eager": True,
            "fully_sharded_loras": False,
            "enable_ray_prometheus_stats": False,
            "gpu_memory_utilization": 0.8,
            "max_num_seqs": 32,
            "remote_inference_engine_urls": (
                [grpo_cfg["vllm_server_url"].replace("http://", "").replace("https://", "")]
                if grpo_cfg.get("vllm_server_url")
                else ["127.0.0.1:8001"]
            ),
            "enable_http_endpoint": False,
            "http_endpoint_host": "127.0.0.1",
            "http_endpoint_port": 8000,
            "served_model_name": None,
            "max_turns": grpo_cfg.get("max_tool_calling_iterations", 15),
            "chat_template": {"source": "name", "name_or_path": None},
            "chat_template_kwargs": {},
            "engine_init_kwargs": {},
            "override_existing_update_group": "auto",
            "sampling_params": {
                "max_generate_length": grpo_cfg.get("max_completion_length", 8192),
                "repetition_penalty": 1.0,
                "temperature": 1.0,
                "top_p": 0.95,
                "min_p": 0.0,
                "top_k": -1,
                # logprobs must be None for external server (remote mode
                # rejects any non-None value); logprobs=1 for local mode.
                "logprobs": None if grpo_cfg.get("vllm_server_url") else 1,
                "stop": None,
                "additional_kwargs": None,
            },
            "use_conversation_multi_turn": True,
            "append_eos_token_after_stop_str_in_multi_turn": True,
            "eval_sampling_params": {
                "max_generate_length": grpo_cfg.get("max_completion_length", 8192),
                "repetition_penalty": 1.0,
                "temperature": 0.6,
                "top_p": 0.95,
                "min_p": 0.0,
                "top_k": -1,
                "logprobs": None if grpo_cfg.get("vllm_server_url") else 0,
                "stop": None,
                "additional_kwargs": None,
            },
            "eval_n_samples_per_prompt": 1,
            "zero_reward_on_non_stop": False,
            "apply_overlong_filtering": False,
            "rope_scaling": None,
            "rope_theta": None,
            "step_wise_trajectories": False,
        },

        # Environment
        "environment": {
            "env_class": "openctf",
            "skyrl_gym": {
                "max_env_workers": 32,
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

    return skyrl_config


def train_grpo(
    model_path: str,
    data_path: str,
    output_dir: str,
    config: Dict[str, Any],
    resume_from: Optional[str] = None,
    challenge_registry: Optional[str] = None,
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
    logger.info("=" * 60)
    logger.info("GRPO TRAINING (SkyRL)")
    logger.info("  Model:  %s", model_path)
    logger.info("  Data:   %s", data_path)
    logger.info("  Output: %s", output_dir)
    logger.info("=" * 60)

    # 1. Convert data to SkyRL format
    registry = None
    if challenge_registry:
        from open_ctf.challenges.registry import ChallengeRegistry
        registry = ChallengeRegistry(challenge_registry)
    converted_data = _convert_grpo_data(data_path, output_dir, registry=registry)

    # 2. Build SkyRL config
    skyrl_config = _build_skyrl_config(model_path, output_dir, config, converted_data)

    if resume_from:
        skyrl_config["trainer"]["resume_path"] = resume_from

    # 3. Write config for reference
    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "skyrl_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(skyrl_config, f, default_flow_style=False)
    logger.info("SkyRL config written to %s", config_path)

    # 4. Launch SkyRL training
    reward_config = config.get("reward", {})
    try:
        _run_skyrl_training(skyrl_config, reward_config)
    except ImportError as e:
        logger.error(
            "SkyRL not installed. Install with: pip install skyrl-train skyrl-gym ray[default] vllm"
        )
        raise

    logger.info("GRPO training complete. Output: %s", output_dir)

    final_dir = os.path.join(output_dir, "final")
    if os.path.exists(final_dir):
        return final_dir
    return output_dir


def _run_skyrl_training(
    config: Dict[str, Any],
    reward_config: Dict[str, Any],
) -> None:
    """Launch SkyRL training with the given config.

    Uses SkyRL's Python API (BasePPOExp). The environment is registered
    inside a Ray remote task so Ray workers can access it.

    Args:
        config: SkyRL config dict.
        reward_config: Serializable reward weight dict (reconstructed
            into CTFReward inside each env instance).

    Key: exp.run() already calls asyncio.run() internally -- do NOT wrap
    in another asyncio.run().
    """
    import ray
    from omegaconf import OmegaConf

    # Convert dict to OmegaConf DictConfig (SkyRL expects this)
    cfg = OmegaConf.create(config)

    # Import SkyRL utilities
    from skyrl_train.entrypoints.main_base import BasePPOExp, validate_cfg
    from skyrl_train.utils import initialize_ray

    # Validate config against SkyRLConfig schema
    validate_cfg(cfg)

    # Initialize Ray cluster
    initialize_ray(cfg)

    # Register env and run training inside a Ray remote task.
    # This ensures the env registration is visible to Ray workers.
    @ray.remote(num_cpus=1)
    def _skyrl_entrypoint(cfg_dict, reward_config):
        from omegaconf import OmegaConf
        from skyrl_gym.envs import register
        from open_ctf.envs.skyrl.openctf_env import OpenCTFTextEnv
        from skyrl_train.entrypoints.main_base import BasePPOExp

        cfg = OmegaConf.create(cfg_dict)

        # Register OpenCTFTextEnv with serializable kwargs only.
        # reward_config is a plain dict of floats -- JSON-safe for SkyRL's
        # EnvSpec._check_can_jsonify(). The env reconstructs CTFReward
        # from this config in __init__().
        register(
            id="openctf",
            entry_point=OpenCTFTextEnv,
            kwargs={
                "reward_config": reward_config,
            },
        )

        # Monkey-patch tokenizer after SkyRL loads it to fix BatchEncoding.
        # Some tokenizers (e.g. Nanbeige) return BatchEncoding from
        # apply_chat_template(tokenize=True) instead of List[int].
        # list(BatchEncoding) returns dict keys ["input_ids", "attention_mask"],
        # not actual token IDs, corrupting all downstream token operations.
        _patch_tokenizer_for_batchencoding()

        exp = BasePPOExp(cfg)
        exp.run()  # Already calls asyncio.run() internally

    # Convert back to dict for serialization through Ray
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    ray.get(_skyrl_entrypoint.remote(cfg_dict, reward_config))


def _patch_tokenizer_for_batchencoding():
    """Monkey-patch SkyRL's generator to handle BatchEncoding from tokenizers.

    Some tokenizers (e.g. Nanbeige4.1-3B) return BatchEncoding (dict-like)
    from apply_chat_template(tokenize=True) instead of List[int].
    SkyRL's skyrl_gym_generator.py calls apply_chat_template in 5+ places
    and expects a plain List[int]. list(BatchEncoding) returns dict keys
    ["input_ids", "attention_mask"], corrupting all token operations.

    Rather than patching each call site, we wrap the generator class's
    _create_agent_loop method to ensure the tokenizer always returns lists.
    """
    try:
        from skyrl_train.generators.skyrl_gym_generator import SkyRLGymGenerator
    except ImportError:
        logger.warning("Could not import SkyRLGymGenerator for BatchEncoding patch")
        return

    _orig_init = SkyRLGymGenerator.__init__

    def _patched_init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        if self.tokenizer is not None:
            _wrap_apply_chat_template(self.tokenizer)

    SkyRLGymGenerator.__init__ = _patched_init
    logger.info("Patched SkyRLGymGenerator.__init__ for BatchEncoding safety")


def _wrap_apply_chat_template(tokenizer):
    """Wrap tokenizer.apply_chat_template to always return List[int].

    When return_dict=True is passed, returns are kept as-is (the caller
    already handles dict access). Otherwise, unwraps BatchEncoding to
    get the raw input_ids list.
    """
    orig_fn = tokenizer.apply_chat_template

    def _safe_apply_chat_template(*args, **kwargs):
        result = orig_fn(*args, **kwargs)
        # If caller requested return_dict=True, leave result as-is
        if kwargs.get("return_dict"):
            return result
        # Unwrap BatchEncoding to List[int]
        if hasattr(result, "input_ids"):
            return list(result.input_ids)
        if isinstance(result, dict) and "input_ids" in result:
            return list(result["input_ids"])
        return result

    tokenizer.apply_chat_template = _safe_apply_chat_template
    logger.info("Wrapped tokenizer.apply_chat_template for BatchEncoding safety")
