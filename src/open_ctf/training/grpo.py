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
    from dataclasses import asdict
    try:
        from skyrl_train.config.config import SkyRLConfig
        skyrl_defaults = asdict(SkyRLConfig())
    except ImportError:
        skyrl_defaults = {}

    model_cfg = config.get("model", {})
    lora_cfg = config.get("lora", {})
    grpo_cfg = config.get("grpo", {})
    output_cfg = config.get("output", {})

    # Reference model path for KL divergence. Defaults to the policy model
    # (standard GRPO), but can be overridden for distillation.
    ref_model_path = config.get("ref_model_path", model_path)

    skyrl_config = {
        # Data
        "data": {
            "train_data": [data_path],
        },

        # Trainer
        "trainer": {
            "strategy": "fsdp2",
            "bf16": True,
            "gradient_checkpointing": True,
            "seed": 42,
            "epochs": grpo_cfg.get("epochs", 1),
            "train_batch_size": grpo_cfg.get("batch_size", 1),
            "policy_mini_batch_size": grpo_cfg.get("batch_size", 1),
            "micro_train_batch_size_per_gpu": 1,
            "max_prompt_length": model_cfg.get("max_seq_length", 8192),
            "ckpt_path": output_dir,
            "ckpt_interval": output_cfg.get("save_steps", 50),
            "log_path": os.path.join(output_dir, "logs"),
            "export_path": os.path.join(output_dir, "final"),
            "project_name": "open-ctf",
            "run_name": "grpo",
            "logger": output_cfg.get("report_to", "none"),

            "placement": {
                "colocate_all": True,
            },

            "policy": {
                "model": {
                    "path": model_path,
                    "lora": {
                        "rank": lora_cfg.get("r", 64),
                        "alpha": lora_cfg.get("alpha", 128),
                        "dropout": lora_cfg.get("dropout", 0.0),
                        "target_modules": ",".join(lora_cfg.get("target_modules", [
                            "q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",
                        ])),
                    },
                },
                "optimizer_config": {
                    "lr": grpo_cfg.get("learning_rate", 5e-6),
                    "weight_decay": grpo_cfg.get("weight_decay", 0.0),
                    "max_grad_norm": grpo_cfg.get("max_grad_norm", 5.0),
                },
            },

            "ref": {
                "model": {
                    "path": ref_model_path,
                },
            },

            "algorithm": {
                "advantage_estimator": grpo_cfg.get("advantage_estimator", "grpo"),
                "policy_loss_type": "regular",
                "kl_loss_coef": grpo_cfg.get("beta", 0.0),
                "use_kl_loss": grpo_cfg.get("beta", 0.0) > 0,
                "loss_reduction": "token_mean",
                "eps_clip_low": 0.2,
                "eps_clip_high": 0.2,
            },
        },

        # Generator (vLLM inference) — all fields from SkyRL 0.3.1 GeneratorConfig
        "generator": {
            "model_name": "",
            "model_dtype": "bfloat16",
            "run_engines_locally": True,
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
            "remote_inference_engine_urls": ["127.0.0.1:8001"],
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
                "logprobs": 1,
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
                "logprobs": 0,
                "stop": None,
                "additional_kwargs": None,
            },
            "eval_n_samples_per_prompt": 1,
            "zero_reward_on_non_stop": False,
            "apply_overlong_filtering": False,
            "rope_scaling": None,
            "rope_theta": None,
            "step_wise_trajectories": False,
            "external_proxy_url": None,
            "external_server_urls": None,
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

        exp = BasePPOExp(cfg)
        exp.run()  # Already calls asyncio.run() internally

    # Convert back to dict for serialization through Ray
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    ray.get(_skyrl_entrypoint.remote(cfg_dict, reward_config))
