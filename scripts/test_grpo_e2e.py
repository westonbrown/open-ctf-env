#!/usr/bin/env python3
"""End-to-end GRPO training loop test on DGX Spark.

Verifies the full SkyRL training pipeline works:
  1. Convert GRPO data (5 samples) to SkyRL format
  2. Build SkyRL OmegaConf config for small test run
  3. Register OpenCTFTextEnv with skyrl_gym
  4. Initialize Ray
  5. Run BasePPOExp (1 epoch, tiny batch)

Uses Nanbeige4.1-3B merged model. Expects to run inside
the open-ctf-grpo container on DGX Spark.

Usage:
    python3 scripts/test_grpo_e2e.py
"""

import json
import logging
import os
import sys
import time
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("grpo_e2e_test")

# ── Paths ──────────────────────────────────────────────────────────────
MODEL_PATH = "/workspace/open-ctf-env/outputs/sft-nanbeige3b-merged"
GRPO_DATA = "/workspace/open-ctf-env/data/grpo_cybench40.jsonl"
CHALLENGE_REGISTRY = "/workspace/open-ctf-env/configs/challenges/cybench.yaml"
OUTPUT_DIR = "/workspace/open-ctf-env/outputs/grpo_e2e_test"
NUM_SAMPLES = 5  # Use just 5 samples for speed

# ── Helpers ────────────────────────────────────────────────────────────

def step_header(n: int, title: str):
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP %d: %s", n, title)
    logger.info("=" * 60)


def main():
    t0 = time.time()
    logger.info("GRPO End-to-End Test Starting")
    logger.info("  Model:     %s", MODEL_PATH)
    logger.info("  Data:      %s", GRPO_DATA)
    logger.info("  Samples:   %d", NUM_SAMPLES)
    logger.info("  Output:    %s", OUTPUT_DIR)

    # ── Step 1: Verify prerequisites ───────────────────────────────
    step_header(1, "Verify prerequisites")

    assert os.path.isdir(MODEL_PATH), f"Model not found: {MODEL_PATH}"
    assert os.path.isfile(GRPO_DATA), f"GRPO data not found: {GRPO_DATA}"
    assert os.path.isfile(CHALLENGE_REGISTRY), f"Registry not found: {CHALLENGE_REGISTRY}"

    # Verify model has required files
    for f in ("config.json", "tokenizer_config.json"):
        fp = os.path.join(MODEL_PATH, f)
        assert os.path.isfile(fp), f"Missing model file: {fp}"
    logger.info("  Model directory: OK (%d files)", len(os.listdir(MODEL_PATH)))

    import torch
    logger.info("  PyTorch: %s", torch.__version__)
    logger.info("  CUDA: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("  GPU: %s", torch.cuda.get_device_name(0))

    logger.info("  Prerequisites: PASS")

    # ── Step 2: Convert GRPO data (subset) ─────────────────────────
    step_header(2, "Convert GRPO data (%d samples)" % NUM_SAMPLES)

    import jsonlines
    from open_ctf.training.grpo import _convert_grpo_data
    from open_ctf.challenges.registry import ChallengeRegistry

    # Create a subset file with only NUM_SAMPLES
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    subset_path = os.path.join(OUTPUT_DIR, "grpo_subset.jsonl")
    with jsonlines.open(GRPO_DATA) as reader, jsonlines.open(subset_path, "w") as writer:
        for i, sample in enumerate(reader):
            if i >= NUM_SAMPLES:
                break
            writer.write(sample)
    logger.info("  Created subset: %s (%d samples)", subset_path, NUM_SAMPLES)

    # Convert to SkyRL format
    registry = ChallengeRegistry(CHALLENGE_REGISTRY)
    converted_path = _convert_grpo_data(subset_path, OUTPUT_DIR, registry=registry)
    logger.info("  Converted data: %s", converted_path)

    # Verify converted data
    samples = list(jsonlines.open(converted_path))
    logger.info("  Converted samples: %d", len(samples))
    for i, s in enumerate(samples):
        logger.info(
            "    [%d] env_class=%s, challenge=%s, flag=%s, target=%s",
            i,
            s.get("env_class"),
            s.get("challenge_id"),
            s.get("ground_truth_flag", "N/A")[:30],
            s.get("target", "N/A"),
        )
        prompt = s.get("prompt", [])
        logger.info("         prompt: %d messages, last_role=%s",
                     len(prompt), prompt[-1]["role"] if prompt else "EMPTY")
    logger.info("  Data conversion: PASS")

    # ── Step 3: Build SkyRL config ─────────────────────────────────
    step_header(3, "Build SkyRL config")

    from omegaconf import OmegaConf
    from skyrl_train.config.config import SkyRLConfig
    import yaml

    # Start from SkyRLConfig defaults (structured config) so all fields exist.
    # Then merge our overrides on top. This avoids missing-key errors.
    cfg = OmegaConf.structured(SkyRLConfig)

    # -- Data --
    cfg.data.train_data = [converted_path]

    # -- Trainer --
    cfg.trainer.strategy = "fsdp2"
    cfg.trainer.bf16 = True
    cfg.trainer.gradient_checkpointing = True
    cfg.trainer.seed = 42
    cfg.trainer.epochs = 1
    cfg.trainer.train_batch_size = 1
    cfg.trainer.micro_train_batch_size_per_gpu = 1
    cfg.trainer.max_prompt_length = 2048
    cfg.trainer.ckpt_path = OUTPUT_DIR
    cfg.trainer.ckpt_interval = 50
    cfg.trainer.log_path = os.path.join(OUTPUT_DIR, "logs")
    cfg.trainer.export_path = os.path.join(OUTPUT_DIR, "final")
    cfg.trainer.project_name = "open-ctf"
    cfg.trainer.run_name = "grpo-e2e-test"
    cfg.trainer.logger = "none"
    cfg.trainer.eval_before_train = False
    cfg.trainer.eval_interval = 999

    # Placement: colocate vLLM + training on same GPU
    cfg.trainer.placement.colocate_all = True

    # Policy model + LoRA
    cfg.trainer.policy.model.path = MODEL_PATH
    cfg.trainer.policy.model.lora.rank = 64
    cfg.trainer.policy.model.lora.alpha = 128
    cfg.trainer.policy.model.lora.dropout = 0.0
    cfg.trainer.policy.model.lora.target_modules = (
        "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
    )
    cfg.trainer.policy.optimizer_config.lr = 5e-6
    cfg.trainer.policy.optimizer_config.weight_decay = 0.0
    cfg.trainer.policy.optimizer_config.max_grad_norm = 5.0

    # Reference model (same as policy for standard GRPO)
    cfg.trainer.ref.model.path = MODEL_PATH

    # Algorithm
    cfg.trainer.algorithm.advantage_estimator = "rloo_n"
    cfg.trainer.algorithm.policy_loss_type = "regular"
    cfg.trainer.algorithm.kl_loss_coef = 0.0
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.loss_reduction = "token_mean"
    cfg.trainer.algorithm.eps_clip_low = 0.2
    cfg.trainer.algorithm.eps_clip_high = 0.2

    # -- Generator (vLLM inference) --
    cfg.generator.model_dtype = "bfloat16"
    cfg.generator.backend = "vllm"
    cfg.generator.run_engines_locally = True
    cfg.generator.num_inference_engines = 1
    cfg.generator.n_samples_per_prompt = 2  # small for test
    cfg.generator.max_input_length = 2048
    cfg.generator.max_turns = 3  # few turns for test
    cfg.generator.inference_engine_tensor_parallel_size = 1
    cfg.generator.gpu_memory_utilization = 0.5  # leave room for training
    cfg.generator.max_num_seqs = 4
    cfg.generator.max_num_batched_tokens = 4096
    cfg.generator.enable_prefix_caching = True
    cfg.generator.enable_chunked_prefill = True
    cfg.generator.enforce_eager = True  # avoid CUDA graph issues on GB10
    cfg.generator.sampling_params.max_generate_length = 512
    cfg.generator.sampling_params.temperature = 1.0
    cfg.generator.sampling_params.top_p = 0.95
    cfg.generator.sampling_params.logprobs = 1

    # -- Environment --
    cfg.environment.env_class = "openctf"
    cfg.environment.skyrl_gym.max_env_workers = 4

    # Write config for inspection
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    config_path = os.path.join(OUTPUT_DIR, "skyrl_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(
            OmegaConf.to_container(cfg, resolve=True),
            f, default_flow_style=False,
        )
    logger.info("  Config written to: %s", config_path)
    logger.info("  OmegaConf config created (from SkyRLConfig defaults + overrides)")
    logger.info("    trainer.strategy: %s", cfg.trainer.strategy)
    logger.info("    trainer.train_batch_size: %s", cfg.trainer.train_batch_size)
    logger.info("    trainer.epochs: %s", cfg.trainer.epochs)
    logger.info("    generator.backend: %s", cfg.generator.backend)
    logger.info("    generator.n_samples_per_prompt: %s", cfg.generator.n_samples_per_prompt)
    logger.info("    generator.gpu_memory_utilization: %s", cfg.generator.gpu_memory_utilization)
    logger.info("    generator.max_turns: %s", cfg.generator.max_turns)
    logger.info("    environment.env_class: %s", cfg.environment.env_class)
    logger.info("  Config build: PASS")

    # ── Step 4: Validate SkyRL config ──────────────────────────────
    step_header(4, "Validate SkyRL config")

    try:
        from skyrl_train.entrypoints.main_base import validate_cfg
        validate_cfg(cfg)
        logger.info("  Config validation: PASS")
    except Exception as e:
        logger.warning("  Config validation FAILED: %s", e)
        logger.warning("  This may be expected -- continuing anyway to see full error chain")

    # ── Step 5: Register env + Initialize Ray ──────────────────────
    step_header(5, "Register env + Initialize Ray")

    from skyrl_gym.envs import register as skyrl_register
    from open_ctf.envs.skyrl.openctf_env import OpenCTFTextEnv

    skyrl_register(
        id="openctf",
        entry_point=OpenCTFTextEnv,
        kwargs={"reward_config": {}},
    )
    logger.info("  OpenCTFTextEnv registered as 'openctf'")

    from skyrl_train.utils import initialize_ray
    initialize_ray(cfg)
    logger.info("  Ray initialized")

    import ray
    logger.info("  Ray cluster: %s", ray.cluster_resources())

    # ── Step 6: Run BasePPOExp ─────────────────────────────────────
    step_header(6, "Run BasePPOExp (training loop)")

    logger.info("  Creating BasePPOExp...")
    from skyrl_train.entrypoints.main_base import BasePPOExp

    try:
        exp = BasePPOExp(cfg)
        logger.info("  BasePPOExp created successfully")
        logger.info("  Tokenizer: %s", type(exp.tokenizer).__name__)
        logger.info("  Train dataset size: %d", len(exp.train_dataset))
        logger.info("  Colocate PG: %s", exp.colocate_pg)

        logger.info("")
        logger.info("  Starting training loop (exp.run())...")
        logger.info("  This will initialize vLLM + FSDP2 and run 1 epoch")
        logger.info("")

        exp.run()

        logger.info("  Training completed successfully!")

    except Exception as e:
        logger.error("  Training FAILED with error:")
        logger.error("  %s: %s", type(e).__name__, e)
        logger.error("")
        traceback.print_exc()

        # Capture the error for reporting
        error_path = os.path.join(OUTPUT_DIR, "error.txt")
        with open(error_path, "w") as f:
            f.write(f"{type(e).__name__}: {e}\n\n")
            traceback.print_exc(file=f)
        logger.info("  Error details saved to: %s", error_path)

    # ── Summary ────────────────────────────────────────────────────
    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info("GRPO E2E TEST COMPLETE")
    logger.info("  Elapsed: %.1f seconds", elapsed)
    logger.info("  Output: %s", OUTPUT_DIR)
    logger.info("=" * 60)

    # Check for output artifacts
    for f in os.listdir(OUTPUT_DIR):
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp)
            logger.info("  %s (%s)", f, _human_size(sz))
        elif os.path.isdir(fp):
            logger.info("  %s/ (dir)", f)


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error("Unhandled error: %s: %s", type(e).__name__, e)
        traceback.print_exc()
        sys.exit(1)
