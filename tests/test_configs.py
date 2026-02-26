"""Smoke tests for YAML configuration files.

Validates:
- All YAML configs load without error
- LlamaFactory SFT configs have required fields
- SkyRL GRPO configs have required fields
- Config values are reasonable (batch sizes, learning rates, etc.)
"""

import pytest
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


# ---------------------------------------------------------------------------
# LlamaFactory SFT configs
# ---------------------------------------------------------------------------


class TestLlamaFactorySFTConfigs:
    SFT_DIR = CONFIGS_DIR / "llamafactory"

    def test_sft_dir_exists(self):
        assert self.SFT_DIR.exists(), f"SFT config dir not found: {self.SFT_DIR}"

    @pytest.fixture(params=[
        "nanbeige_3b.yaml",
        "glm47_flash.yaml",
        "devstral_24b.yaml",
    ])
    def sft_config(self, request):
        path = self.SFT_DIR / request.param
        if not path.exists():
            pytest.skip(f"{request.param} not found")
        return _load_yaml(path), request.param

    def test_loads_without_error(self, sft_config):
        cfg, name = sft_config
        assert isinstance(cfg, dict), f"{name} did not load as dict"

    def test_has_model_name_or_path(self, sft_config):
        cfg, name = sft_config
        assert "model_name_or_path" in cfg, f"{name} missing model_name_or_path"
        assert isinstance(cfg["model_name_or_path"], str)

    def test_has_template(self, sft_config):
        cfg, name = sft_config
        assert "template" in cfg, f"{name} missing template"

    def test_has_cutoff_len(self, sft_config):
        cfg, name = sft_config
        assert "cutoff_len" in cfg, f"{name} missing cutoff_len"
        assert cfg["cutoff_len"] >= 1024, f"{name} cutoff_len too small"

    def test_has_output_dir(self, sft_config):
        cfg, name = sft_config
        assert "output_dir" in cfg, f"{name} missing output_dir"

    def test_stage_is_sft(self, sft_config):
        cfg, name = sft_config
        assert cfg.get("stage") == "sft", f"{name} stage should be 'sft'"

    def test_finetuning_type_is_lora(self, sft_config):
        cfg, name = sft_config
        assert cfg.get("finetuning_type") == "lora", f"{name} should use LoRA"

    def test_lora_rank_reasonable(self, sft_config):
        cfg, name = sft_config
        rank = cfg.get("lora_rank", 0)
        assert 8 <= rank <= 256, f"{name} lora_rank={rank} outside [8, 256]"

    def test_learning_rate_reasonable(self, sft_config):
        cfg, name = sft_config
        lr = cfg.get("learning_rate", 0)
        assert 1e-7 < lr < 1e-2, f"{name} learning_rate={lr} outside reasonable range"

    def test_bf16_enabled(self, sft_config):
        cfg, name = sft_config
        assert cfg.get("bf16") is True, f"{name} should use bf16"


# ---------------------------------------------------------------------------
# SkyRL GRPO configs
# ---------------------------------------------------------------------------


class TestSkyRLGRPOConfigs:
    GRPO_DIR = CONFIGS_DIR / "skyrl"

    def test_grpo_dir_exists(self):
        assert self.GRPO_DIR.exists(), f"GRPO config dir not found: {self.GRPO_DIR}"

    @pytest.fixture(params=[
        "nanbeige_3b.yaml",
        "glm47_flash.yaml",
    ])
    def grpo_config(self, request):
        path = self.GRPO_DIR / request.param
        if not path.exists():
            pytest.skip(f"{request.param} not found")
        return _load_yaml(path), request.param

    def test_loads_without_error(self, grpo_config):
        cfg, name = grpo_config
        assert isinstance(cfg, dict), f"{name} did not load as dict"

    def test_has_trainer_section(self, grpo_config):
        cfg, name = grpo_config
        assert "trainer" in cfg, f"{name} missing 'trainer' section"

    def test_has_generator_section(self, grpo_config):
        cfg, name = grpo_config
        assert "generator" in cfg, f"{name} missing 'generator' section"

    def test_has_environment_section(self, grpo_config):
        cfg, name = grpo_config
        assert "environment" in cfg, f"{name} missing 'environment' section"

    def test_environment_class_is_openctf(self, grpo_config):
        cfg, name = grpo_config
        env = cfg.get("environment", {})
        assert env.get("env_class") == "openctf", (
            f"{name} environment.env_class should be 'openctf'"
        )

    def test_trainer_has_policy(self, grpo_config):
        cfg, name = grpo_config
        trainer = cfg.get("trainer", {})
        assert "policy" in trainer, f"{name} trainer missing 'policy'"

    def test_generator_has_sampling_params(self, grpo_config):
        cfg, name = grpo_config
        gen = cfg.get("generator", {})
        assert "sampling_params" in gen, f"{name} generator missing 'sampling_params'"

    def test_n_samples_per_prompt(self, grpo_config):
        cfg, name = grpo_config
        gen = cfg.get("generator", {})
        n = gen.get("n_samples_per_prompt", 0)
        assert n >= 2, f"{name} n_samples_per_prompt={n} should be >= 2 for GRPO"

    def test_max_turns_set(self, grpo_config):
        cfg, name = grpo_config
        gen = cfg.get("generator", {})
        turns = gen.get("max_turns", 0)
        assert turns >= 5, f"{name} max_turns={turns} should be >= 5"

    def test_algorithm_advantage_estimator(self, grpo_config):
        cfg, name = grpo_config
        algo = cfg.get("trainer", {}).get("algorithm", {})
        assert algo.get("advantage_estimator") in ("grpo", "rloo", "rloo_n"), (
            f"{name} should use GRPO, RLOO, or RLOO-N advantage estimator"
        )


# ---------------------------------------------------------------------------
# All YAML files load
# ---------------------------------------------------------------------------


class TestAllConfigsLoad:
    def test_all_yaml_files_parse(self):
        """Every YAML file under configs/ should parse without error."""
        yaml_files = list(CONFIGS_DIR.rglob("*.yaml"))
        assert len(yaml_files) > 0, "No YAML files found under configs/"

        for path in yaml_files:
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                assert data is not None, f"{path.name} loaded as None"
            except yaml.YAMLError as e:
                pytest.fail(f"{path.name} failed to parse: {e}")
