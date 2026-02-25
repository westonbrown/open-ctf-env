"""Smoke tests for SkyRL GRPO orchestrator.

Validates:
- _convert_grpo_data correctly converts GRPO JSONL to SkyRL format
- _build_skyrl_config produces valid SkyRL config dict
- Config has correct nesting (data.train_data, trainer.policy.model.path, etc.)
"""

import json
import os
import pytest
from pathlib import Path

from open_ctf.training.grpo import (
    _convert_grpo_data,
    _build_skyrl_config,
    _should_force_legacy_inference,
    _resolve_vllm_ready_model_path,
)


# ---------------------------------------------------------------------------
# Sample GRPO data
# ---------------------------------------------------------------------------


def _write_grpo_jsonl(path, samples=None):
    """Write sample GRPO JSONL data."""
    if samples is None:
        samples = [
            {
                "messages": [
                    {"role": "system", "content": "You are a CTF agent."},
                    {"role": "user", "content": "Scan 10.0.0.1 for vulnerabilities."},
                    {"role": "assistant", "content": "Running nmap..."},
                    {"role": "tool", "name": "shell_command", "content": "80/tcp open http"},
                ],
                "ground_truth_flag": "FLAG{test123}",
                "metadata": {
                    "optimal_steps": 5,
                    "challenge_id": "XBEN-001",
                    "task_type": "ctf",
                },
            },
            {
                "messages": [
                    {"role": "system", "content": "You are a CTF agent."},
                    {"role": "user", "content": "Find the flag on the web server."},
                    {"role": "assistant", "content": "Let me check."},
                ],
                "ground_truth_flag": "FLAG{web_flag}",
                "metadata": {
                    "optimal_steps": 3,
                    "challenge_id": "XBEN-002",
                    "task_type": "ctf",
                },
            },
        ]
    import jsonlines
    with jsonlines.open(str(path), "w") as w:
        for s in samples:
            w.write(s)


# ---------------------------------------------------------------------------
# _convert_grpo_data
# ---------------------------------------------------------------------------


class TestConvertGRPOData:
    def test_output_file_created(self, tmp_path):
        src = tmp_path / "grpo.jsonl"
        _write_grpo_jsonl(src)
        output_dir = str(tmp_path / "out")

        result = _convert_grpo_data(str(src), output_dir)
        assert os.path.exists(result)
        assert result.endswith("skyrl_grpo_data.jsonl")

    def test_correct_number_of_samples(self, tmp_path):
        src = tmp_path / "grpo.jsonl"
        _write_grpo_jsonl(src)
        output_dir = str(tmp_path / "out")

        result = _convert_grpo_data(str(src), output_dir)
        import jsonlines
        with jsonlines.open(result) as reader:
            rows = list(reader)
        assert len(rows) == 2

    def test_prompt_extracted(self, tmp_path):
        src = tmp_path / "grpo.jsonl"
        _write_grpo_jsonl(src)
        output_dir = str(tmp_path / "out")

        result = _convert_grpo_data(str(src), output_dir)
        import jsonlines
        with jsonlines.open(result) as reader:
            row = next(iter(reader))

        # Prompt should contain system + user messages before first assistant
        assert isinstance(row["prompt"], list)
        roles = [m["role"] for m in row["prompt"]]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" not in roles

    def test_prompt_ends_with_user(self, tmp_path):
        """SkyRL requires prompt to end with a user message."""
        src = tmp_path / "grpo.jsonl"
        _write_grpo_jsonl(src)
        output_dir = str(tmp_path / "out")

        result = _convert_grpo_data(str(src), output_dir)
        import jsonlines
        with jsonlines.open(result) as reader:
            for row in reader:
                assert row["prompt"][-1]["role"] == "user"

    def test_env_class_set(self, tmp_path):
        src = tmp_path / "grpo.jsonl"
        _write_grpo_jsonl(src)
        output_dir = str(tmp_path / "out")

        result = _convert_grpo_data(str(src), output_dir)
        import jsonlines
        with jsonlines.open(result) as reader:
            for row in reader:
                assert row["env_class"] == "openctf"

    def test_ground_truth_flag_preserved(self, tmp_path):
        src = tmp_path / "grpo.jsonl"
        _write_grpo_jsonl(src)
        output_dir = str(tmp_path / "out")

        result = _convert_grpo_data(str(src), output_dir)
        import jsonlines
        with jsonlines.open(result) as reader:
            row = next(iter(reader))
        assert row["ground_truth_flag"] == "FLAG{test123}"

    def test_metadata_flattened(self, tmp_path):
        src = tmp_path / "grpo.jsonl"
        _write_grpo_jsonl(src)
        output_dir = str(tmp_path / "out")

        result = _convert_grpo_data(str(src), output_dir)
        import jsonlines
        with jsonlines.open(result) as reader:
            row = next(iter(reader))
        assert row["optimal_steps"] == 5
        assert row["challenge_id"] == "XBEN-001"
        assert row["task_type"] == "ctf"

    def test_missing_user_message_gets_default(self, tmp_path):
        """If messages only have system + assistant, a default user msg is added."""
        src = tmp_path / "grpo.jsonl"
        import jsonlines
        with jsonlines.open(str(src), "w") as w:
            w.write({
                "messages": [
                    {"role": "system", "content": "Agent."},
                    {"role": "assistant", "content": "Doing stuff."},
                ],
                "ground_truth_flag": "FLAG{x}",
                "metadata": {},
            })
        output_dir = str(tmp_path / "out")

        result = _convert_grpo_data(str(src), output_dir)
        import jsonlines as jl2
        with jl2.open(result) as reader:
            row = next(iter(reader))
        # Prompt should end with user
        assert row["prompt"][-1]["role"] == "user"
        assert "flag" in row["prompt"][-1]["content"].lower()


# ---------------------------------------------------------------------------
# _build_skyrl_config
# ---------------------------------------------------------------------------


class TestBuildSkyrlConfig:
    @pytest.fixture
    def config(self):
        return {
            "model": {"max_seq_length": 8192},
            "lora": {
                "r": 64,
                "alpha": 128,
                "dropout": 0.0,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "grpo": {
                "learning_rate": 5e-6,
                "num_generations": 4,
                "max_completion_length": 4096,
                "max_tool_calling_iterations": 15,
                "batch_size": 1,
                "epochs": 1,
                "beta": 0.001,
            },
            "output": {"save_steps": 50, "report_to": "none"},
        }

    def test_returns_dict(self, config):
        result = _build_skyrl_config("/path/to/model", "/out", config, "/data.jsonl")
        assert isinstance(result, dict)

    def test_data_train_data_nesting(self, config):
        result = _build_skyrl_config("/path/to/model", "/out", config, "/data.jsonl")
        assert "data" in result
        assert "train_data" in result["data"]
        assert result["data"]["train_data"] == ["/data.jsonl"]

    def test_trainer_policy_model_path(self, config):
        result = _build_skyrl_config("/path/to/model", "/out", config, "/data.jsonl")
        assert result["trainer"]["policy"]["model"]["path"] == "/path/to/model"

    def test_trainer_policy_optimizer_lr(self, config):
        result = _build_skyrl_config("/path/to/model", "/out", config, "/data.jsonl")
        assert result["trainer"]["policy"]["optimizer_config"]["lr"] == 5e-6

    def test_trainer_algorithm_default(self, config):
        result = _build_skyrl_config("/path/to/model", "/out", config, "/data.jsonl")
        algo = result["trainer"]["algorithm"]
        assert algo["advantage_estimator"] == "rloo_n"

    def test_generator_sampling_params(self, config):
        result = _build_skyrl_config("/path/to/model", "/out", config, "/data.jsonl")
        sp = result["generator"]["sampling_params"]
        assert sp["max_generate_length"] == 4096  # From fixture's max_completion_length
        assert sp["temperature"] == 1.0
        assert sp["top_p"] == 0.95
        assert "additional_kwargs" not in sp

    def test_generator_n_samples(self, config):
        result = _build_skyrl_config("/path/to/model", "/out", config, "/data.jsonl")
        # n_samples_per_prompt comes from config's num_generations (4 in fixture)
        assert result["generator"]["n_samples_per_prompt"] == 4

    def test_generator_weight_sync_backend_local_defaults_nccl(self, config):
        result = _build_skyrl_config("/path/to/model", "/out", config, "/data.jsonl")
        assert result["generator"]["run_engines_locally"] is True
        assert result["generator"]["weight_sync_backend"] == "nccl"

    def test_generator_max_turns(self, config):
        result = _build_skyrl_config("/path/to/model", "/out", config, "/data.jsonl")
        assert result["generator"]["max_turns"] == 15

    def test_generator_server_mode_without_url_uses_local_non_colocate(self, config):
        cfg = json.loads(json.dumps(config))
        cfg["grpo"]["vllm_mode"] = "server"
        result = _build_skyrl_config("/path/to/model", "/out", cfg, "/data.jsonl")
        assert result["trainer"]["placement"]["colocate_all"] is False
        assert result["generator"]["run_engines_locally"] is True
        assert result["generator"]["weight_sync_backend"] == "nccl"

    def test_generator_remote_vllm_with_lora_falls_back_to_local(self, config):
        cfg = json.loads(json.dumps(config))
        cfg["grpo"]["vllm_server_url"] = "http://127.0.0.1:9000"
        result = _build_skyrl_config("/path/to/model", "/out", cfg, "/data.jsonl")
        assert result["trainer"]["placement"]["colocate_all"] is False
        assert result["generator"]["run_engines_locally"] is True
        assert result["generator"]["weight_sync_backend"] == "nccl"
        assert result["generator"]["remote_inference_engine_urls"] == ["127.0.0.1:8001"]
        assert result["generator"]["sampling_params"]["logprobs"] == 0

    def test_custom_chat_template_forces_logprobs_none(self, config):
        cfg = json.loads(json.dumps(config))
        cfg["grpo"]["chat_template"] = "qwen3_without_thinking"
        cfg["grpo"]["logprobs"] = 0
        cfg["grpo"]["eval_logprobs"] = 0
        result = _build_skyrl_config("/path/to/model", "/out", cfg, "/data.jsonl")
        assert result["generator"]["sampling_params"]["logprobs"] is None
        assert result["generator"]["eval_sampling_params"]["logprobs"] is None

    def test_generator_remote_vllm_without_lora_uses_broadcast_sync(self, config):
        cfg = json.loads(json.dumps(config))
        cfg["lora"]["r"] = 0
        cfg["grpo"]["vllm_server_url"] = "https://127.0.0.1:9000/"
        result = _build_skyrl_config("/path/to/model", "/out", cfg, "/data.jsonl")
        assert result["trainer"]["placement"]["colocate_all"] is False
        assert result["generator"]["run_engines_locally"] is False
        assert result["generator"]["weight_sync_backend"] == "broadcast"
        assert result["generator"]["remote_inference_engine_urls"] == ["127.0.0.1:9000"]
        assert result["generator"]["sampling_params"]["logprobs"] is None

    def test_environment_env_class(self, config):
        result = _build_skyrl_config("/path/to/model", "/out", config, "/data.jsonl")
        assert result["environment"]["env_class"] == "openctf"

    def test_lora_config_nested(self, config):
        result = _build_skyrl_config("/path/to/model", "/out", config, "/data.jsonl")
        lora = result["trainer"]["policy"]["model"]["lora"]
        assert lora["rank"] == 64
        assert lora["alpha"] == 128
        assert lora["dropout"] == 0.0
        assert lora["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj"]

    def test_lora_target_modules_string_passthrough(self, config):
        cfg = json.loads(json.dumps(config))
        cfg["lora"]["target_modules"] = "q_proj,k_proj"
        result = _build_skyrl_config("/path/to/model", "/out", cfg, "/data.jsonl")
        lora = result["trainer"]["policy"]["model"]["lora"]
        assert lora["target_modules"] == ["q_proj", "k_proj"]

    def test_lora_target_modules_default_all_linear(self):
        cfg = {
            "model": {"max_seq_length": 4096},
            "lora": {"r": 32, "alpha": 64, "dropout": 0.0},
            "grpo": {"batch_size": 1, "epochs": 1, "num_generations": 2},
            "output": {"save_steps": 50},
        }
        result = _build_skyrl_config("/path/to/model", "/out", cfg, "/data.jsonl")
        lora = result["trainer"]["policy"]["model"]["lora"]
        assert lora["target_modules"] == "all-linear"

    def test_kl_loss_enabled_when_beta_positive(self, config):
        result = _build_skyrl_config("/path/to/model", "/out", config, "/data.jsonl")
        algo = result["trainer"]["algorithm"]
        assert algo["use_kl_loss"] is True
        assert algo["kl_loss_coef"] == 0.001

    def test_algorithm_clip_range_uses_grpo_config(self, config):
        cfg = json.loads(json.dumps(config))
        cfg["grpo"]["epsilon_low"] = 0.15
        cfg["grpo"]["epsilon_high"] = 0.28
        result = _build_skyrl_config("/path/to/model", "/out", cfg, "/data.jsonl")
        algo = result["trainer"]["algorithm"]
        assert algo["eps_clip_low"] == 0.15
        assert algo["eps_clip_high"] == 0.28

    def test_algorithm_clip_high_defaults_to_low_when_unspecified(self, config):
        cfg = json.loads(json.dumps(config))
        cfg["grpo"]["epsilon_low"] = 0.12
        cfg["grpo"].pop("epsilon_high", None)
        result = _build_skyrl_config("/path/to/model", "/out", cfg, "/data.jsonl")
        algo = result["trainer"]["algorithm"]
        assert algo["eps_clip_low"] == 0.12
        assert algo["eps_clip_high"] == 0.12

    def test_missing_grpo_section_uses_defaults(self):
        """Config without grpo section should still produce valid output."""
        config = {
            "model": {"max_seq_length": 4096},
            "lora": {"r": 32, "alpha": 64, "dropout": 0.0, "target_modules": ["q_proj"]},
            "output": {"save_steps": 100},
        }
        result = _build_skyrl_config("/model", "/out", config, "/data.jsonl")
        assert isinstance(result, dict)
        assert "trainer" in result
        assert "generator" in result
        # Should have sensible defaults for GRPO params
        assert result["generator"]["n_samples_per_prompt"] >= 1

    def test_environment_env_class_always_openctf(self):
        """env_class should always be 'openctf' regardless of config."""
        config = {
            "model": {"max_seq_length": 4096},
            "lora": {"r": 32, "alpha": 64, "dropout": 0.0, "target_modules": ["q_proj"]},
            "grpo": {"batch_size": 1, "epochs": 1, "num_generations": 2},
            "output": {"save_steps": 50},
        }
        result = _build_skyrl_config("/model", "/out", config, "/data.jsonl")
        assert result["environment"]["env_class"] == "openctf"


class TestInferenceBackendSelection:
    def test_force_legacy_for_text_config(self, monkeypatch):
        import transformers

        class Qwen3_5TextConfig:
            pass

        class DummyAutoConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return Qwen3_5TextConfig()

        monkeypatch.setattr(transformers, "AutoConfig", DummyAutoConfig)
        assert _should_force_legacy_inference("/model") is True

    def test_keep_new_inference_for_standard_config(self, monkeypatch):
        import transformers

        class LlamaConfig:
            pass

        class DummyAutoConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return LlamaConfig()

        monkeypatch.setattr(transformers, "AutoConfig", DummyAutoConfig)
        assert _should_force_legacy_inference("/model") is False

    def test_config_probe_failure_does_not_force_legacy(self, monkeypatch):
        import transformers

        class DummyAutoConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError("boom")

        monkeypatch.setattr(transformers, "AutoConfig", DummyAutoConfig)
        assert _should_force_legacy_inference("/model") is False


class TestModelPathResolution:
    def test_text_wrapper_switches_to_sibling_vllm_path(self, monkeypatch, tmp_path):
        import transformers

        base_path = tmp_path / "model"
        vllm_path = tmp_path / "model_vllm"
        base_path.mkdir()
        vllm_path.mkdir()

        class Qwen3_5TextConfig:
            model_type = "qwen3_5_text"

        class Qwen3_5Config:
            model_type = "qwen3_5"

        class DummyAutoConfig:
            @staticmethod
            def from_pretrained(path, **kwargs):
                if str(path).endswith("_vllm"):
                    return Qwen3_5Config()
                return Qwen3_5TextConfig()

        monkeypatch.setattr(transformers, "AutoConfig", DummyAutoConfig)
        resolved = _resolve_vllm_ready_model_path(str(base_path))
        assert resolved == str(vllm_path)

    def test_non_text_wrapper_keeps_original_path(self, monkeypatch):
        import transformers

        class LlamaConfig:
            model_type = "llama"

        class DummyAutoConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return LlamaConfig()

        monkeypatch.setattr(transformers, "AutoConfig", DummyAutoConfig)
        assert _resolve_vllm_ready_model_path("/model") == "/model"

    def test_text_wrapper_without_sibling_keeps_original_path(self, monkeypatch):
        import transformers

        class Qwen3_5TextConfig:
            model_type = "qwen3_5_text"

        class DummyAutoConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                return Qwen3_5TextConfig()

        monkeypatch.setattr(transformers, "AutoConfig", DummyAutoConfig)
        assert _resolve_vllm_ready_model_path("/model") == "/model"


class TestTargetExtraction:
    def test_target_from_user_message(self, tmp_path):
        """Target URL should be extracted from user message content."""
        src = tmp_path / "with_target.jsonl"
        import jsonlines
        with jsonlines.open(str(src), "w") as w:
            w.write({
                "messages": [
                    {"role": "system", "content": "You are a CTF agent."},
                    {"role": "user", "content": "Solve the challenge at http://localhost:32801"},
                    {"role": "assistant", "content": "Scanning..."},
                ],
                "ground_truth_flag": "FLAG{target_test}",
                "metadata": {"challenge_id": "eval-me", "task_type": "ctf"},
            })
        output_dir = str(tmp_path / "out")
        result = _convert_grpo_data(str(src), output_dir)
        with jsonlines.open(result) as reader:
            row = next(iter(reader))
        assert row["target"] == "http://localhost:32801"

    def test_no_target_produces_none(self, tmp_path):
        """File-based challenges without URLs should have target=None."""
        src = tmp_path / "no_target.jsonl"
        import jsonlines
        with jsonlines.open(str(src), "w") as w:
            w.write({
                "messages": [
                    {"role": "system", "content": "You are a CTF agent."},
                    {"role": "user", "content": "Decrypt the ciphertext."},
                ],
                "ground_truth_flag": "FLAG{crypto}",
                "metadata": {"challenge_id": "Dynastic", "task_type": "ctf"},
            })
        output_dir = str(tmp_path / "out")
        result = _convert_grpo_data(str(src), output_dir)
        with jsonlines.open(result) as reader:
            row = next(iter(reader))
        assert row["target"] is None

    def test_challenge_id_fallback_to_challenge(self, tmp_path):
        """challenge_id should fall back to metadata.challenge if challenge_id missing."""
        src = tmp_path / "fallback.jsonl"
        import jsonlines
        with jsonlines.open(str(src), "w") as w:
            w.write({
                "messages": [
                    {"role": "system", "content": "Agent."},
                    {"role": "user", "content": "Solve it."},
                ],
                "ground_truth_flag": "FLAG{fb}",
                "metadata": {"challenge": "eval-me", "task_type": "ctf"},
            })
        output_dir = str(tmp_path / "out")
        result = _convert_grpo_data(str(src), output_dir)
        with jsonlines.open(result) as reader:
            row = next(iter(reader))
        assert row["challenge_id"] == "eval-me"

    def test_target_from_metadata_fallback(self, tmp_path):
        """If no URL in user message, target should come from metadata."""
        src = tmp_path / "meta_target.jsonl"
        import jsonlines
        with jsonlines.open(str(src), "w") as w:
            w.write({
                "messages": [
                    {"role": "system", "content": "Agent."},
                    {"role": "user", "content": "Solve the challenge."},
                ],
                "ground_truth_flag": "FLAG{mt}",
                "metadata": {"target": "http://localhost:9999", "task_type": "ctf"},
            })
        output_dir = str(tmp_path / "out")
        result = _convert_grpo_data(str(src), output_dir)
        with jsonlines.open(result) as reader:
            row = next(iter(reader))
        assert row["target"] == "http://localhost:9999"


class TestConvertGRPODataEdgeCases:
    def test_data_without_ground_truth_flag(self, tmp_path):
        """Samples without ground_truth_flag should still convert."""
        src = tmp_path / "no_flag.jsonl"
        import jsonlines
        with jsonlines.open(str(src), "w") as w:
            w.write({
                "messages": [
                    {"role": "system", "content": "Agent."},
                    {"role": "user", "content": "Scan the target."},
                ],
                "metadata": {"optimal_steps": 3},
            })
        output_dir = str(tmp_path / "out")
        result = _convert_grpo_data(str(src), output_dir)
        with jsonlines.open(result) as reader:
            row = next(iter(reader))
        # Should have a prompt and env_class even without flag
        assert "prompt" in row
        assert row["env_class"] == "openctf"

    def test_data_with_empty_metadata(self, tmp_path):
        """Samples with empty metadata dict should convert."""
        src = tmp_path / "empty_meta.jsonl"
        import jsonlines
        with jsonlines.open(str(src), "w") as w:
            w.write({
                "messages": [
                    {"role": "system", "content": "Agent."},
                    {"role": "user", "content": "Find the flag."},
                ],
                "ground_truth_flag": "FLAG{empty_meta}",
                "metadata": {},
            })
        output_dir = str(tmp_path / "out")
        result = _convert_grpo_data(str(src), output_dir)
        with jsonlines.open(result) as reader:
            row = next(iter(reader))
        assert row["ground_truth_flag"] == "FLAG{empty_meta}"


class TestRegistryIntegration:
    """Test _convert_grpo_data with ChallengeRegistry integration."""

    def _make_registry(self, tmp_path):
        """Create a simple test registry."""
        import yaml
        registry_data = {
            "challenges": [
                {
                    "id": "eval-me",
                    "category": "misc",
                    "difficulty": "very_easy",
                    "infra_type": "docker",
                    "port": 32805,
                },
                {
                    "id": "Dynastic",
                    "category": "crypto",
                    "difficulty": "very_easy",
                    "infra_type": "static",
                },
            ]
        }
        path = tmp_path / "test_registry.yaml"
        with open(path, "w") as f:
            yaml.dump(registry_data, f)
        from open_ctf.challenges.registry import ChallengeRegistry
        return ChallengeRegistry(str(path))

    def test_registry_provides_target_when_missing(self, tmp_path):
        """Registry should provide target URL when not in user message."""
        registry = self._make_registry(tmp_path)

        src = tmp_path / "grpo.jsonl"
        import jsonlines
        with jsonlines.open(str(src), "w") as w:
            w.write({
                "messages": [
                    {"role": "system", "content": "Agent."},
                    {"role": "user", "content": "Solve the eval-me challenge."},
                ],
                "ground_truth_flag": "FLAG{eval}",
                "metadata": {"challenge_id": "eval-me"},
            })

        output_dir = str(tmp_path / "out")
        result = _convert_grpo_data(str(src), output_dir, registry=registry)
        with jsonlines.open(result) as reader:
            row = next(iter(reader))
        assert row["target"] == "http://localhost:32805"

    def test_url_in_message_takes_precedence_over_registry(self, tmp_path):
        """If URL is in user message, it should override registry."""
        registry = self._make_registry(tmp_path)

        src = tmp_path / "grpo.jsonl"
        import jsonlines
        with jsonlines.open(str(src), "w") as w:
            w.write({
                "messages": [
                    {"role": "system", "content": "Agent."},
                    {"role": "user", "content": "Solve at http://localhost:9999"},
                ],
                "ground_truth_flag": "FLAG{override}",
                "metadata": {"challenge_id": "eval-me"},
            })

        output_dir = str(tmp_path / "out")
        result = _convert_grpo_data(str(src), output_dir, registry=registry)
        with jsonlines.open(result) as reader:
            row = next(iter(reader))
        assert row["target"] == "http://localhost:9999"  # Message URL wins

    def test_static_challenge_gets_none_from_registry(self, tmp_path):
        """Static challenges should get target=None even with registry."""
        registry = self._make_registry(tmp_path)

        src = tmp_path / "grpo.jsonl"
        import jsonlines
        with jsonlines.open(str(src), "w") as w:
            w.write({
                "messages": [
                    {"role": "system", "content": "Agent."},
                    {"role": "user", "content": "Solve the crypto puzzle."},
                ],
                "ground_truth_flag": "FLAG{dyn}",
                "metadata": {"challenge_id": "Dynastic"},
            })

        output_dir = str(tmp_path / "out")
        result = _convert_grpo_data(str(src), output_dir, registry=registry)
        with jsonlines.open(result) as reader:
            row = next(iter(reader))
        assert row["target"] is None  # Static = no URL

    def test_unknown_challenge_in_registry_returns_none(self, tmp_path):
        """Challenge not in registry should not crash, target stays None."""
        registry = self._make_registry(tmp_path)

        src = tmp_path / "grpo.jsonl"
        import jsonlines
        with jsonlines.open(str(src), "w") as w:
            w.write({
                "messages": [
                    {"role": "system", "content": "Agent."},
                    {"role": "user", "content": "Solve it."},
                ],
                "ground_truth_flag": "FLAG{unk}",
                "metadata": {"challenge_id": "unknown-challenge"},
            })

        output_dir = str(tmp_path / "out")
        result = _convert_grpo_data(str(src), output_dir, registry=registry)
        with jsonlines.open(result) as reader:
            row = next(iter(reader))
        assert row["target"] is None
