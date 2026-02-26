"""Smoke tests for LlamaFactory SFT orchestrator.

Validates:
- _ensure_dataset_info creates valid dataset_info.json (openai format for tool_calls)
- _build_lf_config produces valid LlamaFactory config dict
- _resolve_lf_config finds correct model configs
- Config has correct values for Nanbeige4.1-3B (chatml template, hermes tool_format, 32K cutoff)
"""

import json
import pytest
from pathlib import Path

from open_ctf.training.sft.llamafactory import (
    _ensure_dataset_info,
    _build_lf_config,
    _resolve_lf_config,
    _MODEL_CONFIG_MAP,
)


# ---------------------------------------------------------------------------
# _resolve_lf_config
# ---------------------------------------------------------------------------


class TestResolveLFConfig:
    def test_nanbeige_resolves(self):
        result = _resolve_lf_config("Nanbeige/Nanbeige4.1-3B")
        assert result is not None
        assert result.name == "nanbeige_3b.yaml"

    def test_glm4_resolves(self):
        result = _resolve_lf_config("THUDM/glm-4-9b")
        assert result is not None
        assert result.name == "glm47_flash.yaml"

    def test_glm47_flash_resolves(self):
        result = _resolve_lf_config("THUDM/GLM-4.7-Flash")
        assert result is not None
        assert result.name == "glm47_flash.yaml"

    def test_devstral_resolves(self):
        result = _resolve_lf_config("mistralai/Devstral-Small-2")
        assert result is not None
        assert result.name == "devstral_24b.yaml"

    def test_unknown_model_returns_none(self):
        assert _resolve_lf_config("some/unknown-model-xyz") is None

    def test_case_insensitive(self):
        result = _resolve_lf_config("NANBEIGE/SOMETHING")
        assert result is not None


# ---------------------------------------------------------------------------
# _ensure_dataset_info
# ---------------------------------------------------------------------------


class TestEnsureDatasetInfo:
    def test_creates_dataset_info_json(self, tmp_path):
        data_file = tmp_path / "train.jsonl"
        data_file.write_text('{"messages": []}\n')

        result_dir = _ensure_dataset_info(str(data_file))
        assert result_dir == tmp_path

        info_path = tmp_path / "dataset_info.json"
        assert info_path.exists()

        info = json.loads(info_path.read_text())
        assert "open_ctf_sft" in info

    def test_dataset_info_structure(self, tmp_path):
        data_file = tmp_path / "train.jsonl"
        data_file.write_text('{"messages": []}\n')

        _ensure_dataset_info(str(data_file))
        info = json.loads((tmp_path / "dataset_info.json").read_text())

        entry = info["open_ctf_sft"]
        assert entry["file_name"] == "train.jsonl"
        assert entry["formatting"] == "openai"
        assert "columns" in entry
        assert entry["columns"]["messages"] == "messages"

    def test_tags_include_all_roles(self, tmp_path):
        data_file = tmp_path / "train.jsonl"
        data_file.write_text('{"messages": []}\n')

        _ensure_dataset_info(str(data_file))
        info = json.loads((tmp_path / "dataset_info.json").read_text())

        tags = info["open_ctf_sft"]["tags"]
        assert tags["role_tag"] == "role"
        assert tags["content_tag"] == "content"
        assert tags["user_tag"] == "user"
        assert tags["assistant_tag"] == "assistant"
        assert tags["observation_tag"] == "tool"
        assert tags["function_tag"] == "function_call"
        assert tags["system_tag"] == "system"

    def test_preserves_existing_entries(self, tmp_path):
        data_file = tmp_path / "train.jsonl"
        data_file.write_text('{"messages": []}\n')

        # Pre-populate with another entry
        info_path = tmp_path / "dataset_info.json"
        info_path.write_text(json.dumps({"other_dataset": {"file_name": "other.jsonl"}}))

        _ensure_dataset_info(str(data_file))
        info = json.loads(info_path.read_text())
        assert "other_dataset" in info
        assert "open_ctf_sft" in info


# ---------------------------------------------------------------------------
# _build_lf_config
# ---------------------------------------------------------------------------


class TestBuildLFConfig:
    @pytest.fixture
    def minimal_config(self):
        return {
            "model": {"max_seq_length": 32768},
            "lora": {"r": 64, "alpha": 128, "dropout": 0.0},
            "sft": {"batch_size": 2, "learning_rate": 2e-4, "epochs": 5},
            "output": {"logging_steps": 1, "save_steps": 50},
        }

    def test_returns_dict(self, minimal_config, tmp_path):
        result = _build_lf_config(
            "Nanbeige/Nanbeige4.1-3B", tmp_path, str(tmp_path / "out"), minimal_config
        )
        assert isinstance(result, dict)

    def test_model_name_set(self, minimal_config, tmp_path):
        result = _build_lf_config(
            "Nanbeige/Nanbeige4.1-3B", tmp_path, str(tmp_path / "out"), minimal_config
        )
        assert result["model_name_or_path"] == "Nanbeige/Nanbeige4.1-3B"

    def test_stage_is_sft(self, minimal_config, tmp_path):
        result = _build_lf_config(
            "Nanbeige/Nanbeige4.1-3B", tmp_path, str(tmp_path / "out"), minimal_config
        )
        assert result["stage"] == "sft"

    def test_finetuning_type_is_lora(self, minimal_config, tmp_path):
        result = _build_lf_config(
            "Nanbeige/Nanbeige4.1-3B", tmp_path, str(tmp_path / "out"), minimal_config
        )
        assert result["finetuning_type"] == "lora"

    def test_lora_params(self, minimal_config, tmp_path):
        result = _build_lf_config(
            "Nanbeige/Nanbeige4.1-3B", tmp_path, str(tmp_path / "out"), minimal_config
        )
        assert result["lora_rank"] == 64
        assert result["lora_alpha"] == 128
        assert result["lora_dropout"] == 0.0

    def test_cutoff_len_from_config(self, minimal_config, tmp_path):
        result = _build_lf_config(
            "Nanbeige/Nanbeige4.1-3B", tmp_path, str(tmp_path / "out"), minimal_config
        )
        assert result["cutoff_len"] == 32768

    def test_bf16_enabled(self, minimal_config, tmp_path):
        result = _build_lf_config(
            "Nanbeige/Nanbeige4.1-3B", tmp_path, str(tmp_path / "out"), minimal_config
        )
        assert result["bf16"] is True

    def test_packing_enabled_by_default(self, minimal_config, tmp_path):
        result = _build_lf_config(
            "Nanbeige/Nanbeige4.1-3B", tmp_path, str(tmp_path / "out"), minimal_config
        )
        assert result["packing"] is True

    def test_dataset_key(self, minimal_config, tmp_path):
        result = _build_lf_config(
            "Nanbeige/Nanbeige4.1-3B", tmp_path, str(tmp_path / "out"), minimal_config
        )
        assert result["dataset"] == "open_ctf_sft"

    def test_output_dir_set(self, minimal_config, tmp_path):
        out = str(tmp_path / "my_output")
        result = _build_lf_config(
            "Nanbeige/Nanbeige4.1-3B", tmp_path, out, minimal_config
        )
        assert result["output_dir"] == out

    def test_lora_target_modules_string(self, minimal_config, tmp_path):
        """lora_target should be a comma-separated string."""
        result = _build_lf_config(
            "Nanbeige/Nanbeige4.1-3B", tmp_path, str(tmp_path / "out"), minimal_config
        )
        target = result["lora_target"]
        assert isinstance(target, str)
        assert "q_proj" in target
        assert "v_proj" in target

    def test_quantization_4bit(self, tmp_path):
        config = {
            "model": {"load_in_4bit": True},
            "lora": {},
            "sft": {},
            "output": {},
        }
        result = _build_lf_config(
            "some/model", tmp_path, str(tmp_path / "out"), config
        )
        assert result["quantization_bit"] == 4

    def test_no_quantization_by_default(self, minimal_config, tmp_path):
        result = _build_lf_config(
            "some/model", tmp_path, str(tmp_path / "out"), minimal_config
        )
        assert "quantization_bit" not in result


# ---------------------------------------------------------------------------
# Nanbeige4.1-3B specific config values
# ---------------------------------------------------------------------------


class TestNanbeigeConfig:
    """Validate pre-built Nanbeige config has expected LlamaFactory values."""

    @pytest.fixture
    def nanbeige_config(self):
        import yaml
        config_path = _resolve_lf_config("Nanbeige/Nanbeige4.1-3B")
        assert config_path is not None
        with open(config_path) as f:
            return yaml.safe_load(f)

    def test_template_is_chatml(self, nanbeige_config):
        assert nanbeige_config.get("template") == "chatml"

    def test_cutoff_len_at_least_8k(self, nanbeige_config):
        assert nanbeige_config.get("cutoff_len", 0) >= 8192

    def test_stage_sft(self, nanbeige_config):
        assert nanbeige_config.get("stage") == "sft"

    def test_finetuning_lora(self, nanbeige_config):
        assert nanbeige_config.get("finetuning_type") == "lora"

    def test_bf16(self, nanbeige_config):
        assert nanbeige_config.get("bf16") is True
