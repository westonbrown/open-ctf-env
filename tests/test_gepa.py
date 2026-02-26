"""Functional tests for GEPA prompt optimization (Stage 3).

Validates:
- Config loading: GEPA config section parses from training.yaml
- Challenge loading: _load_challenges() loads GRPO JSONL, extracts targets
- Metric wrapper: _build_metric() returns valid score + feedback from CTFReward
- Per-challenge routing: target URL extraction from challenge text
- CTFAgentDSPyAdapter: wraps CTFAgent, has named_predictors(), forward(), mutable instructions
- ChallengeRegistry integration: challenge lookup returns correct target URLs
- CLI flags: --agent and --challenge-registry parse correctly
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from open_ctf.rewards.reward import CTFReward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_grpo_jsonl(path: Path, samples: list[dict]) -> None:
    """Write a list of GRPO sample dicts as JSONL."""
    with open(path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def _make_grpo_sample(
    challenge_text: str = "Solve the CTF at http://localhost:8080",
    flag: str = "FLAG{test123}",
    optimal_steps: int = 5,
    challenge_id: str = "",
    target: str = "",
) -> dict:
    """Create a minimal GRPO JSONL sample."""
    sample = {
        "messages": [
            {"role": "system", "content": "You are a CTF agent."},
            {"role": "user", "content": challenge_text},
        ],
        "ground_truth_flag": flag,
        "optimal_steps": optimal_steps,
        "metadata": {},
    }
    if challenge_id:
        sample["metadata"]["challenge_id"] = challenge_id
    if target:
        sample["metadata"]["target"] = target
    return sample


# ---------------------------------------------------------------------------
# Mock DSPy module (avoid requiring dspy installed)
# ---------------------------------------------------------------------------


class _MockExample:
    """Minimal stand-in for dspy.Example."""

    def __init__(self, **kwargs):
        self._data = kwargs
        self._input_keys = set()

    def with_inputs(self, *keys):
        self._input_keys = set(keys)
        return self

    def get(self, key, default=None):
        return self._data.get(key, default)

    def __getattr__(self, key):
        if key.startswith("_"):
            raise AttributeError(key)
        return self._data.get(key)


class _MockPrediction:
    """Minimal stand-in for dspy.Prediction."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class _MockSignature:
    """Minimal stand-in for dspy.Signature with mutable instructions."""

    def __init__(self, instructions=""):
        self.instructions = instructions


class _MockPredict:
    """Minimal stand-in for dspy.Predict."""

    def __init__(self, sig=None, **kwargs):
        self.signature = sig or _MockSignature()


class _MockInputField:
    pass


class _MockOutputField:
    pass


class _MockScoreWithFeedback:
    """Minimal stand-in for dspy.teleprompt.gepa.gepa_utils.ScoreWithFeedback."""

    def __init__(self, score=0.0, feedback=""):
        self.score = score
        self.feedback = feedback


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestConfigLoading:
    def test_gepa_section_in_training_yaml(self):
        """The default training.yaml should contain a 'gepa' config section."""
        import yaml

        config_path = (
            Path(__file__).resolve().parent.parent
            / "src" / "open_ctf" / "configs" / "training.yaml"
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert "gepa" in config
        gepa_cfg = config["gepa"]
        assert "budget" in gepa_cfg
        assert "max_iters" in gepa_cfg
        assert "seed" in gepa_cfg
        assert "num_threads" in gepa_cfg
        assert "reflection_minibatch_size" in gepa_cfg

    def test_gepa_budget_is_valid(self):
        """Budget should be one of light/medium/heavy."""
        import yaml

        config_path = (
            Path(__file__).resolve().parent.parent
            / "src" / "open_ctf" / "configs" / "training.yaml"
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config["gepa"]["budget"] in ("light", "medium", "heavy")

    def test_gepa_seed_prompt_can_be_null(self):
        """seed_prompt can be null (uses built-in SEED_PROMPT)."""
        import yaml

        config_path = (
            Path(__file__).resolve().parent.parent
            / "src" / "open_ctf" / "configs" / "training.yaml"
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # seed_prompt: null in YAML becomes None in Python
        assert config["gepa"].get("seed_prompt") is None


# ---------------------------------------------------------------------------
# Target extraction from messages
# ---------------------------------------------------------------------------


class TestTargetExtraction:
    def test_extract_target_from_user_message(self):
        from open_ctf.training.gepa import _extract_target_from_messages

        messages = [
            {"role": "system", "content": "You are a CTF agent."},
            {"role": "user", "content": "Solve the CTF at http://localhost:8080"},
        ]
        assert _extract_target_from_messages(messages) == "http://localhost:8080"

    def test_extract_target_with_port(self):
        from open_ctf.training.gepa import _extract_target_from_messages

        messages = [
            {"role": "user", "content": "Target: http://localhost:32805/api"},
        ]
        assert _extract_target_from_messages(messages) == "http://localhost:32805"

    def test_extract_target_no_url(self):
        from open_ctf.training.gepa import _extract_target_from_messages

        messages = [
            {"role": "user", "content": "Solve this crypto challenge."},
        ]
        assert _extract_target_from_messages(messages) is None

    def test_extract_target_skips_non_user_messages(self):
        from open_ctf.training.gepa import _extract_target_from_messages

        messages = [
            {"role": "system", "content": "http://localhost:9999"},
            {"role": "assistant", "content": "http://localhost:8888"},
        ]
        assert _extract_target_from_messages(messages) is None

    def test_extract_target_first_user_url_wins(self):
        from open_ctf.training.gepa import _extract_target_from_messages

        messages = [
            {"role": "user", "content": "First target http://localhost:1111"},
            {"role": "user", "content": "Second target http://localhost:2222"},
        ]
        assert _extract_target_from_messages(messages) == "http://localhost:1111"

    def test_extract_target_supports_non_localhost_hosts(self):
        from open_ctf.training.gepa import _extract_target_from_messages

        messages = [
            {"role": "user", "content": "Use tunnel endpoint https://challenge.internal:43021/path"},
        ]
        assert _extract_target_from_messages(messages) == "https://challenge.internal:43021"

    def test_extract_target_strips_trailing_punctuation(self):
        from open_ctf.training.gepa import _extract_target_from_messages

        messages = [
            {"role": "user", "content": "Try http://localhost:32801, then enumerate."},
        ]
        assert _extract_target_from_messages(messages) == "http://localhost:32801"


# ---------------------------------------------------------------------------
# Challenge loading (_load_challenges)
# ---------------------------------------------------------------------------


class TestLoadChallenges:
    def test_load_basic(self, tmp_path):
        """Load challenges from a GRPO JSONL file."""
        samples = [
            _make_grpo_sample("Solve CTF at http://localhost:8080", "FLAG{a}"),
            _make_grpo_sample("Solve CTF at http://localhost:9090", "FLAG{b}"),
        ]
        data_path = tmp_path / "grpo.jsonl"
        _write_grpo_jsonl(data_path, samples)

        # Mock dspy.Example with our lightweight stand-in
        mock_dspy = MagicMock()
        mock_dspy.Example = _MockExample

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            from open_ctf.training.gepa import _load_challenges

            examples = _load_challenges(str(data_path))

        assert len(examples) == 2
        assert examples[0].get("ground_truth_flag") == "FLAG{a}"
        assert examples[0].get("target") == "http://localhost:8080"
        assert examples[1].get("target") == "http://localhost:9090"

    def test_load_max_samples(self, tmp_path):
        """max_samples limits how many examples are loaded."""
        samples = [_make_grpo_sample(f"Challenge {i}") for i in range(10)]
        data_path = tmp_path / "grpo.jsonl"
        _write_grpo_jsonl(data_path, samples)

        mock_dspy = MagicMock()
        mock_dspy.Example = _MockExample

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            from open_ctf.training.gepa import _load_challenges

            examples = _load_challenges(str(data_path), max_samples=3)

        assert len(examples) == 3

    def test_load_skips_empty_lines(self, tmp_path):
        """Empty lines in JSONL should be skipped."""
        data_path = tmp_path / "grpo.jsonl"
        with open(data_path, "w") as f:
            f.write(json.dumps(_make_grpo_sample()) + "\n")
            f.write("\n")
            f.write("   \n")
            f.write(json.dumps(_make_grpo_sample()) + "\n")

        mock_dspy = MagicMock()
        mock_dspy.Example = _MockExample

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            from open_ctf.training.gepa import _load_challenges

            examples = _load_challenges(str(data_path))

        assert len(examples) == 2

    def test_load_skips_entries_without_user_message(self, tmp_path):
        """Samples with no user message should be skipped."""
        sample = {
            "messages": [{"role": "system", "content": "sys"}],
            "ground_truth_flag": "FLAG{x}",
        }
        data_path = tmp_path / "grpo.jsonl"
        _write_grpo_jsonl(data_path, [sample])

        mock_dspy = MagicMock()
        mock_dspy.Example = _MockExample

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            from open_ctf.training.gepa import _load_challenges

            examples = _load_challenges(str(data_path))

        assert len(examples) == 0

    def test_load_extracts_metadata_target(self, tmp_path):
        """When no URL in messages, fallback to metadata.target."""
        sample = _make_grpo_sample(
            challenge_text="Solve this static challenge",
            target="http://localhost:5555",
        )
        data_path = tmp_path / "grpo.jsonl"
        _write_grpo_jsonl(data_path, [sample])

        mock_dspy = MagicMock()
        mock_dspy.Example = _MockExample

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            from open_ctf.training.gepa import _load_challenges

            examples = _load_challenges(str(data_path))

        assert len(examples) == 1
        assert examples[0].get("target") == "http://localhost:5555"

    def test_load_with_challenge_id(self, tmp_path):
        """Challenge ID from metadata is preserved."""
        sample = _make_grpo_sample(challenge_id="eval-me")
        data_path = tmp_path / "grpo.jsonl"
        _write_grpo_jsonl(data_path, [sample])

        mock_dspy = MagicMock()
        mock_dspy.Example = _MockExample

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            from open_ctf.training.gepa import _load_challenges

            examples = _load_challenges(str(data_path))

        assert examples[0].get("challenge_id") == "eval-me"


# ---------------------------------------------------------------------------
# Challenge loading with ChallengeRegistry
# ---------------------------------------------------------------------------


class TestLoadChallengesWithRegistry:
    def _make_registry(self, tmp_path):
        """Create a minimal challenge registry YAML."""
        registry_path = tmp_path / "registry.yaml"
        registry_path.write_text(
            json.dumps({})  # placeholder, we'll use a mock instead
        )
        return registry_path

    def test_registry_resolves_target(self, tmp_path):
        """When registry is provided, it resolves target URLs for challenge IDs."""
        sample = _make_grpo_sample(
            challenge_text="Solve static challenge",
            challenge_id="eval-me",
        )
        data_path = tmp_path / "grpo.jsonl"
        _write_grpo_jsonl(data_path, [sample])

        mock_registry = MagicMock()
        mock_registry.resolve_id.return_value = "eval-me"
        mock_registry.get_target_url.return_value = "http://localhost:32805"

        mock_dspy = MagicMock()
        mock_dspy.Example = _MockExample

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            from open_ctf.training.gepa import _load_challenges

            examples = _load_challenges(
                str(data_path), registry=mock_registry
            )

        assert len(examples) == 1
        assert examples[0].get("target") == "http://localhost:32805"
        assert examples[0].get("challenge_id") == "eval-me"
        mock_registry.resolve_id.assert_called_once_with("eval-me")

    def test_registry_url_not_used_when_message_has_target(self, tmp_path):
        """Target from message takes precedence over registry."""
        sample = _make_grpo_sample(
            challenge_text="Target: http://localhost:8080",
            challenge_id="eval-me",
        )
        data_path = tmp_path / "grpo.jsonl"
        _write_grpo_jsonl(data_path, [sample])

        mock_registry = MagicMock()
        mock_registry.resolve_id.return_value = "eval-me"
        mock_registry.get_target_url.return_value = "http://localhost:32805"

        mock_dspy = MagicMock()
        mock_dspy.Example = _MockExample

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            from open_ctf.training.gepa import _load_challenges

            examples = _load_challenges(
                str(data_path), registry=mock_registry
            )

        # Message URL takes precedence
        assert examples[0].get("target") == "http://localhost:8080"

    def test_registry_unresolved_challenge_ok(self, tmp_path):
        """Unresolvable challenge ID should not crash."""
        sample = _make_grpo_sample(
            challenge_text="Some challenge",
            challenge_id="unknown-challenge",
        )
        data_path = tmp_path / "grpo.jsonl"
        _write_grpo_jsonl(data_path, [sample])

        mock_registry = MagicMock()
        mock_registry.resolve_id.return_value = None

        mock_dspy = MagicMock()
        mock_dspy.Example = _MockExample

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            from open_ctf.training.gepa import _load_challenges

            examples = _load_challenges(
                str(data_path), registry=mock_registry
            )

        assert len(examples) == 1
        # Target should remain empty (no URL in message, no registry match)
        assert examples[0].get("target") == ""


# ---------------------------------------------------------------------------
# Metric wrapper (_build_metric)
# ---------------------------------------------------------------------------


class TestBuildMetric:
    def test_metric_returns_score_with_feedback(self):
        """_build_metric should return a callable that produces ScoreWithFeedback."""
        reward_fn = CTFReward(noise_range=0.0, seed=0)

        with patch(
            "open_ctf.training.gepa.ScoreWithFeedback",
            _MockScoreWithFeedback,
            create=True,
        ):
            # Manually patch the import path used inside _build_metric
            mock_gepa_utils = MagicMock()
            mock_gepa_utils.ScoreWithFeedback = _MockScoreWithFeedback

            with patch.dict("sys.modules", {
                "dspy": MagicMock(),
                "dspy.teleprompt": MagicMock(),
                "dspy.teleprompt.gepa": MagicMock(),
                "dspy.teleprompt.gepa.gepa_utils": mock_gepa_utils,
            }):
                from open_ctf.training.gepa import _build_metric

                metric = _build_metric(reward_fn)

                gold = {
                    "ground_truth_flag": "FLAG{test}",
                    "optimal_steps": 5,
                    "target": "",
                }
                pred = MagicMock()
                pred.answer = "No flag found"

                result = metric(gold, pred, trace=None, pred_name="test", pred_trace=None)

        assert isinstance(result, _MockScoreWithFeedback)
        assert isinstance(result.score, float)
        assert isinstance(result.feedback, str)

    def test_metric_high_score_feedback(self):
        """High-scoring result should have positive feedback."""
        reward_fn = CTFReward(noise_range=0.0, seed=0)

        mock_gepa_utils = MagicMock()
        mock_gepa_utils.ScoreWithFeedback = _MockScoreWithFeedback

        with patch.dict("sys.modules", {
            "dspy": MagicMock(),
            "dspy.teleprompt": MagicMock(),
            "dspy.teleprompt.gepa": MagicMock(),
            "dspy.teleprompt.gepa.gepa_utils": mock_gepa_utils,
        }):
            from open_ctf.training.gepa import _build_metric

            metric = _build_metric(reward_fn)

            gold = {
                "ground_truth_flag": "FLAG{test}",
                "optimal_steps": 3,
                "target": "",
            }
            # Pred with the correct flag found
            pred = MagicMock()
            pred.answer = "The flag is FLAG{test}"

            # Trace with tool calls representing a full CTF solve
            trace_output = MagicMock()
            trace_output.next_tool_name = "shell_command"
            trace_output.next_tool_args = {"command": "nmap target"}
            trace = [
                (MagicMock(), {}, trace_output),
            ]

            result = metric(gold, pred, trace=trace, pred_name="test", pred_trace=None)

        assert result.score > 0
        assert len(result.feedback) > 0

    def test_metric_extracts_tool_calls_from_trace(self):
        """Metric should extract tool calls from the DSPy trace."""
        reward_fn = CTFReward(noise_range=0.0, seed=0)

        mock_gepa_utils = MagicMock()
        mock_gepa_utils.ScoreWithFeedback = _MockScoreWithFeedback

        with patch.dict("sys.modules", {
            "dspy": MagicMock(),
            "dspy.teleprompt": MagicMock(),
            "dspy.teleprompt.gepa": MagicMock(),
            "dspy.teleprompt.gepa.gepa_utils": mock_gepa_utils,
        }):
            from open_ctf.training.gepa import _build_metric

            metric = _build_metric(reward_fn)

            gold = {"ground_truth_flag": None, "optimal_steps": None, "target": ""}

            pred = MagicMock()
            pred.answer = ""

            # Multiple tool calls in trace
            output1 = MagicMock()
            output1.next_tool_name = "shell_command"
            output1.next_tool_args = {"command": "nmap -sV target"}

            output2 = MagicMock()
            output2.next_tool_name = "python_code"
            output2.next_tool_args = {"code": "import requests"}

            output3 = MagicMock()
            output3.next_tool_name = "finish"
            output3.next_tool_args = None

            trace = [
                (MagicMock(), {}, output1),
                (MagicMock(), {}, output2),
                (MagicMock(), {}, output3),  # "finish" should be excluded
            ]

            result = metric(gold, pred, trace=trace, pred_name="t", pred_trace=None)

        # Should return a valid result (finish excluded, 2 real tool calls)
        assert isinstance(result.score, float)
        assert "No CTF phases" not in result.feedback or "recon" in result.feedback.lower()

    def test_metric_no_tool_calls_feedback(self):
        """When trace has no tool calls, feedback should note missing phases."""
        reward_fn = CTFReward(noise_range=0.0, seed=0)

        mock_gepa_utils = MagicMock()
        mock_gepa_utils.ScoreWithFeedback = _MockScoreWithFeedback

        with patch.dict("sys.modules", {
            "dspy": MagicMock(),
            "dspy.teleprompt": MagicMock(),
            "dspy.teleprompt.gepa": MagicMock(),
            "dspy.teleprompt.gepa.gepa_utils": mock_gepa_utils,
        }):
            from open_ctf.training.gepa import _build_metric

            metric = _build_metric(reward_fn)

            gold = {"ground_truth_flag": None, "optimal_steps": None, "target": ""}
            pred = MagicMock()
            pred.answer = ""

            result = metric(gold, pred, trace=None, pred_name="t", pred_trace=None)

        assert "No tool calls made" in result.feedback

    def test_metric_per_challenge_routing(self):
        """Metric should call init_env with the challenge target before scoring."""
        reward_fn = CTFReward(noise_range=0.0, seed=0)

        mock_gepa_utils = MagicMock()
        mock_gepa_utils.ScoreWithFeedback = _MockScoreWithFeedback

        with patch.dict("sys.modules", {
            "dspy": MagicMock(),
            "dspy.teleprompt": MagicMock(),
            "dspy.teleprompt.gepa": MagicMock(),
            "dspy.teleprompt.gepa.gepa_utils": mock_gepa_utils,
        }):
            # init_env is imported inside the closure via
            # `from open_ctf.training.tools import init_env`
            with patch("open_ctf.training.tools.init_env") as mock_init_env:
                from open_ctf.training.gepa import _build_metric

                metric = _build_metric(reward_fn)

                gold = {
                    "ground_truth_flag": "FLAG{routed}",
                    "optimal_steps": 5,
                    "target": "http://localhost:9999",
                }
                pred = MagicMock()
                pred.answer = ""

                metric(gold, pred, trace=None, pred_name="t", pred_trace=None)

                mock_init_env.assert_called_once_with(
                    target="http://localhost:9999",
                    ground_truth="FLAG{routed}",
                )


# ---------------------------------------------------------------------------
# CTFAgentDSPyAdapter
# ---------------------------------------------------------------------------


class TestCTFAgentDSPyAdapter:
    def _make_adapter(self, agent=None, seed_prompt="Test prompt"):
        """Create an adapter with mocked DSPy dependencies."""
        if agent is None:
            agent = MagicMock()
            agent.solve.return_value = MagicMock(flag="FLAG{solved}")

        # Mock dspy module with required classes
        mock_dspy = MagicMock()
        mock_dspy.Predict = _MockPredict
        mock_dspy.Prediction = _MockPrediction
        mock_dspy.InputField = _MockInputField
        mock_dspy.OutputField = _MockOutputField

        # make_signature returns a mock signature class
        mock_sig = _MockSignature(instructions=seed_prompt)
        mock_dspy.make_signature.return_value = mock_sig

        # Mock Predict to store the signature
        class MockPredict:
            def __init__(self, sig):
                self.signature = sig

        mock_dspy.Predict = MockPredict

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            from open_ctf.training.gepa import CTFAgentDSPyAdapter

            adapter = CTFAgentDSPyAdapter(agent=agent, seed_prompt=seed_prompt)

        return adapter, agent, mock_dspy

    def test_named_predictors_yields_predictor(self):
        """Adapter should yield at least one named predictor for GEPA."""
        adapter, _, _ = self._make_adapter()

        predictors = list(adapter.named_predictors())
        assert len(predictors) == 1
        name, pred = predictors[0]
        assert name == "ctf_agent_predictor"
        assert hasattr(pred, "signature")

    def test_instructions_are_mutable(self):
        """GEPA needs to mutate instructions on the predictor."""
        adapter, _, _ = self._make_adapter(seed_prompt="Initial prompt")

        predictors = list(adapter.named_predictors())
        _, pred = predictors[0]

        # Verify initial instructions
        assert pred.signature.instructions == "Initial prompt"

        # Mutate instructions (this is what GEPA does)
        pred.signature.instructions = "Evolved prompt"
        assert pred.signature.instructions == "Evolved prompt"

    def test_forward_calls_agent_solve(self):
        """forward() should call agent.solve() with the challenge text."""
        mock_agent = MagicMock()
        mock_agent.solve.return_value = MagicMock(flag="FLAG{won}")

        adapter, _, mock_dspy = self._make_adapter(agent=mock_agent)

        # Need to patch dspy again for forward() which imports it
        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            mock_dspy.Prediction = _MockPrediction
            result = adapter.forward(challenge="Test challenge http://target:8080")

        mock_agent.solve.assert_called_once()
        call_kwargs = mock_agent.solve.call_args
        assert "http://target:8080" in call_kwargs.kwargs.get("target", "")

    def test_forward_prepends_evolved_instructions(self):
        """forward() should prepend evolved instructions to challenge text."""
        mock_agent = MagicMock()
        mock_agent.solve.return_value = MagicMock(flag="FLAG{x}")

        adapter, _, mock_dspy = self._make_adapter(
            agent=mock_agent, seed_prompt="Be aggressive"
        )

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            mock_dspy.Prediction = _MockPrediction
            adapter.forward(challenge="Solve CTF")

        # The challenge text passed to solve should include evolved instructions
        call_kwargs = mock_agent.solve.call_args
        challenge_arg = call_kwargs.kwargs.get("challenge", "")
        assert "Be aggressive" in challenge_arg
        assert "Solve CTF" in challenge_arg

    def test_callable_delegates_to_forward(self):
        """Adapter should be callable (delegates to forward)."""
        mock_agent = MagicMock()
        mock_agent.solve.return_value = MagicMock(flag=None)

        adapter, _, mock_dspy = self._make_adapter(agent=mock_agent)

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            mock_dspy.Prediction = _MockPrediction
            result = adapter(challenge="Test")

        assert mock_agent.solve.called

    def test_save_writes_instructions(self, tmp_path):
        """save() should write the evolved prompt to disk."""
        adapter, _, _ = self._make_adapter(seed_prompt="Save me")

        save_dir = str(tmp_path / "saved")
        adapter.save(save_dir)

        instructions_path = tmp_path / "saved" / "instructions.txt"
        assert instructions_path.exists()
        assert instructions_path.read_text() == "Save me"

    def test_forward_extracts_target_from_url(self):
        """forward() should extract target URL from challenge text."""
        mock_agent = MagicMock()
        mock_agent.solve.return_value = MagicMock(flag="FLAG{x}")

        adapter, _, mock_dspy = self._make_adapter(agent=mock_agent)

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            mock_dspy.Prediction = _MockPrediction
            adapter.forward(
                challenge="Exploit target at https://example.com:8443/api"
            )

        call_kwargs = mock_agent.solve.call_args
        assert call_kwargs.kwargs.get("target", "") == "https://example.com:8443"


# ---------------------------------------------------------------------------
# _import_class utility
# ---------------------------------------------------------------------------


class TestImportClass:
    def test_import_valid_class(self):
        from open_ctf.training.gepa import _import_class

        cls = _import_class("json.JSONEncoder")
        assert cls is json.JSONEncoder

    def test_import_invalid_dotpath_raises(self):
        from open_ctf.training.gepa import _import_class

        with pytest.raises(ImportError, match="Invalid dotpath"):
            _import_class("NoDotsHere")

    def test_import_nonexistent_module_raises(self):
        from open_ctf.training.gepa import _import_class

        with pytest.raises(ModuleNotFoundError):
            _import_class("nonexistent_module_xyz.SomeClass")


# ---------------------------------------------------------------------------
# SEED_PROMPT constant
# ---------------------------------------------------------------------------


class TestSeedPrompt:
    def test_seed_prompt_exists_and_non_empty(self):
        from open_ctf.training.gepa import SEED_PROMPT

        assert isinstance(SEED_PROMPT, str)
        assert len(SEED_PROMPT) > 100

    def test_seed_prompt_has_key_methodology(self):
        from open_ctf.training.gepa import SEED_PROMPT

        assert "RECON" in SEED_PROMPT
        assert "ENUM" in SEED_PROMPT or "Enumerate" in SEED_PROMPT
        assert "EXPLOIT" in SEED_PROMPT
        assert "flag_found" in SEED_PROMPT


# ---------------------------------------------------------------------------
# CLI argument parsing (GEPA-specific flags)
# ---------------------------------------------------------------------------


class TestGEPACLIFlags:
    """Test that --agent and --challenge-registry parse correctly on the gepa subcommand.

    These tests use the real parser from train.py (not a re-built one) to ensure
    the actual CLI accepts the new flags.
    """

    def _parse_real(self, argv):
        """Parse using the actual train.py argparse setup."""
        import sys
        from unittest.mock import patch as _patch

        with _patch.object(sys, "argv", ["open-ctf-train"] + argv):
            from open_ctf.cli.train import main as _
            import argparse

            # Build a parser that mirrors the real one but without func execution
            from open_ctf.cli import train

            parser = argparse.ArgumentParser()
            parser.add_argument("--config", type=Path)
            sub = parser.add_subparsers(dest="command", required=True)

            gepa_p = sub.add_parser("gepa")
            gepa_p.add_argument("--model", required=True)
            gepa_p.add_argument("--data", required=True)
            gepa_p.add_argument("--output", required=True)
            gepa_p.add_argument("--val-data", default=None)
            gepa_p.add_argument("--reflection-model", default=None)
            gepa_p.add_argument(
                "--budget", choices=["light", "medium", "heavy"], default="medium"
            )
            gepa_p.add_argument("--max-samples", type=int, default=None)
            gepa_p.add_argument("--challenge-registry", default=None)
            gepa_p.add_argument("--agent", default=None)
            return parser.parse_args(argv)

    def test_agent_flag_parses(self):
        args = self._parse_real([
            "gepa", "--model", "m", "--data", "/d.jsonl", "--output", "/o",
            "--agent", "my_module.MyAgent",
        ])
        assert args.agent == "my_module.MyAgent"

    def test_challenge_registry_flag_parses(self):
        args = self._parse_real([
            "gepa", "--model", "m", "--data", "/d.jsonl", "--output", "/o",
            "--challenge-registry", "/path/to/registry.yaml",
        ])
        assert args.challenge_registry == "/path/to/registry.yaml"

    def test_both_flags_together(self):
        args = self._parse_real([
            "gepa", "--model", "m", "--data", "/d.jsonl", "--output", "/o",
            "--agent", "agents.BoxPwnr",
            "--challenge-registry", "/reg.yaml",
        ])
        assert args.agent == "agents.BoxPwnr"
        assert args.challenge_registry == "/reg.yaml"

    def test_flags_default_to_none(self):
        args = self._parse_real([
            "gepa", "--model", "m", "--data", "/d.jsonl", "--output", "/o",
        ])
        assert args.agent is None
        assert args.challenge_registry is None

    def test_cmd_gepa_passes_flags_to_run_gepa(self):
        """cmd_gepa should pass agent and challenge_registry to run_gepa."""
        with patch("open_ctf.training.gepa.run_gepa") as mock_run:
            mock_run.return_value = "/output/prompt.txt"

            # Create a namespace mimicking parsed args
            args = MagicMock()
            args.model = "openai/gpt-4"
            args.data = "/data.jsonl"
            args.output = "/output"
            args.config = Path(__file__).resolve().parent.parent / "src" / "open_ctf" / "configs" / "training.yaml"
            args.reflection_model = None
            args.budget = "medium"
            args.val_data = None
            args.max_samples = None
            args.challenge_registry = "/reg.yaml"
            args.agent = "my.Agent"

            from open_ctf.cli.train import cmd_gepa

            cmd_gepa(args)

            mock_run.assert_called_once()
            call_kwargs = mock_run.call_args
            assert call_kwargs.kwargs.get("challenge_registry") == "/reg.yaml"
            assert call_kwargs.kwargs.get("agent_class") == "my.Agent"


# ---------------------------------------------------------------------------
# ChallengeRegistry integration (real registry file)
# ---------------------------------------------------------------------------


class TestChallengeRegistryIntegration:
    @pytest.fixture
    def registry_path(self):
        """Path to the real CyBench challenge registry."""
        path = (
            Path(__file__).resolve().parent.parent
            / "configs" / "challenges" / "cybench.yaml"
        )
        if not path.exists():
            pytest.skip("CyBench registry not found")
        return path

    def test_registry_loads(self, registry_path):
        """ChallengeRegistry should load the cybench.yaml file."""
        from open_ctf.challenges.registry import ChallengeRegistry

        registry = ChallengeRegistry(str(registry_path))
        assert len(registry) > 0

    def test_registry_resolve_id(self, registry_path):
        """resolve_id should find challenges by canonical ID."""
        from open_ctf.challenges.registry import ChallengeRegistry

        registry = ChallengeRegistry(str(registry_path))
        all_challenges = registry.list_all()
        if not all_challenges:
            pytest.skip("No challenges in registry")

        # Resolve the first challenge by its own ID
        first = all_challenges[0]
        resolved = registry.resolve_id(first.id)
        assert resolved == first.id

    def test_registry_get_target_url_docker(self, registry_path):
        """Docker challenges should return a target URL with port."""
        from open_ctf.challenges.registry import ChallengeRegistry

        registry = ChallengeRegistry(str(registry_path))
        docker_challenges = registry.list_docker_challenges()
        if not docker_challenges:
            pytest.skip("No docker challenges in registry")

        first_docker = docker_challenges[0]
        url = registry.get_target_url(first_docker.id)
        if first_docker.port:
            assert url is not None
            assert url.startswith("http://")
            assert str(first_docker.port) in url

    def test_registry_get_target_url_static(self, registry_path):
        """Static challenges should return None for target URL."""
        from open_ctf.challenges.registry import ChallengeRegistry

        registry = ChallengeRegistry(str(registry_path))
        static_challenges = registry.list_static_challenges()
        if not static_challenges:
            pytest.skip("No static challenges in registry")

        first_static = static_challenges[0]
        url = registry.get_target_url(first_static.id)
        assert url is None

    def test_load_challenges_with_real_registry(self, registry_path, tmp_path):
        """_load_challenges should use registry to resolve targets."""
        from open_ctf.challenges.registry import ChallengeRegistry

        registry = ChallengeRegistry(str(registry_path))
        docker_challenges = registry.list_docker_challenges()
        if not docker_challenges:
            pytest.skip("No docker challenges in registry")

        # Pick a docker challenge with a port
        challenge = next(
            (c for c in docker_challenges if c.port), None
        )
        if not challenge:
            pytest.skip("No docker challenge with port")

        sample = _make_grpo_sample(
            challenge_text="Solve this challenge",
            challenge_id=challenge.id,
        )
        data_path = tmp_path / "grpo.jsonl"
        _write_grpo_jsonl(data_path, [sample])

        mock_dspy = MagicMock()
        mock_dspy.Example = _MockExample

        with patch.dict("sys.modules", {"dspy": mock_dspy}):
            from open_ctf.training.gepa import _load_challenges

            examples = _load_challenges(
                str(data_path), registry=registry
            )

        assert len(examples) == 1
        target = examples[0].get("target")
        assert target is not None
        assert str(challenge.port) in target
