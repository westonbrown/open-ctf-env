"""Tests for CTFReward function.

Validates all four reward components (flag, grammar, efficiency, format)
plus integration tests against grpo_sample.jsonl data.
"""

import json
import os
from pathlib import Path

import pytest

from open_ctf.rewards.ctf_reward import CTFReward, _classify_tool_call

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sample"


@pytest.fixture
def reward():
    """CTFReward with fixed seed for deterministic noise."""
    return CTFReward(seed=42)


@pytest.fixture
def reward_no_noise():
    """CTFReward with zero noise for exact value testing."""
    return CTFReward(noise_range=0.0, seed=0)


@pytest.fixture
def grpo_samples():
    """Load all GRPO samples from data/sample/grpo_sample.jsonl."""
    path = DATA_DIR / "grpo_sample.jsonl"
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


# ---------------------------------------------------------------------------
# _classify_tool_call tests
# ---------------------------------------------------------------------------


class TestClassifyToolCall:
    def test_nmap_is_recon(self):
        assert _classify_tool_call("shell_command", '{"command": "nmap -sV target"}') == "recon"

    def test_gobuster_is_enum(self):
        assert _classify_tool_call("shell_command", '{"command": "gobuster dir -u http://target"}') == "enum"

    def test_sqlmap_is_exploit(self):
        assert _classify_tool_call("shell_command", '{"command": "sqlmap -u http://target"}') == "exploit"

    def test_flag_found_is_exploit(self):
        assert _classify_tool_call("flag_found", '{"content": "FLAG{test}"}') == "exploit"

    def test_python_code_is_exploit(self):
        assert _classify_tool_call("python_code", '{"code": "import requests"}') == "exploit"

    def test_curl_default_is_recon(self):
        assert _classify_tool_call("shell_command", '{"command": "curl -s http://target/"}') == "recon"

    def test_curl_with_exploit_keyword(self):
        assert _classify_tool_call("shell_command", '{"command": "curl -X POST -d exploit http://target"}') == "exploit"

    def test_curl_with_dirb_is_enum(self):
        assert _classify_tool_call("shell_command", '{"command": "curl http://target/robots.txt"}') == "enum"

    def test_unknown_tool_returns_none(self):
        assert _classify_tool_call("unknown_tool", '{"key": "value"}') is None

    def test_ffuf_is_enum(self):
        assert _classify_tool_call("shell_command", '{"command": "ffuf -u http://target/FUZZ"}') == "enum"

    def test_hydra_is_exploit(self):
        assert _classify_tool_call("shell_command", '{"command": "hydra -l admin -P pass.txt target ssh"}') == "exploit"


# ---------------------------------------------------------------------------
# _flag_score tests
# ---------------------------------------------------------------------------


class TestFlagScore:
    def test_exact_match(self, reward_no_noise):
        score = reward_no_noise._flag_score("Found FLAG{test_123}", "FLAG{test_123}")
        assert score == 1.0

    def test_exact_match_embedded(self, reward_no_noise):
        text = "Welcome admin! Your flag is FLAG{abc_def} congrats!"
        score = reward_no_noise._flag_score(text, "FLAG{abc_def}")
        assert score == 1.0

    def test_pattern_match_no_gt(self, reward_no_noise):
        score = reward_no_noise._flag_score("FLAG{some_random_flag}", None)
        assert score == 0.1

    def test_pattern_match_wrong_gt(self, reward_no_noise):
        score = reward_no_noise._flag_score("FLAG{wrong_flag}", "FLAG{correct_flag}")
        assert score == 0.1

    def test_no_flag(self, reward_no_noise):
        score = reward_no_noise._flag_score("No flag here at all", "FLAG{test}")
        assert score == 0.0

    def test_placeholder_not_matched(self, reward_no_noise):
        score = reward_no_noise._flag_score("FLAG{...}", None)
        assert score == 0.0

    def test_empty_text(self, reward_no_noise):
        score = reward_no_noise._flag_score("", "FLAG{test}")
        assert score == 0.0

    def test_no_ground_truth_no_flag(self, reward_no_noise):
        score = reward_no_noise._flag_score("hello world", None)
        assert score == 0.0


# ---------------------------------------------------------------------------
# _grammar_score tests
# ---------------------------------------------------------------------------


class TestGrammarScore:
    def _tc(self, name, args="{}"):
        return {"name": name, "arguments": args}

    def test_correct_order_full(self, reward_no_noise):
        tool_calls = [
            self._tc("shell_command", '{"command": "nmap target"}'),
            self._tc("shell_command", '{"command": "gobuster dir -u http://target"}'),
            self._tc("shell_command", '{"command": "sqlmap -u http://target"}'),
            self._tc("flag_found", '{"content": "FLAG{x}"}'),
        ]
        score = reward_no_noise._grammar_score(tool_calls)
        assert score == 1.0

    def test_wrong_order_exploit_first(self, reward_no_noise):
        tool_calls = [
            self._tc("flag_found", '{"content": "FLAG{x}"}'),
            self._tc("shell_command", '{"command": "nmap target"}'),
            self._tc("shell_command", '{"command": "gobuster dir -u http://target"}'),
        ]
        score = reward_no_noise._grammar_score(tool_calls)
        # phases_seen = [exploit, recon, enum]
        # Presence: recon(0.2)+enum(0.2)+exploit(0.2) = 0.6
        # Order: recon(1) < enum(2) => +0.2, but enum(2) > exploit(0) => +0.0
        # Total = 0.8
        assert score == pytest.approx(0.8)

    def test_recon_only(self, reward_no_noise):
        tool_calls = [
            self._tc("shell_command", '{"command": "nmap target"}'),
            self._tc("shell_command", '{"command": "ping target"}'),
        ]
        score = reward_no_noise._grammar_score(tool_calls)
        assert score == 0.2  # Only recon present

    def test_empty_tool_calls(self, reward_no_noise):
        score = reward_no_noise._grammar_score([])
        assert score == 0.0

    def test_missing_enum_phase(self, reward_no_noise):
        tool_calls = [
            self._tc("shell_command", '{"command": "nmap target"}'),
            self._tc("shell_command", '{"command": "sqlmap -u http://target"}'),
        ]
        score = reward_no_noise._grammar_score(tool_calls)
        # recon(0.2) + exploit(0.2) = 0.4, no enum
        # recon before exploit would need enum in between, so no order bonus
        assert score == 0.4

    def test_recon_enum_no_exploit(self, reward_no_noise):
        tool_calls = [
            self._tc("shell_command", '{"command": "nmap target"}'),
            self._tc("shell_command", '{"command": "gobuster dir -u http://target"}'),
        ]
        score = reward_no_noise._grammar_score(tool_calls)
        # recon(0.2) + enum(0.2) + order bonus recon<enum(0.2) = 0.6
        assert score == pytest.approx(0.6)

    def test_unclassified_tools_ignored(self, reward_no_noise):
        tool_calls = [
            self._tc("unknown_tool", '{"key": "value"}'),
            self._tc("another_tool", '{"key": "value"}'),
        ]
        score = reward_no_noise._grammar_score(tool_calls)
        assert score == 0.0


# ---------------------------------------------------------------------------
# _efficiency_score tests
# ---------------------------------------------------------------------------


class TestEfficiencyScore:
    def test_optimal(self, reward_no_noise):
        score = reward_no_noise._efficiency_score(3, 3)
        assert score == 1.0

    def test_double_optimal(self, reward_no_noise):
        score = reward_no_noise._efficiency_score(6, 3)
        assert score == 0.5

    def test_under_optimal(self, reward_no_noise):
        # Fewer steps than optimal should cap at 1.0
        score = reward_no_noise._efficiency_score(2, 3)
        assert score == 1.0

    def test_no_metadata(self, reward_no_noise):
        score = reward_no_noise._efficiency_score(5, None)
        assert score == 0.5  # Neutral score when metadata unavailable

    def test_zero_steps(self, reward_no_noise):
        score = reward_no_noise._efficiency_score(0, 3)
        assert score == 0.0

    def test_many_steps(self, reward_no_noise):
        score = reward_no_noise._efficiency_score(20, 4)
        assert score == pytest.approx(0.2)

    def test_large_optimal(self, reward_no_noise):
        score = reward_no_noise._efficiency_score(15, 15)
        assert score == 1.0


# ---------------------------------------------------------------------------
# _format_score tests
# ---------------------------------------------------------------------------


class TestFormatScore:
    def _tc(self, name, args):
        return {"name": name, "arguments": args}

    def test_all_valid_json(self, reward_no_noise):
        tool_calls = [
            self._tc("shell_command", '{"command": "ls"}'),
            self._tc("flag_found", '{"content": "FLAG{x}"}'),
        ]
        score = reward_no_noise._format_score(tool_calls)
        assert score == 1.0

    def test_all_invalid_json(self, reward_no_noise):
        tool_calls = [
            self._tc("shell_command", "not json"),
            self._tc("flag_found", "also not json"),
        ]
        score = reward_no_noise._format_score(tool_calls)
        assert score == 0.5  # Each gets 0.5 credit

    def test_empty(self, reward_no_noise):
        score = reward_no_noise._format_score([])
        assert score == 0.0

    def test_mixed_valid_invalid(self, reward_no_noise):
        tool_calls = [
            self._tc("shell_command", '{"command": "ls"}'),
            self._tc("flag_found", "broken"),
        ]
        score = reward_no_noise._format_score(tool_calls)
        assert score == pytest.approx(0.75)  # (1.0 + 0.5) / 2

    def test_empty_arguments(self, reward_no_noise):
        tool_calls = [
            self._tc("shell_command", ""),
        ]
        score = reward_no_noise._format_score(tool_calls)
        assert score == 0.0  # Empty args = invalid

    def test_missing_name(self, reward_no_noise):
        tool_calls = [
            self._tc("", '{"command": "ls"}'),
        ]
        score = reward_no_noise._format_score(tool_calls)
        assert score == 0.0  # Empty name = invalid


# ---------------------------------------------------------------------------
# _extract tests
# ---------------------------------------------------------------------------


class TestExtract:
    def test_string_input(self):
        text, tcs = CTFReward._extract("hello FLAG{test}")
        assert text == "hello FLAG{test}"
        assert tcs == []

    def test_message_list_with_tool_calls(self):
        msgs = [
            {"role": "assistant", "content": "thinking...", "tool_calls": [
                {"function": {"name": "shell_command", "arguments": '{"command": "ls"}'}}
            ]},
            {"role": "tool", "content": "file1.txt\nfile2.txt"},
        ]
        text, tcs = CTFReward._extract(msgs)
        assert "thinking..." in text
        assert len(tcs) == 1
        assert tcs[0]["name"] == "shell_command"

    def test_message_list_no_tool_calls(self):
        msgs = [
            {"role": "assistant", "content": "I failed to find the flag."},
        ]
        text, tcs = CTFReward._extract(msgs)
        assert "failed" in text
        assert tcs == []

    def test_dict_arguments_converted(self):
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "shell_command", "arguments": {"command": "ls"}}}
            ]},
        ]
        text, tcs = CTFReward._extract(msgs)
        assert len(tcs) == 1
        assert tcs[0]["arguments"] == '{"command": "ls"}'


# ---------------------------------------------------------------------------
# Noise tests
# ---------------------------------------------------------------------------


class TestNoise:
    def test_noise_adds_variance(self):
        reward = CTFReward(noise_range=0.05, seed=42)
        completion = "FLAG{test}"
        results = []
        for _ in range(20):
            r = reward([completion], ground_truth_flag=["FLAG{test}"], optimal_steps=[1])
            results.append(r[0])
        # With noise, not all values should be identical
        assert len(set(round(r, 6) for r in results)) > 1

    def test_zero_noise_deterministic(self):
        reward = CTFReward(noise_range=0.0, seed=0)
        completion = "FLAG{test}"
        r1 = reward([completion], ground_truth_flag=["FLAG{test}"], optimal_steps=[1])
        r2 = reward([completion], ground_truth_flag=["FLAG{test}"], optimal_steps=[1])
        assert r1[0] == r2[0]


# ---------------------------------------------------------------------------
# Integration: full __call__ tests
# ---------------------------------------------------------------------------


class TestCallIntegration:
    def test_successful_trace_scores_high(self, reward_no_noise):
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "shell_command", "arguments": '{"command": "nmap target"}'}}
            ]},
            {"role": "tool", "content": "80/tcp open http"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "shell_command", "arguments": '{"command": "gobuster dir -u http://target"}'}}
            ]},
            {"role": "tool", "content": "/admin (200)"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "shell_command", "arguments": '{"command": "sqlmap -u http://target"}'}}
            ]},
            {"role": "tool", "content": "FLAG{test_flag}"},
            {"role": "assistant", "content": "Found FLAG{test_flag}", "tool_calls": [
                {"function": {"name": "flag_found", "arguments": '{"content": "FLAG{test_flag}"}'}}
            ]},
        ]
        scores = reward_no_noise(
            [msgs],
            ground_truth_flag=["FLAG{test_flag}"],
            optimal_steps=[4],
        )
        assert len(scores) == 1
        assert scores[0] > 0.5

    def test_failed_trace_scores_low(self, reward_no_noise):
        # Trace with no tool calls at all -- pure text failure
        msgs = [
            {"role": "assistant", "content": "I cannot solve this challenge."},
        ]
        scores = reward_no_noise(
            [msgs],
            ground_truth_flag=["FLAG{secret}"],
            optimal_steps=[3],
        )
        assert len(scores) == 1
        assert scores[0] < 0.1

    def test_failed_trace_with_tools_lower_than_success(self, reward_no_noise):
        # Failed trace with tool calls still gets some score from format/efficiency
        # but should be notably lower than a successful trace
        fail_msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "shell_command", "arguments": '{"command": "curl http://target"}'}}
            ]},
            {"role": "tool", "content": "200 OK"},
            {"role": "assistant", "content": "Could not find the flag."},
        ]
        success_msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "shell_command", "arguments": '{"command": "curl http://target"}'}}
            ]},
            {"role": "tool", "content": "FLAG{secret}"},
            {"role": "assistant", "content": "FLAG{secret}", "tool_calls": [
                {"function": {"name": "flag_found", "arguments": '{"content": "FLAG{secret}"}'}}
            ]},
        ]
        fail_scores = reward_no_noise(
            [fail_msgs], ground_truth_flag=["FLAG{secret}"], optimal_steps=[3],
        )
        success_scores = reward_no_noise(
            [success_msgs], ground_truth_flag=["FLAG{secret}"], optimal_steps=[3],
        )
        assert success_scores[0] > fail_scores[0]

    def test_batch_scoring(self, reward_no_noise):
        success_msgs = [
            {"role": "assistant", "content": "FLAG{test}", "tool_calls": [
                {"function": {"name": "flag_found", "arguments": '{"content": "FLAG{test}"}'}}
            ]},
        ]
        fail_msgs = [
            {"role": "assistant", "content": "I failed."},
        ]
        scores = reward_no_noise(
            [success_msgs, fail_msgs],
            ground_truth_flag=["FLAG{test}", "FLAG{test}"],
            optimal_steps=[1, 1],
        )
        assert len(scores) == 2
        assert scores[0] > scores[1]

    def test_string_completions(self, reward_no_noise):
        scores = reward_no_noise(
            ["FLAG{exact_match}", "no flag here"],
            ground_truth_flag=["FLAG{exact_match}", "FLAG{exact_match}"],
        )
        assert scores[0] > scores[1]

    def test_no_kwargs(self, reward_no_noise):
        scores = reward_no_noise(["hello world"])
        assert len(scores) == 1
        # No ground truth, no optimal steps = low score
        assert scores[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration: grpo_sample.jsonl validation
# ---------------------------------------------------------------------------


class TestGRPOSamples:
    def test_all_samples_load(self, grpo_samples):
        assert len(grpo_samples) >= 15, f"Expected >= 15 GRPO samples, got {len(grpo_samples)}"

    def test_all_have_required_fields(self, grpo_samples):
        for i, sample in enumerate(grpo_samples):
            assert "messages" in sample, f"Sample {i} missing 'messages'"
            assert "ground_truth_flag" in sample, f"Sample {i} missing 'ground_truth_flag'"
            assert "optimal_steps" in sample, f"Sample {i} missing 'optimal_steps'"
            assert sample["ground_truth_flag"].startswith("FLAG{"), f"Sample {i} has invalid flag format"
            assert isinstance(sample["optimal_steps"], int), f"Sample {i} optimal_steps not int"
            assert sample["optimal_steps"] >= 1, f"Sample {i} optimal_steps < 1"

    def test_mix_of_successes_and_failures(self, grpo_samples):
        successes = sum(1 for s in grpo_samples if s["metadata"]["success"])
        failures = sum(1 for s in grpo_samples if not s["metadata"]["success"])
        assert successes >= 5, f"Need >= 5 successes, got {successes}"
        assert failures >= 5, f"Need >= 5 failures, got {failures}"

    def test_rewards_in_range(self, grpo_samples):
        reward = CTFReward(noise_range=0.05, seed=42)
        for i, sample in enumerate(grpo_samples):
            completions = [sample["messages"]]
            scores = reward(
                completions,
                ground_truth_flag=[sample["ground_truth_flag"]],
                optimal_steps=[sample["optimal_steps"]],
            )
            # With noise_range=0.05, scores can go slightly below 0
            assert scores[0] >= -0.05, f"Sample {i} score {scores[0]} too low"
            assert scores[0] <= 1.05, f"Sample {i} score {scores[0]} too high"

    def test_successes_score_higher_than_failures(self, grpo_samples):
        reward = CTFReward(noise_range=0.0, seed=0)

        success_scores = []
        failure_scores = []

        for sample in grpo_samples:
            completions = [sample["messages"]]
            scores = reward(
                completions,
                ground_truth_flag=[sample["ground_truth_flag"]],
                optimal_steps=[sample["optimal_steps"]],
            )
            if sample["metadata"]["success"]:
                success_scores.append(scores[0])
            else:
                failure_scores.append(scores[0])

        avg_success = sum(success_scores) / len(success_scores)
        avg_failure = sum(failure_scores) / len(failure_scores)

        assert avg_success > avg_failure, (
            f"Average success ({avg_success:.3f}) should be > average failure ({avg_failure:.3f})"
        )

    def test_successful_traces_above_threshold(self, grpo_samples):
        reward = CTFReward(noise_range=0.0, seed=0)
        for sample in grpo_samples:
            if not sample["metadata"]["success"]:
                continue
            completions = [sample["messages"]]
            scores = reward(
                completions,
                ground_truth_flag=[sample["ground_truth_flag"]],
                optimal_steps=[sample["optimal_steps"]],
            )
            assert scores[0] > 0.3, (
                f"Success sample '{sample['metadata']['challenge']}' scored only {scores[0]:.3f}"
            )

    def test_failed_traces_below_success_average(self, grpo_samples):
        """Failed traces should score below the average success score.

        Note: Failed traces with valid tool calls still earn format/efficiency
        points. The key invariant is that failures score lower than successes
        on average, not that they fall below an arbitrary threshold.
        """
        reward = CTFReward(noise_range=0.0, seed=0)

        success_scores = []
        failure_scores = []

        for sample in grpo_samples:
            completions = [sample["messages"]]
            scores = reward(
                completions,
                ground_truth_flag=[sample["ground_truth_flag"]],
                optimal_steps=[sample["optimal_steps"]],
            )
            if sample["metadata"]["success"]:
                success_scores.append(scores[0])
            else:
                failure_scores.append(scores[0])

        avg_success = sum(success_scores) / len(success_scores)

        for i, score in enumerate(failure_scores):
            assert score < avg_success, (
                f"Failure {i} scored {score:.3f} >= avg success {avg_success:.3f}"
            )

    def test_noise_adds_variance_across_samples(self, grpo_samples):
        reward = CTFReward(noise_range=0.05, seed=42)
        # Run same sample twice
        sample = grpo_samples[0]
        results = []
        for _ in range(10):
            scores = reward(
                [sample["messages"]],
                ground_truth_flag=[sample["ground_truth_flag"]],
                optimal_steps=[sample["optimal_steps"]],
            )
            results.append(scores[0])
        unique = len(set(round(r, 8) for r in results))
        assert unique > 1, "Noise should produce variance across repeated evaluations"

    def test_optimal_steps_vary(self, grpo_samples):
        steps = [s["optimal_steps"] for s in grpo_samples]
        unique_steps = set(steps)
        assert len(unique_steps) >= 3, f"Need varied optimal_steps, got {unique_steps}"
