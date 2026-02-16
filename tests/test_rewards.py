"""Tests for CTFReward function.

Validates all four reward components (flag, uniqueness, efficiency, format)
plus integration tests against data/grpo.jsonl.
"""

import json
import os
from pathlib import Path

import pytest

from open_ctf.rewards.reward import CTFReward

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


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
    """Load GRPO samples from data/grpo.jsonl."""
    path = DATA_DIR / "grpo.jsonl"
    if not path.exists():
        pytest.skip("data/grpo.jsonl not found (run open-ctf-convert + open-ctf-split)")
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


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

    def test_metadata_success_overrides(self, reward_no_noise):
        # metadata.success=True should give 1.0 even without flag in text
        score = reward_no_noise._flag_score("no flag here", None, metadata_success=True)
        assert score == 1.0

    def test_metadata_success_with_placeholder_flag(self, reward_no_noise):
        # The critical fix: placeholder flag + metadata.success should score 1.0
        score = reward_no_noise._flag_score(
            "no flag text", "content_of_flag_here", metadata_success=True
        )
        assert score == 1.0

    def test_placeholder_flag_skipped(self, reward_no_noise):
        # Placeholder flag should not match even if present in text
        score = reward_no_noise._flag_score(
            "content_of_flag_here", "content_of_flag_here"
        )
        assert score == 0.0

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

    def test_metadata_false_no_flag(self, reward_no_noise):
        score = reward_no_noise._flag_score("no flag", None, metadata_success=False)
        assert score == 0.0

    def test_metadata_false_blocks_exact_match(self, reward_no_noise):
        """metadata.success=False should block exact match fallback."""
        score = reward_no_noise._flag_score(
            "Found FLAG{real_flag}", "FLAG{real_flag}", metadata_success=False
        )
        # Flag IS in text, but metadata says not solved -> only pattern credit
        assert score == 0.1

    def test_metadata_false_with_flag_pattern(self, reward_no_noise):
        """metadata.success=False with FLAG{} pattern gives 0.1 credit."""
        score = reward_no_noise._flag_score(
            "FLAG{some_flag}", None, metadata_success=False
        )
        assert score == 0.1

    def test_check_placeholder_skipped(self, reward_no_noise):
        """ground_truth_flag='CHECK' should be treated as placeholder."""
        score = reward_no_noise._flag_score("answer CHECK here", "CHECK")
        assert score == 0.0

    def test_template_text_flag_skipped(self, reward_no_noise):
        """Multiline flags containing 'content_of_flag_here' are placeholders."""
        template_flag = "tags:\n  <FLAG>content_of_flag_here"
        score = reward_no_noise._flag_score(template_flag, template_flag)
        assert score == 0.0


# ---------------------------------------------------------------------------
# _uniqueness_score tests
# ---------------------------------------------------------------------------


class TestUniquenessScore:
    def _tc(self, name, args="{}"):
        return {"name": name, "arguments": args}

    def test_all_unique_commands(self, reward_no_noise):
        tool_calls = [
            self._tc("shell_command", '{"command": "nmap target"}'),
            self._tc("shell_command", '{"command": "curl http://target"}'),
            self._tc("shell_command", '{"command": "gobuster dir -u http://target"}'),
        ]
        score = reward_no_noise._uniqueness_score(tool_calls)
        assert score == 1.0

    def test_all_repeated_commands(self, reward_no_noise):
        tool_calls = [
            self._tc("shell_command", '{"command": "ls"}'),
            self._tc("shell_command", '{"command": "ls"}'),
            self._tc("shell_command", '{"command": "ls"}'),
        ]
        score = reward_no_noise._uniqueness_score(tool_calls)
        assert score == pytest.approx(1.0 / 3.0)

    def test_half_repeated(self, reward_no_noise):
        tool_calls = [
            self._tc("shell_command", '{"command": "nmap target"}'),
            self._tc("shell_command", '{"command": "nmap target"}'),
            self._tc("shell_command", '{"command": "curl http://target"}'),
            self._tc("shell_command", '{"command": "curl http://target"}'),
        ]
        score = reward_no_noise._uniqueness_score(tool_calls)
        assert score == pytest.approx(0.5)

    def test_empty_tool_calls(self, reward_no_noise):
        score = reward_no_noise._uniqueness_score([])
        assert score == 0.0

    def test_single_command(self, reward_no_noise):
        tool_calls = [
            self._tc("shell_command", '{"command": "id"}'),
        ]
        score = reward_no_noise._uniqueness_score(tool_calls)
        assert score == 1.0

    def test_no_extractable_commands(self, reward_no_noise):
        tool_calls = [
            self._tc("unknown_tool", ""),
            self._tc("another_tool", ""),
        ]
        score = reward_no_noise._uniqueness_score(tool_calls)
        assert score == 0.5  # Neutral

    def test_python_code_extracted(self, reward_no_noise):
        tool_calls = [
            self._tc("python_code", '{"code": "import os; os.system(\\"id\\")"}'),
            self._tc("python_code", '{"code": "print(open(\\"/etc/passwd\\").read())"}'),
        ]
        score = reward_no_noise._uniqueness_score(tool_calls)
        assert score == 1.0

    def test_flag_found_extracted(self, reward_no_noise):
        tool_calls = [
            self._tc("flag_found", '{"content": "FLAG{test}"}'),
            self._tc("flag_found", '{"content": "FLAG{test}"}'),
        ]
        score = reward_no_noise._uniqueness_score(tool_calls)
        assert score == 0.5  # Same flag submitted twice


# ---------------------------------------------------------------------------
# _extract_command tests
# ---------------------------------------------------------------------------


class TestExtractCommand:
    def _tc(self, name, args):
        return {"name": name, "arguments": args}

    def test_shell_command(self):
        cmd = CTFReward._extract_command(
            self._tc("shell_command", '{"command": "nmap -sV target"}')
        )
        assert cmd == "nmap -sV target"

    def test_python_code(self):
        cmd = CTFReward._extract_command(
            self._tc("python_code", '{"code": "print(1)"}')
        )
        assert cmd == "print(1)"

    def test_flag_found(self):
        cmd = CTFReward._extract_command(
            self._tc("flag_found", '{"content": "FLAG{test}"}')
        )
        assert cmd == "FLAG{test}"

    def test_empty_args(self):
        cmd = CTFReward._extract_command(self._tc("tool", ""))
        assert cmd == ""

    def test_plain_string_args(self):
        cmd = CTFReward._extract_command(self._tc("tool", "ls -la"))
        assert cmd == "ls -la"

    def test_dict_with_path(self):
        cmd = CTFReward._extract_command(
            self._tc("read_file", '{"path": "/etc/passwd"}')
        )
        assert cmd == "/etc/passwd"


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
                {"function": {"name": "shell_command", "arguments": '{"command": "curl http://target"}'}}
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

    def test_successful_trace_with_metadata(self, reward_no_noise):
        """metadata.success=True should score high even without flag text."""
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "shell_command", "arguments": '{"command": "nmap target"}'}}
            ]},
            {"role": "tool", "content": "80/tcp open"},
            {"role": "assistant", "content": "solved"},
        ]
        scores = reward_no_noise(
            [msgs],
            ground_truth_flag=["content_of_flag_here"],
            optimal_steps=[2],
            metadata=[{"success": True}],
        )
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
        # No ground truth, no optimal steps, no metadata = low score
        assert scores[0] == pytest.approx(0.0)

    def test_metadata_success_via_kwargs(self, reward_no_noise):
        """Verify metadata.success is extracted from nested metadata dict."""
        scores = reward_no_noise(
            ["no flag in text at all"],
            metadata=[{"success": True}],
            optimal_steps=[5],
        )
        # metadata.success=True → flag_sc=1.0
        # efficiency = min(5/0, 1.0) → 0 steps (no tool calls) → 0.0
        # uniqueness = 0.0 (no tool calls)
        # format = 0.0 (no tool calls)
        # Total = 0.50 * 1.0 = 0.50
        assert scores[0] == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# Integration: data/grpo.jsonl validation
# ---------------------------------------------------------------------------


class TestGRPOSamples:
    def test_all_samples_load(self, grpo_samples):
        assert len(grpo_samples) >= 15, f"Expected >= 15 GRPO samples, got {len(grpo_samples)}"

    def test_all_have_required_fields(self, grpo_samples):
        for i, sample in enumerate(grpo_samples):
            assert "messages" in sample, f"Sample {i} missing 'messages'"
            assert "optimal_steps" in sample, f"Sample {i} missing 'optimal_steps'"
            assert isinstance(sample["optimal_steps"], int), f"Sample {i} optimal_steps not int"
            assert sample["optimal_steps"] >= 0, f"Sample {i} optimal_steps < 0"
            # metadata.success is the authoritative signal
            meta = sample.get("metadata", {})
            assert "success" in meta, f"Sample {i} missing metadata.success"

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
                metadata=[sample["metadata"]],
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
                metadata=[sample["metadata"]],
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
        """Successful traces (per metadata) should score above 0.3."""
        reward = CTFReward(noise_range=0.0, seed=0)
        for sample in grpo_samples:
            if not sample["metadata"]["success"]:
                continue
            completions = [sample["messages"]]
            scores = reward(
                completions,
                ground_truth_flag=[sample["ground_truth_flag"]],
                optimal_steps=[sample["optimal_steps"]],
                metadata=[sample["metadata"]],
            )
            assert scores[0] > 0.3, (
                f"Success sample '{sample['metadata']['challenge']}' scored only {scores[0]:.3f}"
            )

    def test_failed_traces_below_success_average(self, grpo_samples):
        """Average failure score should be below average success score."""
        reward = CTFReward(noise_range=0.0, seed=0)

        success_scores = []
        failure_scores = []

        for sample in grpo_samples:
            completions = [sample["messages"]]
            scores = reward(
                completions,
                ground_truth_flag=[sample["ground_truth_flag"]],
                optimal_steps=[sample["optimal_steps"]],
                metadata=[sample["metadata"]],
            )
            if sample["metadata"]["success"]:
                success_scores.append(scores[0])
            else:
                failure_scores.append(scores[0])

        avg_success = sum(success_scores) / len(success_scores)
        avg_failure = sum(failure_scores) / len(failure_scores)

        assert avg_failure < avg_success, (
            f"Avg failure ({avg_failure:.3f}) should be < avg success ({avg_success:.3f})"
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
                metadata=[sample["metadata"]],
            )
            results.append(scores[0])
        unique = len(set(round(r, 8) for r in results))
        assert unique > 1, "Noise should produce variance across repeated evaluations"

    def test_optimal_steps_vary(self, grpo_samples):
        steps = [s["optimal_steps"] for s in grpo_samples]
        unique_steps = set(steps)
        assert len(unique_steps) >= 3, f"Need varied optimal_steps, got {unique_steps}"

    def test_placeholder_flags_handled(self, grpo_samples):
        """Samples with placeholder flags should still score correctly via metadata."""
        reward = CTFReward(noise_range=0.0, seed=0)
        placeholder_successes = [
            s for s in grpo_samples
            if s["metadata"]["success"]
            and s.get("ground_truth_flag") == "content_of_flag_here"
        ]
        if not placeholder_successes:
            pytest.skip("No placeholder-flag successes found")

        for sample in placeholder_successes[:5]:  # Test first 5
            scores = reward(
                [sample["messages"]],
                ground_truth_flag=[sample["ground_truth_flag"]],
                optimal_steps=[sample["optimal_steps"]],
                metadata=[sample["metadata"]],
            )
            assert scores[0] > 0.3, (
                f"Placeholder-flag success '{sample['metadata']['challenge']}' "
                f"scored only {scores[0]:.3f} (metadata.success should give 1.0)"
            )

    def test_grpo_readiness(self, grpo_samples):
        """All 4 GRPO readiness checks must pass on actual trace data."""
        reward = CTFReward(noise_range=0.0, seed=0)

        success_scores = []
        failure_scores = []
        all_scores = []

        for sample in grpo_samples:
            scores = reward(
                [sample["messages"]],
                ground_truth_flag=[sample["ground_truth_flag"]],
                optimal_steps=[sample["optimal_steps"]],
                metadata=[sample["metadata"]],
            )
            all_scores.append(scores[0])
            if sample["metadata"]["success"]:
                success_scores.append(scores[0])
            else:
                failure_scores.append(scores[0])

        avg_success = sum(success_scores) / len(success_scores)
        avg_failure = sum(failure_scores) / len(failure_scores)
        gap = avg_success - avg_failure

        import statistics
        variance = statistics.variance(all_scores)

        assert gap > 0.50, f"Gap {gap:.4f} should be > 0.50"
        assert variance > 0.02, f"Variance {variance:.4f} should be > 0.02"
        assert avg_failure < 0.20, f"Failure mean {avg_failure:.4f} should be < 0.20"
        assert avg_success > avg_failure, (
            f"Success mean {avg_success:.4f} should be > failure mean {avg_failure:.4f}"
        )
