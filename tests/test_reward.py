"""Smoke tests for CTFReward function.

Validates the 8-signal reward function:
- Flag capture (exact match, pattern match, env verification, no match)
- Efficiency (optimal vs actual steps, gating at <3 steps)
- Progression (RECON->ENUM->EXPLOIT ordering)
- Exploration (exponentially-decayed novelty)
- Uniqueness (command diversity)
- Format compliance (valid JSON args, known tool names)
- Recovery (pivots after stuck runs)
- Cognitive (words-per-action density)
- Hallucination penalty (wrong flag submission)
- Noise (variance guarantee for GRPO)
- Weight validation (must sum to 1.0)
"""

import json
import pytest

from open_ctf.rewards.reward import CTFReward


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tc(name: str, args: dict | str = "{}") -> dict:
    """Build a tool call dict for internal scoring methods."""
    if isinstance(args, dict):
        args = json.dumps(args)
    return {"name": name, "arguments": args}


def _shell(cmd: str) -> dict:
    return _tc("shell_command", {"command": cmd})


def _completion_with_tools(tool_calls: list[dict], text: str = "") -> list[dict]:
    """Build a ChatML completion with tool calls."""
    msgs = []
    for i, t in enumerate(tool_calls):
        msgs.append({
            "role": "assistant",
            "content": text if i == 0 else "",
            "tool_calls": [{"function": t}],
        })
        msgs.append({
            "role": "tool",
            "tool_call_id": f"call_{i}",
            "content": f"output_{i}",
        })
    return msgs


# ---------------------------------------------------------------------------
# Construction and weight validation
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_default_weights_sum_to_one(self):
        r = CTFReward()
        total = (
            r.flag_weight + r.efficiency_weight + r.progression_weight
            + r.exploration_weight + r.uniqueness_weight + r.format_weight
            + r.recovery_weight + r.cognitive_weight
        )
        assert abs(total - 1.0) < 1e-6

    def test_invalid_weights_rejected(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            CTFReward(flag_weight=0.5, efficiency_weight=0.5, progression_weight=0.5)

    def test_custom_weights(self):
        r = CTFReward(
            flag_weight=0.30, efficiency_weight=0.15, progression_weight=0.10,
            exploration_weight=0.10, uniqueness_weight=0.05, format_weight=0.10,
            recovery_weight=0.10, cognitive_weight=0.10,
        )
        assert r.flag_weight == 0.30


# ---------------------------------------------------------------------------
# Flag capture signal
# ---------------------------------------------------------------------------


class TestFlagCapture:
    @pytest.fixture
    def r(self):
        return CTFReward(noise_range=0.0, seed=0)

    def test_exact_match(self, r):
        assert r._flag_score("Found FLAG{abc}", "FLAG{abc}") == 1.0

    def test_env_verified_correct(self, r):
        assert r._flag_score("Correct! Flag verified: FLAG{x}", None) == 1.0

    def test_env_verified_incorrect(self, r):
        assert r._flag_score("Incorrect submission: FLAG{x}", None) == 0.0

    def test_pattern_match_no_ground_truth(self, r):
        assert r._flag_score("FLAG{some_real_flag}", None) == 0.1

    def test_pattern_match_wrong_flag(self, r):
        assert r._flag_score("FLAG{wrong}", "FLAG{correct}") == 0.1

    def test_no_flag(self, r):
        assert r._flag_score("no flag at all", "FLAG{test}") == 0.0

    def test_placeholder_flag_not_matched(self, r):
        assert r._flag_score("content_of_flag_here", "content_of_flag_here") == 0.0

    def test_check_placeholder_skipped(self, r):
        assert r._flag_score("answer CHECK here", "CHECK") == 0.0

    def test_empty_text(self, r):
        assert r._flag_score("", "FLAG{test}") == 0.0


# ---------------------------------------------------------------------------
# Efficiency signal
# ---------------------------------------------------------------------------


class TestEfficiency:
    @pytest.fixture
    def r(self):
        return CTFReward(noise_range=0.0, seed=0)

    def test_optimal_path(self, r):
        assert r._efficiency_score(5, 5, flag_found=True) == 1.0

    def test_double_optimal(self, r):
        assert r._efficiency_score(10, 5, flag_found=True) == 0.5

    def test_zero_steps(self, r):
        assert r._efficiency_score(0, 5) == 0.0

    def test_fewer_than_3_steps(self, r):
        assert r._efficiency_score(2, 5) == 0.0

    def test_no_metadata(self, r):
        assert r._efficiency_score(5, None) == 0.3

    def test_capped_without_flag(self, r):
        score = r._efficiency_score(5, 5, flag_found=False)
        assert score <= 0.3


# ---------------------------------------------------------------------------
# Progression signal (RECON->ENUM->EXPLOIT)
# ---------------------------------------------------------------------------


class TestProgression:
    @pytest.fixture
    def r(self):
        return CTFReward(noise_range=0.0, seed=0)

    def test_perfect_ordering(self, r):
        tool_calls = [
            _shell("nmap target"),      # recon
            _shell("curl target"),      # enum
            _tc("python_code", {"code": "exploit()"}),  # exploit
        ]
        assert r._progression_score(tool_calls) == 1.0

    def test_no_tool_calls(self, r):
        assert r._progression_score([]) == 0.0

    def test_only_enum(self, r):
        tool_calls = [_shell("curl target"), _shell("gobuster dir target")]
        score = r._progression_score(tool_calls)
        assert 0.0 < score < 1.0  # Has enum but missing recon/exploit


# ---------------------------------------------------------------------------
# Exploration signal
# ---------------------------------------------------------------------------


class TestExploration:
    @pytest.fixture
    def r(self):
        return CTFReward(noise_range=0.0, seed=0)

    def test_empty(self, r):
        assert r._exploration_score([]) == 0.0

    def test_all_unique_known_tools(self, r):
        tool_calls = [
            _tc("shell_command", {"command": "nmap"}),
            _tc("python_code", {"code": "x"}),
            _tc("read_file", {"path": "/etc/passwd"}),
        ]
        score = r._exploration_score(tool_calls)
        assert score > 0.8  # All unique and early = high

    def test_all_same_tool(self, r):
        tool_calls = [_tc("shell_command", {"command": "ls"})] * 5
        score = r._exploration_score(tool_calls)
        assert score < 0.5  # Only 1 unique out of 5


# ---------------------------------------------------------------------------
# Uniqueness signal
# ---------------------------------------------------------------------------


class TestUniqueness:
    @pytest.fixture
    def r(self):
        return CTFReward(noise_range=0.0, seed=0)

    def test_all_unique(self, r):
        tool_calls = [
            _shell("nmap target"),
            _shell("curl target"),
            _shell("gobuster dir target"),
        ]
        assert r._uniqueness_score(tool_calls) == 1.0

    def test_all_same(self, r):
        tool_calls = [_shell("ls")] * 4
        assert r._uniqueness_score(tool_calls) == 0.25

    def test_empty(self, r):
        assert r._uniqueness_score([]) == 0.0


# ---------------------------------------------------------------------------
# Format compliance signal
# ---------------------------------------------------------------------------


class TestFormat:
    @pytest.fixture
    def r(self):
        return CTFReward(noise_range=0.0, seed=0)

    def test_valid_json_known_tool(self, r):
        tool_calls = [_tc("shell_command", {"command": "ls"})]
        assert r._format_score(tool_calls) == 1.0

    def test_invalid_json(self, r):
        tool_calls = [{"name": "shell_command", "arguments": "not json"}]
        assert r._format_score(tool_calls) == 0.5  # Partial credit

    def test_unknown_tool_excluded(self, r):
        tool_calls = [{"name": "totally_fake", "arguments": '{"x": 1}'}]
        assert r._format_score(tool_calls) == 0.0

    def test_empty(self, r):
        assert r._format_score([]) == 0.0


# ---------------------------------------------------------------------------
# Recovery signal
# ---------------------------------------------------------------------------


class TestRecovery:
    @pytest.fixture
    def r(self):
        return CTFReward(noise_range=0.0, seed=0)

    def test_too_short(self, r):
        tool_calls = [_shell("nmap target"), _shell("curl target")]
        assert r._recovery_score(tool_calls) == 0.5  # Neutral

    def test_no_stuck_runs(self, r):
        tool_calls = [
            _shell("nmap target"),
            _shell("curl target"),
            _tc("python_code", {"code": "x"}),
        ]
        assert r._recovery_score(tool_calls) == 0.5  # Neutral

    def test_stuck_then_pivot(self, r):
        tool_calls = [
            _shell("curl target/admin"),
            _shell("curl target/admin"),
            _shell("nmap target"),  # Pivot
        ]
        score = r._recovery_score(tool_calls)
        assert score == 1.0  # 1 stuck run, 1 pivot = 1.0

    def test_stuck_no_pivot(self, r):
        tool_calls = [
            _shell("curl target"),
            _shell("curl target"),
            _shell("curl target"),
        ]
        score = r._recovery_score(tool_calls)
        assert score == 0.0  # Stuck but never pivoted


# ---------------------------------------------------------------------------
# Cognitive (words-per-action) signal
# ---------------------------------------------------------------------------


class TestCognitive:
    @pytest.fixture
    def r(self):
        return CTFReward(noise_range=0.0, seed=0)

    def test_no_tool_calls(self, r):
        assert r._cognitive_score("some text", []) == 0.0

    def test_too_few_words(self, r):
        assert r._cognitive_score("hi", [_shell("ls")]) == 0.5  # Neutral

    def test_optimal_wpa(self, r):
        # 42 words per 1 tool call = optimal
        text = " ".join(["word"] * 42)
        score = r._cognitive_score(text, [_shell("ls")])
        assert score == pytest.approx(1.0)

    def test_severe_underthinking(self, r):
        # 5 words per 1 tool call = WPA 5 < 10
        text = " ".join(["word"] * 5)
        score = r._cognitive_score(text, [_shell("ls")])
        assert score == pytest.approx(0.1)

    def test_severe_overthinking(self, r):
        # 100 words per 1 tool call = WPA 100 > 80
        text = " ".join(["word"] * 100)
        score = r._cognitive_score(text, [_shell("ls")])
        assert score == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Hallucination penalty
# ---------------------------------------------------------------------------


class TestHallucination:
    @pytest.fixture
    def r(self):
        return CTFReward(noise_range=0.0, seed=0)

    def test_correct_flag_no_penalty(self, r):
        tool_calls = [_tc("flag_found", {"content": "FLAG{x}"})]
        assert r._hallucination_score(tool_calls, flag_sc=1.0) == 0.0

    def test_wrong_flag_penalty(self, r):
        tool_calls = [_tc("flag_found", {"content": "FLAG{wrong}"})]
        penalty = r._hallucination_score(tool_calls, flag_sc=0.1)
        assert penalty < 0  # Negative
        assert penalty == -r.hallucination_penalty

    def test_no_flag_submission_no_penalty(self, r):
        tool_calls = [_shell("nmap target")]
        assert r._hallucination_score(tool_calls, flag_sc=0.0) == 0.0


# ---------------------------------------------------------------------------
# Noise and variance
# ---------------------------------------------------------------------------


class TestNoise:
    def test_noise_provides_variance(self):
        r = CTFReward(noise_range=0.05, seed=42)
        scores = [r(["hello"])[0] for _ in range(20)]
        unique = len(set(round(s, 8) for s in scores))
        assert unique > 1, "Noise should create different scores"

    def test_zero_noise_deterministic(self):
        r = CTFReward(noise_range=0.0, seed=0)
        s1 = r(["hello"])[0]
        s2 = r(["hello"])[0]
        assert s1 == s2


# ---------------------------------------------------------------------------
# Full __call__ integration
# ---------------------------------------------------------------------------


class TestCallIntegration:
    @pytest.fixture
    def r(self):
        return CTFReward(noise_range=0.0, seed=0)

    def test_return_type(self, r):
        scores = r(["hello", "world"])
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_success_higher_than_failure(self, r):
        success = _completion_with_tools(
            [_shell("nmap t"), _shell("curl t"), _tc("python_code", {"code": "x"}),
             _tc("flag_found", {"content": "FLAG{win}"})],
            text="FLAG{win}",
        )
        failure = [{"role": "assistant", "content": "I failed."}]
        scores = r(
            [success, failure],
            ground_truth_flag=["FLAG{win}", "FLAG{win}"],
            optimal_steps=[4, 4],
        )
        assert scores[0] > scores[1]

    def test_batch_length_matches_input(self, r):
        scores = r(["a", "b", "c"])
        assert len(scores) == 3

    def test_no_kwargs(self, r):
        scores = r(["hello"])
        assert len(scores) == 1
        assert isinstance(scores[0], float)


