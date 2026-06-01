"""Tests for open_ctf.cli.evaluate wiring."""

from __future__ import annotations

import logging
import time
import types
from argparse import Namespace

import yaml


def test_cmd_run_forwards_agent_argument(monkeypatch, tmp_path):
    """`open-ctf eval run --agent ...` should reach ModelEvaluator."""
    from open_ctf.cli import evaluate as cli_evaluate

    captured = {}

    class StubEvaluator:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def run_all(self):
            return types.SimpleNamespace(
                solved=1,
                total_challenges=1,
                solve_rate=1.0,
                avg_turns=3.0,
                avg_time_seconds=2.5,
            )

        def save(self, report, output_dir):
            del report
            (tmp_path / "saved.txt").write_text(str(output_dir), encoding="utf-8")

    fake_mod = types.SimpleNamespace(ModelEvaluator=StubEvaluator)
    monkeypatch.setitem(__import__("sys").modules, "open_ctf.eval.evaluator", fake_mod)

    args = Namespace(
        model="Nanbeige/Nanbeige4.1-3B",
        output=str(tmp_path / "out"),
        challenges="configs/challenges/cybench.yaml",
        platform="cybench",
        strategy="chat_tools",
        max_turns=20,
        max_time=10,
        traces_dir=str(tmp_path / "traces"),
        reasoning_effort="medium",
        attempts=1,
        agent="custom:demo_agent.MyAgent",
    )
    cli_evaluate.cmd_run(args)

    assert captured["agent"] == "custom:demo_agent.MyAgent"


def test_model_evaluator_custom_agent_path(tmp_path, monkeypatch):
    """ModelEvaluator should execute custom CTFAgent paths in eval mode."""
    from open_ctf.eval.evaluator import ModelEvaluator

    module_path = tmp_path / "demo_agent_mod.py"
    module_path.write_text(
        "from open_ctf.agent.protocol import AgentResult\n"
        "class DemoAgent:\n"
        "    def __init__(self, **kwargs):\n"
        "        self.kwargs = kwargs\n"
        "    def solve(self, challenge, target, ground_truth_flag='', max_steps=30, timeout=300):\n"
        "        return AgentResult(success=True, flag='FLAG{demo}', steps=2)\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "challenges.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "challenges": [
                    {
                        "id": "demo-challenge",
                        "platform": "cybench",
                        "vuln_type": "misc",
                        "difficulty": "very_easy",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.syspath_prepend(str(tmp_path))
    evaluator = ModelEvaluator(
        model="Nanbeige/Nanbeige4.1-3B",
        challenges_yaml=str(config_path),
        agent="custom:demo_agent_mod.DemoAgent",
        traces_dir=str(tmp_path / "traces"),
        max_turns=10,
        max_time=1,
    )

    result = evaluator.run_challenge(
        {
            "id": "demo-challenge",
            "platform": "cybench",
            "vuln_type": "misc",
            "difficulty": "very_easy",
            "target": "http://localhost:32805",
        }
    )
    assert result.solved is True
    assert result.turns == 2


def test_skip_list_filters_and_emits_skipped_line(tmp_path, caplog):
    """``skip`` block excludes ids from the run and logs ``SKIPPED: <id> (<reason>)``.

    Verifies the four acceptance points from the XBEN-042 skip-list task:
    skipped ids do not appear in run_all results, a SKIPPED log line is
    emitted, the EvalReport.skipped list carries the audit metadata, and
    the markdown summary renders a Skipped section.
    """
    from open_ctf.eval.evaluator import ModelEvaluator, SkippedChallenge

    config_path = tmp_path / "challenges.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "skip": {
                    "XBEN-042-24": {
                        "reason": "qemu-slow",
                        "opened_on": "2026-05-22",
                        "linked_issue": "vecna-item:45a28ece",
                    },
                },
                "challenges": [
                    {
                        "id": "XBEN-042-24",
                        "platform": "xbow",
                        "vuln_type": "pwn",
                        "difficulty": "hard",
                    },
                    {
                        "id": "XBEN-003-24",
                        "platform": "xbow",
                        "vuln_type": "idor",
                        "difficulty": "medium",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    evaluator = ModelEvaluator(
        model="stub",
        challenges_yaml=str(config_path),
        traces_dir=str(tmp_path / "traces"),
        max_turns=1,
        max_time=1,
    )

    # load_challenges directly drops skipped entries.
    challenges = evaluator.load_challenges()
    assert [c["id"] for c in challenges] == ["XBEN-003-24"]

    # _load_challenges_and_skips returns the SkippedChallenge record with metadata.
    _, skipped, _ = evaluator._load_challenges_and_skips()
    assert skipped == [
        SkippedChallenge(
            challenge_id="XBEN-042-24",
            reason="qemu-slow",
            opened_on="2026-05-22",
            linked_issue="vecna-item:45a28ece",
        )
    ]

    # run_all emits the auditable SKIPPED log line and records it in the report.
    # Stub out run_challenge so we don't actually invoke BoxPwnr.
    from open_ctf.eval.evaluator import ChallengeResult

    def _fake_run(challenge):
        return ChallengeResult(
            challenge_id=challenge["id"],
            platform=challenge["platform"],
            vuln_type=challenge["vuln_type"],
            difficulty=challenge["difficulty"],
            solved=False,
            turns=0,
            elapsed_seconds=0.0,
        )

    evaluator.run_challenge = _fake_run  # type: ignore[assignment]
    evaluator._run_runtime_preflight = lambda _c: None  # type: ignore[assignment]

    with caplog.at_level(logging.INFO, logger="open_ctf.eval.evaluator"):
        report = evaluator.run_all()

    assert "SKIPPED: XBEN-042-24 (qemu-slow)" in caplog.text
    assert report.total_challenges == 1
    assert [r["challenge_id"] for r in report.results] == ["XBEN-003-24"]
    assert report.skipped == [
        {
            "challenge_id": "XBEN-042-24",
            "reason": "qemu-slow",
            "opened_on": "2026-05-22",
            "linked_issue": "vecna-item:45a28ece",
            "note": None,
        }
    ]

    md = ModelEvaluator._format_markdown(report)
    assert "## Skipped" in md
    assert "SKIPPED: XBEN-042-24" in md
    assert "qemu-slow" in md


# ---------------------------------------------------------------------------
# Emulation-slow queue + per-class cycle timeout
# ---------------------------------------------------------------------------


def _stub_run(challenge):
    """Drop-in for ModelEvaluator.run_challenge that returns a clean failure."""
    from open_ctf.eval.evaluator import ChallengeResult

    return ChallengeResult(
        challenge_id=challenge["id"],
        platform=challenge.get("platform", "xbow"),
        vuln_type=challenge.get("vuln_type", "unknown"),
        difficulty=challenge.get("difficulty", "unknown"),
        solved=False,
        turns=1,
        elapsed_seconds=0.1,
        status="failed",
        queue_class=challenge.get("class", "default"),
    )


def _write_queue_config(tmp_path, *, cycle_timeouts=None, extra_challenges=None):
    """Write an eval YAML containing the slow-queue + fast-queue mix."""
    payload = {
        "challenges": [
            {
                "id": "XBEN-003-24",
                "platform": "xbow",
                "vuln_type": "idor",
                "difficulty": "medium",
            },
            {
                "id": "XBEN-042-24",
                "platform": "xbow",
                "vuln_type": "pwn",
                "difficulty": "hard",
                "class": "emulation-slow",
            },
        ],
    }
    if cycle_timeouts is not None:
        payload["cycle_timeouts"] = cycle_timeouts
    if extra_challenges:
        payload["challenges"].extend(extra_challenges)
    path = tmp_path / "queues.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _make_evaluator(tmp_path, config_path):
    from open_ctf.eval.evaluator import ModelEvaluator

    ev = ModelEvaluator(
        model="stub",
        challenges_yaml=str(config_path),
        traces_dir=str(tmp_path / "traces"),
        max_turns=1,
        max_time=1,
    )
    ev.run_challenge = _stub_run  # type: ignore[assignment]
    ev._run_runtime_preflight = lambda _c: None  # type: ignore[assignment]
    return ev


def test_partition_queues_routes_by_class_tag(tmp_path):
    """Challenges with ``class: emulation-slow`` route to the slow queue."""
    from open_ctf.eval.evaluator import ModelEvaluator

    cfg = _write_queue_config(tmp_path)
    ev = _make_evaluator(tmp_path, cfg)
    challenges, _, cycle_timeouts = ev._load_challenges_and_skips()
    queues = ModelEvaluator._partition_queues(challenges)

    assert set(queues) == {"default", "emulation-slow"}
    assert [c["id"] for c in queues["default"]] == ["XBEN-003-24"]
    assert [c["id"] for c in queues["emulation-slow"]] == ["XBEN-042-24"]
    # cycle_timeouts default to the module-level mapping when YAML omits the block.
    assert cycle_timeouts["default"] == 3600.0
    assert cycle_timeouts["emulation-slow"] == 21600.0


def test_cycle_timeouts_yaml_override(tmp_path):
    """Explicit ``cycle_timeouts`` values override module defaults."""
    cfg = _write_queue_config(
        tmp_path,
        cycle_timeouts={"default": 1800, "emulation-slow": 14400},
    )
    ev = _make_evaluator(tmp_path, cfg)
    _, _, cycle_timeouts = ev._load_challenges_and_skips()

    assert cycle_timeouts["default"] == 1800.0
    assert cycle_timeouts["emulation-slow"] == 14400.0


def test_cycle_timeouts_rejects_non_positive(tmp_path):
    """Zero / negative cycle timeouts must fail loudly at load time."""
    cfg = _write_queue_config(
        tmp_path,
        cycle_timeouts={"default": 0},
    )
    ev = _make_evaluator(tmp_path, cfg)
    import pytest

    with pytest.raises(ValueError, match="must be > 0"):
        ev._load_challenges_and_skips()


def test_run_all_emits_queue_report_per_class(tmp_path, caplog):
    """``EvalReport.queues`` contains one entry per scheduled queue class."""
    from open_ctf.eval.evaluator import DEFAULT_QUEUE_CLASS

    cfg = _write_queue_config(tmp_path)
    ev = _make_evaluator(tmp_path, cfg)

    with caplog.at_level(logging.INFO, logger="open_ctf.eval.evaluator"):
        report = ev.run_all()

    # default queue runs before emulation-slow (fast-queue results flush first).
    classes_in_order = [q["queue_class"] for q in report.queues]
    assert classes_in_order == [DEFAULT_QUEUE_CLASS, "emulation-slow"]
    # All challenges executed under their declared class — no cycle timeouts.
    by_class = {q["queue_class"]: q for q in report.queues}
    assert by_class["default"]["cycle_expired"] is False
    assert by_class["emulation-slow"]["cycle_expired"] is False
    assert by_class["default"]["cycle_timeouts"] == 0
    assert by_class["emulation-slow"]["cycle_timeouts"] == 0
    # Per-challenge results carry queue_class.
    by_id = {r["challenge_id"]: r for r in report.results}
    assert by_id["XBEN-003-24"]["queue_class"] == DEFAULT_QUEUE_CLASS
    assert by_id["XBEN-042-24"]["queue_class"] == "emulation-slow"


def test_cycle_timeout_classifies_unrun_challenges(tmp_path, caplog):
    """When a queue's cycle budget expires, remaining challenges get ``cycle_timeout``."""
    # Three slow-queue challenges; the cycle budget is so small that only
    # the first one runs and the rest are abandoned as cycle_timeout.
    cfg = _write_queue_config(
        tmp_path,
        cycle_timeouts={"default": 3600, "emulation-slow": 0.01},
        extra_challenges=[
            {
                "id": "XBEN-091-24",
                "platform": "xbow",
                "vuln_type": "pwn",
                "difficulty": "hard",
                "class": "emulation-slow",
            },
            {
                "id": "XBEN-092-24",
                "platform": "xbow",
                "vuln_type": "pwn",
                "difficulty": "hard",
                "class": "emulation-slow",
            },
        ],
    )
    ev = _make_evaluator(tmp_path, cfg)

    # Make _stub_run cost 0.05s so the 0.01s slow-queue budget trips
    # after the first challenge but before the second.
    real_stub = ev.run_challenge

    def _slow_stub(challenge):
        time.sleep(0.05)
        return real_stub(challenge)

    ev.run_challenge = _slow_stub  # type: ignore[assignment]

    # The default queue's tiny challenge count must still run cleanly —
    # default budget is 3600s.
    with caplog.at_level(logging.WARNING, logger="open_ctf.eval.evaluator"):
        report = ev.run_all()

    # Fast queue: XBEN-003-24 ran under its 3600s budget.
    fast = next(q for q in report.queues if q["queue_class"] == "default")
    assert fast["cycle_expired"] is False
    assert fast["solved"] == 0  # stub returns failed
    assert fast["cycle_timeouts"] == 0

    # Slow queue: the first challenge ran; the remaining two are
    # cycle_timeout. Wall-clock budget expired triggers the warning log.
    slow = next(q for q in report.queues if q["queue_class"] == "emulation-slow")
    assert slow["cycle_expired"] is True
    assert slow["cycle_timeouts"] == 2
    assert "cycle budget exhausted" in caplog.text

    # Per-challenge statuses: first slow-queue challenge has status failed,
    # remaining two are cycle_timeout.
    by_id = {r["challenge_id"]: r for r in report.results}
    assert by_id["XBEN-042-24"]["status"] == "failed"
    assert by_id["XBEN-091-24"]["status"] == "cycle_timeout"
    assert by_id["XBEN-092-24"]["status"] == "cycle_timeout"
    assert by_id["XBEN-091-24"]["queue_class"] == "emulation-slow"

    # cycle_timeout entries do not count as solved and do not invoke the
    # solver — turns=0, elapsed=0.
    assert by_id["XBEN-091-24"]["solved"] is False
    assert by_id["XBEN-091-24"]["turns"] == 0
    assert by_id["XBEN-091-24"]["elapsed_seconds"] == 0.0

    # The fast queue's result is present in the report regardless of the
    # slow queue's failure — slow-queue timeout does not block fast-queue
    # flushing.
    assert "XBEN-003-24" in by_id
    assert by_id["XBEN-003-24"]["status"] == "failed"


def test_solver_vs_cycle_timeout_are_distinct(tmp_path):
    """``solver_timeout`` and ``cycle_timeout`` must surface as separate statuses.

    The regression gate (vecna-item:4568f82f-934b-4451-93b3-da18a87f696f)
    relies on this distinction to tell "harness gave up" from "solver gave
    up". A challenge whose elapsed wall-clock reaches the solver's
    per-challenge budget is ``solver_timeout``; a challenge that never
    ran because the queue's cycle budget expired is ``cycle_timeout``.
    """
    from open_ctf.eval.evaluator import ChallengeResult, ModelEvaluator

    ev = ModelEvaluator(
        model="stub",
        challenges_yaml="unused",
        max_turns=1,
        max_time=1,  # 60s solver budget
    )

    # Elapsed equals the solver budget => solver_timeout.
    assert ev._classify_status(solved=False, elapsed_seconds=60.0, error_msg=None) \
        == "solver_timeout"
    # Elapsed comfortably under the budget => failed (not solver_timeout).
    assert ev._classify_status(solved=False, elapsed_seconds=5.0, error_msg=None) \
        == "failed"
    # Success regardless of elapsed.
    assert ev._classify_status(solved=True, elapsed_seconds=60.0, error_msg=None) \
        == "solved"
    # Exception path => error.
    assert ev._classify_status(solved=False, elapsed_seconds=1.0, error_msg="boom") \
        == "error"

    # cycle_timeout results are synthesized — they never flow through
    # _classify_status. Confirm the synthesized row carries the right shape.
    row = ModelEvaluator._cycle_timeout_result(
        {"id": "XBEN-999-24", "platform": "xbow", "vuln_type": "pwn", "difficulty": "hard"},
        queue_class="emulation-slow",
    )
    assert isinstance(row, ChallengeResult)
    assert row.status == "cycle_timeout"
    assert row.queue_class == "emulation-slow"
    assert row.solved is False
    assert row.turns == 0
    assert row.elapsed_seconds == 0.0


def test_queue_class_helper_handles_missing_or_empty(tmp_path):
    """``_queue_class_for`` falls back to ``default`` for absent/empty class keys."""
    from open_ctf.eval.evaluator import DEFAULT_QUEUE_CLASS, ModelEvaluator

    assert ModelEvaluator._queue_class_for({}) == DEFAULT_QUEUE_CLASS
    assert ModelEvaluator._queue_class_for({"class": ""}) == DEFAULT_QUEUE_CLASS
    assert ModelEvaluator._queue_class_for({"class": "   "}) == DEFAULT_QUEUE_CLASS
    assert ModelEvaluator._queue_class_for({"class": "emulation-slow"}) == "emulation-slow"


def test_markdown_renders_queue_section(tmp_path):
    """``_format_markdown`` adds a ``## Queues`` section when queue reports exist."""
    cfg = _write_queue_config(tmp_path)
    ev = _make_evaluator(tmp_path, cfg)
    report = ev.run_all()

    md = ev._format_markdown(report)
    assert "## Queues" in md
    assert "emulation-slow" in md
    assert "Cycle Budget" in md
    # Per-challenge results gain a ``Queue`` column.
    assert "| Queue |" in md or "| default |" in md  # header or row presence

    # Unused param.
    del tmp_path


def test_default_yaml_registers_xben042_as_emulation_slow():
    """``configs/challenges/eval_default.yaml`` registers XBEN-042-24 in the slow queue.

    This is the durable replacement for the temporary skip-list entry —
    once the slow queue clears three nightly cycles, a follow-up PR
    removes the skip and this registration goes live.
    """
    import pathlib
    import yaml as _yaml

    cfg_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "configs" / "challenges" / "eval_default.yaml"
    )
    data = _yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    # cycle_timeouts block present, default unchanged, slow queue at 6x.
    assert data["cycle_timeouts"]["default"] == 3600
    assert data["cycle_timeouts"]["emulation-slow"] == 21600

    # XBEN-042-24 is registered with the slow-queue class tag.
    matches = [c for c in data["challenges"] if c["id"] == "XBEN-042-24"]
    assert len(matches) == 1, "XBEN-042-24 must be registered exactly once"
    assert matches[0]["class"] == "emulation-slow"

    # Skip-list entry is still present (belt-and-suspenders for this PR);
    # removal lands in the follow-up once the slow queue is green.
    assert "XBEN-042-24" in data["skip"]
