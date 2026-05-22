"""Tests for open_ctf.cli.evaluate wiring."""

from __future__ import annotations

import logging
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
    _, skipped = evaluator._load_challenges_and_skips()
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
