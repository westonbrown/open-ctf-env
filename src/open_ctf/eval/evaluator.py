"""Model evaluation harness for CTF challenges.

Runs a model (base or fine-tuned) against a set of challenges using BoxPwnr's
Solver, collects per-challenge statistics, and produces a JSON report plus a
human-readable markdown summary table.

Usage (programmatic):
    from open_ctf.eval import ModelEvaluator
    ev = ModelEvaluator(model="ollama/nanbeige4.1-3b")
    report = ev.run_all()
    ev.save(report, "outputs/eval")

See also ``open_ctf.cli.evaluate`` for the CLI wrapper.
"""

import importlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_PACKAGE_DIR = Path(__file__).resolve().parent.parent  # src/open_ctf/


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


#: Default per-class cycle wall-clock budgets in seconds. Used when the
#: challenges YAML omits ``cycle_timeouts:``. The default-queue value MUST
#: stay aligned with the current nightly's pre-change baseline; it protects
#: the "green nightly = healthy" signal that the bench is standing up to
#: defend. The emulation-slow value is the proposal in
#: vecna-item:6d4b587a-7c6c-4e00-81e6-1a1c915a71bc (6× default) and should
#: be revisited after three green nightly cycles.
DEFAULT_CYCLE_TIMEOUTS: dict[str, float] = {
    "default": 3600.0,         # 1h — fast queue
    "emulation-slow": 21600.0, # 6h — slow queue (6× default)
}

#: Default queue class for challenges without an explicit ``class:`` key.
DEFAULT_QUEUE_CLASS = "default"

#: Allowed values for ``ChallengeResult.status``. ``cycle_timeout`` and
#: ``solver_timeout`` are kept distinct so the regression gate can tell
#: "harness gave up because the queue's cycle wall-clock expired" from
#: "solver gave up because its own per-challenge budget expired".
RESULT_STATUSES = frozenset({
    "solved",
    "failed",
    "solver_timeout",
    "cycle_timeout",
    "error",
})


@dataclass
class ChallengeResult:
    """Result for a single challenge attempt."""

    challenge_id: str
    platform: str
    vuln_type: str
    difficulty: str
    solved: bool
    turns: int
    elapsed_seconds: float
    error: str | None = None
    #: Terminal status — one of :data:`RESULT_STATUSES`. ``solved`` mirrors
    #: ``solved=True``; ``solver_timeout`` means the per-challenge solver
    #: budget expired; ``cycle_timeout`` means the queue's per-class cycle
    #: wall-clock expired before this challenge ran (or while it was
    #: running); ``error`` is set when the runner raised; ``failed`` is the
    #: default for a clean no-flag finish.
    status: str = "failed"
    #: Name of the queue this challenge was assigned to. Defaults to
    #: :data:`DEFAULT_QUEUE_CLASS`.
    queue_class: str = DEFAULT_QUEUE_CLASS


@dataclass
class SkippedChallenge:
    """Record for a challenge filtered out by the YAML ``skip`` block."""

    challenge_id: str
    reason: str
    opened_on: str | None = None
    linked_issue: str | None = None
    note: str | None = None


@dataclass
class QueueReport:
    """Per-queue execution accounting.

    Populated once per queue class run in :meth:`ModelEvaluator.run_all` so
    the regression gate can attribute timeouts and wall-clock to the right
    class without parsing per-challenge results.
    """

    queue_class: str
    cycle_timeout_seconds: float
    elapsed_seconds: float
    cycle_expired: bool
    total: int
    solved: int
    solver_timeouts: int
    cycle_timeouts: int
    errors: int


@dataclass
class EvalReport:
    """Aggregate evaluation report."""

    model: str
    strategy: str
    timestamp: str
    total_challenges: int
    solved: int
    solve_rate: float
    avg_turns: float
    avg_time_seconds: float
    results: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    #: Per-queue accounting; one entry per queue class that was scheduled.
    queues: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


class ModelEvaluator:
    """Run BoxPwnr solver across a challenge set and collect metrics.

    Args:
        model: LLM model identifier (e.g. ``ollama/nanbeige4.1-3b``).
        challenges_yaml: Path to YAML file listing challenges.
        platform: Default platform for challenges (overridden per-challenge).
        strategy: BoxPwnr strategy name (``chat_tools`` or ``chat``).
        max_turns: Maximum conversation turns per challenge.
        max_time: Maximum time in minutes per challenge.
        traces_dir: Directory for BoxPwnr trace output.
        reasoning_effort: Reasoning effort level for supported models.
        attempts: Number of attempts per challenge.
    """

    def __init__(
        self,
        model: str = "openrouter/openai/gpt-oss-120b",
        challenges_yaml: str = str(_PACKAGE_DIR / "configs" / "challenges.yaml"),
        platform: str = "xbow",
        strategy: str = "chat_tools",
        max_turns: int = 50,
        max_time: int = 30,
        traces_dir: str = "./targets",
        reasoning_effort: str = "medium",
        attempts: int = 1,
        agent: str = "boxpwnr",
        challenge_registry: str | None = None,
        target_map: str | None = None,
        host: str = "localhost",
    ) -> None:
        self.model = model
        self.challenges_yaml = challenges_yaml
        self.platform = platform
        self.strategy = strategy
        self.max_turns = max_turns
        self.max_time = max_time
        self.traces_dir = traces_dir
        self.reasoning_effort = reasoning_effort
        self.attempts = attempts
        self.agent = agent
        self.challenge_registry = challenge_registry
        self.target_map = target_map
        self.host = host

    # ------------------------------------------------------------------
    # Challenge loading
    # ------------------------------------------------------------------

    def load_challenges(self) -> list[dict[str, Any]]:
        """Load challenge list from YAML config.

        Returns:
            List of challenge dicts with keys: id, platform, vuln_type, difficulty.
        """
        challenges, _, _ = self._load_challenges_and_skips()
        return challenges

    def _load_challenges_and_skips(
        self,
    ) -> tuple[list[dict[str, Any]], list[SkippedChallenge], dict[str, float]]:
        """Load challenges + skip-list + per-class cycle timeouts from YAML.

        Returns:
            Tuple of ``(non-skipped challenges, SkippedChallenge records,
            cycle_timeouts)``. ``cycle_timeouts`` is a mapping from queue
            class name to wall-clock seconds; classes absent from the YAML
            fall back to :data:`DEFAULT_CYCLE_TIMEOUTS`.

        The skip block is a top-level ``skip:`` mapping keyed by challenge id::

            skip:
              XBEN-042-24:
                reason: qemu-slow
                opened_on: "2026-05-22"
                linked_issue: "vecna-item:..."

        Skipped entries are removed from the challenge list returned to the
        runner. The caller is expected to log + record them in the report so
        the absence is auditable rather than silent.

        Per-class cycle timeouts may be overridden via::

            cycle_timeouts:
              default: 3600
              emulation-slow: 21600

        Each challenge may carry an optional ``class:`` key. Challenges
        without a class fall into :data:`DEFAULT_QUEUE_CLASS` and inherit
        the default-queue cycle timeout. Slow-queue tagging keeps long
        challenges (e.g. QEMU full-system emulation) off the fast queue's
        wall-clock so they cannot push fast-queue p99 around.
        """
        yaml_path = Path(self.challenges_yaml)
        if not yaml_path.is_absolute():
            yaml_path = _PACKAGE_DIR / yaml_path

        if not yaml_path.exists():
            raise FileNotFoundError(f"Challenges config not found: {yaml_path}")

        with open(yaml_path) as f:
            data = yaml.safe_load(f) or {}

        all_challenges = data.get("challenges", [])
        if not all_challenges:
            raise ValueError(f"No challenges found in {yaml_path}")

        raw_skip = data.get("skip") or {}
        if not isinstance(raw_skip, dict):
            raise ValueError(
                f"`skip` in {yaml_path} must be a mapping keyed by challenge id, "
                f"got {type(raw_skip).__name__}"
            )

        skip_records: list[SkippedChallenge] = []
        for cid, meta in raw_skip.items():
            meta = meta or {}
            if not isinstance(meta, dict):
                raise ValueError(
                    f"skip[{cid}] in {yaml_path} must be a mapping, "
                    f"got {type(meta).__name__}"
                )
            reason = meta.get("reason")
            if not reason:
                raise ValueError(
                    f"skip[{cid}] in {yaml_path} is missing required `reason` field"
                )
            skip_records.append(
                SkippedChallenge(
                    challenge_id=str(cid),
                    reason=str(reason),
                    opened_on=(str(meta["opened_on"]) if meta.get("opened_on") else None),
                    linked_issue=(
                        str(meta["linked_issue"]) if meta.get("linked_issue") else None
                    ),
                    note=(str(meta["note"]) if meta.get("note") else None),
                )
            )

        skip_ids = {s.challenge_id for s in skip_records}
        challenges = [c for c in all_challenges if str(c.get("id")) not in skip_ids]

        cycle_timeouts = self._parse_cycle_timeouts(data.get("cycle_timeouts"), yaml_path)

        if skip_records:
            logger.info(
                "Loaded %d challenges from %s (%d skipped: %s)",
                len(challenges),
                yaml_path,
                len(skip_records),
                ", ".join(sorted(skip_ids)),
            )
        else:
            logger.info("Loaded %d challenges from %s", len(challenges), yaml_path)
        return challenges, skip_records, cycle_timeouts

    @staticmethod
    def _parse_cycle_timeouts(
        raw: Any,
        yaml_path: Path,
    ) -> dict[str, float]:
        """Parse the optional top-level ``cycle_timeouts`` block.

        Falls back to :data:`DEFAULT_CYCLE_TIMEOUTS` when the key is absent
        or empty. Missing classes inherit their default; explicit values
        override. Values must be positive numbers (seconds).
        """
        result: dict[str, float] = dict(DEFAULT_CYCLE_TIMEOUTS)
        if raw is None:
            return result
        if not isinstance(raw, dict):
            raise ValueError(
                f"`cycle_timeouts` in {yaml_path} must be a mapping "
                f"class -> seconds, got {type(raw).__name__}"
            )
        for cls, seconds in raw.items():
            try:
                value = float(seconds)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"cycle_timeouts[{cls!r}] in {yaml_path} must be a "
                    f"positive number of seconds, got {seconds!r}"
                ) from exc
            if value <= 0:
                raise ValueError(
                    f"cycle_timeouts[{cls!r}] in {yaml_path} must be > 0, "
                    f"got {value}"
                )
            result[str(cls)] = value
        return result

    @staticmethod
    def _queue_class_for(challenge: dict[str, Any]) -> str:
        """Return the queue class for a challenge dict.

        Reads the ``class`` key (string) and falls back to
        :data:`DEFAULT_QUEUE_CLASS` when absent or empty.
        """
        raw = challenge.get("class")
        if raw is None:
            return DEFAULT_QUEUE_CLASS
        name = str(raw).strip()
        return name or DEFAULT_QUEUE_CLASS

    # ------------------------------------------------------------------
    # Single challenge run
    # ------------------------------------------------------------------

    def run_challenge(self, challenge: dict[str, Any]) -> ChallengeResult:
        """Run a single challenge and return the result.

        Args:
            challenge: Dict with at least ``id``. Optional: ``platform``,
                ``vuln_type``, ``difficulty``.

        Returns:
            ChallengeResult with solve status, turns, and timing.
        """
        cid = challenge["id"]
        platform = challenge.get("platform", self.platform)
        vuln_type = challenge.get("vuln_type", "unknown")
        difficulty = challenge.get("difficulty", "unknown")
        queue_class = self._queue_class_for(challenge)

        logger.info("Running challenge %s (%s, %s)", cid, vuln_type, difficulty)

        start = time.time()
        solved = False
        turns = 0
        error_msg = None

        try:
            if str(self.agent).strip().lower() == "boxpwnr":
                from open_ctf.agent.runner import AgentRunner

                runner = AgentRunner(
                    platform=platform,
                    model=self.model,
                    strategy=self.strategy,
                    max_turns=self.max_turns,
                    max_time=self.max_time,
                    traces_dir=self.traces_dir,
                    reasoning_effort=self.reasoning_effort,
                    attempts=self.attempts,
                )
                runner.run(target=cid)
                # Check trace output for success
                solved, turns = self._parse_trace(cid, platform)
            else:
                solved, turns = self._run_with_custom_agent(challenge, platform)
        except Exception as e:
            error_msg = str(e)
            logger.warning("Challenge %s failed: %s", cid, error_msg)

        elapsed = time.time() - start
        status = self._classify_status(
            solved=solved,
            elapsed_seconds=elapsed,
            error_msg=error_msg,
        )

        return ChallengeResult(
            challenge_id=cid,
            platform=platform,
            vuln_type=vuln_type,
            difficulty=difficulty,
            solved=solved,
            turns=turns,
            elapsed_seconds=round(elapsed, 1),
            error=error_msg,
            status=status,
            queue_class=queue_class,
        )

    def _classify_status(
        self,
        solved: bool,
        elapsed_seconds: float,
        error_msg: str | None,
    ) -> str:
        """Classify a finished challenge run into a terminal status.

        ``solver_timeout`` is the heuristic for "the solver hit its own
        per-challenge budget" — currently inferred from elapsed wall-clock
        reaching ``max_time``. The regression gate (sibling
        vecna-item:4568f82f-934b-4451-93b3-da18a87f696f) uses this to keep
        solver-budget exhaustion separate from harness cycle exhaustion.
        """
        if error_msg:
            return "error"
        if solved:
            return "solved"
        # ``max_time`` is per-challenge solver budget in minutes; treat
        # elapsed within 1% of that bound as solver-side timeout.
        solver_budget_s = float(self.max_time) * 60.0
        if solver_budget_s > 0 and elapsed_seconds >= solver_budget_s * 0.99:
            return "solver_timeout"
        return "failed"

    def _run_with_custom_agent(
        self,
        challenge: dict[str, Any],
        platform: str,
    ) -> tuple[bool, int]:
        """Run evaluation with a custom CTFAgent implementation."""
        from open_ctf.agent.protocol import CTFAgent

        agent_spec = str(self.agent or "").strip()
        if agent_spec.lower().startswith("custom:"):
            agent_spec = agent_spec.split(":", 1)[1].strip()
        if not agent_spec:
            raise ValueError("Custom agent spec is empty")

        module_name, _, class_name = agent_spec.rpartition(".")
        if not module_name or not class_name:
            raise ValueError(
                f"Invalid --agent value {self.agent!r}. "
                "Use 'boxpwnr' or 'custom:module.ClassName'."
            )

        module = importlib.import_module(module_name)
        agent_cls = getattr(module, class_name)
        agent = agent_cls(
            model=self.model,
            platform=platform,
            strategy=self.strategy,
            traces_dir=self.traces_dir,
            reasoning_effort=self.reasoning_effort,
            attempts=self.attempts,
        )
        if not isinstance(agent, CTFAgent):
            raise TypeError(f"Resolved agent {agent_spec} does not satisfy CTFAgent protocol")

        cid = challenge["id"]
        target = (
            challenge.get("target")
            or challenge.get("target_url")
            or challenge.get("url")
            or cid
        )
        result = agent.solve(
            challenge=cid,
            target=str(target),
            max_steps=self.max_turns,
            timeout=max(1, int(self.max_time)) * 60,
        )
        solved = bool(getattr(result, "success", False))
        turns = int(getattr(result, "steps", 0) or 0)
        return solved, turns

    # ------------------------------------------------------------------
    # Batch run
    # ------------------------------------------------------------------

    def run_all(self) -> EvalReport:
        """Run all challenges and produce an evaluation report.

        Challenges are partitioned by ``class:`` into independent queues;
        each queue is bound by its own per-class cycle wall-clock budget
        (see :data:`DEFAULT_CYCLE_TIMEOUTS`). A slow queue blowing its
        budget does not block fast-queue results from flushing — fast
        queues run first and their results are recorded before any slow
        queue starts. Challenges left unrun when a queue's budget expires
        are emitted as :class:`ChallengeResult` rows with
        ``status='cycle_timeout'`` so the regression gate can attribute
        the loss to the harness rather than the solver.

        Returns:
            EvalReport with aggregate statistics, per-challenge results,
            and per-queue accounting.
        """
        challenges, skipped, cycle_timeouts = self._load_challenges_and_skips()
        for s in skipped:
            # Auditable single-line marker for the run log; matches the format
            # asserted by tests/test_cli_evaluate.py.
            logger.info("SKIPPED: %s (%s)", s.challenge_id, s.reason)
        self._run_runtime_preflight(challenges)

        queues = self._partition_queues(challenges)
        results: list[ChallengeResult] = []
        queue_reports: list[QueueReport] = []

        # Schedule the default queue before any non-default queue. The
        # contract is: fast-queue results flush regardless of slow-queue
        # behaviour. Within remaining queues we sort alphabetically for
        # determinism.
        ordered_classes = self._order_queue_classes(queues.keys())
        for queue_class in ordered_classes:
            queue_challenges = queues[queue_class]
            budget = cycle_timeouts.get(queue_class)
            if budget is None:
                # Unknown class — fall back to default-queue budget rather
                # than crash; log loudly so the typo is auditable.
                budget = cycle_timeouts.get(
                    DEFAULT_QUEUE_CLASS, DEFAULT_CYCLE_TIMEOUTS[DEFAULT_QUEUE_CLASS]
                )
                logger.warning(
                    "Queue class %r has no cycle_timeouts entry; "
                    "inheriting default-queue budget %.0fs",
                    queue_class, budget,
                )
            queue_results, queue_report = self._run_queue(
                queue_class=queue_class,
                queue_challenges=queue_challenges,
                cycle_timeout_seconds=budget,
            )
            results.extend(queue_results)
            queue_reports.append(queue_report)

        solved_count = sum(1 for r in results if r.solved)
        total = len(results)
        avg_turns = sum(r.turns for r in results) / max(total, 1)
        avg_time = sum(r.elapsed_seconds for r in results) / max(total, 1)

        report = EvalReport(
            model=self.model,
            strategy=self.strategy,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            total_challenges=total,
            solved=solved_count,
            solve_rate=round(solved_count / max(total, 1), 4),
            avg_turns=round(avg_turns, 1),
            avg_time_seconds=round(avg_time, 1),
            results=[asdict(r) for r in results],
            skipped=[asdict(s) for s in skipped],
            queues=[asdict(q) for q in queue_reports],
        )

        return report

    # ------------------------------------------------------------------
    # Queue scheduling
    # ------------------------------------------------------------------

    @staticmethod
    def _partition_queues(
        challenges: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        """Group challenges into queues keyed by their ``class:`` tag."""
        queues: dict[str, list[dict[str, Any]]] = {}
        for ch in challenges:
            qc = ModelEvaluator._queue_class_for(ch)
            queues.setdefault(qc, []).append(ch)
        return queues

    @staticmethod
    def _order_queue_classes(queue_classes: Any) -> list[str]:
        """Return queue classes in execution order.

        :data:`DEFAULT_QUEUE_CLASS` runs first so the fast queue's results
        flush before any non-default queue starts; remaining classes run
        in sorted order for determinism.
        """
        names = list(queue_classes)
        rest = sorted(n for n in names if n != DEFAULT_QUEUE_CLASS)
        if DEFAULT_QUEUE_CLASS in names:
            return [DEFAULT_QUEUE_CLASS, *rest]
        return rest

    def _run_queue(
        self,
        queue_class: str,
        queue_challenges: list[dict[str, Any]],
        cycle_timeout_seconds: float,
    ) -> tuple[list[ChallengeResult], QueueReport]:
        """Run a single queue under a per-class cycle wall-clock budget.

        When the cycle deadline passes between challenges, the remaining
        challenges are emitted as ``status='cycle_timeout'`` rows without
        invoking the solver, and the queue terminates early. A challenge
        that overshoots the deadline mid-flight is still recorded with
        whatever status the solver reports — the cycle bound is enforced
        at the boundary between challenges, not in-flight.
        """
        logger.info(
            "Queue %r: %d challenge(s), cycle budget %.0fs",
            queue_class, len(queue_challenges), cycle_timeout_seconds,
        )
        results: list[ChallengeResult] = []
        cycle_expired = False
        queue_start = time.monotonic()

        for idx, challenge in enumerate(queue_challenges):
            elapsed = time.monotonic() - queue_start
            if elapsed >= cycle_timeout_seconds:
                cycle_expired = True
                remaining = queue_challenges[idx:]
                logger.warning(
                    "Queue %r: cycle budget exhausted at %.1fs; "
                    "%d challenge(s) emitted as cycle_timeout",
                    queue_class, elapsed, len(remaining),
                )
                for ch in remaining:
                    results.append(self._cycle_timeout_result(ch, queue_class))
                break

            result = self.run_challenge(challenge)
            # run_challenge sets queue_class from the dict; keep it explicit
            # in case a caller overrides via a future hook.
            result.queue_class = queue_class
            results.append(result)
            log_status = result.status.upper()
            logger.info(
                "  %s %s (turns=%d, time=%.1fs, queue=%s)",
                log_status, result.challenge_id, result.turns,
                result.elapsed_seconds, queue_class,
            )

        queue_elapsed = time.monotonic() - queue_start
        queue_report = QueueReport(
            queue_class=queue_class,
            cycle_timeout_seconds=cycle_timeout_seconds,
            elapsed_seconds=round(queue_elapsed, 1),
            cycle_expired=cycle_expired,
            total=len(results),
            solved=sum(1 for r in results if r.status == "solved"),
            solver_timeouts=sum(1 for r in results if r.status == "solver_timeout"),
            cycle_timeouts=sum(1 for r in results if r.status == "cycle_timeout"),
            errors=sum(1 for r in results if r.status == "error"),
        )
        return results, queue_report

    @staticmethod
    def _cycle_timeout_result(
        challenge: dict[str, Any],
        queue_class: str,
    ) -> ChallengeResult:
        """Synthesize a ``cycle_timeout`` result for an un-run challenge."""
        return ChallengeResult(
            challenge_id=str(challenge["id"]),
            platform=str(challenge.get("platform", "unknown")),
            vuln_type=str(challenge.get("vuln_type", "unknown")),
            difficulty=str(challenge.get("difficulty", "unknown")),
            solved=False,
            turns=0,
            elapsed_seconds=0.0,
            error=None,
            status="cycle_timeout",
            queue_class=queue_class,
        )

    def _run_runtime_preflight(self, challenges: list[dict[str, Any]]) -> None:
        """Fail fast on registry/target/port/container mismatches for cybench runs."""
        cybench_ids = [
            str(challenge["id"])
            for challenge in challenges
            if str(challenge.get("platform", self.platform)).lower() == "cybench"
        ]
        if not cybench_ids:
            return

        registry_path = self.challenge_registry
        if not registry_path:
            default_registry = (
                Path(__file__).resolve().parents[3]
                / "configs"
                / "challenges"
                / "cybench.yaml"
            )
            if default_registry.exists():
                registry_path = str(default_registry)
        if not registry_path:
            raise ValueError(
                "Cybench evaluation requires a challenge registry "
                "(pass --challenge-registry)."
            )

        from open_ctf.challenges.preflight import validate_runtime_preflight
        from open_ctf.challenges.registry import ChallengeRegistry

        registry = ChallengeRegistry(str(registry_path))
        if self.target_map:
            registry.load_target_overrides(self.target_map, strict=False)
        validate_runtime_preflight(
            registry,
            host=self.host,
            challenge_ids=cybench_ids,
            require_reachable=True,
            strict_container_check=True,
        )

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save(self, report: EvalReport, output_dir: str) -> None:
        """Save evaluation report as JSON and markdown.

        Args:
            report: EvalReport to save.
            output_dir: Directory to write report files.
        """
        os.makedirs(output_dir, exist_ok=True)

        # JSON
        json_path = os.path.join(output_dir, "eval_report.json")
        with open(json_path, "w") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info("JSON report saved to %s", json_path)

        # Markdown
        md_path = os.path.join(output_dir, "eval_report.md")
        with open(md_path, "w") as f:
            f.write(self._format_markdown(report))
        logger.info("Markdown report saved to %s", md_path)

    @staticmethod
    def _format_markdown(report: EvalReport) -> str:
        """Format an EvalReport as a markdown table."""
        lines = [
            "# Evaluation Report",
            "",
            f"- **Model:** {report.model}",
            f"- **Strategy:** {report.strategy}",
            f"- **Timestamp:** {report.timestamp}",
            f"- **Challenges:** {report.total_challenges}",
            f"- **Solved:** {report.solved}/{report.total_challenges} "
            f"({report.solve_rate * 100:.1f}%)",
            f"- **Avg Turns:** {report.avg_turns}",
            f"- **Avg Time:** {report.avg_time_seconds}s",
            "",
        ]

        if report.queues:
            lines.extend([
                "## Queues",
                "",
                "| Queue | Cycle Budget (s) | Elapsed (s) | Cycle Expired "
                "| Solved | Solver-Timeouts | Cycle-Timeouts | Errors |",
                "|-------|------------------|-------------|---------------"
                "|--------|-----------------|----------------|--------|",
            ])
            for q in report.queues:
                lines.append(
                    f"| {q['queue_class']} | {q['cycle_timeout_seconds']:.0f} "
                    f"| {q['elapsed_seconds']} "
                    f"| {'Yes' if q['cycle_expired'] else 'No'} "
                    f"| {q['solved']} | {q['solver_timeouts']} "
                    f"| {q['cycle_timeouts']} | {q['errors']} |"
                )
            lines.append("")

        lines.extend([
            "## Per-Challenge Results",
            "",
            "| Challenge | Queue | Platform | Vuln Type | Difficulty "
            "| Status | Turns | Time (s) | Error |",
            "|-----------|-------|----------|-----------|------------"
            "|--------|-------|----------|-------|",
        ])

        for r in report.results:
            status_str = r.get("status") or ("solved" if r["solved"] else "failed")
            error_str = r.get("error") or ""
            if len(error_str) > 40:
                error_str = error_str[:37] + "..."
            lines.append(
                f"| {r['challenge_id']} | {r.get('queue_class', 'default')} "
                f"| {r['platform']} | {r['vuln_type']} "
                f"| {r['difficulty']} | {status_str} | {r['turns']} "
                f"| {r['elapsed_seconds']} | {error_str} |"
            )

        if report.skipped:
            lines.extend([
                "",
                "## Skipped",
                "",
                "| Challenge | Reason | Opened | Linked Issue |",
                "|-----------|--------|--------|--------------|",
            ])
            for s in report.skipped:
                lines.append(
                    f"| SKIPPED: {s['challenge_id']} "
                    f"| {s['reason']} "
                    f"| {s.get('opened_on') or ''} "
                    f"| {s.get('linked_issue') or ''} |"
                )

        lines.append("")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Trace parsing
    # ------------------------------------------------------------------

    def _parse_trace(self, challenge_id: str, platform: str) -> tuple:
        """Parse BoxPwnr trace directory for solve status and turn count.

        Returns:
            (solved: bool, turns: int)
        """
        trace_dir = Path(self.traces_dir) / platform / challenge_id
        if not trace_dir.exists():
            return False, 0

        # BoxPwnr writes a stats.json in the trace directory
        results_file = trace_dir / "stats.json"
        if results_file.exists():
            try:
                with open(results_file) as f:
                    data = json.load(f)
                solved = data.get("status") == "success"
                turns = data.get("total_turns", 0)
                return solved, turns
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: scan for flag in conversation log
        conv_file = trace_dir / "conversation.json"
        if conv_file.exists():
            try:
                with open(conv_file) as f:
                    conv = json.load(f)
                turns = len([m for m in conv if m.get("role") == "assistant"])
                # Check if flag_found was called
                for msg in conv:
                    for tc in msg.get("tool_calls", []):
                        if tc.get("function", {}).get("name") == "flag_found":
                            return True, turns
            except (json.JSONDecodeError, KeyError):
                pass

        return False, 0


# ---------------------------------------------------------------------------
# Comparison helper
# ---------------------------------------------------------------------------


def compare_reports(
    base_path: str,
    tuned_path: str,
) -> str:
    """Compare two evaluation reports and produce a markdown diff table.

    Args:
        base_path: Path to base model eval_report.json.
        tuned_path: Path to fine-tuned model eval_report.json.

    Returns:
        Markdown string with comparison table.
    """
    with open(base_path) as f:
        base = json.load(f)
    with open(tuned_path) as f:
        tuned = json.load(f)

    lines = [
        "# Model Comparison",
        "",
        "| Metric | Base | Fine-Tuned | Delta |",
        "|--------|------|------------|-------|",
    ]

    for key, label in [
        ("solve_rate", "Solve Rate"),
        ("avg_turns", "Avg Turns"),
        ("avg_time_seconds", "Avg Time (s)"),
    ]:
        b = base.get(key, 0)
        t = tuned.get(key, 0)
        delta = t - b
        if key == "solve_rate":
            lines.append(f"| {label} | {b*100:.1f}% | {t*100:.1f}% | {delta*100:+.1f}% |")
        else:
            lines.append(f"| {label} | {b:.1f} | {t:.1f} | {delta:+.1f} |")

    # Per-challenge comparison
    base_results = {r["challenge_id"]: r for r in base.get("results", [])}
    tuned_results = {r["challenge_id"]: r for r in tuned.get("results", [])}
    all_ids = sorted(set(base_results) | set(tuned_results))

    lines.extend([
        "",
        "## Per-Challenge",
        "",
        "| Challenge | Base | Fine-Tuned |",
        "|-----------|------|------------|",
    ])

    for cid in all_ids:
        b_solved = "Yes" if base_results.get(cid, {}).get("solved") else "No"
        t_solved = "Yes" if tuned_results.get(cid, {}).get("solved") else "No"
        lines.append(f"| {cid} | {b_solved} | {t_solved} |")

    lines.append("")
    return "\n".join(lines)
