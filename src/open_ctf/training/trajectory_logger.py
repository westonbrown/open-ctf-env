"""Structured trajectory logging for GRPO training post-run analysis.

Provides per-generation JSONL logging, step summaries, and a challenge
scoreboard so that after a training run you can:
  1. Replay what the model generated per step
  2. See which reward signals fired and how much each contributed
  3. Know which challenges the model is learning to solve vs struggling with

All data is written to ``{output_dir}/trajectories/`` as JSONL files.
The scoreboard is saved as ``{output_dir}/challenge_scoreboard.json``.

When ``tensorboard_dir`` is provided, CTF-specific scalars are written
alongside SkyRL's native training metrics (loss, KL, gradients).

No external dependencies beyond stdlib + json.  TensorBoard writing is
optional and gracefully degrades if ``tensorboard`` is not installed.
"""

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class TrajectoryLogger:
    """Saves per-generation GRPO data as structured JSONL.

    Thread-safe: multiple SkyRL env workers may call log_generation()
    concurrently from different threads.

    Usage::

        tl = TrajectoryLogger("/path/to/output")
        tl.log_generation(
            global_step=34,
            generation_idx=2,
            challenge_id="forensics_urgent",
            prompt_messages=[...],
            model_output="...",
            tool_calls=[{"name": "shell_command", "args": {...}, "output": "..."}],
            reward_total=0.36,
            reward_breakdown={"flag": 0.0, "format": 0.15, ...},
            flag_found=False,
            ground_truth_flag="FLAG{...}",
        )
        tl.log_step_summary(global_step=34, rewards=[0.36, 0.12, 0.0, 0.0])
        tl.save_scoreboard()
    """

    def __init__(
        self,
        output_dir: str,
        enabled: bool = True,
        tensorboard_dir: Optional[str] = None,
    ) -> None:
        self._output_dir = output_dir
        self._enabled = enabled
        self._trajectories_dir = os.path.join(output_dir, "trajectories")
        self._lock = threading.Lock()
        # Challenge scoreboard: {challenge_id: {attempts, solves, rewards, ...}}
        self._scoreboard: Dict[str, Dict[str, Any]] = {}
        self._tb_writer = None

        if self._enabled:
            os.makedirs(self._trajectories_dir, exist_ok=True)
            logger.info("TrajectoryLogger initialized: %s", self._trajectories_dir)

        # Optional TensorBoard writer for CTF-specific scalars.
        if tensorboard_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=tensorboard_dir)
                logger.info("TensorBoard CTF metrics: %s", tensorboard_dir)
            except ImportError:
                logger.info("tensorboard not installed; CTF scalars disabled")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log_generation(
        self,
        global_step: int,
        generation_idx: int = 0,
        challenge_id: Optional[str] = None,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        target: Optional[str] = None,
        prompt_messages: Optional[List[Dict[str, Any]]] = None,
        model_output: Optional[str] = None,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        reward_total: float = 0.0,
        reward_breakdown: Optional[Dict[str, float]] = None,
        flag_found: bool = False,
        flag_submitted: Optional[str] = None,
        ground_truth_flag: Optional[str] = None,
        response_length: int = 0,
        num_tool_calls: int = 0,
        **extra: Any,
    ) -> None:
        """Log a single generation (one rollout) to the step JSONL file.

        Args:
            global_step: Current training step number.
            generation_idx: Index within the step's generation batch.
            challenge_id: Challenge identifier.
            category: Challenge category (web, forensics, crypto, etc.).
            difficulty: Challenge difficulty level.
            target: Target URL.
            prompt_messages: Input prompt messages.
            model_output: Raw model output text.
            tool_calls: List of {name, args, output} dicts.
            reward_total: Total reward score.
            reward_breakdown: Per-signal reward breakdown.
            flag_found: Whether the flag was found.
            flag_submitted: Flag string that was submitted, if any.
            ground_truth_flag: Expected flag string.
            response_length: Length of model response in characters.
            num_tool_calls: Number of tool calls executed.
            **extra: Additional fields to include in the log entry.
        """
        if not self._enabled:
            return

        entry = {
            "global_step": global_step,
            "generation_idx": generation_idx,
            "challenge_id": challenge_id,
            "category": category,
            "difficulty": difficulty,
            "target": target,
            "prompt_messages": prompt_messages,
            "model_output": _truncate(model_output, max_len=50000),
            "tool_calls": tool_calls,
            "reward_total": reward_total,
            "reward_breakdown": reward_breakdown,
            "flag_found": flag_found,
            "flag_submitted": flag_submitted,
            "ground_truth_flag": ground_truth_flag,
            "response_length": response_length,
            "num_tool_calls": num_tool_calls,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            entry.update(extra)

        filepath = os.path.join(
            self._trajectories_dir, f"step_{global_step}.jsonl"
        )
        line = json.dumps(entry, default=str, ensure_ascii=False) + "\n"

        with self._lock:
            with open(filepath, "a") as f:
                f.write(line)

    def log_step_summary(
        self,
        global_step: int,
        rewards: Optional[List[float]] = None,
        flag_found_count: int = 0,
        total_generations: int = 0,
        avg_tool_calls: float = 0.0,
        avg_response_length: float = 0.0,
        challenge_ids: Optional[List[str]] = None,
        **extra: Any,
    ) -> None:
        """Log aggregate statistics for a training step.

        Written to ``{trajectories_dir}/step_summaries.jsonl``.
        """
        if not self._enabled:
            return

        rewards = rewards or []
        avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
        min_reward = min(rewards) if rewards else 0.0
        max_reward = max(rewards) if rewards else 0.0
        reward_std = _std(rewards) if len(rewards) > 1 else 0.0

        summary = {
            "global_step": global_step,
            "total_generations": total_generations,
            "flag_found_count": flag_found_count,
            "flag_found_rate": (
                flag_found_count / total_generations
                if total_generations > 0
                else 0.0
            ),
            "avg_reward": avg_reward,
            "min_reward": min_reward,
            "max_reward": max_reward,
            "reward_std": reward_std,
            "avg_tool_calls": avg_tool_calls,
            "avg_response_length": avg_response_length,
            "unique_challenges": (
                len(set(challenge_ids)) if challenge_ids else 0
            ),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            summary.update(extra)

        filepath = os.path.join(self._trajectories_dir, "step_summaries.jsonl")
        line = json.dumps(summary, default=str, ensure_ascii=False) + "\n"

        with self._lock:
            with open(filepath, "a") as f:
                f.write(line)

        # Write CTF-specific scalars to TensorBoard.
        if self._tb_writer is not None:
            step = global_step
            self._tb_writer.add_scalar("ctf/avg_reward", avg_reward, step)
            self._tb_writer.add_scalar("ctf/min_reward", min_reward, step)
            self._tb_writer.add_scalar("ctf/max_reward", max_reward, step)
            self._tb_writer.add_scalar("ctf/reward_std", reward_std, step)
            self._tb_writer.add_scalar(
                "ctf/flag_found_rate",
                flag_found_count / total_generations if total_generations > 0 else 0.0,
                step,
            )
            self._tb_writer.add_scalar("ctf/avg_tool_calls", avg_tool_calls, step)
            self._tb_writer.add_scalar("ctf/avg_response_length", avg_response_length, step)

    def log_challenge_result(
        self,
        challenge_id: str,
        category: Optional[str] = None,
        difficulty: Optional[str] = None,
        reward: float = 0.0,
        flag_found: bool = False,
    ) -> None:
        """Accumulate a result for the challenge scoreboard.

        Thread-safe. Call save_scoreboard() at training end to persist.
        """
        if not self._enabled or not challenge_id:
            return

        with self._lock:
            if challenge_id not in self._scoreboard:
                self._scoreboard[challenge_id] = {
                    "attempts": 0,
                    "solves": 0,
                    "rewards": [],
                    "category": category,
                    "difficulty": difficulty,
                }
            entry = self._scoreboard[challenge_id]
            entry["attempts"] += 1
            if flag_found:
                entry["solves"] += 1
            entry["rewards"].append(reward)
            # Update category/difficulty if not set
            if category and not entry.get("category"):
                entry["category"] = category
            if difficulty and not entry.get("difficulty"):
                entry["difficulty"] = difficulty

    def save_scoreboard(self) -> Optional[str]:
        """Write the challenge scoreboard to JSON.

        Returns:
            Path to the scoreboard file, or None if disabled/empty.
        """
        if not self._enabled:
            return None

        with self._lock:
            if not self._scoreboard:
                return None

            scoreboard = {}
            for cid, data in self._scoreboard.items():
                rewards = data["rewards"]
                scoreboard[cid] = {
                    "attempts": data["attempts"],
                    "solves": data["solves"],
                    "solve_rate": (
                        data["solves"] / data["attempts"]
                        if data["attempts"] > 0
                        else 0.0
                    ),
                    "avg_reward": (
                        sum(rewards) / len(rewards) if rewards else 0.0
                    ),
                    "best_reward": max(rewards) if rewards else 0.0,
                    "worst_reward": min(rewards) if rewards else 0.0,
                    "category": data.get("category"),
                    "difficulty": data.get("difficulty"),
                }

        filepath = os.path.join(self._output_dir, "challenge_scoreboard.json")
        with open(filepath, "w") as f:
            json.dump(scoreboard, f, indent=2, default=str, ensure_ascii=False)

        logger.info(
            "Challenge scoreboard saved: %s (%d challenges)",
            filepath,
            len(scoreboard),
        )
        return filepath

    def get_scoreboard(self) -> Dict[str, Dict[str, Any]]:
        """Return a copy of the current scoreboard data."""
        with self._lock:
            result = {}
            for cid, data in self._scoreboard.items():
                rewards = data["rewards"]
                result[cid] = {
                    "attempts": data["attempts"],
                    "solves": data["solves"],
                    "solve_rate": (
                        data["solves"] / data["attempts"]
                        if data["attempts"] > 0
                        else 0.0
                    ),
                    "avg_reward": (
                        sum(rewards) / len(rewards) if rewards else 0.0
                    ),
                    "best_reward": max(rewards) if rewards else 0.0,
                    "category": data.get("category"),
                    "difficulty": data.get("difficulty"),
                }
            return result


    def flush_scoreboard_to_tensorboard(self, global_step: int = 0) -> None:
        """Write per-challenge solve rates to TensorBoard as a bar chart."""
        if self._tb_writer is None:
            return
        with self._lock:
            for cid, data in self._scoreboard.items():
                attempts = data["attempts"]
                if attempts == 0:
                    continue
                solve_rate = data["solves"] / attempts
                avg_r = sum(data["rewards"]) / len(data["rewards"]) if data["rewards"] else 0.0
                safe_cid = cid.replace("/", "_").replace(" ", "_")
                self._tb_writer.add_scalar(f"ctf_challenge/{safe_cid}/solve_rate", solve_rate, global_step)
                self._tb_writer.add_scalar(f"ctf_challenge/{safe_cid}/avg_reward", avg_r, global_step)

    def close(self) -> None:
        """Flush and close TensorBoard writer if active."""
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
            self._tb_writer = None


def _truncate(text: Optional[str], max_len: int = 50000) -> Optional[str]:
    """Truncate text to max_len characters with an indicator."""
    if text is None or len(text) <= max_len:
        return text
    return text[:max_len] + f"... [truncated, {len(text)} total chars]"


def _std(values: List[float]) -> float:
    """Compute sample standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return variance ** 0.5
