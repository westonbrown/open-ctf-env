"""CTFReward adapter for SkyRL's per-step reward protocol.

SkyRL expects environments to return a reward at each step (float).
Our CTFReward is designed for batch scoring at episode end. This adapter
bridges the two:

  - Non-terminal steps (step_wise_trajectories=false): returns 0.0.
    All reward signal comes from the terminal CTFReward computation.
    This matches OpenThoughts-Agent methodology: intermediate rewards
    dilute the RLOO-N advantage signal because the estimator sums all
    per-token rewards into a scalar score for normalization.

  - Non-terminal steps (step_wise_trajectories=true): returns a small
    shaping reward based on format compliance and phase progression.
    The terminal CTFReward (8 signals) still provides the dominant
    learning signal; per-step rewards are kept intentionally small
    (+/- 0.02 to 0.03) to avoid diluting the terminal gap.

  - Terminal step: full CTFReward 8-signal score (unchanged).

This module also provides a factory function to create a CTFReward
instance from a training config dict.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


_VALID_REWARD_KEYS = frozenset({
    "flag_weight", "efficiency_weight", "progression_weight",
    "format_weight", "exploration_weight", "uniqueness_weight",
    "recovery_weight", "cognitive_weight", "hallucination_penalty",
    "noise_range", "exploration_gamma", "seed", "use_gdpo",
})


def create_reward_fn(config: Dict[str, Any]):
    """Create a CTFReward instance from training config.

    Args:
        config: Full training config dict (has 'reward' key).

    Returns:
        CTFReward instance.

    Raises:
        KeyError: If reward config contains unrecognized keys.
    """
    from open_ctf.rewards import CTFReward

    reward_cfg = config.get("reward", {})

    unknown_keys = set(reward_cfg.keys()) - _VALID_REWARD_KEYS
    if unknown_keys:
        raise KeyError(
            f"Unrecognized reward config keys: {sorted(unknown_keys)}. "
            f"Valid keys: {sorted(_VALID_REWARD_KEYS)}"
        )

    kwargs = {}
    for key in _VALID_REWARD_KEYS:
        if key in reward_cfg:
            if key == "seed":
                kwargs[key] = int(reward_cfg[key]) if reward_cfg[key] is not None else None
            else:
                kwargs[key] = float(reward_cfg[key])

    kwargs["use_gdpo"] = bool(reward_cfg.get("use_gdpo", False))

    return CTFReward(**kwargs)


# ---------------------------------------------------------------------------
# CTF attack phase classification for phase-progression reward
# ---------------------------------------------------------------------------

# Map tool names to attack phases. Phases are ordered:
# recon (0) -> enumerate (1) -> exploit (2) -> post-exploit (3)
_TOOL_PHASE: Dict[str, int] = {
    # Phase 0: Reconnaissance — gathering initial information
    "web_search": 0,
    "file_search": 0,
    "grep": 0,
    "read_file": 0,
    # Phase 1: Enumeration — active probing and scanning
    "shell_command": 1,
    "execute_command": 1,
    "exec_command": 1,
    "list_sessions": 1,
    # Phase 2: Exploitation — interactive exploitation
    "python_code": 2,
    "write_stdin": 2,
    "apply_patch": 2,
    # Phase 3: Post-exploitation — flag submission
    "flag_found": 3,
    "submit_flag": 3,
    "close_session": 3,
}


def _get_phase(tool_name: str) -> int:
    """Return the attack phase (0-3) for a tool name."""
    return _TOOL_PHASE.get(tool_name, 1)


def per_step_reward(
    tool_calls_so_far: List[Dict[str, str]],
    step: int,
    max_steps: int,
    *,
    step_tool_call_count: int = 0,
    step_wise: bool = False,
) -> float:
    """Per-step reward during GRPO rollouts.

    When step_wise=False (default), returns 0.0 for all non-terminal steps.
    All reward signal comes from the terminal CTFReward computation.

    When step_wise=True (enabled via grpo.step_wise_trajectories config),
    returns a small shaping reward:
      - Format compliance: +0.02 for valid tool call, -0.02 for no tool call
      - Phase progression: +0.03 if this step advances to a new attack phase

    These signals are intentionally small so the terminal CTFReward
    (8-signal, typical range 0.0-2.0) remains the dominant learning signal.

    Args:
        tool_calls_so_far: All tool calls accumulated in the episode so far.
        step: Current step number (1-indexed).
        max_steps: Maximum allowed steps.
        step_tool_call_count: Number of tool calls parsed in this step
            (0 means no valid tool call was found).
        step_wise: Whether step-wise trajectory rewards are enabled.
    """
    if not step_wise:
        return 0.0

    reward = 0.0

    # Signal 1: Format compliance (+0.02 / -0.02)
    # Did the model produce a well-formed tool call this step?
    if step_tool_call_count > 0:
        reward += 0.02
    else:
        reward -= 0.02

    # Signal 2: Phase progression (+0.03 if new phase reached)
    # Check if this step's tool calls advanced to a phase not seen before.
    if len(tool_calls_so_far) >= 2 and step_tool_call_count > 0:
        # Phases seen before this step's tool calls
        prev_calls = tool_calls_so_far[:-step_tool_call_count] if step_tool_call_count > 0 else tool_calls_so_far
        prev_phases = {_get_phase(tc.get("name", "")) for tc in prev_calls} if prev_calls else set()

        # Phases from this step's tool calls
        current_calls = tool_calls_so_far[-step_tool_call_count:]
        current_phases = {_get_phase(tc.get("name", "")) for tc in current_calls}

        # Reward if any new phase was reached
        new_phases = current_phases - prev_phases
        if new_phases:
            reward += 0.03

    return reward
