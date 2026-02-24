"""CTFReward adapter for SkyRL's per-step reward protocol.

SkyRL expects environments to return a reward at each step (float).
Our CTFReward is designed for batch scoring at episode end. This adapter
bridges the two:

  - Non-terminal steps: returns 0.0 (binary terminal reward approach).
    All reward signal comes from the terminal CTFReward computation.
    This matches OpenThoughts-Agent methodology: intermediate rewards
    dilute the RLOO-N advantage signal because the estimator sums all
    per-token rewards into a scalar score for normalization.
  - Terminal step: full CTFReward 8-signal score.

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

    kwargs["use_gdpo"] = bool(reward_cfg.get("use_gdpo", True))

    return CTFReward(**kwargs)


def per_step_reward(
    tool_calls_so_far: List[Dict[str, str]],
    step: int,
    max_steps: int,
) -> float:
    """Per-step reward during GRPO rollouts.

    Returns 0.0 for all non-terminal steps.  All reward signal comes from
    the terminal CTFReward computation, matching OpenThoughts-Agent's
    binary terminal reward approach.

    Why not intermediate rewards?  SkyRL's RLOO-N advantage estimator sums
    per-token rewards into a scalar score per trajectory, then normalizes
    within the prompt group.  Non-zero intermediate rewards accumulate
    across steps and dilute the gap between successful (flag found) and
    failed trajectories, making it harder for the estimator to assign
    credit correctly.
    """
    return 0.0
