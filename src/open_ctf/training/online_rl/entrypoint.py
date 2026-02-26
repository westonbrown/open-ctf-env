"""Online RL training entrypoints.

This module is the semantic entrypoint for stage-2 training. The underlying
algorithm can be GRPO/RLOO/etc, but the pipeline component is "online RL".

Backward compatibility:
- ``train_grpo`` remains available and forwards to the same implementation.
- Config can use either ``online_rl`` (preferred) or ``grpo`` (legacy).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .runtime import train_grpo as _train_grpo_impl


def train_online_rl(
    model_path: str,
    data_path: str,
    output_dir: str,
    config: Dict[str, Any],
    resume_from: Optional[str] = None,
    challenge_registry: Optional[str] = None,
    agent_class: Optional[str] = None,
) -> str:
    """Run stage-2 online RL training (SkyRL backend)."""
    return _train_grpo_impl(
        model_path=model_path,
        data_path=data_path,
        output_dir=output_dir,
        config=config,
        resume_from=resume_from,
        challenge_registry=challenge_registry,
        agent_class=agent_class,
    )


def train_grpo(
    model_path: str,
    data_path: str,
    output_dir: str,
    config: Dict[str, Any],
    resume_from: Optional[str] = None,
    challenge_registry: Optional[str] = None,
    agent_class: Optional[str] = None,
) -> str:
    """Legacy alias for callers still using GRPO naming."""
    return train_online_rl(
        model_path=model_path,
        data_path=data_path,
        output_dir=output_dir,
        config=config,
        resume_from=resume_from,
        challenge_registry=challenge_registry,
        agent_class=agent_class,
    )


__all__ = ["train_online_rl", "train_grpo"]
