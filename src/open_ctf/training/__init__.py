"""Training modules for SFT and GRPO stages.

Imports are lazy to avoid requiring all dependencies (e.g. unsloth)
when only one training stage is used.
"""

import logging

__all__ = ["train_sft", "train_grpo", "check_wandb_available"]

logger = logging.getLogger(__name__)


def check_wandb_available(report_to: str) -> str:
    """Return *report_to* unchanged if wandb is usable, else ``"none"``.

    Checks both that the ``wandb`` package can be imported **and** that an
    API key is configured. Used by both SFT and GRPO training loops.
    """
    if report_to == "none":
        return "none"
    try:
        import wandb
        if not wandb.api.api_key:
            logger.info("wandb installed but no API key configured, disabling")
            return "none"
    except (ImportError, AttributeError):
        return "none"
    return report_to


def __getattr__(name):
    if name == "train_sft":
        from .sft import train_sft
        return train_sft
    if name == "train_grpo":
        from .grpo import train_grpo
        return train_grpo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
