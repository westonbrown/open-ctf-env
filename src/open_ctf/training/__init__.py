"""Training modules for SFT and GRPO stages.

Imports are lazy to avoid requiring all dependencies (e.g. unsloth)
when only one training stage is used.
"""

__all__ = ["train_sft", "train_grpo"]


def __getattr__(name):
    if name == "train_sft":
        from .sft import train_sft
        return train_sft
    if name == "train_grpo":
        from .grpo import train_grpo
        return train_grpo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
