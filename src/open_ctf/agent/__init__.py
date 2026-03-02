"""CTF agent interface and implementations."""

from .boxpwnr_adapter import BoxPwnrAgent
from .default_agent import DefaultStepAgent
from .protocol import AgentResult, CTFAgent, StepAgent, StepResult, validate_step_agent
from .rollout_status import RolloutStatus, normalize_rollout_status
from .runner import AgentRunner

__all__ = [
    "AgentResult",
    "AgentRunner",
    "BoxPwnrAgent",
    "CTFAgent",
    "DefaultStepAgent",
    "RolloutStatus",
    "StepAgent",
    "StepResult",
    "normalize_rollout_status",
    "validate_step_agent",
]
