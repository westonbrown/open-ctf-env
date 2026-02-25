"""CTF agent interface and implementations."""

from .protocol import AgentResult, CTFAgent, StepAgent, StepResult
from .default_agent import DefaultStepAgent
from .runner import AgentRunner
from .boxpwnr_adapter import BoxPwnrAgent

__all__ = [
    "AgentResult",
    "AgentRunner",
    "BoxPwnrAgent",
    "CTFAgent",
    "DefaultStepAgent",
    "StepAgent",
    "StepResult",
]
