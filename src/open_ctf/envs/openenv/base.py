"""Minimal base classes for the agent environment.

Adapted from OpenEnv core patterns (Meta Platforms). These are standalone
copies so the open-ctf-env project does NOT depend on the full OpenEnv package.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, Optional, TypeVar, Union


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass(kw_only=True)
class Action:
    """Base class for all environment actions."""

    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class Observation:
    """Base class for all environment observations."""

    done: bool = False
    reward: Union[int, float, None] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class State:
    """Base class for environment state."""

    episode_id: Optional[str] = None
    step_count: int = 0


# ---------------------------------------------------------------------------
# StepResult (client-side wrapper)
# ---------------------------------------------------------------------------

ObsT = TypeVar("ObsT")


@dataclass
class StepResult(Generic[ObsT]):
    """Result of one environment step, returned by the HTTP client."""

    observation: ObsT
    reward: Optional[float] = None
    done: bool = False


# ---------------------------------------------------------------------------
# Environment ABC
# ---------------------------------------------------------------------------

class Environment(ABC):
    """Base class for all environment servers (Gym-style API)."""

    @abstractmethod
    def reset(self) -> Observation:
        """Reset the environment and return the initial observation."""

    @abstractmethod
    def step(self, action: Action) -> Observation:
        """Take a step in the environment."""

    @property
    @abstractmethod
    def state(self) -> State:
        """Get the current environment state."""
