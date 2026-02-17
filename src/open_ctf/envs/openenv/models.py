"""Generic agent-environment action/observation/state types.

These are intentionally NOT CTF-specific. The naming uses "Tool" because the
environment exposes configurable tools (shell_command, python_code, submit_flag,
etc.) that an agent can invoke.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import Action, Observation, State


@dataclass(kw_only=True)
class ToolAction(Action):
    """A tool invocation from the agent."""

    tool_name: str  # e.g. "shell_command", "python_code", "submit_flag"
    arguments: Dict[str, Any] = field(default_factory=dict)


@dataclass(kw_only=True)
class ToolObservation(Observation):
    """Result of executing a tool call."""

    stdout: str = ""
    stderr: str = ""
    exit_code: int = 0
    tool_name: str = ""
    step_number: int = 0
    flag_submitted: bool = False
    flag_correct: bool = False


@dataclass
class ToolState(State):
    """Tracks episode progress."""

    target: str = ""
    tools_used: int = 0
    tools_available: List[str] = field(default_factory=list)
    flag_submitted: bool = False
    flag_correct: bool = False
    ground_truth: str = ""
    max_steps: int = 30
