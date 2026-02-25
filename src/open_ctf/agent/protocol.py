"""CTFAgent protocol — minimal interface for pluggable CTF agents.

SkyRL owns generation during GRPO training. This protocol is for:
  - Evaluation (open-ctf-eval)
  - GEPA trace collection
  - Standalone agent runs (open-ctf-agent)

Any class implementing solve() satisfies CTFAgent via structural subtyping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# StepAgent — pluggable tool-execution agent for GRPO training loop
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """Result of a single agent step (tool parsing + execution).

    The env owns reward computation (SkyRL contract). The agent returns
    observations and done status only.
    """

    observations: List[Dict[str, str]]  # [{role: "user", content: "[Tool: name]\noutput"}]
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class StepAgent(Protocol):
    """Pluggable agent for the GRPO training loop.

    During GRPO training, SkyRL owns generation (vLLM). The StepAgent
    owns tool parsing + execution. This lets users swap in custom tool
    handlers, different parsing logic, or entirely different execution
    backends without touching the env or reward code.

    Example::

        class MyAgent:
            def reset(self, target="", ground_truth_flag="", max_steps=30, **kw):
                self.target = target

            def step(self, action: str) -> StepResult:
                # Parse tool calls YOUR way
                # Execute tools YOUR way
                return StepResult(observations=[...], done=False)

            def close(self):
                pass

            @property
            def tools(self):
                # Return None to use defaults, or provide your own:
                return [{"type": "function", "function": {"name": "my_tool", ...}}]

        assert isinstance(MyAgent(), StepAgent)
    """

    def reset(
        self,
        target: str = "",
        ground_truth_flag: str = "",
        max_steps: int = 30,
        **kwargs: Any,
    ) -> None:
        """Reset agent state for a new episode."""
        ...

    def step(self, action: str) -> StepResult:
        """Parse tool calls from LLM output and execute them.

        Args:
            action: Raw LLM text output (may contain tool calls).

        Returns:
            StepResult with observations and done flag.
        """
        ...

    def close(self) -> None:
        """Release resources."""
        ...

    @property
    def tools(self) -> Optional[List[Dict[str, Any]]]:
        """Tool schemas for prompt injection (OpenAI function format).

        Return None to use the environment's default tool schemas.
        Return a list of tool dicts to override with your own tools.

        Each dict should follow OpenAI function calling format::

            {"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}
        """
        ...


# ---------------------------------------------------------------------------
# CTFAgent — full agent protocol for eval/GEPA (owns generation too)
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """Result of an agent solving a CTF challenge."""
    success: bool
    flag: Optional[str] = None
    steps: int = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class CTFAgent(Protocol):
    """Minimal protocol for pluggable CTF agents.

    Any class with a matching ``solve()`` signature satisfies this protocol.
    No base class inheritance required.

    Example::

        class MyAgent:
            def solve(self, challenge, target, ground_truth_flag="",
                      max_steps=30, timeout=300) -> AgentResult:
                # ... your logic ...
                return AgentResult(success=True, flag="FLAG{...}")

        assert isinstance(MyAgent(), CTFAgent)
    """

    def solve(
        self,
        challenge: str,
        target: str,
        ground_truth_flag: str = "",
        max_steps: int = 30,
        timeout: int = 300,
    ) -> AgentResult:
        """Attempt to solve a CTF challenge.

        Args:
            challenge: Challenge identifier (e.g. "eval-me", "XBEN-003-24").
            target: Target URL or file path for the challenge.
            ground_truth_flag: Expected flag for validation (empty = unknown).
            max_steps: Maximum tool-use steps before giving up.
            timeout: Maximum wall-clock seconds.

        Returns:
            AgentResult with success status, captured flag, and metadata.
        """
        ...
