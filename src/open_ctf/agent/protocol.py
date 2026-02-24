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
