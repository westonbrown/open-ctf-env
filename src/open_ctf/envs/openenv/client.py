"""HTTP client for the agent environment.

Talks to a running AgentEnvironment server via POST /reset, POST /step,
GET /state endpoints. Mirrors the OpenEnv HTTPEnvClient pattern but is
fully self-contained.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from .base import StepResult
from .models import ToolAction, ToolObservation, ToolState


class AgentEnv:
    """HTTP client that wraps an AgentEnvironment server.

    Usage::

        env = AgentEnv("http://localhost:8100")
        result = env.reset()
        print(result.observation.stdout)

        result = env.step(ToolAction(tool_name="shell_command",
                                     arguments={"command": "id"}))
        print(result.observation.stdout, result.reward, result.done)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8100",
        request_timeout_s: float = 60.0,
        default_headers: Optional[Dict[str, str]] = None,
    ):
        self._base = base_url.rstrip("/")
        self._timeout = float(request_timeout_s)
        self._http = requests.Session()
        if default_headers:
            self._http.headers.update(default_headers)

    # -- Public API --------------------------------------------------------

    def reset(self) -> StepResult[ToolObservation]:
        r = self._http.post(
            f"{self._base}/reset",
            json={},
            timeout=self._timeout,
        )
        r.raise_for_status()
        return self._parse_result(r.json())

    def step(self, action: ToolAction) -> StepResult[ToolObservation]:
        body: Dict[str, Any] = {
            "action": self._step_payload(action),
            "timeout_s": int(self._timeout),
        }
        r = self._http.post(
            f"{self._base}/step",
            json=body,
            timeout=self._timeout,
        )
        r.raise_for_status()
        return self._parse_result(r.json())

    def state(self) -> ToolState:
        r = self._http.get(
            f"{self._base}/state",
            timeout=self._timeout,
        )
        r.raise_for_status()
        return self._parse_state(r.json())

    def close(self) -> None:
        self._http.close()

    # -- Serialization helpers ---------------------------------------------

    @staticmethod
    def _step_payload(action: ToolAction) -> dict:
        return {
            "tool_name": action.tool_name,
            "arguments": action.arguments,
        }

    @staticmethod
    def _parse_result(payload: dict) -> StepResult[ToolObservation]:
        obs = payload.get("observation", {})
        return StepResult(
            observation=ToolObservation(
                stdout=obs.get("stdout", ""),
                stderr=obs.get("stderr", ""),
                exit_code=obs.get("exit_code", 0),
                tool_name=obs.get("tool_name", ""),
                step_number=obs.get("step_number", 0),
                flag_submitted=obs.get("flag_submitted", False),
                flag_correct=obs.get("flag_correct", False),
            ),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    @staticmethod
    def _parse_state(payload: dict) -> ToolState:
        return ToolState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            target=payload.get("target", ""),
            tools_used=payload.get("tools_used", 0),
            tools_available=payload.get("tools_available", []),
            flag_submitted=payload.get("flag_submitted", False),
            flag_correct=payload.get("flag_correct", False),
            ground_truth=payload.get("ground_truth", ""),
            max_steps=payload.get("max_steps", 30),
        )
