"""Agent environment server with real tool execution.

Exposes configurable tools (shell_command, python_code, submit_flag) via
a Gym-style reset/step/state API served over FastAPI.

For local/Docker training the server runs inside the same container and the
TRL rollout loop talks to it via localhost HTTP.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Type

from .base import Action, Environment, Observation
from .models import ToolAction, ToolObservation, ToolState


# ---------------------------------------------------------------------------
# Default tool handlers
# ---------------------------------------------------------------------------

def _default_shell(command: str, timeout: int) -> tuple[str, str, int]:
    """Execute a shell command via bash."""
    try:
        result = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Command timed out after {timeout}s", 124
    except Exception as e:
        return "", str(e), 1


def _default_python(code: str, timeout: int) -> tuple[str, str, int]:
    """Execute Python code."""
    try:
        result = subprocess.run(
            ["python3", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", f"Code timed out after {timeout}s", 124
    except Exception as e:
        return "", str(e), 1


# ---------------------------------------------------------------------------
# AgentEnvironment
# ---------------------------------------------------------------------------

# Type alias for a tool handler: (arguments_dict, timeout) -> (stdout, stderr, exit_code)
ToolHandler = Callable[[Dict[str, Any], int], tuple[str, str, int]]


class AgentEnvironment(Environment):
    """Generic agent environment with configurable tools.

    Args:
        target: Description or URL of the target (passed to the agent on reset).
        ground_truth: Expected answer/flag for reward computation.
        max_steps: Maximum steps per episode.
        command_timeout: Per-command timeout in seconds.
        tools: List of tool names to enable. Defaults to
               ["shell_command", "python_code", "submit_flag"].
        tool_handlers: Optional dict mapping tool names to custom handlers.
            Each handler receives ``(arguments_dict, timeout)`` and returns
            ``(stdout, stderr, exit_code)``.
        stdout_limit: Max characters kept from stdout (default 4096).
        stderr_limit: Max characters kept from stderr (default 1024).
    """

    # Built-in tool set
    BUILTIN_TOOLS = {"shell_command", "python_code", "submit_flag"}

    def __init__(
        self,
        target: Optional[str] = None,
        ground_truth: str = "",
        max_steps: int = 30,
        command_timeout: int = 30,
        tools: Optional[List[str]] = None,
        tool_handlers: Optional[Dict[str, ToolHandler]] = None,
        stdout_limit: int = 4096,
        stderr_limit: int = 1024,
    ):
        self.target = target or os.getenv("CHALLENGE_TARGET", "http://localhost:8080")
        self.ground_truth = ground_truth or os.getenv("GROUND_TRUTH", "")
        self.max_steps = max_steps
        self.command_timeout = command_timeout
        self.tools = tools or ["shell_command", "python_code", "submit_flag"]
        self.stdout_limit = stdout_limit
        self.stderr_limit = stderr_limit
        self._state = ToolState()

        # Register handlers
        self._handlers: Dict[str, ToolHandler] = {}
        self._handlers["shell_command"] = self._wrap_shell
        self._handlers["python_code"] = self._wrap_python
        # submit_flag is handled inline (special logic)

        if tool_handlers:
            self._handlers.update(tool_handlers)

    # -- Gym-style API -----------------------------------------------------

    def reset(self) -> Observation:
        self._state = ToolState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            target=self.target,
            tools_used=0,
            tools_available=list(self.tools),
            flag_submitted=False,
            flag_correct=False,
            ground_truth=self.ground_truth,
            max_steps=self.max_steps,
        )
        return ToolObservation(
            stdout=f"Environment ready. Target: {self.target}",
            stderr="",
            exit_code=0,
            tool_name="reset",
            step_number=0,
            done=False,
            reward=0.0,
        )

    def step(self, action: Action) -> Observation:
        if not isinstance(action, ToolAction):
            raise ValueError(f"Expected ToolAction, got {type(action)}")

        self._state.step_count += 1
        self._state.tools_used += 1

        tool = action.tool_name
        args = action.arguments

        # Validate tool is enabled
        if tool not in self.tools:
            return ToolObservation(
                stdout="",
                stderr=f"Tool '{tool}' is not available. Available: {self.tools}",
                exit_code=1,
                tool_name=tool,
                step_number=self._state.step_count,
                done=self._state.step_count >= self.max_steps,
                reward=-0.02,
            )

        # Handle flag submission separately
        if tool == "submit_flag":
            return self._handle_flag_submission(args.get("content", ""))

        # Dispatch to registered handler
        handler = self._handlers.get(tool)
        if handler is None:
            return ToolObservation(
                stdout="",
                stderr=f"No handler registered for tool '{tool}'",
                exit_code=1,
                tool_name=tool,
                step_number=self._state.step_count,
                done=self._state.step_count >= self.max_steps,
                reward=-0.02,
            )

        stdout, stderr, exit_code = handler(args, self.command_timeout)

        done = self._state.step_count >= self.max_steps
        reward = 0.05 if exit_code == 0 else -0.02

        return ToolObservation(
            stdout=stdout[: self.stdout_limit],
            stderr=stderr[: self.stderr_limit],
            exit_code=exit_code,
            tool_name=tool,
            step_number=self._state.step_count,
            done=done,
            reward=reward,
        )

    @property
    def state(self) -> ToolState:
        return self._state

    # -- Flag submission ---------------------------------------------------

    def _handle_flag_submission(self, submitted: str) -> ToolObservation:
        submitted = submitted.strip()
        correct = bool(self.ground_truth) and submitted == self.ground_truth

        self._state.flag_submitted = True
        self._state.flag_correct = correct

        if correct:
            stdout = f"Correct! Flag verified: {submitted}"
            reward = 1.0
        else:
            stdout = f"Incorrect submission: {submitted}"
            reward = -0.2

        return ToolObservation(
            stdout=stdout,
            stderr="",
            exit_code=0 if correct else 1,
            tool_name="submit_flag",
            step_number=self._state.step_count,
            flag_submitted=True,
            flag_correct=correct,
            done=True,  # Episode ends on flag submission
            reward=reward,
        )

    # -- Built-in tool wrappers --------------------------------------------

    def _wrap_shell(self, args: Dict[str, Any], timeout: int) -> tuple[str, str, int]:
        return _default_shell(args.get("command", "echo 'no command'"), timeout)

    def _wrap_python(self, args: Dict[str, Any], timeout: int) -> tuple[str, str, int]:
        return _default_python(args.get("code", "print('no code')"), timeout)


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def create_app(
    env: Optional[AgentEnvironment] = None,
    **env_kwargs: Any,
) -> Any:
    """Create a FastAPI application for the agent environment.

    Args:
        env: Pre-built AgentEnvironment, or None to create one from env_kwargs.
        **env_kwargs: Forwarded to ``AgentEnvironment(...)`` when *env* is None.

    Returns:
        FastAPI application with /reset, /step, /state, /health endpoints.
    """
    try:
        from fastapi import Body, FastAPI
    except ImportError:
        raise ImportError(
            "FastAPI is required for the server. Install with: pip install fastapi uvicorn"
        )

    if env is None:
        env = AgentEnvironment(**env_kwargs)

    app = FastAPI(title="Agent Environment Server")

    @app.post("/reset")
    async def reset(request: Dict[str, Any] = Body(default={})) -> Dict[str, Any]:
        observation = env.reset()
        return _serialize(observation)

    @app.post("/step")
    async def step(request: Dict[str, Any] = Body(...)) -> Dict[str, Any]:
        action_data = request.get("action", {})
        metadata = action_data.pop("metadata", {})
        action = ToolAction(**action_data)
        action.metadata = metadata
        observation = env.step(action)
        return _serialize(observation)

    @app.get("/state")
    async def get_state() -> Dict[str, Any]:
        return asdict(env.state)

    @app.get("/health")
    async def health() -> Dict[str, str]:
        return {"status": "healthy"}

    return app


def _serialize(observation: Observation) -> Dict[str, Any]:
    """Serialize an Observation into the wire format expected by AgentEnv client."""
    d = asdict(observation)
    reward = d.pop("reward", None)
    done = d.pop("done", False)
    d.pop("metadata", None)
    return {"observation": d, "reward": reward, "done": done}


# ---------------------------------------------------------------------------
# Module-level app instance for ``uvicorn open_ctf.envs.openenv.server:app``
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI as _FastAPI  # noqa: F811
    app = create_app()
except ImportError:
    app = None  # FastAPI not installed; server mode unavailable

# ---------------------------------------------------------------------------
# Convenience: ``python -m open_ctf.envs.openenv.server``
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    _app = app or create_app()
    uvicorn.run(_app, host="0.0.0.0", port=8100)
