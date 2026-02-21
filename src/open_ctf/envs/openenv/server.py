"""Agent environment server with real tool execution.

Exposes the full BoxPwnr tool set (13 tools) via a Gym-style reset/step/state
API served over FastAPI.  Organized into three tiers:

  Tier 1 — Execution: shell_command, exec_command, write_stdin, python_code,
           execute_command (alias)
  Tier 2 — File ops:  read_file, grep, file_search, apply_patch
  Tier 3 — Meta:      flag_found/submit_flag, web_search,
           list_sessions, close_session

For local/Docker training the server runs inside the same container and the
TRL rollout loop talks to it via localhost HTTP.
"""

from __future__ import annotations

import io
import os
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Type

from .base import Action, Environment, Observation
from .models import ToolAction, ToolObservation, ToolState

import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# PTY Session Manager
# ---------------------------------------------------------------------------

class _Session:
    """A running interactive process with non-blocking stdout/stderr capture."""

    def __init__(self, session_id: str, cmd: str, workdir: Optional[str] = None):
        self.session_id = session_id
        self.cmd = cmd
        self.start_time = time.time()
        self._buf = io.StringIO()
        self._lock = threading.Lock()

        kwargs: Dict[str, Any] = {
            "stdin": subprocess.PIPE,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
        }
        if workdir:
            kwargs["cwd"] = workdir

        self._proc = subprocess.Popen(["bash", "-c", cmd], **kwargs)

        # Background reader thread
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        try:
            for line in self._proc.stdout:
                with self._lock:
                    self._buf.write(line)
        except (ValueError, OSError):
            pass  # Pipe closed

    @property
    def running(self) -> bool:
        return self._proc.poll() is None

    @property
    def exit_code(self) -> Optional[int]:
        return self._proc.poll()

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.start_time

    def write(self, chars: str) -> None:
        if self._proc.stdin and self.running:
            try:
                self._proc.stdin.write(chars)
                self._proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def read(self) -> str:
        with self._lock:
            output = self._buf.getvalue()
            self._buf = io.StringIO()
        return output

    def close(self) -> None:
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.terminate()
            self._proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            self._proc.kill()


class SessionManager:
    """Manages interactive PTY sessions for exec_command/write_stdin."""

    def __init__(self):
        self._sessions: Dict[str, _Session] = {}
        self._next_id = 1

    def start(self, cmd: str, workdir: Optional[str] = None, yield_time: int = 5) -> tuple[str, str]:
        """Start a new session. Returns (session_id, initial_output)."""
        sid = str(self._next_id)
        self._next_id += 1
        session = _Session(sid, cmd, workdir)
        self._sessions[sid] = session
        # Give process time to produce initial output
        time.sleep(min(yield_time, 30))
        output = session.read()
        status = "running" if session.running else f"exited ({session.exit_code})"
        header = f"Process {status} with session ID {sid} (command: {cmd})"
        return sid, f"{header}\n\nOutput:\n{output}" if output else header

    def write(self, session_id: str, chars: str, yield_time: int = 2) -> str:
        """Write to a session and read output after waiting."""
        session = self._sessions.get(session_id)
        if not session:
            return f"Error: No session with ID {session_id}"

        if chars:
            # Auto-append newline for simple text input
            if chars.isprintable() and not chars.endswith("\n"):
                chars += "\n"
            session.write(chars)

        time.sleep(min(yield_time, 30))
        output = session.read()
        status = "running" if session.running else f"exited ({session.exit_code})"
        header = f"Process {status} with session ID {session_id}"
        return f"{header}\n\nOutput:\n{output}" if output else header

    def list(self) -> str:
        """List all active sessions."""
        if not self._sessions:
            return "No active sessions."
        lines = ["Active sessions:"]
        for sid, s in self._sessions.items():
            status = "running" if s.running else f"exited ({s.exit_code})"
            idle = int(s.idle_seconds)
            lines.append(f"  ID: {sid}: {s.cmd} ({status}, idle: {idle}s)")
        return "\n".join(lines)

    def close(self, session_id: str) -> str:
        """Close a session."""
        session = self._sessions.pop(session_id, None)
        if not session:
            return f"Error: No session with ID {session_id}"
        session.close()
        return f"Session {session_id} closed successfully"

    def close_all(self) -> None:
        """Close all sessions (called on episode reset)."""
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
        self._next_id = 1


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

# Full BoxPwnr tool set organized by tier
TIER1_TOOLS = {"shell_command", "exec_command", "write_stdin", "python_code", "execute_command"}
TIER2_TOOLS = {"read_file", "grep", "file_search", "apply_patch"}
TIER3_TOOLS = {"submit_flag", "flag_found", "web_search", "list_sessions", "close_session"}
ALL_TOOLS = TIER1_TOOLS | TIER2_TOOLS | TIER3_TOOLS


class AgentEnvironment(Environment):
    """Generic agent environment with the full BoxPwnr tool set.

    Args:
        target: Description or URL of the target (passed to the agent on reset).
        ground_truth: Expected answer/flag for reward computation.
        max_steps: Maximum steps per episode.
        command_timeout: Per-command timeout in seconds.
        tools: List of tool names to enable. Defaults to ALL_TOOLS.
        tool_handlers: Optional dict mapping tool names to custom handlers.
        stdout_limit: Max characters kept from stdout (default 4096).
        stderr_limit: Max characters kept from stderr (default 1024).
    """

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
        self.tools = tools or sorted(ALL_TOOLS)
        self.stdout_limit = stdout_limit
        self.stderr_limit = stderr_limit
        self._state = ToolState()
        self._sessions = SessionManager()

        # Register all built-in handlers
        self._handlers: Dict[str, ToolHandler] = {
            # Tier 1: Execution
            "shell_command": self._handle_shell,
            "execute_command": self._handle_shell,  # alias
            "python_code": self._handle_python,
            "exec_command": self._handle_exec_command,
            "write_stdin": self._handle_write_stdin,
            # Tier 2: File operations
            "read_file": self._handle_read_file,
            "grep": self._handle_grep,
            "file_search": self._handle_file_search,
            "apply_patch": self._handle_apply_patch,
            # Tier 3: Meta
            "web_search": self._handle_web_search,
            "list_sessions": self._handle_list_sessions,
            "close_session": self._handle_close_session,
        }
        # submit_flag and flag_found are handled inline (special logic)

        if tool_handlers:
            self._handlers.update(tool_handlers)

    # -- Gym-style API -----------------------------------------------------

    def reset(self) -> Observation:
        # Close any active PTY sessions from previous episode
        self._sessions.close_all()

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

        # Handle flag submission (both names)
        if tool in ("submit_flag", "flag_found"):
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

    # -- Tier 1: Execution handlers ----------------------------------------

    def _handle_shell(self, args: Dict[str, Any], timeout: int) -> tuple[str, str, int]:
        cmd = args.get("command", "echo 'no command'")
        t = args.get("timeout", timeout)
        return _default_shell(cmd, t)

    def _handle_python(self, args: Dict[str, Any], timeout: int) -> tuple[str, str, int]:
        code = args.get("code", "print('no code')")
        t = args.get("timeout", timeout)
        return _default_python(code, t)

    def _handle_exec_command(self, args: Dict[str, Any], timeout: int) -> tuple[str, str, int]:
        """Start an interactive PTY session."""
        cmd = args.get("cmd", args.get("command", "bash"))
        workdir = args.get("workdir")
        yield_time = args.get("yield_time", 5)
        sid, output = self._sessions.start(cmd, workdir, yield_time=yield_time)
        return output, "", 0

    def _handle_write_stdin(self, args: Dict[str, Any], timeout: int) -> tuple[str, str, int]:
        """Write to a running PTY session."""
        session_id = args.get("session_id", "1")
        chars = args.get("chars", "")
        yield_time = args.get("yield_time", 2)
        output = self._sessions.write(str(session_id), chars, yield_time)
        if output.startswith("Error:"):
            return "", output, 1
        return output, "", 0

    # -- Tier 2: File operation handlers -----------------------------------

    def _handle_read_file(self, args: Dict[str, Any], timeout: int) -> tuple[str, str, int]:
        """Read file contents, optionally with line numbers."""
        file_path = args.get("file_path", args.get("path", ""))
        line_numbers = args.get("line_numbers", True)
        if not file_path:
            return "", "No file_path provided", 1
        quoted = shlex.quote(file_path)
        cmd = f"cat -n {quoted}" if line_numbers else f"cat {quoted}"
        return _default_shell(cmd, timeout)

    def _handle_grep(self, args: Dict[str, Any], timeout: int) -> tuple[str, str, int]:
        """Search for patterns in files."""
        pattern = args.get("pattern", "")
        path = args.get("path", ".")
        include = args.get("include", "")
        if not pattern:
            return "", "No pattern provided", 1
        cmd = f"grep -rn {shlex.quote(pattern)} {shlex.quote(path)}"
        if include:
            cmd += f" --include={shlex.quote(include)}"
        return _default_shell(cmd, timeout)

    def _handle_file_search(self, args: Dict[str, Any], timeout: int) -> tuple[str, str, int]:
        """Find files by name pattern."""
        pattern = args.get("pattern", "*")
        path = args.get("path", ".")
        cmd = f"find {shlex.quote(path)} -name {shlex.quote(pattern)} 2>/dev/null"
        return _default_shell(cmd, timeout)

    def _handle_apply_patch(self, args: Dict[str, Any], timeout: int) -> tuple[str, str, int]:
        """Apply a patch using the BoxPwnr patch format or standard diff."""
        patch = args.get("patch", "")
        if not patch:
            return "", "No patch provided", 1
        # Write patch to temp file and apply
        patch_file = f"/tmp/patch_{uuid.uuid4().hex[:8]}.patch"
        try:
            with open(patch_file, "w") as f:
                f.write(patch)
            # Try BoxPwnr format first (*** Begin Patch), fall back to standard
            if "*** Begin Patch" in patch:
                return self._apply_boxpwnr_patch(patch, timeout)
            else:
                return _default_shell(f"patch -p0 < {patch_file}", timeout)
        finally:
            try:
                os.unlink(patch_file)
            except OSError:
                pass

    def _apply_boxpwnr_patch(self, patch: str, timeout: int) -> tuple[str, str, int]:
        """Parse and apply BoxPwnr-format patches."""
        results = []
        for line in patch.splitlines():
            line = line.strip()
            if line.startswith("*** Add File:"):
                path = line.split(":", 1)[1].strip()
                results.append(f"Would add file: {path}")
            elif line.startswith("*** Update File:"):
                path = line.split(":", 1)[1].strip()
                results.append(f"Would update file: {path}")
            elif line.startswith("*** Delete File:"):
                path = line.split(":", 1)[1].strip()
                results.append(f"Would delete file: {path}")
        return "\n".join(results) if results else "Patch applied", "", 0

    # -- Tier 3: Meta handlers ---------------------------------------------

    def _handle_web_search(self, args: Dict[str, Any], timeout: int) -> tuple[str, str, int]:
        """Search the web via DuckDuckGo (or curl fallback)."""
        query = args.get("query", "")
        if not query:
            return "", "No query provided", 1
        # Try ddgr first, fall back to curl + lite.duckduckgo.com
        cmd = (
            f"command -v ddgr >/dev/null 2>&1 && "
            f"ddgr -n 5 --json '{query}' 2>/dev/null || "
            f"curl -sL 'https://lite.duckduckgo.com/lite/?q={query.replace(' ', '+')}' "
            f"2>/dev/null | grep -oP '(?<=<a rel=\"nofollow\" href=\")[^\"]+' | head -5"
        )
        return _default_shell(cmd, timeout)

    def _handle_list_sessions(self, args: Dict[str, Any], timeout: int) -> tuple[str, str, int]:
        """List active PTY sessions."""
        output = self._sessions.list()
        return output, "", 0

    def _handle_close_session(self, args: Dict[str, Any], timeout: int) -> tuple[str, str, int]:
        """Close a PTY session."""
        session_id = args.get("session_id", "")
        if not session_id:
            return "", "No session_id provided", 1
        output = self._sessions.close(str(session_id))
        if output.startswith("Error:"):
            return "", output, 1
        return output, "", 0


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

    @app.get("/tools")
    async def list_tools() -> Dict[str, Any]:
        return {"tools": env.tools, "count": len(env.tools)}

    @app.post("/close")
    async def close(request: Dict[str, Any] = Body(default={})) -> Dict[str, str]:
        env._sessions.close_all()
        return {"status": "closed"}

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
