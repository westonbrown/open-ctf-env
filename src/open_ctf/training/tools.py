"""TRL tool wrappers for OpenEnv CTF environment interaction.

These functions are designed to be passed to TRL's GRPOTrainer via the
``tools=`` parameter.  TRL auto-generates JSON tool schemas from the
Google-style docstrings and type annotations.

Full BoxPwnr tool set (13 tools, organized by tier):

  Tier 1 — Execution:  shell_command, exec_command, write_stdin,
                        python_code, execute_command
  Tier 2 — File ops:   read_file, grep, file_search, apply_patch
  Tier 3 — Meta:       flag_found, submit_flag, web_search,
                        list_sessions, close_session

Usage::

    from open_ctf.training.tools import get_all_tools, init_env

    init_env("http://localhost:8000")
    tools = get_all_tools()  # returns list of callables for TRL

    trainer = GRPOTrainer(
        model=model,
        tools=tools,
        reward_funcs=[reward_fn],
        args=GRPOConfig(max_tool_calling_iterations=15),
    )

The environment client is a module-level singleton initialized by
``init_env(base_url)``.  Each training step should call ``reset_env()``
to start a fresh episode.
"""

import logging
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level environment client
# ---------------------------------------------------------------------------

_base_url: Optional[str] = None
_session: Optional[requests.Session] = None
_episode_id: Optional[str] = None


def init_env(base_url: str) -> None:
    """Initialize the environment client.

    Must be called before any tool function is used. Creates an HTTP
    session for connection pooling.

    Args:
        base_url: Base URL of the OpenEnv HTTP server
            (e.g. ``http://localhost:8000``).
    """
    global _base_url, _session
    _base_url = base_url.rstrip("/")
    _session = requests.Session()
    _session.headers["Content-Type"] = "application/json"
    logger.info("OpenEnv client initialized: %s", _base_url)


def reset_env(challenge_id: Optional[str] = None) -> str:
    """Reset the environment for a new episode.

    Args:
        challenge_id: Optional challenge identifier. If provided,
            the server will load this specific challenge.

    Returns:
        The initial observation text from the environment.
    """
    global _episode_id
    _ensure_initialized()
    payload = {}
    if challenge_id:
        payload["challenge_id"] = challenge_id
    resp = _post("/reset", payload)
    _episode_id = resp.get("episode_id")
    return resp.get("observation", {}).get("stdout", "Environment reset.")


def close_env() -> None:
    """Close the environment client and release resources."""
    global _base_url, _session, _episode_id
    if _session is not None:
        try:
            _post("/close", {})
        except Exception:
            pass
        _session.close()
    _base_url = None
    _session = None
    _episode_id = None
    logger.info("OpenEnv client closed")


def get_last_step_info() -> dict:
    """Return metadata from the last environment step."""
    _ensure_initialized()
    try:
        return _get("/status")
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Tool collections (pass to GRPOTrainer tools=)
# ---------------------------------------------------------------------------


def get_all_tools() -> list:
    """Return all tool functions for TRL's tools= parameter.

    Returns the full BoxPwnr-compatible tool set. Each function has
    proper type annotations and docstrings for TRL schema generation.
    """
    return [
        # Tier 1: Execution
        shell_command,
        exec_command,
        write_stdin,
        python_code,
        execute_command,
        # Tier 2: File operations
        read_file,
        grep,
        file_search,
        apply_patch,
        # Tier 3: Meta
        flag_found,
        web_search,
        list_sessions,
        close_session,
    ]


def get_core_tools() -> list:
    """Return only the core 3 tools (backward compatible).

    Suitable for simple challenges that only need shell + python + flag.
    """
    return [shell_command, python_code, flag_found]


# ---------------------------------------------------------------------------
# Tier 1: Execution tools
# ---------------------------------------------------------------------------


def shell_command(command: str, timeout: int = 30) -> str:
    """Run a shell command in the CTF attacker container and return output.

    Use this to execute reconnaissance, enumeration, and exploitation
    commands against the target challenge. Supports pipes, redirects,
    and multi-line scripts.

    Args:
        command: The shell command to execute (e.g. ``nmap -sV target``).
        timeout: Maximum execution time in seconds. Defaults to 30.

    Returns:
        The combined stdout and stderr output from the command.
    """
    return _step("shell_command", {"command": command, "timeout": timeout})


def exec_command(cmd: str, workdir: str = "", yield_time: int = 5) -> str:
    """Start an interactive process in a PTY session and return its output.

    Returns a session ID for ongoing interaction via write_stdin. Use this
    for interactive programs like bash, python3, ssh, gdb, or netcat.
    For non-interactive commands, prefer shell_command instead.

    Args:
        cmd: Shell command to execute (e.g. ``python3``, ``ssh user@host``).
        workdir: Optional working directory to run the command in.
        yield_time: Seconds to wait for initial output before returning.
            Defaults to 5.

    Returns:
        Session ID and initial output from the process.
    """
    args = {"cmd": cmd, "yield_time": yield_time}
    if workdir:
        args["workdir"] = workdir
    return _step("exec_command", args)


def write_stdin(session_id: str, chars: str = "", yield_time: int = 2) -> str:
    """Send input to a running PTY session and return new output.

    Use this to interact with processes started via exec_command.
    Pass empty chars to poll for new output without sending input.

    Args:
        session_id: Numeric ID of the session (e.g. ``1``, ``2``).
        chars: Text to write to stdin. May be empty to just poll output.
        yield_time: Seconds to wait for output after writing. Defaults to 2.

    Returns:
        Process status and any new output from the session.
    """
    return _step("write_stdin", {
        "session_id": session_id, "chars": chars, "yield_time": yield_time,
    })


def python_code(code: str, timeout: int = 120) -> str:
    """Execute Python code in the CTF attacker container.

    Use this for complex exploits, payload generation, encoding/decoding,
    crypto operations, or data processing. Use print() to see output.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds. Defaults to 120.

    Returns:
        The output from executing the Python code.
    """
    return _step("python_code", {"code": code, "timeout": timeout})


def execute_command(command: str, timeout: int = 30) -> str:
    """Execute a non-interactive command and return complete output.

    Similar to shell_command. Use this for commands that produce output
    and exit. For interactive programs, use exec_command instead.

    Args:
        command: Shell command to execute.
        timeout: Maximum execution time in seconds. Defaults to 30.

    Returns:
        The combined stdout and stderr output from the command.
    """
    return _step("execute_command", {"command": command, "timeout": timeout})


# ---------------------------------------------------------------------------
# Tier 2: File operation tools
# ---------------------------------------------------------------------------


def read_file(file_path: str, line_numbers: bool = True) -> str:
    """Read the contents of a file in the CTF environment.

    Returns file content with optional line numbers for reference.

    Args:
        file_path: Path to the file to read.
        line_numbers: Whether to include line numbers. Defaults to True.

    Returns:
        The file contents, optionally with line numbers.
    """
    return _step("read_file", {"file_path": file_path, "line_numbers": line_numbers})


def grep(pattern: str, path: str = ".", include: str = "") -> str:
    """Search for a regex pattern in files.

    Returns matching lines with file paths and line numbers.

    Args:
        pattern: Regex pattern to search for.
        path: Directory to search in. Defaults to current directory.
        include: File pattern to include (e.g. ``*.py``). Optional.

    Returns:
        Grep results with line numbers and context.
    """
    args = {"pattern": pattern, "path": path}
    if include:
        args["include"] = include
    return _step("grep", args)


def file_search(pattern: str, path: str = ".") -> str:
    """Find files by name pattern using the find command.

    Args:
        pattern: Glob pattern to match file names (e.g. ``*.py``, ``flag*``).
        path: Directory to search in. Defaults to current directory.

    Returns:
        List of matching file paths.
    """
    return _step("file_search", {"pattern": pattern, "path": path})


def apply_patch(patch: str) -> str:
    """Apply a patch to modify files in the CTF environment.

    Supports BoxPwnr patch format (*** Begin Patch) and standard diff.

    Args:
        patch: The patch content to apply.

    Returns:
        Result of applying the patch.
    """
    return _step("apply_patch", {"patch": patch})


# ---------------------------------------------------------------------------
# Tier 3: Meta tools
# ---------------------------------------------------------------------------


def flag_found(content: str) -> str:
    """Submit a discovered flag for verification.

    Call this when you have found the flag. The environment will verify
    correctness and end the episode if the flag is correct.

    Args:
        content: The flag string you discovered (e.g. ``FLAG{example}``).

    Returns:
        Verification result indicating whether the flag is correct.
    """
    return _step("flag_found", {"content": content})


def web_search(query: str) -> str:
    """Search the web for information using DuckDuckGo.

    Use this to find documentation, CVE details, exploit code, or other
    public information relevant to the challenge.

    Args:
        query: The search query string.

    Returns:
        Search results with titles, snippets, and URLs.
    """
    return _step("web_search", {"query": query})


def list_sessions() -> str:
    """List all active interactive shell sessions.

    Shows session IDs, commands, and current status. Use this to check
    which PTY sessions are still running.

    Returns:
        List of active sessions with their status.
    """
    return _step("list_sessions", {})


def close_session(session_id: str) -> str:
    """Close an interactive shell session and free resources.

    Use this when done with a session started via exec_command.

    Args:
        session_id: Numeric ID of the session to close (e.g. ``1``).

    Returns:
        Confirmation that the session was closed.
    """
    return _step("close_session", {"session_id": session_id})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _step(tool_name: str, arguments: dict) -> str:
    """Send a tool call to the OpenEnv server and return the output string."""
    _ensure_initialized()
    resp = _post("/step", {"action": {
        "tool_name": tool_name,
        "arguments": arguments,
    }})
    obs = resp.get("observation", {})
    stdout = obs.get("stdout", "")
    stderr = obs.get("stderr", "")
    if stderr:
        return f"{stdout}\n[stderr] {stderr}"
    return stdout


# Keep submit_flag as an alias for backward compatibility
def submit_flag(flag: str) -> str:
    """Submit a captured flag for verification. Alias for flag_found.

    Args:
        flag: The flag string to submit.

    Returns:
        Verification result.
    """
    return flag_found(flag)


def _ensure_initialized() -> None:
    """Raise if init_env() has not been called."""
    if _base_url is None or _session is None:
        raise RuntimeError(
            "OpenEnv client not initialized. Call init_env(base_url) first."
        )


def _post(path: str, payload: dict) -> dict:
    """Send a POST request to the environment server."""
    url = f"{_base_url}{path}"
    try:
        resp = _session.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError as e:
        logger.error("Cannot connect to OpenEnv server at %s: %s", url, e)
        return {"observation": {"stdout": f"[ERROR] Connection failed: {e}", "stderr": "", "exit_code": 1}}
    except requests.exceptions.Timeout as e:
        logger.error("OpenEnv request timed out: %s", e)
        return {"observation": {"stdout": "[ERROR] Request timed out", "stderr": "", "exit_code": 1}}
    except Exception as e:
        logger.error("OpenEnv request failed: %s", e)
        return {"observation": {"stdout": f"[ERROR] {e}", "stderr": "", "exit_code": 1}}


def _get(path: str) -> dict:
    """Send a GET request to the environment server."""
    url = f"{_base_url}{path}"
    try:
        resp = _session.get(url, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error("OpenEnv GET request failed: %s", e)
        return {}
