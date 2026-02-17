"""TRL tool wrappers for OpenEnv CTF environment interaction.

These functions are designed to be passed to TRL's GRPOTrainer via the
``tools=`` parameter.  TRL auto-generates JSON tool schemas from the
Google-style docstrings and type annotations.

Usage::

    from open_ctf.training.tools import shell_command, python_code, submit_flag, init_env

    init_env("http://localhost:8000")

    trainer = GRPOTrainer(
        model=model,
        tools=[shell_command, python_code, submit_flag],
        reward_funcs=[env_reward_fn],
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
    """Return metadata from the last environment step.

    Returns a dict with keys: ``flag_captured``, ``done``, ``reward``,
    ``exit_code``.  Returns empty dict if no step has been taken.
    """
    _ensure_initialized()
    try:
        resp = _get("/status")
        return resp
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# TRL tool functions (passed to GRPOTrainer tools=[...])
#
# Each function has Google-style docstrings so TRL can auto-generate
# the JSON schema that gets injected into the model's prompt.
# ---------------------------------------------------------------------------


def shell_command(command: str, timeout: int = 30) -> str:
    """Run a shell command in the CTF attacker container and return output.

    Use this to execute reconnaissance, enumeration, and exploitation
    commands against the target challenge.

    Args:
        command: The shell command to execute (e.g. ``curl http://target/``).
        timeout: Maximum execution time in seconds. Defaults to 30.

    Returns:
        The combined stdout and stderr output from the command.
    """
    _ensure_initialized()
    resp = _post("/step", {"action": {
        "tool_name": "shell_command",
        "arguments": {"command": command, "timeout": timeout},
    }})
    obs = resp.get("observation", {})
    stdout = obs.get("stdout", "")
    stderr = obs.get("stderr", "")
    if stderr:
        return f"{stdout}\n[stderr] {stderr}"
    return stdout


def python_code(code: str, timeout: int = 120) -> str:
    """Execute Python code in the CTF attacker container.

    Use this for complex exploits, payload generation, or data processing
    that is easier to express in Python than shell commands.

    Args:
        code: Python source code to execute.
        timeout: Maximum execution time in seconds. Defaults to 120.

    Returns:
        The output from executing the Python code.
    """
    _ensure_initialized()
    resp = _post("/step", {"action": {
        "tool_name": "python_code",
        "arguments": {"code": code, "timeout": timeout},
    }})
    obs = resp.get("observation", {})
    stdout = obs.get("stdout", "")
    stderr = obs.get("stderr", "")
    if stderr:
        return f"{stdout}\n[stderr] {stderr}"
    return stdout


def submit_flag(flag: str) -> str:
    """Submit a captured flag for verification.

    Call this when you have found a flag (e.g. ``FLAG{...}``). The
    environment will verify correctness and end the episode if correct.

    Args:
        flag: The flag string to submit (e.g. ``FLAG{example_flag}``).

    Returns:
        Verification result indicating whether the flag is correct.
    """
    _ensure_initialized()
    resp = _post("/step", {"action": {
        "tool_name": "submit_flag",
        "arguments": {"content": flag},
    }})
    obs = resp.get("observation", {})
    correct = obs.get("flag_correct", False)
    if correct:
        return "Correct! Flag verified successfully."
    return f"Incorrect flag. {obs.get('stdout', '')}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
