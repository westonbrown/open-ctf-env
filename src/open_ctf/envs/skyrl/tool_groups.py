"""SkyRL-Gym ToolGroup wrapper for CTF tools.

Wraps our tool functions as a SkyRL-Gym ToolGroup for tool schema
discovery. SkyRL uses ToolGroups to inject tool definitions into the
system prompt and validate tool calls during generation.

The actual tool execution happens in OpenCTFTextEnv.step() via
ToolExecutor — this module only provides the schema metadata.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# Tool definitions matching our ToolExecutor tools (from tool_executor.py).
# These are OpenAI-compatible function schemas that SkyRL can inject
# into prompts and use for tool-call validation.
OPENCTF_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "shell_command",
            "description": (
                "Run a shell command in the CTF attacker container. "
                "Supports pipes, redirects, and multi-line scripts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Max seconds", "default": 30},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "python_code",
            "description": (
                "Execute Python code for exploits, payload generation, "
                "encoding/decoding, crypto operations. Use print() for output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source code"},
                    "timeout": {"type": "integer", "description": "Max seconds", "default": 120},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read file contents with optional line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to file"},
                    "line_numbers": {"type": "boolean", "default": True},
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for regex pattern in files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regex pattern"},
                    "path": {"type": "string", "description": "Directory to search", "default": "."},
                    "include": {"type": "string", "description": "File glob filter"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_search",
            "description": "Find files by name pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                    "path": {"type": "string", "description": "Search directory", "default": "."},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": "Apply a patch to modify files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {"type": "string", "description": "Patch content"},
                },
                "required": ["patch"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_found",
            "description": (
                "Submit a discovered flag for verification. "
                "Call when you have found the flag."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The flag string"},
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for CVE details, exploits, documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": (
                "Start an interactive command session (e.g. netcat, ssh, gdb). "
                "Returns a session ID for subsequent write_stdin calls."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string", "description": "Command to run interactively"},
                    "workdir": {"type": "string", "description": "Working directory"},
                    "yield_time": {
                        "type": "integer",
                        "description": "Seconds to wait for initial output",
                        "default": 5,
                    },
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_stdin",
            "description": "Write input to an interactive session started with exec_command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session ID from exec_command"},
                    "chars": {"type": "string", "description": "Characters to send to stdin"},
                    "yield_time": {
                        "type": "integer",
                        "description": "Seconds to wait for output after writing",
                        "default": 2,
                    },
                },
                "required": ["session_id", "chars"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": (
                "Run a shell command in the CTF attacker container. "
                "Alias for shell_command. Supports pipes, redirects, and multi-line scripts."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "timeout": {"type": "integer", "description": "Max seconds", "default": 30},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sessions",
            "description": "List all active interactive sessions started with exec_command.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "close_session",
            "description": "Close an active interactive session.",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session ID to close"},
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_flag",
            "description": (
                "Submit a discovered flag for verification. "
                "Alias for flag_found. Call when you have found the flag."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "The flag string"},
                },
                "required": ["content"],
            },
        },
    },
]


def get_tool_schemas() -> List[Dict[str, Any]]:
    """Return OpenAI-compatible tool schemas for all 14 CTF tools.

    Covers the full ToolExecutor tool set (13 unique tools + 2 aliases):
      Tier 1: shell_command, exec_command, write_stdin, python_code, execute_command
      Tier 2: read_file, grep, file_search, apply_patch
      Tier 3: flag_found, submit_flag, web_search, list_sessions, close_session

    These can be injected into SkyRL's chat template via the tools parameter
    or used for tool-call validation during generation.
    """
    return OPENCTF_TOOLS


def get_tool_names() -> List[str]:
    """Return the list of available tool names."""
    return [t["function"]["name"] for t in OPENCTF_TOOLS]
