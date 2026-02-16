"""BoxPwnr tool schemas in OpenAI function-calling format.

Extracted from BoxPwnr's Pydantic input models (boxpwnr/tools/tools.py).
These schemas are used during training data generation and at inference time
to provide tool definitions to the model.
"""

from typing import Any, Dict, List, Optional


BOXPWNR_TOOLS: List[Dict[str, Any]] = [
    # ─── One-shot execution ───────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "shell_command",
            "description": (
                "Runs a shell script (string) and returns its output when finished. "
                "Use this for non-interactive commands, including pipes/redirects and multi-line scripts. "
                "For interactive or long-running programs, use exec_command + write_stdin instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell script to execute as a single string.",
                    },
                    "workdir": {
                        "type": "string",
                        "description": (
                            "Optional working directory to run the script in; "
                            "defaults to current directory."
                        ),
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            "Optional timeout in seconds. If omitted, the executor "
                            "default timeout is used."
                        ),
                        "minimum": 1,
                        "maximum": 300,
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_command",
            "description": (
                "Execute a command using subprocess.run() and return the complete output "
                "when finished. Use this for non-interactive commands. For interactive "
                "commands (shells, sessions, real-time tools), use tmux_* tools instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Maximum execution time in seconds (1-300).",
                        "default": 30,
                        "minimum": 1,
                        "maximum": 300,
                    },
                },
                "required": ["command"],
            },
        },
    },
    # ─── PTY session management (Codex architecture) ──────────────────
    {
        "type": "function",
        "function": {
            "name": "exec_command",
            "description": (
                "Runs a command directly in a PTY and returns output. "
                "Returns a session ID for ongoing interaction via write_stdin. "
                "For most non-interactive commands, prefer shell_command instead. "
                "For interactive programs: exec_command('bash'), exec_command('python3'), "
                "exec_command('ssh user@host')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "description": "Shell command to execute.",
                    },
                    "workdir": {
                        "type": "string",
                        "description": (
                            "Optional working directory to run the command in; "
                            "defaults to the current directory."
                        ),
                    },
                    "yield_time": {
                        "type": "integer",
                        "description": (
                            "How long to wait (in seconds) for output before yielding."
                        ),
                        "default": 5,
                        "minimum": 1,
                        "maximum": 300,
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
            "description": "Bytes to write to stdin (may be empty to poll).",
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": (
                            "Numeric identifier of the running session (e.g., 1, 2)."
                        ),
                    },
                    "chars": {
                        "type": "string",
                        "description": "Bytes to write to stdin (may be empty to poll).",
                        "default": "",
                    },
                    "yield_time": {
                        "type": "integer",
                        "description": "Seconds to wait for output.",
                        "default": 2,
                        "minimum": 1,
                        "maximum": 300,
                    },
                },
                "required": ["session_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_sessions",
            "description": (
                "List all active shell sessions with their status. "
                "Shows session IDs, commands, and current state."
            ),
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
            "description": (
                "Close a shell session and free resources. "
                "Use this when you're done with a session."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "Numeric session ID to close (e.g., '1', '2').",
                    },
                },
                "required": ["session_id"],
            },
        },
    },
    # ─── Tmux tools ───────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "tmux_send_and_read",
            "description": (
                "Type text into the current tmux window and read the output. "
                "Use this to execute commands or provide input to running programs. "
                "Set press_enter=False if you want to type without executing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "Text to type in the current tmux window. "
                            "Do not wrap the text in quotes."
                        ),
                    },
                    "press_enter": {
                        "type": "boolean",
                        "description": (
                            "Whether to press Enter after typing. Set to True to "
                            "execute commands (default), or False to type text "
                            "without executing."
                        ),
                        "default": True,
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tmux_wait_and_read",
            "description": (
                "Wait for a specified number of seconds and then read any new output. "
                "Use this after starting a command that needs time to produce output "
                "(e.g., long-running scans, network operations)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": (
                            "Number of seconds to wait before reading output (1-300)."
                        ),
                        "minimum": 1,
                        "maximum": 300,
                    },
                },
                "required": ["seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "tmux_read_output",
            "description": (
                "Read only the NEW output from the current tmux window since the "
                "last read. Use this to check if there's any new output after "
                "waiting, or to see what's currently displayed without sending "
                "any input."
            ),
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
            "name": "tmux_cancel_command",
            "description": (
                "Send Ctrl-C to abort the currently running command. Use this when "
                "a command is stuck, taking too long, or when you want to interrupt "
                "it to run something else. Returns the output after cancellation."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    # ─── Flag submission ──────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "flag_found",
            "description": "Submit a flag that you have discovered during exploitation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The flag string you discovered.",
                    },
                },
                "required": ["content"],
            },
        },
    },
    # ─── Code execution ──────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "python_code",
            "description": (
                "Execute Python code inside the execution environment (Docker container). "
                "Use this for data processing, encoding/decoding, crypto operations, or "
                "any Python computation. Use print() to see output values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Valid Python code to execute.",
                    },
                    "timeout": {
                        "type": "integer",
                        "description": (
                            "Maximum execution time in seconds (1-120). Code will be "
                            "terminated if it exceeds this limit."
                        ),
                        "default": 120,
                        "minimum": 1,
                        "maximum": 120,
                    },
                },
                "required": ["code"],
            },
        },
    },
    # ─── File operations ─────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file. Returns the content with line numbers "
                "by default, which is useful for editing."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to read.",
                    },
                    "line_numbers": {
                        "type": "boolean",
                        "description": (
                            "Whether to include line numbers in the output."
                        ),
                        "default": True,
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search for a pattern in files using grep. Returns matches "
                "with line numbers and context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in.",
                        "default": ".",
                    },
                    "include": {
                        "type": "string",
                        "description": "File pattern to include (e.g., '*.py').",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "file_search",
            "description": "Find files by name using the 'find' command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Pattern to search for in file names (e.g., '*.py')."
                        ),
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory to search in.",
                        "default": ".",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    # ─── Web search ──────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for information using DuckDuckGo. Returns results "
                "with titles, snippets, and links. Use this to find documentation, "
                "exploits, or other public information."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    # ─── Patching ────────────────────────────────────────────────────
    {
        "type": "function",
        "function": {
            "name": "apply_patch",
            "description": (
                "Apply a patch to files. The patch format is:\n"
                "*** Begin Patch\n"
                "*** Add File: <path>\n"
                "<lines to add with + prefix>\n"
                "*** Update File: <path>\n"
                "@@ ...\n"
                "<hunk with context>\n"
                "*** End Patch\n\n"
                "For updates:\n"
                "- Provide 3 lines of context before and after changes.\n"
                "- Use + for new lines, - for removed lines, and space for context.\n"
                "- Context lines must match the file content EXACTLY."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "patch": {
                        "type": "string",
                        "description": (
                            "The patch string using the custom format "
                            "(*** Begin Patch ...)."
                        ),
                    },
                },
                "required": ["patch"],
            },
        },
    },
]

# Index by name for fast lookup.
_TOOL_INDEX: Dict[str, Dict[str, Any]] = {
    t["function"]["name"]: t for t in BOXPWNR_TOOLS
}


def get_tool_by_name(name: str) -> Optional[Dict[str, Any]]:
    """Return a single tool schema by name, or None if not found."""
    return _TOOL_INDEX.get(name)


def get_tools_by_names(names: List[str]) -> List[Dict[str, Any]]:
    """Return a list of tool schemas for the given names.

    Raises KeyError for unknown tool names.
    """
    tools = []
    for name in names:
        tool = _TOOL_INDEX.get(name)
        if tool is None:
            raise KeyError(f"Unknown tool: {name}")
        tools.append(tool)
    return tools
