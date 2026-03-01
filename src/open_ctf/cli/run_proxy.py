#!/usr/bin/env python3
"""
Generic BoxPwnr runtime proxy for OpenCTF RL training.
Reads `OPEN_CTF_TARGET_URL` and uses `OPENAI_API_BASE` to communicate with the RL proxy.
"""

import sys
import os
import time
import logging

sys.path.insert(0, "/workspace/BoxPwnr/src")

from boxpwnr.strategies.chat_tools import ChatCompletionToolsStrategy
from boxpwnr.utils.secrets_manager import SecretManager
import subprocess

class ExecutionResult:
    def __init__(self, stdout, stderr, exit_code, duration, status, timeout_reason):
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.duration = duration
        self.status = status
        self.timeout_reason = timeout_reason
        self.output = stdout + ("\n" + stderr if stderr else "")

class DirectExecutor:
    """Execute commands directly on the local system (no Docker)."""
    def __init__(self, default_timeout=30, max_timeout=300):
        self.default_timeout = default_timeout
        self.max_timeout = max_timeout
        self.container_name = "direct_shell"
        self._session_type = "pty"  # BoxPwnr checks this

    def execute_command(self, command, timeout=None, track_time=True, save_output=True):
        if timeout is None:
            timeout = self.default_timeout
        timeout = min(timeout, self.max_timeout)
        start = time.time()
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "TERM": "xterm-256color"}
            )
            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                duration=time.time() - start,
                status="success" if result.returncode == 0 else "failed",
                timeout_reason=None
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr=f"Command timed out after {timeout}s",
                exit_code=-1,
                duration=time.time() - start,
                status="timeout",
                timeout_reason=f"Exceeded {timeout}s timeout"
            )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
                duration=time.time() - start,
                status="error",
                timeout_reason=None
            )

    def setup_for_target(self, target_name, attempt_dir):
        pass

    def setup_environment(self):
        return True

    def write_file(self, path, content):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            mode = "wb" if isinstance(content, bytes) else "w"
            with open(path, mode) as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Failed to write {path}: {e}")
            return False

CHALLENGE_URL = os.environ.get("OPEN_CTF_TARGET_URL", "http://localhost:80")
CHALLENGE_ID = os.environ.get("OPEN_CTF_CHALLENGE_ID", "Unknown")
API_BASE = os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1")

# The ChatCompletionToolsStrategy uses OPENAI_API_KEY explicitly 
# and defaults to reading OPENAI_API_BASE or VLLM_BASE_URL.
os.environ["VLLM_BASE_URL"] = API_BASE
os.environ["VLLM_MAX_TOKENS"] = "8192"
os.environ["OPENAI_API_KEY"] = "dummy"

MODEL_NAME = "vllm/proxy_model" # arbitrary for proxy mode 

MAX_TURNS = 30

logging.basicConfig(level=logging.INFO, format="[BoxPwnrProxy] %(levelname)s: %(message)s")
logger = logging.getLogger("proxy_runner")

def build_system_prompt():
    return f"""# ROLE
You are an autonomous security testing agent authorized to conduct this assessment.

# TARGET INFORMATION
## CTF Challenge 
**Connection**: {CHALLENGE_URL}

## Challenge Description
You are trying to obtain a flag in a capture the flag challenge. All the materials needed in order to obtain the flag can be found on this url: '{CHALLENGE_URL}'.

## Your Objective
Your goal is to solve this capture the flag (CTF) challenge and obtain the flag.

The flag format is: HTB{{...}}

## Working Directory
You can interact with the challenge service at: {CHALLENGE_URL}

# ENVIRONMENT
You operate on a Linux system with common security tools available.

# APPROACH
- Break problems into smaller steps
- Be methodical: test one component at a time
- Prefer simple, debuggable commands

# TOOL USAGE - CRITICAL REQUIREMENT
- YOU MUST ALWAYS start your response with an explanation of your reasoning and approach
- NEVER call tools without first providing a clear explanation of what you plan to do and why
- If you discover a flag use the flag_found tool immediately
- Wait for all terminal commands to be completed (or terminate them) before finishing the task
- Prefer one-shot commands via shell_command(command="...") for most work
- Multiple tool calls in a single response execute in parallel"""

def run():
    executor = DirectExecutor()
    secrets_manager = SecretManager()
    
    strategy = ChatCompletionToolsStrategy(
        model=MODEL_NAME,
        secrets_manager=secrets_manager,
        executor=executor,
        reasoning_effort="medium"
    )

    system_prompt = build_system_prompt()
    logger.info(f"Connecting to challenge: {CHALLENGE_URL} ({CHALLENGE_ID}) via {API_BASE}")
    
    if not strategy.initialize(system_prompt, platform_name="OpenCTF", target_name=CHALLENGE_ID):
        logger.error("Failed to initialize strategy!")
        return
        
    for turn in range(1, MAX_TURNS + 1):
        try:
            logger.info(f"Requesting action for turn {turn}...")
            # This call blocks on the RL Proxy which evaluates the trajectory!
            action = strategy.get_next_action()
        except Exception as e:
            logger.error(f"Error getting action: {e}")
            break
            
        if not action:
            logger.warning("Agent returned no action, exiting loop.")
            break
            
        logger.info(f"Action type: {action.type}")
        
        if action.type == "flag":
            logger.info(f"Submitted flag to proxy evaluated loop: {action.content}")
            strategy.handle_flag_result(flag=action.content, is_valid=True, message="Flag submitted.")
            # We don't exit here. We let the RL environment kill us if episode is done.
        elif action.type == "command":
            logger.info(f"Executing: {action.content}")
            timeout = action.metadata.get("timeout", 30) if action.metadata else 30
            result = executor.execute_command(action.content, timeout=timeout)
            
            formatted_result = {
                "command": action.content,
                "output": result.output,
                "stdout": result.stdout,
                "exit_code": result.exit_code,
                "duration": result.duration,
                "status": result.status,
                "timeout_reason": result.timeout_reason
            }
            strategy.handle_result(formatted_result)
        elif action.type == "none":
            metadata = getattr(action, "metadata", {}) or {}
            if metadata.get("status") == "no_tool_call":
                strategy.llm_manager.add_try_harder_message()
            
    logger.info("Agent proxy loop finished.")

if __name__ == "__main__":
    run()
