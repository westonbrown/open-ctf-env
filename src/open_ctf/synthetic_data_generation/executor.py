"""
Offline Simulated environment mimicking the ToolExecutor interface for fast, scalable generation.
"""

import re
import uuid
import logging
from typing import Dict, Any, Optional

from ..envs.tool_executor import BaseExecutor, ALL_TOOLS
from .manifest import WorldManifest, FileNode

logger = logging.getLogger(__name__)

class SimulatedEnvironmentExecutor(BaseExecutor):
    """
    Offline executor that mocks all tool effects based on a WorldManifest.
    Removes realistic container execution to eliminate simulation drift.
    """
    def __init__(self, manifest: WorldManifest, **kwargs):
        super().__init__(**kwargs)
        self.manifest_template = manifest
        # Setup specific to this process/actor instance
        self.tools = kwargs.get("tools") or sorted(ALL_TOOLS)
        self.max_steps = kwargs.get("max_steps", 30)
        self.ground_truth = manifest.ground_truth_flag
        
        # State
        self._step_count = 0
        self._done = False
        self._current_manifest: WorldManifest = self.manifest_template.clone()

    def reset(self, generation_id: Optional[str] = None) -> dict:
        self._step_count = 0
        self._done = False
        self._current_manifest = self.manifest_template.clone()
        return {
            "stdout": f"Simulated environment loaded: {self._current_manifest.name}",
            "stderr": "",
            "exit_code": 0,
            "done": False,
            "reward": 0.0,
        }

    def step(self, tool_name: str, arguments: dict, generation_id: Optional[str] = None) -> dict:
        self._step_count += 1
        
        if tool_name not in self.tools:
            return {
                "stdout": "",
                "stderr": f"Tool '{tool_name}' is not available.",
                "exit_code": 1,
                "done": self._step_count >= self.max_steps,
                "reward": -0.02,
            }

        # Handle mock flag submission
        if tool_name in ("submit_flag", "flag_found"):
            submitted = arguments.get("content", "").strip()
            correct = bool(self.ground_truth) and submitted == self.ground_truth
            self._done = True
            
            # Injection point: Optional failure perturbation could artificially flip 'correct' 
            # for robustness training, though SFT usually wants optimal paths unless labeled failure.
            
            if correct:
                return {"stdout": f"Correct! Flag verified: {submitted}", "stderr": "", "exit_code": 0, "done": True, "reward": 1.0}
            else:
                return {"stdout": f"Incorrect submission: {submitted}", "stderr": "", "exit_code": 1, "done": True, "reward": -0.2}

        # Dispatch to mock handling
        stdout, stderr, exit_code = self._handle_mock(tool_name, arguments)
        
        done = self._step_count >= self.max_steps
        reward = 0.05 if exit_code == 0 else -0.02
        
        return {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "done": done,
            "reward": reward,
        }

    def close(self) -> None:
        """No PTY sessions to clean up in mock."""
        pass

    # -- Mock Dispatchers --
    def _handle_mock(self, tool_name: str, args: Dict[str, Any]) -> tuple[str, str, int]:
        """Routes tool call dynamically based on manifest mappings or defined logic."""
        
        # 1. Check exact/regex tool match overrides from manifest
        # This allows users to perfectly script arbitrary tool_responses like `execute_sql`
        command_query = str(args.get("command", args.get("code", args.get("path", ""))))
        if tool_name in self._current_manifest.tool_responses:
            for pattern, response in self._current_manifest.tool_responses[tool_name].items():
                if re.search(pattern, command_query) or pattern in command_query:
                    return response, "", 0
                    
        # 2. Builtin file fallbacks
        if tool_name == "read_file":
            path = args.get("path", args.get("file_path", ""))
            if path in self._current_manifest.files:
                return self._current_manifest.files[path].content, "", 0
            return "", f"cat: {path}: No such file or directory", 1
            
        # 3. Builtin shell/exec fallbacks
        if tool_name in ("shell_command", "execute_command", "exec_command"):
            cmd = args.get("command", args.get("cmd", ""))
            
            # Simulated typical Recon Tools
            if "nmap" in cmd:
                return self._mock_nmap(cmd)
            elif "whoami" in cmd:
                return "root\n", "", 0
            elif "ls" in cmd:
                return self._mock_ls(cmd)
            
            return f"{cmd}: mock fallback response\n", "", 0
            
        return "Simulated execution success", "", 0

    def _mock_nmap(self, cmd: str) -> tuple[str, str, int]:
        lines = ["Starting Nmap ( https://nmap.org )"]
        lines.append("Nmap scan report for target")
        lines.append("Host is up (0.0001s latency).")
        lines.append("PORT     STATE SERVICE")
        
        for host in self._current_manifest.hosts.values():
            for srv in host.services:
                lines.append(f"{srv.port}/tcp open  {srv.name}")
        
        return "\n".join(lines), "", 0
        
    def _mock_ls(self, cmd: str) -> tuple[str, str, int]:
        output = []
        for path in self._current_manifest.files.keys():
            output.append(path.split('/')[-1])
        return "  ".join(output), "", 0
