"""Default StepAgent — extracts tool parsing + execution from OpenCTFTextEnv.

This is the line-for-line equivalent of OpenCTFTextEnv.step() logic, packaged
as a pluggable StepAgent. It uses parse_tool_calls() for model-agnostic parsing
and SubprocessExecutor for tool execution.

Users who want custom tool handling can implement StepAgent and swap this out
via ``agent_class`` in the GRPO config or ``--agent`` on the CLI.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from open_ctf.agent.protocol import StepResult

logger = logging.getLogger(__name__)


class DefaultStepAgent:
    """Default StepAgent using parse_tool_calls + SubprocessExecutor.

    Implements the StepAgent protocol via structural subtyping (no inheritance
    required). Logic is extracted from OpenCTFTextEnv.step() lines 329-433.

    Attributes exposed for reward computation by the env:
        tool_calls_history: List of {name, arguments} dicts.
        tool_outputs: List of output strings.
        all_text: Concatenated LLM + tool output text.
        episode_done: Whether the flag was submitted successfully.
        turns: Number of steps taken.
    """

    def __init__(self, **kwargs: Any):
        self._executor = None
        self._executor_type: str = kwargs.get("executor_type", "subprocess")
        self._executor_kwargs: Dict[str, Any] = kwargs

        # Episode state (exposed as properties for reward computation)
        self.tool_calls_history: List[Dict[str, str]] = []
        self.tool_outputs: List[str] = []
        self.all_text: str = ""
        self.episode_done: bool = False
        self.turns: int = 0
        self.max_steps: int = 30

    def reset(
        self,
        target: str = "",
        ground_truth_flag: str = "",
        max_steps: int = 30,
        **kwargs: Any,
    ) -> None:
        """Reset agent state and executor for a new episode."""
        # Close previous executor if any
        if self._executor is not None:
            self._executor.close()

        # Create executor
        from open_ctf.envs.tool_executor import (
            BaseExecutor,
            RemoteBatchExecutor,
            SubprocessExecutor,
        )

        executor_type = kwargs.get("executor_type", self._executor_type)
        if executor_type == "remote":
            self._executor: BaseExecutor = RemoteBatchExecutor(
                target=target,
                ground_truth=ground_truth_flag,
                max_steps=max_steps * 5,
            )
        else:
            self._executor: BaseExecutor = SubprocessExecutor(
                target=target,
                ground_truth=ground_truth_flag,
                max_steps=max_steps * 5,
            )

        self._executor.reset()

        # Reset episode state
        self.tool_calls_history = []
        self.tool_outputs = []
        self.all_text = ""
        self.episode_done = False
        self.turns = 0
        self.max_steps = max_steps

    def step(self, action: str) -> StepResult:
        """Parse tool calls from LLM output and execute them.

        Logic is line-for-line identical to OpenCTFTextEnv.step().
        """
        from open_ctf.envs.skyrl.openctf_env import parse_tool_calls

        self.turns += 1
        self.all_text += "\n" + action

        # Parse tool calls from LLM output
        tool_calls = parse_tool_calls(action)

        if not tool_calls:
            # No tool calls — model just generated text.
            done = self.turns >= self.max_steps
            if done:
                return StepResult(observations=[], done=True, info={"tool_calls": 0, "step": self.turns})
            return StepResult(
                observations=[
                    {"role": "user", "content": "No tool call detected. Use a tool to make progress."}
                ],
                done=False,
                info={"tool_calls": 0, "step": self.turns},
            )

        # Execute each tool call via executor
        obs_messages: List[Dict[str, str]] = []
        for tc in tool_calls:
            if self.episode_done:
                output = "[EPISODE COMPLETE] Flag already submitted."
            else:
                # Track for reward computation
                self.tool_calls_history.append({
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"]) if isinstance(tc["arguments"], dict) else str(tc["arguments"]),
                })

                try:
                    resp = self._executor.step(tc["name"], tc["arguments"])
                    stdout = resp.get("stdout", "")
                    stderr = resp.get("stderr", "")
                    env_done = resp.get("done", False)
                except Exception as exc:
                    logger.warning("Tool execution error: %s", exc)
                    stdout = f"[ERROR] Tool execution failed: {exc}"
                    stderr = ""
                    env_done = False

                output = stdout
                if stderr:
                    output += f"\n[stderr] {stderr}"

                self.tool_outputs.append(output)
                self.all_text += "\n" + output

                # Check for flag submission success
                if env_done or (tc["name"] == "flag_found" and "correct" in stdout.lower()):
                    self.episode_done = True
                    logger.info("Episode done at step %d (flag submitted)", self.turns)

            obs_messages.append({
                "role": "user",
                "content": f"[Tool: {tc['name']}]\n{output}",
            })

        done = self.episode_done or self.turns >= self.max_steps

        if done:
            return StepResult(
                observations=[],
                done=True,
                info={
                    "tool_calls": len(tool_calls),
                    "step": self.turns,
                    "episode_done": self.episode_done,
                },
            )

        return StepResult(
            observations=obs_messages,
            done=False,
            info={
                "tool_calls": len(tool_calls),
                "step": self.turns,
                "episode_done": self.episode_done,
            },
        )

    @property
    def tools(self):
        """Use environment default tool schemas."""
        return None

    def close(self) -> None:
        """Release executor resources."""
        if self._executor is not None:
            self._executor.close()
            self._executor = None
