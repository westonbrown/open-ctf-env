import asyncio
import logging
import os
from typing import Any

from open_ctf.agent.default_agent import DefaultStepAgent
from open_ctf.agent.protocol import StepResult
from open_ctf.agent.rollout_status import RolloutStatus
from open_ctf.envs.skyrl.rl_proxy_runtime_bridge import RLProxyRuntimeBridge
from open_ctf.parsing import parse_tool_calls

logger = logging.getLogger(__name__)

import socket


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

class ProxyStepAgent(DefaultStepAgent):
    """StepAgent that uses the ROCK ModelService proxy architecture.
    
    Instead of parsing tools directly, this agent forwards the LLM action 
    to the external BYO agent process via an HTTP Proxy and blocks until
    the external agent makes its next POST /v1/chat/completions request.
    """
    def __init__(self, agent_cmd: str | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.port = _get_free_port()

        self.agent_cmd = agent_cmd or os.environ.get(
            "OPEN_CTF_AGENT_CMD",
            "python examples/adapters/langgraph_native_runtime_adapter.py"
        )
        self.bridge = RLProxyRuntimeBridge(
            agent_cmd=self.agent_cmd, max_steps=self.max_steps, port=self.port
        )
        self.loop = asyncio.new_event_loop()

        self.last_prompt_len = 0
        self._initial_prompt: list[dict[str, Any]] | None = None

    def reset(self, *args, **kwargs):
        super().reset(*args, **kwargs)
        target = kwargs.get("target") or "http://localhost:80"
        challenge_id = kwargs.get("challenge_id") or "Unknown"
        logger.info(f"Starting ProxyStepAgent bridge on port {self.port} for target {target} with CMD: {self.agent_cmd}")
        self.loop.run_until_complete(self.bridge.start(target=target, challenge_id=challenge_id))

        # Block until the BYO agent boots and sends its very first prompt
        logger.info("Waiting for initial prompt from BYO Agent...")
        initial_prompt = self.loop.run_until_complete(self.bridge.get_next_prompt(wait_timeout=20.0))

        if initial_prompt:
            self._initial_prompt = initial_prompt
            self.last_prompt_len = len(initial_prompt)
            # The agent controls the prompt entirely!
        else:
            logger.error("BYO Agent failed to send initial prompt.")
            self.episode_done = False

    def get_initial_prompt(self) -> list[dict[str, Any]] | None:
        """Called by openctf_env.py to rewrite the starting prompt to match the agent."""
        return self._initial_prompt

    def step(self, action: str) -> StepResult:
        # 1. We just got text from vLLM/Megatron. Send it back to the resting BYO Agent.
        logger.debug(f"Sending action to proxy bridge: {action[:100]}")

        # We must append the model's text (and mock tool outputs in our local memory to match BaseAgent)
        self.all_text += action

        self.loop.run_until_complete(self.bridge.send_action(action))

        # 2. Block until the BYO Agent executes its tool and asks for the next generation
        new_prompt = self.loop.run_until_complete(self.bridge.get_next_prompt(wait_timeout=30.0))

        info = {"rollout_status": RolloutStatus.OK.value}

        if self.bridge.is_done or new_prompt is None:
            logger.info(f"Proxy bridge is done. Terminal Reward assigned: {self.bridge.terminal_reward}")
            self.episode_done = (self.bridge.terminal_reward > 0)
            if self.bridge.terminal_reward < 0:
                info["rollout_status"] = RolloutStatus.TOOL_TIMEOUT.value

            return StepResult(observations=[], done=True, info=info)

        # 3. Calculate delta to return as standard gym observations
        delta = new_prompt[self.last_prompt_len:]
        self.last_prompt_len = len(new_prompt)

        # Parse delta messages to populate tool_calls_history AND tool_outputs
        # for CTFReward scoring.  Without this, format and recovery reward
        # signals are always 0.0 because the reward function reads
        # agent.tool_outputs and agent.tool_calls_history.
        for msg in delta:
            role = msg.get("role", "")
            content = msg.get("content", "") or ""

            if role == "assistant" and content:
                # Extract tool call names/args for reward metrics.
                # We do NOT execute anything locally — the BYO agent
                # already handled execution.
                self._extract_metrics_only(content)

            elif role in ("user", "tool") and content:
                # Tool response messages from the BYO agent carry the
                # execution output.  Append to tool_outputs so CTFReward
                # has data for format, recovery, and interaction-quality
                # signals.
                truncated = self._truncate_tool_output(content)
                self.tool_outputs.append(truncated)
                self.all_text += "\n" + truncated

        return StepResult(observations=delta, done=False, info=info)

    def _extract_metrics_only(self, text: str):
        """Parse tool calls from assistant text for reward-signal metrics.

        Delegates to :func:`open_ctf.parsing.parse_tool_calls` which handles
        all 5 formats (Hermes JSON, Qwen3.5 XML, GLM-4 XML, bare JSON,
        Python-style) instead of a single permissive regex.
        """
        calls = parse_tool_calls(text)
        for call in calls:
            name = call.get("name", "unknown")
            args = call.get("arguments", {})
            self.tool_calls_history.append({"name": name, "arguments": args})

    def close(self):
        super().close()
        self.loop.run_until_complete(self.bridge.stop())
        self.loop.close()
