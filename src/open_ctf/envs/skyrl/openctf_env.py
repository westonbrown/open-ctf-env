"""SkyRL-Gym BaseTextEnv subclass bridging SkyRL to execution environments.

Each SkyRL agent loop gets its own env instance with a BaseExecutor
(SubprocessExecutor or RemoteBatchExecutor) for tool execution.

Architecture:
    SkyRL SkyRLGymGenerator -> agent_loop()
        -> env.init(prompt) -> BaseExecutor.reset()
        -> env.step(action) -> parse tool calls, BaseExecutor.step(),
                               compute reward via CTFReward
        -> env.close() -> BaseExecutor.close()

The env receives raw LLM text output, parses tool calls from it using
regex patterns compatible with Hermes/GLM4/Qwen3 formats, executes
them via BaseExecutor, and returns observations + rewards.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Type aliases matching SkyRL's ConversationType
ConversationType = List[Dict[str, Any]]

# ---------------------------------------------------------------------------
# Tool call parsing patterns (model-agnostic)
# ---------------------------------------------------------------------------

# Hermes/Qwen3/Nanbeige: <tool_call>{"name": ..., "arguments": ...}</tool_call>
_HERMES_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)

# GLM-4 MoE XML: <tool_call>func_name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
_GLM4_TC_PATTERN = re.compile(
    r"<tool_call>(\S+?)((?:<arg_key>.*?</arg_key><arg_value>.*?</arg_value>)*)\s*</tool_call>",
    re.DOTALL,
)
_GLM4_ARG_PATTERN = re.compile(
    r"<arg_key>(.*?)</arg_key><arg_value>(.*?)</arg_value>", re.DOTALL,
)

# Bare JSON fallback: {"name": "...", "arguments": {...}}
# Supports one level of nested braces in arguments (e.g. {"headers": {"X-UserId": "10052"}})
_BARE_JSON_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{(?:[^{}]|\{[^{}]*\})*\})\s*\}',
    re.DOTALL,
)


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from LLM output text.

    Supports Hermes JSON, GLM4 XML, and bare JSON formats.
    Returns list of {"name": str, "arguments": dict} dicts.
    """
    tool_calls = []

    # 1. Hermes/Qwen3/Nanbeige JSON format
    for m in _HERMES_PATTERN.finditer(text):
        try:
            d = json.loads(m.group(1))
            name = d.get("name", "")
            args = d.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name:
                tool_calls.append({"name": name, "arguments": args})
        except json.JSONDecodeError:
            continue

    if tool_calls:
        return tool_calls

    # 2. GLM-4 MoE XML format
    for m in _GLM4_TC_PATTERN.finditer(text):
        name = m.group(1).strip()
        args = {}
        for am in _GLM4_ARG_PATTERN.finditer(m.group(2)):
            key = am.group(1).strip()
            val = am.group(2).strip()
            try:
                val = json.loads(val)
            except (ValueError, json.JSONDecodeError):
                pass
            args[key] = val
        if name:
            tool_calls.append({"name": name, "arguments": args})

    if tool_calls:
        return tool_calls

    # 3. Bare JSON fallback
    for m in _BARE_JSON_PATTERN.finditer(text):
        name = m.group(1)
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            args = {}
        if name:
            tool_calls.append({"name": name, "arguments": args})

    return tool_calls


# ---------------------------------------------------------------------------
# Lazy import of BaseTextEnv — avoids hard dependency on skyrl_gym at
# module load time (allows running validate / other CLIs without skyrl).
# ---------------------------------------------------------------------------

def _get_base_class():
    """Import BaseTextEnv lazily to avoid hard dep on skyrl_gym."""
    try:
        from skyrl_gym.envs.base_text_env import BaseTextEnv
        return BaseTextEnv
    except ImportError:
        # Fallback: return object so the class can still be defined
        # (won't pass SkyRL isinstance checks but allows unit testing)
        logger.warning("skyrl_gym not installed — OpenCTFTextEnv will not register with SkyRL")
        return object


# Build the class dynamically to handle missing skyrl_gym gracefully
_Base = _get_base_class()


class OpenCTFTextEnv(_Base):
    """SkyRL-Gym BaseTextEnv for CTF challenges via BaseExecutor.

    Each instance manages one episode with execution via SubprocessExecutor
    or RemoteBatchExecutor. SkyRL creates a new instance per trajectory.

    SkyRL's ``make()`` merges registered kwargs (static config) with
    per-sample kwargs from the dataset:

    - **Static** (from ``register(kwargs=...)``)::

        reward_config: dict of CTFReward weight overrides

    - **Per-sample** (from dataset ``extras``)::

        ground_truth_flag: expected flag string
        optimal_steps: optimal step count for efficiency reward
        challenge_id: challenge identifier

    Both arrive as keyword args; per-sample data is nested under ``extras``.
    """

    def __init__(
        self,
        env_config: Any = None,
        extras: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        if _Base is not object:
            super().__init__()

        extras = extras or {}

        self.max_turns = extras.get("max_turns") or kwargs.get("max_turns", 15)
        self.turns = 0

        # Tool schemas — SkyRL uses these for prompt injection.
        from open_ctf.envs.skyrl.tool_groups import OPENCTF_TOOLS
        self.tools = OPENCTF_TOOLS
        self.tool_groups = []

        self._episode_id: Optional[str] = None
        self._done = False
        self._tool_calls_history: List[Dict[str, str]] = []
        self._tool_outputs: List[str] = []
        self._all_text = ""

        # Tool call format for prompt injection.
        # "hermes" (default): <tool_call>{"name": ..., "arguments": ...}</tool_call>
        # "glm4":  <tool_call>func_name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
        self._tool_call_format: str = (
            extras.get("tool_call_format")
            or kwargs.get("tool_call_format", "hermes")
        )

        # Reconstruct reward function from serializable config dict.
        reward_config = kwargs.get("reward_config") or extras.get("reward_config")
        if reward_config and isinstance(reward_config, dict):
            try:
                from open_ctf.training.step_reward import create_reward_fn
                self._reward_fn = create_reward_fn({"reward": reward_config})
            except Exception as exc:
                logger.warning("Failed to create reward function: %s — using binary fallback", exc)
                self._reward_fn = None
        else:
            self._reward_fn = None

        # Per-episode data (set from dataset sample extras)
        self._ground_truth_flag: Optional[str] = extras.get("ground_truth_flag")
        self._optimal_steps: Optional[int] = extras.get("optimal_steps")
        self._challenge_id: Optional[str] = extras.get("challenge_id")

        # Support different executor backends for BYOE
        executor_type = extras.get("executor_type", kwargs.get("executor_type", "subprocess"))
        
        from open_ctf.envs.tool_executor import BaseExecutor, RemoteBatchExecutor, SubprocessExecutor
        if executor_type == "remote":
            self._executor: BaseExecutor = RemoteBatchExecutor(
                target=extras.get("target", kwargs.get("target", "")),
                ground_truth=self._ground_truth_flag or "",
                max_steps=self.max_turns * 5,
            )
        else:
            self._executor: BaseExecutor = SubprocessExecutor(
                target=extras.get("target", kwargs.get("target", "")),
                ground_truth=self._ground_truth_flag or "",
                max_steps=self.max_turns * 5,  # generous step limit per episode
            )

    def init(self, prompt: ConversationType) -> tuple:
        """Initialize episode: reset executor, return prompt with tool schemas.

        Args:
            prompt: Initial conversation (system + user messages).

        Returns:
            (prompt, metadata) — prompt with tool schemas injected, metadata has episode_id.
        """
        # Update ground truth if set per-episode
        if self._ground_truth_flag:
            self._executor.ground_truth = self._ground_truth_flag

        resp = self._executor.reset()
        self._episode_id = None  # no longer tracked via server
        self.turns = 0
        self._done = False
        self._tool_calls_history = []
        self._tool_outputs = []
        self._all_text = ""

        # Inject tool schemas into the system message so the model knows
        # what tools are available during GRPO rollouts. SkyRL's generator
        # may pass tools= to apply_chat_template for structured tool calling,
        # but for text-based parsing the model needs schemas in the prompt.
        prompt = self._inject_tool_schemas(prompt)

        logger.debug(
            "OpenCTFTextEnv initialized: challenge=%s",
            self._challenge_id,
        )
        return prompt, {"episode_id": self._episode_id}

    def _inject_tool_schemas(self, prompt: ConversationType) -> ConversationType:
        """Prepend tool schemas to the system message in the prompt.

        If no system message exists, one is created. If the system message
        already contains tool schema text, injection is skipped to avoid
        duplication.

        Returns a new list (does not mutate the input).
        """
        if not self.tools:
            return prompt

        # Format tool schemas as a concise block
        tool_lines = []
        for tool_def in self.tools:
            fn = tool_def.get("function", {})
            name = fn.get("name", "")
            desc = fn.get("description", "")
            params = fn.get("parameters", {})
            required = params.get("required", [])
            props = params.get("properties", {})

            param_parts = []
            for pname, pschema in props.items():
                ptype = pschema.get("type", "string")
                req_marker = " [required]" if pname in required else ""
                param_parts.append(f"  - {pname}: {ptype}{req_marker}")

            param_str = "\n".join(param_parts) if param_parts else "  (no parameters)"
            tool_lines.append(f"- {name}: {desc}\n{param_str}")

        # Model-aware format instruction
        _FORMAT_INSTRUCTIONS = {
            "hermes": (
                'Call tools using: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>'
            ),
            "glm4": (
                "Call tools using: <tool_call>tool_name"
                "<arg_key>param</arg_key><arg_value>value</arg_value>"
                "</tool_call>"
            ),
        }
        fmt_instruction = _FORMAT_INSTRUCTIONS.get(
            self._tool_call_format, _FORMAT_INSTRUCTIONS["hermes"]
        )

        tools_block = (
            "\n\n# Available Tools\n\n"
            + fmt_instruction + "\n\n"
            + "\n".join(tool_lines)
            + "\n"
        )

        # Make a shallow copy to avoid mutating the original
        prompt = list(prompt)

        # Find or create system message
        if prompt and prompt[0].get("role") == "system":
            sys_content = prompt[0].get("content", "")
            # Skip if tool schemas already present
            if "# Available Tools" not in sys_content:
                prompt[0] = {
                    **prompt[0],
                    "content": sys_content + tools_block,
                }
        else:
            # No system message — prepend one with tool schemas
            prompt.insert(0, {
                "role": "system",
                "content": "You are a CTF agent with access to the following tools." + tools_block,
            })

        return prompt

    def step(self, action: str) -> Dict[str, Any]:
        """Process LLM output: parse tool calls, execute via executor.

        Args:
            action: Raw LLM text output (may contain tool calls).

        Returns:
            BaseTextEnvStepOutput dict with observations, reward, done, metadata.
            Observations use role="user" with tool name embedded in content
            to match SkyRL's apply_chat_template expectations.
        """
        self.turns += 1
        self._all_text += "\n" + action

        # Parse tool calls from LLM output
        tool_calls = parse_tool_calls(action)

        if not tool_calls:
            # No tool calls — model just generated text.
            # Use role="user" here since this is a system prompt, not a tool result.
            done = self.turns >= self.max_turns
            if done:
                return {
                    "observations": [],
                    "reward": self._compute_reward(done),
                    "done": True,
                    "metadata": {"tool_calls": 0, "step": self.turns},
                }
            return {
                "observations": [
                    {"role": "user", "content": "No tool call detected. Use a tool to make progress."}
                ],
                "reward": 0.0,
                "done": False,
                "metadata": {"tool_calls": 0, "step": self.turns},
            }

        # Execute each tool call via executor
        obs_messages: ConversationType = []
        for tc in tool_calls:
            if self._done:
                output = "[EPISODE COMPLETE] Flag already submitted."
            else:
                # Track for reward computation
                self._tool_calls_history.append({
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

                self._tool_outputs.append(output)
                self._all_text += "\n" + output

                # Check for flag submission success
                if env_done or (tc["name"] == "flag_found" and "correct" in stdout.lower()):
                    self._done = True
                    logger.info("Episode done at step %d (flag submitted)", self.turns)

            # Build observation with role="user" to match SkyRL's
            # apply_chat_template expectations (all built-in SkyRL envs
            # return observations with role="user").
            obs_messages.append({
                "role": "user",
                "content": f"[Tool: {tc['name']}]\n{output}",
            })

        done = self._done or self.turns >= self.max_turns
        reward = self._compute_reward(done)

        if done:
            return {
                "observations": [],
                "reward": reward,
                "done": True,
                "metadata": {
                    "tool_calls": len(tool_calls),
                    "step": self.turns,
                    "episode_done": self._done,
                },
            }

        return {
            "observations": obs_messages,
            "reward": reward,
            "done": False,
            "metadata": {
                "tool_calls": len(tool_calls),
                "step": self.turns,
                "episode_done": self._done,
            },
        }

    def _compute_reward(self, done: bool) -> float:
        """Compute reward for the current step."""
        if not done:
            from open_ctf.training.step_reward import per_step_reward
            return per_step_reward(
                self._tool_calls_history, self.turns, self.max_turns,
            )

        # Terminal: compute full reward
        if self._reward_fn is not None:
            completion_msgs = []
            for i, tc in enumerate(self._tool_calls_history):
                completion_msgs.append({
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": tc["name"],
                            "arguments": tc["arguments"],
                        }
                    }],
                })
                if i < len(self._tool_outputs):
                    completion_msgs.append({
                        "role": "tool",
                        "content": self._tool_outputs[i],
                        "name": tc["name"],
                    })
            completion_msgs.append({
                "role": "assistant",
                "content": self._all_text,
            })

            rewards = self._reward_fn(
                completions=[completion_msgs],
                ground_truth_flag=[self._ground_truth_flag],
                optimal_steps=[self._optimal_steps],
            )
            return rewards[0] if rewards else 0.0

        # Fallback: binary flag reward
        return 1.0 if self._done else 0.0

    def close(self):
        """Close the episode and release resources."""
        if self._executor:
            self._executor.close()
        logger.debug("OpenCTFTextEnv closed (episode=%s)", self._episode_id)

    def get_metrics(self) -> Dict[str, Any]:
        """Return episode-level metrics."""
        return {
            "total_steps": self.turns,
            "total_tool_calls": len(self._tool_calls_history),
            "flag_found": self._done,
            "unique_tools": len(set(tc["name"] for tc in self._tool_calls_history)),
        }

    @staticmethod
    def aggregate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics across multiple episodes."""
        if not metrics:
            return {}
        n = len(metrics)
        return {
            "avg_steps": sum(m.get("total_steps", 0) for m in metrics) / n,
            "avg_tool_calls": sum(m.get("total_tool_calls", 0) for m in metrics) / n,
            "flag_found_rate": sum(1 for m in metrics if m.get("flag_found")) / n,
            "avg_unique_tools": sum(m.get("unique_tools", 0) for m in metrics) / n,
            "num_episodes": n,
        }
