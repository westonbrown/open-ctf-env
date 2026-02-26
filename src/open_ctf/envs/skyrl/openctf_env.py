"""SkyRL-Gym BaseTextEnv subclass bridging SkyRL to execution environments.

Each SkyRL agent loop gets its own env instance with a pluggable StepAgent
for tool parsing + execution. The default agent (DefaultStepAgent) preserves
the original behavior.

Architecture:
    SkyRL SkyRLGymGenerator -> agent_loop()
        -> env.init(prompt) -> agent.reset()
        -> env.step(action) -> agent.step(), compute reward via CTFReward
        -> env.close() -> agent.close()

The env receives raw LLM text output, delegates tool parsing + execution
to the StepAgent, and computes rewards from agent state.
"""

import ast
import importlib
import json
import logging
import os
import re
import shutil
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

# Qwen3.5 Coder XML: <tool_call><function=func_name><parameter=k>v</parameter>...</function></tool_call>
_QWEN35_CODER_PATTERN = re.compile(
    r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", re.DOTALL,
)
_QWEN35_PARAM_PATTERN = re.compile(
    r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL,
)

# Bare JSON fallback: {"name": "...", "arguments": {...}}
# Supports one level of nested braces in arguments (e.g. {"headers": {"X-UserId": "10052"}})
_BARE_JSON_PATTERN = re.compile(
    r'\{\s*"name"\s*:\s*"([^"]+)"\s*,\s*"arguments"\s*:\s*(\{(?:[^{}]|\{[^{}]*\})*\})\s*\}',
    re.DOTALL,
)

# Thinking block pattern: <think>...</think> (Qwen3.5, Qwen3, DeepSeek-R1, etc.)
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

# Python-style function call fallback (e.g. shell_command(command="ls -la"))
_PY_CALL_LINE_PATTERN = re.compile(r"^\s*([A-Za-z_]\w*)\((.*)\)\s*$")
_TOOL_ARG_ORDER: Dict[str, List[str]] = {
    "shell_command": ["command", "timeout"],
    "execute_command": ["command", "timeout"],
    "python_code": ["code", "timeout"],
    "read_file": ["file_path", "line_numbers"],
    "grep": ["pattern", "path", "include"],
    "file_search": ["pattern", "path"],
    "apply_patch": ["patch"],
    "flag_found": ["content"],
    "submit_flag": ["content"],
    "web_search": ["query"],
    "exec_command": ["cmd", "workdir", "yield_time"],
    "write_stdin": ["session_id", "chars", "yield_time"],
    "list_sessions": [],
    "close_session": ["session_id"],
}
_KNOWN_TOOL_NAMES = set(_TOOL_ARG_ORDER.keys())


def _coerce_scalar(raw: str) -> Any:
    """Best-effort parse of scalar argument text."""
    text = raw.strip()
    if not text:
        return ""
    try:
        return ast.literal_eval(text)
    except Exception:
        pass
    try:
        return json.loads(text)
    except Exception:
        return text


def _parse_python_style_calls(text: str) -> List[Dict[str, Any]]:
    """Parse python-style tool calls from assistant text.

    Supports lines like:
      shell_command(command="ls -la")
      read_file("/root/challenge/index.php")
      flag_found(content="HTB{...}")
    """
    calls: List[Dict[str, Any]] = []
    cleaned = text.replace("```python", "").replace("```json", "").replace("```", "")
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = line.lstrip("-*").strip()
        m = _PY_CALL_LINE_PATTERN.match(line)
        if not m:
            continue
        name = m.group(1).strip()
        if name not in _TOOL_ARG_ORDER:
            continue
        args_src = m.group(2).strip()
        args: Dict[str, Any] = {}

        # Parse kwargs/positional args robustly with AST first.
        call_src = f"{name}({args_src})"
        try:
            parsed = ast.parse(call_src, mode="eval")
            expr = parsed.body
            if isinstance(expr, ast.Call):
                positional = []
                for node in expr.args:
                    seg = ast.get_source_segment(call_src, node) or ""
                    positional.append(_coerce_scalar(seg))
                ordered = _TOOL_ARG_ORDER.get(name, [])
                for idx, value in enumerate(positional):
                    key = ordered[idx] if idx < len(ordered) else f"arg{idx}"
                    args[key] = value
                for kw in expr.keywords:
                    if kw.arg is None:
                        continue
                    seg = ast.get_source_segment(call_src, kw.value) or ""
                    args[kw.arg] = _coerce_scalar(seg)
        except Exception:
            # Fallback for malformed-but-common style: read_file(/path/file)
            if args_src:
                ordered = _TOOL_ARG_ORDER.get(name, [])
                key = ordered[0] if ordered else "arg0"
                args[key] = _coerce_scalar(args_src)

        calls.append({"name": name, "arguments": args})

    return calls


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    """Extract tool calls from LLM output text.

    Strips ``<think>...</think>`` blocks first to prevent regex confusion
    when thinking content contains tool-call-like patterns. The original
    text is not modified — only the copy used for parsing is cleaned.

    Supports Hermes JSON, Qwen3.5 Coder XML, GLM4 XML, and bare JSON formats.
    Returns list of {"name": str, "arguments": dict} dicts.
    """
    # Strip thinking blocks before parsing (model generates <think>...</think>
    # by default in Qwen3.5/Qwen3 thinking mode). Thinking content may contain
    # JSON, XML, or tool-call-like patterns that confuse the parsers.
    text = _THINK_PATTERN.sub("", text)

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
            if name in _KNOWN_TOOL_NAMES:
                tool_calls.append({"name": name, "arguments": args})
        except json.JSONDecodeError:
            continue

    if tool_calls:
        return tool_calls

    # 2. Qwen3.5 Coder XML format
    for m in _QWEN35_CODER_PATTERN.finditer(text):
        name = m.group(1).strip()
        args = {}
        for pm in _QWEN35_PARAM_PATTERN.finditer(m.group(2)):
            key = pm.group(1).strip()
            val = pm.group(2).strip()
            try:
                val = json.loads(val)
            except (ValueError, json.JSONDecodeError):
                pass
            args[key] = val
        if name in _KNOWN_TOOL_NAMES:
            tool_calls.append({"name": name, "arguments": args})

    if tool_calls:
        return tool_calls

    # 3. GLM-4 MoE XML format
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
        if name in _KNOWN_TOOL_NAMES:
            tool_calls.append({"name": name, "arguments": args})

    if tool_calls:
        return tool_calls

    # 4. Bare JSON fallback
    for m in _BARE_JSON_PATTERN.finditer(text):
        name = m.group(1)
        try:
            args = json.loads(m.group(2))
        except json.JSONDecodeError:
            args = {}
        if name in _KNOWN_TOOL_NAMES:
            tool_calls.append({"name": name, "arguments": args})

    if tool_calls:
        return tool_calls

    # 5. Python-style fallback (common in BoxPwnr-authored prompts)
    tool_calls = _parse_python_style_calls(text)
    if tool_calls:
        return tool_calls

    return tool_calls


# ---------------------------------------------------------------------------
# Agent class resolution
# ---------------------------------------------------------------------------

def _resolve_class(dotpath: Optional[str]):
    """Resolve a dotted path string to a class.

    Example: "my_module.MyAgent" -> <class my_module.MyAgent>

    Returns None if dotpath is None or empty.
    Raises ImportError/AttributeError if the path is invalid.
    """
    if not dotpath:
        return None
    parts = dotpath.rsplit(".", 1)
    if len(parts) == 2:
        module = importlib.import_module(parts[0])
        return getattr(module, parts[1])
    # Single name — try importing as module (unlikely for a class)
    return importlib.import_module(dotpath)


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
    """SkyRL-Gym BaseTextEnv for CTF challenges via pluggable StepAgent.

    Each instance manages one episode. Tool parsing + execution is delegated
    to a StepAgent (default: DefaultStepAgent). The env owns reward computation,
    tool schema injection, and SkyRL protocol compliance.

    SkyRL's ``make()`` merges registered kwargs (static config) with
    per-sample kwargs from the dataset:

    - **Static** (from ``register(kwargs=...)``)::

        reward_config: dict of CTFReward weight overrides
        agent_class: dotted path to StepAgent class (optional)
        agent_kwargs: dict of kwargs for StepAgent constructor (optional)

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
        self._base_max_turns = int(self.max_turns)
        self.turns = 0

        # Optional progressive horizon schedule:
        #   {"rounds": [12, 24, 40, 60], "step_interval": 80}
        # If set, max_turns is reduced/expanded by global_step stage.
        self._horizon_schedule: Optional[Dict[str, Any]] = (
            extras.get("horizon_schedule")
            or kwargs.get("horizon_schedule")
        )

        # Tool schemas — SkyRL uses these for prompt injection.
        from open_ctf.envs.skyrl.tool_groups import OPENCTF_TOOLS
        self.tools = OPENCTF_TOOLS
        self.tool_groups = []

        self._episode_id: Optional[str] = None
        self._done = False

        # Step-wise trajectory rewards: when enabled, per-step rewards
        # include small format-compliance and phase-progression signals
        # instead of returning 0.0 for all non-terminal steps.
        self._step_wise: bool = bool(
            extras.get("step_wise_trajectories")
            or kwargs.get("step_wise_trajectories", False)
        )

        # Track tool call count before each step for per-step reward.
        self._prev_tool_call_count: int = 0

        # Tool call format for prompt injection.
        # "hermes" (default): <tool_call>{"name": ..., "arguments": ...}</tool_call>
        # "qwen3_coder": <tool_call><function=name><parameter=k>v</parameter></function></tool_call>
        # "glm4":  <tool_call>func_name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
        self._tool_call_format: str = (
            extras.get("tool_call_format")
            or kwargs.get("tool_call_format", "hermes")
        )

        # Reconstruct reward function from serializable config dict.
        # If reward_config is missing/empty, use CTFReward defaults instead of
        # silently falling back to binary terminal reward.
        reward_config = kwargs.get("reward_config", extras.get("reward_config", {}))
        if reward_config is None:
            reward_config = {}
        if not isinstance(reward_config, dict):
            logger.warning(
                "Invalid reward_config type %s; using CTFReward defaults.",
                type(reward_config).__name__,
            )
            reward_config = {}
        try:
            from open_ctf.training.online_rl.step_reward import create_reward_fn
            self._reward_fn = create_reward_fn({"reward": reward_config})
        except Exception as exc:
            logger.warning("Failed to create reward function: %s — using binary fallback", exc)
            self._reward_fn = None

        # Per-episode data (set from dataset sample extras)
        self._ground_truth_flag: Optional[str] = extras.get("ground_truth_flag")
        self._optimal_steps: Optional[int] = extras.get("optimal_steps")
        self._challenge_id: Optional[str] = extras.get("challenge_id")
        self._infra_type: str = str(
            extras.get("infra_type")
            or kwargs.get("infra_type")
            or "docker"
        )
        self._path_hint: Optional[str] = (
            extras.get("path_hint")
            or kwargs.get("path_hint")
        )

        # Target URL for the challenge
        raw_target = extras.get("target", kwargs.get("target", ""))
        self._target: str = str(raw_target).strip() if raw_target else ""

        # Trajectory logging (optional, for post-run analysis).
        # The output_dir is a plain string path, not a logger object, because
        # this must be serializable through Ray. Each env worker creates its
        # own TrajectoryLogger instance writing to the shared output dir.
        self._trajectory_output_dir: Optional[str] = (
            kwargs.get("trajectory_output_dir")
            or extras.get("trajectory_output_dir")
        )
        self._trajectory_logger = None
        if self._trajectory_output_dir:
            try:
                from open_ctf.training.online_rl.trajectory_logger import TrajectoryLogger
                # Reuse the same TensorBoard logdir set by _resolve_skyrl_logger()
                # so environment scalars appear alongside SkyRL training metrics.
                tb_dir = os.environ.get("TENSORBOARD_LOGDIR")
                self._trajectory_logger = TrajectoryLogger(
                    self._trajectory_output_dir,
                    enabled=True,
                    tensorboard_dir=tb_dir,
                )
            except Exception as exc:
                logger.warning("Failed to create TrajectoryLogger: %s", exc)

        # Global step counter (set from env extras per sample).
        self._global_step: int = extras.get("global_step", 0)
        self._generation_idx: int = extras.get("generation_idx", 0)
        # Challenge metadata for logging
        self._category: Optional[str] = extras.get("category")
        self._difficulty: Optional[str] = extras.get("difficulty")
        # Prompt messages for logging (set in init())
        self._prompt_messages: Optional[list] = None
        self._last_rollout_status: str = "ok"
        self._status_counts: Dict[str, int] = {}
        self._timing_totals: Dict[str, float] = {
            "parse_s": 0.0,
            "execute_s": 0.0,
            "total_s": 0.0,
        }
        # Rollout quality filter: statuses that should not contribute reward.
        hard_mask_statuses = (
            extras.get("hard_mask_statuses")
            or kwargs.get("hard_mask_statuses")
            or ["infra_unreachable", "target_mismatch", "parser_error", "tool_timeout"]
        )
        self._hard_mask_statuses = {str(s).strip() for s in hard_mask_statuses if str(s).strip()}
        # Optional warmup mode: clamp non-positive rollout rewards to 0 during
        # early steps, which avoids strong negative gradients before the policy
        # learns to issue stable tool calls.
        raw_positive_only_until_step = (
            extras.get("positive_only_until_step")
            if extras.get("positive_only_until_step") is not None
            else kwargs.get("positive_only_until_step", 0)
        )
        try:
            self._positive_only_until_step = int(raw_positive_only_until_step or 0)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid positive_only_until_step=%r; defaulting to 0.",
                raw_positive_only_until_step,
            )
            self._positive_only_until_step = 0

        raw_positive_only_reward_floor = (
            extras.get("positive_only_reward_floor")
            if extras.get("positive_only_reward_floor") is not None
            else kwargs.get("positive_only_reward_floor", 0.0)
        )
        try:
            self._positive_only_reward_floor = float(raw_positive_only_reward_floor or 0.0)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid positive_only_reward_floor=%r; defaulting to 0.0.",
                raw_positive_only_reward_floor,
            )
            self._positive_only_reward_floor = 0.0

        # Resolve and create the pluggable StepAgent.
        # agent_class is a dotted path string (Ray-safe serialization).
        agent_class_path = kwargs.get("agent_class") or extras.get("agent_class")
        agent_cls = _resolve_class(agent_class_path)
        if agent_cls is None:
            from open_ctf.agent.default_agent import DefaultStepAgent
            agent_cls = DefaultStepAgent

        agent_kwargs = dict(kwargs.get("agent_kwargs") or extras.get("agent_kwargs") or {})
        # Pass executor config to agent if not already specified
        if "executor_type" not in agent_kwargs:
            agent_kwargs["executor_type"] = extras.get(
                "executor_type", kwargs.get("executor_type", "subprocess")
            )

        self._agent = agent_cls(**agent_kwargs)

        # Let agent override tool schemas if it provides them
        agent_tools = getattr(self._agent, "tools", None)
        if agent_tools is not None:
            self.tools = agent_tools

    @property
    def _all_text(self) -> str:
        """Proxy to agent's all_text for backward compatibility."""
        return getattr(self._agent, "all_text", "")

    @property
    def _tool_calls_history(self) -> list:
        """Proxy to agent's tool_calls_history for backward compatibility."""
        return getattr(self._agent, "tool_calls_history", [])

    @property
    def _tool_outputs(self) -> list:
        """Proxy to agent's tool_outputs for backward compatibility."""
        return getattr(self._agent, "tool_outputs", [])

    def init(self, prompt: ConversationType) -> tuple:
        """Initialize episode: reset agent, return prompt with tool schemas.

        Args:
            prompt: Initial conversation (system + user messages).

        Returns:
            (prompt, metadata) — prompt with tool schemas injected, metadata has episode_id.
        """
        # Progressive horizon: clamp max turns by stage if schedule is enabled.
        self.max_turns = self._resolve_max_turns_for_step(self._global_step)
        # For static challenges, stage challenge assets into /root/challenge.
        if self._infra_type == "static":
            self._prepare_static_workspace()
        # Avoid empty target fallback to localhost for static rows.
        if not self._target:
            if self._infra_type == "static":
                self._target = "file:///root/challenge/"
            else:
                self._target = os.getenv("CHALLENGE_TARGET", "http://localhost:8080")
        self._agent.reset(
            target=self._target,
            ground_truth_flag=self._ground_truth_flag or "",
            max_steps=self.max_turns,
        )

        self._episode_id = None  # no longer tracked via server
        self.turns = 0
        self._done = False
        self._prev_tool_call_count = 0
        self._last_rollout_status = "ok"
        self._status_counts = {}
        self._timing_totals = {"parse_s": 0.0, "execute_s": 0.0, "total_s": 0.0}

        # Inject tool schemas into the system message so the model knows
        # what tools are available during GRPO rollouts. SkyRL's generator
        # may pass tools= to apply_chat_template for structured tool calling,
        # but for text-based parsing the model needs schemas in the prompt.
        prompt = self._inject_tool_schemas(prompt)

        # Capture prompt for trajectory logging (shallow copy to avoid mutation).
        self._prompt_messages = list(prompt)

        logger.debug(
            "OpenCTFTextEnv initialized: challenge=%s",
            self._challenge_id,
        )
        return prompt, {"episode_id": self._episode_id}

    def _resolve_max_turns_for_step(self, global_step: int) -> int:
        """Return effective max turns for current training step."""
        schedule = self._horizon_schedule
        if not isinstance(schedule, dict):
            return self._base_max_turns
        rounds_raw = schedule.get("rounds")
        if not isinstance(rounds_raw, list):
            return self._base_max_turns
        rounds = []
        for item in rounds_raw:
            try:
                v = int(item)
            except (TypeError, ValueError):
                continue
            if v > 0:
                rounds.append(v)
        if not rounds:
            return self._base_max_turns
        try:
            step_interval = int(schedule.get("step_interval", 1))
        except (TypeError, ValueError):
            step_interval = 1
        step_interval = max(1, step_interval)
        stage = max(0, int(global_step) // step_interval)
        if stage >= len(rounds):
            stage = len(rounds) - 1
        return max(1, min(self._base_max_turns, int(rounds[stage])))

    def _resolve_static_source_path(self) -> Optional[str]:
        """Resolve static challenge source path from path_hint + known roots."""
        hint = (self._path_hint or "").strip()
        if not hint:
            return None
        if os.path.isabs(hint) and os.path.exists(hint):
            return hint

        roots: List[str] = []
        env_roots = os.getenv("OPENCTF_BENCHMARK_ROOTS", "")
        if env_roots:
            roots.extend([p.strip() for p in env_roots.split(":") if p.strip()])
        roots.extend(
            [
                "/workspace/benchmarks/cybench",
                "/workspace/cybench",
                "/workspace/cybench-patched",
                "/workspace/open-ctf-env",
            ]
        )

        candidates: List[str] = []
        for root in roots:
            candidates.append(os.path.join(root, hint))
            if hint.startswith("benchmark/"):
                candidates.append(os.path.join(root, hint[len("benchmark/") :]))

        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _prepare_static_workspace(self) -> None:
        """Stage static challenge files into /root/challenge for tool access."""
        src = self._resolve_static_source_path()
        target_dir = "/root/challenge"
        self._target = "file:///root/challenge/"
        if not src:
            logger.warning(
                "Static challenge path not found for challenge=%s path_hint=%r",
                self._challenge_id,
                self._path_hint,
            )
            return

        # Prefer release/challenge payload directories when present.
        payload = src
        for subdir in ("release", "challenge", "dist"):
            candidate = os.path.join(src, subdir)
            if os.path.isdir(candidate):
                payload = candidate
                break

        os.makedirs(target_dir, exist_ok=True)
        for name in os.listdir(target_dir):
            path = os.path.join(target_dir, name)
            try:
                if os.path.isdir(path) and not os.path.islink(path):
                    shutil.rmtree(path)
                else:
                    os.remove(path)
            except FileNotFoundError:
                continue

        for name in os.listdir(payload):
            src_path = os.path.join(payload, name)
            dst_path = os.path.join(target_dir, name)
            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_path)

        logger.info(
            "Prepared static workspace: challenge=%s source=%s payload=%s target=%s",
            self._challenge_id,
            src,
            payload,
            target_dir,
        )

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
            "qwen3_coder": (
                "Call tools using: <tool_call><function=tool_name>"
                "<parameter=param>value</parameter>"
                "</function></tool_call>"
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
            # Skip if tool schemas already present (check multiple variants:
            # "# Available Tools" from our injection, "Available tools:" from
            # GRPO training data system prompts, "<tools>" from Nanbeige/Qwen
            # native format).  Prevents double injection that wastes ~800
            # context tokens.
            has_tools = (
                "# Available Tools" in sys_content
                or "Available tools:" in sys_content
                or "Available tools\n" in sys_content
                or "<tools>" in sys_content
            )
            has_format_instruction = (
                "Call tools using:" in sys_content
                or "<tool_call>" in sys_content
            )
            if not has_tools:
                prompt[0] = {
                    **prompt[0],
                    "content": sys_content + tools_block,
                }
            elif not has_format_instruction:
                prompt[0] = {
                    **prompt[0],
                    "content": (
                        sys_content
                        + "\n\n# Tool Call Format\n\n"
                        + fmt_instruction
                        + "\n"
                    ),
                }
        else:
            # No system message — prepend one with tool schemas
            prompt.insert(0, {
                "role": "system",
                "content": "You are a CTF agent with access to the following tools." + tools_block,
            })

        return prompt

    def step(self, action: str) -> Dict[str, Any]:
        """Process LLM output: delegate to agent, compute reward.

        Args:
            action: Raw LLM text output (may contain tool calls).

        Returns:
            BaseTextEnvStepOutput dict with observations, reward, done, metadata.
        """
        self.turns += 1

        # Snapshot tool call count before agent processes this step,
        # so we can compute how many new tool calls this step produced
        # (needed for step-wise trajectory rewards).
        self._prev_tool_call_count = len(
            getattr(self._agent, "tool_calls_history", [])
        )

        result = self._agent.step(action)
        info = dict(result.info or {})
        rollout_status = str(info.get("rollout_status", "ok") or "ok")
        self._last_rollout_status = rollout_status
        self._status_counts[rollout_status] = int(self._status_counts.get(rollout_status, 0)) + 1
        timing = info.get("timing", {})
        if isinstance(timing, dict):
            for key in ("parse_s", "execute_s", "total_s"):
                try:
                    self._timing_totals[key] += float(timing.get(key, 0.0))
                except (TypeError, ValueError):
                    continue

        # Sync done state from agent
        if result.done:
            self._done = hasattr(self._agent, "episode_done") and self._agent.episode_done

        done = result.done or self.turns >= self.max_turns
        reward = self._compute_reward(done)

        if done:
            # Minimal episode-level logging for diagnostics (avoid huge payloads).
            tool_calls = getattr(self._agent, "tool_calls_history", [])
            tool_outputs = getattr(self._agent, "tool_outputs", [])
            last_calls = [
                tc.get("name", "") if isinstance(tc, dict) else str(tc)
                for tc in tool_calls[-5:]
            ]
            last_outputs = [out[:200] for out in tool_outputs[-3:]]
            logger.info(
                "Episode done: challenge=%s target=%s steps=%s tool_calls=%s unique_tools=%s flag_found=%s "
                "last_tools=%s last_outputs=%s",
                self._challenge_id,
                self._target,
                self.turns,
                len(tool_calls),
                len(
                    set(
                        tc.get("name", "") if isinstance(tc, dict) else str(tc)
                        for tc in tool_calls
                    )
                ),
                getattr(self._agent, "episode_done", False),
                last_calls,
                last_outputs,
            )
            return {
                "observations": [],
                "reward": reward,
                "done": True,
                "metadata": info,
            }

        return {
            "observations": result.observations,
            "reward": reward,
            "done": False,
            "metadata": info,
        }

    def _compute_reward(self, done: bool) -> float:
        """Compute reward for the current step.

        Reads tool_calls_history, tool_outputs, all_text from the agent
        for reward computation. Falls back gracefully if the agent doesn't
        expose these attributes (custom agents may not).
        """
        # Read agent state (DefaultStepAgent exposes these; custom agents may not)
        tool_calls_history = getattr(self._agent, "tool_calls_history", [])
        tool_outputs = getattr(self._agent, "tool_outputs", [])
        all_text = getattr(self._agent, "all_text", "")
        episode_done = getattr(self._agent, "episode_done", False)

        if not done:
            from open_ctf.training.online_rl.step_reward import per_step_reward
            # Compute how many new tool calls were added by this step.
            step_tool_call_count = len(tool_calls_history) - self._prev_tool_call_count
            return per_step_reward(
                tool_calls_history, self.turns, self.max_turns,
                step_tool_call_count=max(0, step_tool_call_count),
                step_wise=self._step_wise,
            )

        # Terminal: compute full reward
        reward = 0.0
        breakdown = None

        if self._reward_fn is not None:
            completion_msgs = []
            for i, tc in enumerate(tool_calls_history):
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
                if i < len(tool_outputs):
                    completion_msgs.append({
                        "role": "tool",
                        "content": tool_outputs[i],
                        "name": tc["name"],
                    })
            completion_msgs.append({
                "role": "assistant",
                "content": all_text,
            })

            # Pass challenge metadata so reward function can adjust weights
            # (e.g. crypto/rev/forensics skip RECON->ENUM->EXPLOIT progression).
            reward_metadata = [{"task_category": self._category or "web", "success": episode_done}]

            # Use compute_with_breakdown if available for trajectory logging.
            if self._trajectory_logger and hasattr(self._reward_fn, "compute_with_breakdown"):
                results = self._reward_fn.compute_with_breakdown(
                    completions=[completion_msgs],
                    ground_truth_flag=[self._ground_truth_flag],
                    optimal_steps=[self._optimal_steps],
                    metadata=reward_metadata,
                )
                if results:
                    reward, breakdown = results[0]
            else:
                rewards = self._reward_fn(
                    completions=[completion_msgs],
                    ground_truth_flag=[self._ground_truth_flag],
                    optimal_steps=[self._optimal_steps],
                    metadata=reward_metadata,
                )
                reward = rewards[0] if rewards else 0.0
        else:
            # Fallback: binary flag reward
            reward = 1.0 if episode_done else 0.0

        # Quality filter: hard-mask known infra/runtime-invalid rollouts so they
        # do not inject misleading gradients into policy updates.
        if self._last_rollout_status in self._hard_mask_statuses:
            logger.info(
                "Hard-masking reward due to rollout_status=%s (challenge=%s step=%s)",
                self._last_rollout_status,
                self._challenge_id,
                self._global_step,
            )
            reward = 0.0
        elif (
            self._positive_only_until_step > 0
            and int(self._global_step) < self._positive_only_until_step
            and float(reward) <= self._positive_only_reward_floor
        ):
            logger.info(
                "Positive-only warmup masking reward=%s at step=%s (challenge=%s)",
                reward,
                self._global_step,
                self._challenge_id,
            )
            reward = 0.0

        # Log trajectory data when episode ends.
        self._log_episode_trajectory(
            reward_total=reward,
            reward_breakdown=breakdown,
            tool_calls_history=tool_calls_history,
            tool_outputs=tool_outputs,
            all_text=all_text,
            episode_done=episode_done,
            rollout_status=self._last_rollout_status,
        )

        return reward

    def _log_episode_trajectory(
        self,
        reward_total: float,
        reward_breakdown: Optional[Dict[str, float]],
        tool_calls_history: list,
        tool_outputs: list,
        all_text: str,
        episode_done: bool,
        rollout_status: str,
    ) -> None:
        """Log episode trajectory and update challenge scoreboard."""
        if not self._trajectory_logger:
            return

        try:
            # Build structured tool call list for logging
            logged_tool_calls = []
            for i, tc in enumerate(tool_calls_history):
                if not isinstance(tc, dict):
                    continue
                entry = {
                    "name": tc.get("name", ""),
                    "args": tc.get("arguments", ""),
                }
                if i < len(tool_outputs):
                    # Truncate long outputs to avoid huge JSONL files
                    output = tool_outputs[i]
                    if len(output) > 2000:
                        output = output[:2000] + "... [truncated]"
                    entry["output"] = output
                logged_tool_calls.append(entry)

            # Detect flag_submitted from tool calls
            flag_submitted = None
            for tc in tool_calls_history:
                if not isinstance(tc, dict):
                    continue
                if tc.get("name") in ("flag_found", "submit_flag"):
                    try:
                        args = tc.get("arguments", "")
                        if isinstance(args, str):
                            import json as _json
                            args = _json.loads(args)
                        flag_submitted = args.get("content", "") if isinstance(args, dict) else str(args)
                    except Exception:
                        flag_submitted = str(tc.get("arguments", ""))

            self._trajectory_logger.log_generation(
                global_step=self._global_step,
                generation_idx=self._generation_idx,
                challenge_id=self._challenge_id,
                category=self._category,
                difficulty=self._difficulty,
                target=self._target,
                prompt_messages=self._prompt_messages,
                model_output=all_text,
                tool_calls=logged_tool_calls,
                reward_total=reward_total,
                reward_breakdown=reward_breakdown,
                flag_found=episode_done,
                flag_submitted=flag_submitted,
                ground_truth_flag=self._ground_truth_flag,
                response_length=len(all_text),
                num_tool_calls=len(tool_calls_history),
                rollout_status=rollout_status,
                timing=dict(self._timing_totals),
                status_counts=dict(self._status_counts),
                max_turns=self.max_turns,
            )

            # Update challenge scoreboard
            if self._challenge_id:
                self._trajectory_logger.log_challenge_result(
                    challenge_id=self._challenge_id,
                    category=self._category,
                    difficulty=self._difficulty,
                    reward=reward_total,
                    flag_found=episode_done,
                )
        except Exception as exc:
            logger.warning("Trajectory logging failed: %s", exc)

    def close(self):
        """Close the episode and release resources."""
        if self._agent:
            self._agent.close()
        logger.debug("OpenCTFTextEnv closed (episode=%s)", self._episode_id)

    def get_metrics(self) -> Dict[str, Any]:
        """Return episode-level metrics."""
        tool_calls_history = getattr(self._agent, "tool_calls_history", [])
        episode_done = getattr(self._agent, "episode_done", False)
        return {
            "total_steps": self.turns,
            "total_tool_calls": len(tool_calls_history),
            "flag_found": episode_done,
            "unique_tools": len(
                set(
                    tc.get("name", "") if isinstance(tc, dict) else str(tc)
                    for tc in tool_calls_history
                )
            ),
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
