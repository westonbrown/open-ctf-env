"""GEPA prompt optimization stage (Stage 3).

Uses DSPy's GEPA optimizer (Genetic-Pareto reflective prompt evolution)
to evolve the CTF agent's system prompt without weight updates.

Pipeline position: SFT -> GRPO -> **GEPA** -> Deploy

- SFT teaches format and domain knowledge (weight updates)
- GRPO optimizes tool-calling efficiency (weight updates)
- GEPA evolves the system prompt instructions (no weight updates)

How GEPA improves over time:
  1. Seed prompt → evaluate agent on a minibatch of challenges → score each
  2. Reflection LM analyzes execution traces and proposes improved instructions
  3. New candidate prompts are evaluated on the next minibatch
  4. Pareto selection keeps prompts that are best on at least one challenge
  5. Repeat until budget is exhausted → output the best prompt

The reflection LM defaults to the same model as the agent. Both can run
on a local vLLM server — no cloud APIs required. For better mutations,
serve a stronger model for reflection on a separate port.

Outperforms GRPO by ~6% avg with 4-35x fewer rollouts (ICLR 2026 Oral).

Tools execute via ToolExecutor (direct subprocess, no HTTP server needed).

References:
  - Paper: https://arxiv.org/abs/2507.19457
  - GEPA library: https://github.com/gepa-ai/gepa
  - DSPy integration: dspy.teleprompt.gepa.GEPA
"""

import importlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Default seed prompt for GEPA prompt evolution.
# GEPA evolves this instruction to maximize task solve performance.
# Override with a domain-specific seed via gepa.seed_prompt in config.
SEED_PROMPT = """\
You are an autonomous agent authorized to solve the given task.

# APPROACH
- Break problems into smaller steps
- Be methodical: test one component at a time
- Prefer simple, debuggable actions
- After discovering useful information, act on it immediately
- Enumerate systematically before attempting solutions
- Use the available tools to gather information and take action
- When you find the answer, submit it immediately

# STRATEGY
- Start with reconnaissance: understand the problem space
- Enumerate: discover endpoints, files, parameters, or entry points
- Act: use what you've discovered to reach the goal
- Verify: confirm your answer before submitting"""


def _strip_react_boilerplate(instruction: str) -> str:
    """Strip DSPy ReAct agent framing, keeping only the CTF instructions.

    DSPy ReAct prepends boilerplate like::

        You are an Agent designed to complete any task by
        calling tools. At each step, ...

    followed by tool schemas and the original seed instructions.  We want
    just the CTF-specific part (evolved or original) — i.e. everything that
    came from our seed prompt or GEPA mutations.

    Strategy: look for known boilerplate markers and strip them.  If the
    instruction doesn't contain boilerplate, return it as-is.
    """
    if not instruction:
        return instruction

    # Pattern 1: DSPy ReAct wraps instructions in a larger agent prompt.
    # The original seed instructions typically appear after the tool list.
    # Look for the last occurrence of known seed markers.
    markers = [
        "You are an autonomous",
        "# APPROACH",
        "# STRATEGY",
        "# ENVIRONMENT",
    ]

    for marker in markers:
        idx = instruction.find(marker)
        if idx >= 0:
            # Found our seed content — extract from this point
            extracted = instruction[idx:].strip()
            if len(extracted) > 50:  # sanity check — not a trivial fragment
                return extracted

    # Pattern 2: If "You are an Agent designed to" appears, strip everything
    # before the actual CTF instructions.  Look for double-newline separation.
    agent_marker = "You are an Agent designed to"
    if agent_marker in instruction:
        # Try to find where the ReAct boilerplate ends and content begins.
        # DSPy typically puts the original instructions after tool schemas.
        # Look for sections that look like our domain content.
        sections = re.split(r"\n{2,}", instruction)
        content_sections = []
        in_boilerplate = True
        for section in sections:
            stripped = section.strip()
            if not stripped:
                continue
            # Boilerplate indicators
            if in_boilerplate and any(bp in stripped for bp in [
                "You are an Agent designed to",
                "At each step,",
                "Tool Name:",
                "Tool Description:",
                "Tool Arguments:",
            ]):
                continue
            in_boilerplate = False
            content_sections.append(stripped)

        if content_sections:
            return "\n\n".join(content_sections)

    # No boilerplate detected — return as-is
    return instruction.strip()


def _extract_first_url(text: str) -> Optional[str]:
    """Extract the first HTTP(S) URL and normalize to scheme://host[:port]."""
    if not text:
        return None
    match = re.search(r"https?://[^\s)]+", text)
    if not match:
        return None
    raw = match.group(0).rstrip(".,;:!?")
    parsed = urlparse(raw)
    if not parsed.scheme or not parsed.netloc:
        return raw
    return f"{parsed.scheme}://{parsed.netloc}"


# ---------------------------------------------------------------------------
# GEPA metric (wraps reward function for trajectory scoring + feedback)
# ---------------------------------------------------------------------------


def _build_metric(reward_fn):
    """Wrap a reward function as a GEPA feedback metric.

    Returns a callable matching the ``GEPAFeedbackMetric`` protocol::

        (gold, pred, trace, pred_name, pred_trace) -> ScoreWithFeedback

    Produces both a numeric score (from the reward function) and textual
    diagnostic feedback (tool trace, diversity analysis) that the GEPA
    reflection LM uses to propose improved instructions.
    """
    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

    def ctf_metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
        # DSPy Evaluate calls metric(gold, pred) with 2 args.
        # GEPA reflection calls metric(gold, pred, trace, pred_name, pred_trace)
        # with 5 args.  Accept both by defaulting trace/pred_name/pred_trace.

        # Reinitialize env for this challenge's target before scoring.
        # This ensures tool calls (if re-executed during reflection) hit
        # the correct challenge endpoint.
        target = gold.get("target", "")
        gt_flag = gold.get("ground_truth_flag", "")
        if target:
            from open_ctf.training.tools import init_env
            init_env(target=target, ground_truth=gt_flag)

        # Extract tool calls from the DSPy trace.
        # DSPy trace shape: [(predictor, inputs_dict, Prediction), ...]
        tool_calls = []
        if trace:
            for _predictor, _inputs, output in trace:
                tool_name = getattr(output, "next_tool_name", None)
                tool_args = getattr(output, "next_tool_args", None)
                if tool_name and tool_name != "finish":
                    if isinstance(tool_args, dict):
                        args_str = json.dumps(tool_args)
                    else:
                        args_str = str(tool_args or "{}")
                    tool_calls.append({"name": tool_name, "arguments": args_str})

        # Build completion in a format CTFReward._extract() understands.
        answer_text = getattr(pred, "answer", "") or ""
        completion = [{
            "content": answer_text,
            "tool_calls": [
                {
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"],
                    }
                }
                for tc in tool_calls
            ],
        }]

        gt_flag = gold.get("ground_truth_flag", None)
        optimal = gold.get("optimal_steps", None)

        scores = reward_fn(
            completions=completion,
            ground_truth_flag=[gt_flag],
            optimal_steps=[optimal],
        )
        score = scores[0] if scores else 0.0

        # Build diagnostic feedback for GEPA reflection.
        # GEPA uses this textual feedback to propose better instructions.
        # Include concrete trace data so the reflection LM knows exactly
        # what the agent did and where it got stuck.
        feedback_parts = []

        # Score bucket
        if score >= 0.8:
            feedback_parts.append(
                "Strong performance (score={:.2f}). Flag was captured.".format(score)
            )
        elif score >= 0.5:
            feedback_parts.append(
                "Moderate performance (score={:.2f}). Some phases completed "
                "but flag not captured.".format(score)
            )
        elif score >= 0.2:
            feedback_parts.append(
                "Weak performance (score={:.2f}). Agent attempted tool calls "
                "but lacked a structured approach.".format(score)
            )
        else:
            feedback_parts.append(
                "Very weak performance (score={:.2f}). Agent failed to engage "
                "effectively.".format(score)
            )

        # Concrete trace summary — show what commands ran and what happened.
        # This gives the reflection LM actionable context for mutations.
        # Generic: works with any tool set, not just BoxPwnr tools.
        if tool_calls:
            trace_lines = []
            for i, tc in enumerate(tool_calls[:10], 1):  # cap at 10 for brevity
                name = tc["name"]
                try:
                    args = json.loads(tc["arguments"])
                except (json.JSONDecodeError, TypeError):
                    args = tc["arguments"]
                # Generic formatting: show tool name + truncated arguments
                if isinstance(args, dict):
                    # Show first meaningful arg value for readability
                    first_val = next(
                        (str(v)[:100] for v in args.values() if v),
                        str(args)[:100],
                    )
                    trace_lines.append(f"  {i}. {name}: {first_val}")
                else:
                    trace_lines.append(f"  {i}. {name}({str(args)[:100]})")
            if len(tool_calls) > 10:
                trace_lines.append(f"  ... and {len(tool_calls) - 10} more calls")
            feedback_parts.append(
                "Tool trace:\n" + "\n".join(trace_lines)
            )

        # Tool diversity analysis (generic — works with any tool set)
        tool_names = {tc["name"] for tc in tool_calls}
        if len(tool_calls) == 0:
            feedback_parts.append(
                "No tool calls made. The instruction should encourage "
                "active use of available tools."
            )
        elif len(tool_names) <= 1:
            feedback_parts.append(
                f"Only used tool: {next(iter(tool_names))}. "
                "The instruction should encourage diverse tool usage."
            )
        elif len(tool_calls) > 25:
            feedback_parts.append(
                f"Used {len(tool_calls)} tool calls ({len(tool_names)} unique tools). "
                "The instruction should emphasize efficiency."
            )
        else:
            feedback_parts.append(
                f"Used {len(tool_calls)} tool calls across "
                f"{len(tool_names)} tools: {', '.join(sorted(tool_names))}."
            )

        feedback = "\n".join(feedback_parts)
        return ScoreWithFeedback(score=score, feedback=feedback)

    return ctf_metric


# ---------------------------------------------------------------------------
# Challenge data loader
# ---------------------------------------------------------------------------


def _extract_target_from_messages(messages: List[Dict[str, str]]) -> Optional[str]:
    """Extract target URL from user messages."""
    for msg in messages:
        if msg.get("role") == "user":
            target = _extract_first_url(msg.get("content", ""))
            if target:
                return target
    return None


def _load_challenges(
    data_path: str,
    max_samples: Optional[int] = None,
    registry=None,
) -> list:
    """Load challenges from GRPO JSONL as DSPy Examples.

    Each example contains:
    - ``challenge``: The CTF challenge description (from user message)
    - ``ground_truth_flag``: The expected flag (for scoring)
    - ``optimal_steps``: Minimum steps to solve (for efficiency scoring)
    - ``target``: Target URL for the challenge (extracted or from registry)
    - ``challenge_id``: Canonical challenge ID (if resolvable)

    Args:
        data_path: Path to GRPO JSONL file.
        max_samples: Maximum examples to load.
        registry: Optional ChallengeRegistry for target URL resolution.
    """
    import dspy

    examples = []
    with open(data_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages", [])
            metadata = row.get("metadata", {})

            # Extract challenge description from the user message
            challenge_text = ""
            for msg in messages:
                if msg.get("role") == "user":
                    challenge_text = msg.get("content", "")
                    break

            if not challenge_text:
                continue

            # Extract target URL from user messages (same as online_rl/runtime.py)
            target = _extract_target_from_messages(messages)
            if not target:
                target = metadata.get("target")

            # Resolve challenge ID and target from registry
            challenge_id = metadata.get("challenge_id") or metadata.get("challenge")
            if registry and challenge_id:
                resolved = registry.resolve_id(str(challenge_id))
                if resolved is not None:
                    challenge_id = resolved
                    if not target:
                        try:
                            target = registry.get_target_url(resolved)
                        except KeyError:
                            pass

            ex = dspy.Example(
                challenge=challenge_text,
                ground_truth_flag=row.get("ground_truth_flag", ""),
                optimal_steps=row.get("optimal_steps"),
                target=target or "",
                challenge_id=challenge_id or "",
            ).with_inputs("challenge")

            examples.append(ex)

            if max_samples and len(examples) >= max_samples:
                break

    return examples


# ---------------------------------------------------------------------------
# CTFAgent -> DSPy Module adapter (for --agent flag)
# ---------------------------------------------------------------------------


def _import_class(dotpath: str):
    """Import a class from a dotted path like ``my_module.MyClass``."""
    module_path, _, cls_name = dotpath.rpartition(".")
    if not module_path:
        raise ImportError(f"Invalid dotpath (need module.Class): {dotpath!r}")
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


class CTFAgentDSPyAdapter:
    """Wraps any CTFAgent as a DSPy Module for GEPA optimization.

    GEPA evolves the ``instructions`` field on the internal predictor.
    Each forward() call delegates to ``agent.solve()`` with the evolved
    prompt prepended to the challenge text.
    """

    def __init__(self, agent, seed_prompt: str = ""):
        import dspy

        self._agent = agent
        # The predictor's instructions field is what GEPA evolves.
        self._predictor = dspy.Predict(
            dspy.make_signature(
                {"challenge": dspy.InputField(), "answer": dspy.OutputField()},
                instructions=seed_prompt,
            )
        )

    def named_predictors(self):
        """Yield (name, predictor) pairs — GEPA iterates over these."""
        yield ("ctf_agent_predictor", self._predictor)

    def __call__(self, challenge: str = "", **kwargs):
        return self.forward(challenge=challenge, **kwargs)

    def forward(self, challenge: str = "", **kwargs):
        import dspy

        # Build the prompt GEPA is evolving.
        evolved_prompt = self._predictor.signature.instructions

        # Prepend evolved instructions to the challenge text so the agent
        # sees GEPA's optimized framing even if it doesn't accept system_prompt.
        augmented_challenge = f"{evolved_prompt}\n\n{challenge}" if evolved_prompt else challenge

        # Extract target URL from challenge text.
        target = _extract_first_url(challenge) or ""

        result = self._agent.solve(
            challenge=augmented_challenge,
            target=target,
            ground_truth_flag=kwargs.get("ground_truth_flag", ""),
        )
        return dspy.Prediction(answer=result.flag or "")

    def save(self, path: str):
        """Save just the optimized prompt (the agent itself is stateless)."""
        Path(path).mkdir(parents=True, exist_ok=True)
        prompt_path = Path(path) / "instructions.txt"
        prompt_path.write_text(self._predictor.signature.instructions)


# ---------------------------------------------------------------------------
# Environment-aware ReAct wrapper
# ---------------------------------------------------------------------------


class _EnvAwareReAct:
    """Wraps a DSPy ReAct module to initialize the ToolExecutor before each episode.

    The core problem: ``init_env()`` is called once at module setup with empty
    ``ground_truth``, so ``flag_found()`` always rejects submitted flags.
    This wrapper intercepts each ``__call__`` to:

    1. Look up the ground-truth flag for this challenge's target URL
    2. Call ``mark_step_begin(ground_truth=...)`` to reset episode state and
       set the correct flag for verification
    3. Delegate to the inner ReAct module

    All DSPy introspection methods (``named_predictors``, ``predictors``,
    ``parameters``, ``save``, ``load``) delegate to the inner module so GEPA
    can still inspect and evolve the prompt.
    """

    def __init__(self, inner: "dspy.ReAct", challenge_flags: Dict[str, str]):
        self._inner = inner
        self._challenge_flags = challenge_flags

    def _resolve_ground_truth(self, challenge: str) -> str:
        """Resolve ground-truth flag from target URL or challenge text."""
        # Try to extract target URL from the challenge text
        target = _extract_first_url(challenge)
        if target and target in self._challenge_flags:
            return self._challenge_flags[target]

        # Fallback: match by challenge text prefix
        prefix = challenge[:128] if challenge else ""
        if prefix in self._challenge_flags:
            return self._challenge_flags[prefix]

        # Last resort: if we only have one challenge, use its flag
        flags = list(set(self._challenge_flags.values()))
        if len(flags) == 1:
            return flags[0]

        return ""

    def __call__(self, challenge: str = "", **kwargs):
        from open_ctf.training.tools import mark_step_begin

        gt_flag = self._resolve_ground_truth(challenge)
        if gt_flag:
            mark_step_begin(ground_truth=gt_flag)
            logger.debug(
                "Episode initialized: target extracted, ground_truth=%s...%s",
                gt_flag[:6], gt_flag[-4:],
            )
        else:
            mark_step_begin()
            logger.warning(
                "No ground_truth found for challenge (first 80 chars: %s)",
                challenge[:80],
            )

        return self._inner(challenge=challenge, **kwargs)

    # --- DSPy introspection delegation ---
    # GEPA needs these to inspect and evolve the prompt on the inner ReAct.

    def named_predictors(self):
        return self._inner.named_predictors()

    def predictors(self):
        return self._inner.predictors()

    def parameters(self):
        if hasattr(self._inner, "parameters"):
            return self._inner.parameters()
        return []

    def named_parameters(self):
        if hasattr(self._inner, "named_parameters"):
            return self._inner.named_parameters()
        return []

    def save(self, path, *args, **kwargs):
        return self._inner.save(path, *args, **kwargs)

    def load(self, path, *args, **kwargs):
        return self._inner.load(path, *args, **kwargs)

    def __deepcopy__(self, memo):
        import copy
        return _EnvAwareReAct(
            copy.deepcopy(self._inner, memo),
            self._challenge_flags.copy(),
        )

    def __getattr__(self, name):
        """Delegate any other attribute access to the inner ReAct module."""
        return getattr(self._inner, name)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_gepa(
    model_id: str,
    data_path: str,
    output_dir: str,
    config: Dict[str, Any],
    reflection_model: Optional[str] = None,
    budget: str = "medium",
    val_data_path: Optional[str] = None,
    max_samples: Optional[int] = None,
    challenge_registry: Optional[str] = None,
    agent_class: Optional[str] = None,
    tools: Optional[list] = None,
) -> str:
    """Run GEPA prompt optimization.

    Evolves the agent's system prompt by reflecting on execution traces.
    Uses DSPy ReAct with pluggable tools and the CTFReward metric for
    scoring.

    Tools execute via ToolExecutor (direct subprocess, no HTTP server)
    by default. Pass custom ``tools`` to use any callable set.

    Args:
        model_id: LLM model identifier for ``dspy.LM``
            (e.g. ``openai/my-model`` with ``OPENAI_API_BASE``
            pointing at a local vLLM server).
        data_path: Path to JSONL data (challenges with flags).
        output_dir: Directory for optimized prompts and logs.
        config: Merged config dict (may contain ``gepa:`` section).
        reflection_model: LLM for GEPA reflection. Defaults to
            ``model_id`` (same model). For stronger mutations, point
            at a larger local model on a separate vLLM port.
        budget: GEPA budget preset (``light`` / ``medium`` / ``heavy``).
        val_data_path: Optional separate validation data path.
        max_samples: Maximum number of training examples to load.
        challenge_registry: Path to challenge registry YAML for target
            URL resolution.
        agent_class: Dotted path to a CTFAgent class. When set, wraps
            the agent in a DSPy Module adapter instead of using DSPy ReAct.
        tools: Optional list of callable tools for DSPy ReAct. When
            ``None``, loads the default tool set from ToolExecutor.

    Returns:
        Path to saved optimized prompt file.
    """
    import dspy
    from dspy.teleprompt.gepa import GEPA

    from open_ctf.rewards import CTFReward

    gepa_cfg = config.get("gepa", {})
    budget = budget or gepa_cfg.get("budget", "medium")

    logger.info("=" * 60)
    logger.info("GEPA PROMPT OPTIMIZATION (Stage 3)")
    logger.info("  Model:      %s", model_id)
    logger.info("  Reflection: %s", reflection_model or model_id)
    logger.info("  Data:       %s", data_path)
    logger.info("  Budget:     %s", budget)
    logger.info("  Output:     %s", output_dir)
    if agent_class:
        logger.info("  Agent:      %s", agent_class)
    if challenge_registry:
        logger.info("  Registry:   %s", challenge_registry)
    logger.info("=" * 60)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Configure DSPy LM ------------------------------------------------
    lm = dspy.LM(model=model_id, temperature=0.7, max_tokens=4096)
    dspy.configure(lm=lm)

    # Reflection LM — defaults to same model (no cloud APIs needed).
    # Override via --reflection-model CLI flag or gepa.reflection_model in config.
    ref_model = reflection_model or gepa_cfg.get("reflection_model") or model_id
    reflection_lm = dspy.LM(model=ref_model, temperature=1.0, max_tokens=32000)

    # --- Load challenge registry (if provided) ----------------------------
    registry = None
    if challenge_registry:
        from open_ctf.challenges.registry import ChallengeRegistry
        registry = ChallengeRegistry(challenge_registry)
        logger.info("Challenge registry loaded: %d challenges", len(registry))

    # --- Load challenge data -----------------------------------------------
    max_n = max_samples or gepa_cfg.get("max_samples")
    trainset = _load_challenges(data_path, max_samples=max_n, registry=registry)
    valset = None
    if val_data_path and Path(val_data_path).exists():
        valset = _load_challenges(val_data_path, registry=registry)

    logger.info("Loaded %d training examples", len(trainset))
    targets_found = sum(1 for ex in trainset if ex.get("target"))
    logger.info("  %d/%d have target URLs", targets_found, len(trainset))
    if valset:
        logger.info("Loaded %d validation examples", len(valset))

    # --- Build CTF agent ---------------------------------------------------
    seed = gepa_cfg.get("seed_prompt") or SEED_PROMPT

    if agent_class:
        # Wrap a CTFAgent in a DSPy Module adapter.
        AgentCls = _import_class(agent_class)
        ctf_agent = AgentCls()
        agent = CTFAgentDSPyAdapter(agent=ctf_agent, seed_prompt=seed)
        logger.info("Using CTFAgent adapter: %s", agent_class)
    else:
        # Default: DSPy ReAct with direct tool execution.
        if tools is not None:
            # Custom tools provided by caller — use directly.
            logger.info("Using %d custom tools", len(tools))
        else:
            # Fall back to default ToolExecutor tools.
            from open_ctf.training.tools import get_all_tools, init_env
            init_env()
            tools = get_all_tools()
            logger.info("Tools initialized (default ToolExecutor, %d tools)", len(tools))

        class CTFAgentSignature(dspy.Signature):
            """Placeholder instructions (replaced by seed prompt below)."""

            challenge: str = dspy.InputField(
                desc="CTF challenge description and target information",
            )
            answer: str = dspy.OutputField(
                desc="The captured flag or final answer",
            )

        # max_iters controls how many tool calls per ReAct episode.
        # Each iteration = 1 LLM call + 1 tool execution, so this directly
        # affects latency.  Default 15; override via gepa.max_iters in config.
        inner_react = dspy.ReAct(
            signature=CTFAgentSignature.with_instructions(seed),
            tools=tools,
            max_iters=gepa_cfg.get("max_iters", 15),
        )

        # Build target→ground_truth lookup so the wrapper can initialize the
        # ToolExecutor with the correct flag BEFORE each ReAct episode.
        challenge_flags = {}
        for ex in trainset:
            target = ex.get("target", "")
            gt = ex.get("ground_truth_flag", "")
            challenge_text = ex.get("challenge", "")
            if gt:
                if target:
                    challenge_flags[target] = gt
                # Also key by challenge text prefix (128 chars) for fallback
                if challenge_text:
                    challenge_flags[challenge_text[:128]] = gt

        agent = _EnvAwareReAct(inner_react, challenge_flags)

    # --- Build metric ------------------------------------------------------
    reward_fn = CTFReward()
    metric = _build_metric(reward_fn)

    # --- Run GEPA ----------------------------------------------------------
    num_threads = gepa_cfg.get("num_threads", 1)

    # Budget: prefer explicit max_metric_calls (from config or small datasets),
    # fall back to auto preset.  auto="light" with default minibatch_size=35
    # produces ~736 rollouts even for 1 challenge — far too many for a smoke test.
    max_metric_calls = gepa_cfg.get("max_metric_calls")
    if max_metric_calls is None and len(trainset) <= 3:
        # Small dataset heuristic: each metric call = a full ReAct agent loop
        # (up to max_iters LLM calls + tool executions).  auto="light" with
        # default minibatch_size=35 produces ~736 calls even for 1 challenge.
        # 10 per challenge is enough for 1-2 GEPA reflection cycles.
        max_metric_calls = max(10 * len(trainset), 10)
        logger.info(
            "Small dataset (%d examples) — using max_metric_calls=%d "
            "(override via gepa.max_metric_calls in config)",
            len(trainset),
            max_metric_calls,
        )

    budget_kwargs = {}
    if max_metric_calls is not None:
        budget_kwargs["max_metric_calls"] = int(max_metric_calls)
    else:
        budget_kwargs["auto"] = budget

    optimizer = GEPA(
        metric=metric,
        **budget_kwargs,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=gepa_cfg.get("reflection_minibatch_size", 3),
        log_dir=str(out_dir / "gepa_logs"),
        track_stats=True,
        seed=gepa_cfg.get("seed", 42),
        num_threads=num_threads,
    )

    optimized = optimizer.compile(
        student=agent,
        trainset=trainset,
        valset=valset,
    )

    # --- Save optimized prompt ---------------------------------------------
    prompt_path = out_dir / "optimized_prompt.txt"
    optimized_instruction = ""
    for _name, pred in optimized.named_predictors():
        optimized_instruction = pred.signature.instructions
        break

    # Strip DSPy ReAct boilerplate from the instruction.
    # DSPy prepends agent framing like "You are an Agent designed to..." and
    # tool schemas.  We only want the CTF-specific evolved instructions.
    optimized_instruction = _strip_react_boilerplate(optimized_instruction)

    prompt_path.write_text(optimized_instruction)
    logger.info("Optimized prompt saved to %s", prompt_path)

    # Also save the raw (unstripped) instruction for debugging
    raw_path = out_dir / "optimized_prompt_raw.txt"
    raw_instruction = ""
    for _name, pred in optimized.named_predictors():
        raw_instruction = pred.signature.instructions
        break
    raw_path.write_text(raw_instruction)
    logger.info("Raw instruction (with ReAct framing) saved to %s", raw_path)

    # Save detailed results
    if hasattr(optimized, "detailed_results") and optimized.detailed_results:
        results = optimized.detailed_results.to_dict()
        results_path = out_dir / "gepa_results.json"
        results_path.write_text(json.dumps(results, indent=2, default=str))
        logger.info("Detailed results saved to %s", results_path)
        logger.info(
            "Best score: %.4f (candidate %d of %d)",
            optimized.detailed_results.val_aggregate_scores[
                optimized.detailed_results.best_idx
            ],
            optimized.detailed_results.best_idx,
            len(optimized.detailed_results.candidates),
        )

    # Save the optimized DSPy module for reuse
    try:
        optimized.save(str(out_dir / "optimized_agent"))
        logger.info("Optimized DSPy module saved to %s", out_dir / "optimized_agent")
    except Exception as e:
        logger.warning("Could not save DSPy module: %s", e)

    return str(prompt_path)
