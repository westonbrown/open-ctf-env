"""GEPA prompt optimization stage (Stage 3).

Uses DSPy's GEPA optimizer (Genetic-Pareto reflective prompt evolution)
to evolve the CTF agent's system prompt without weight updates.

Pipeline position: SFT -> GRPO -> **GEPA** -> Deploy

- SFT teaches format and domain knowledge (weight updates)
- GRPO optimizes tool-calling efficiency (weight updates)
- GEPA evolves the system prompt instructions (no weight updates)

GEPA reflects on execution traces to propose improved instructions,
using Pareto-based candidate selection to avoid local optima.
Outperforms GRPO by ~6% avg with 4-35x fewer rollouts (ICLR 2026 Oral).

Two modes:

  **Offline** (default): Stub tools return acknowledgment strings.
  The metric scores on trajectory STRUCTURE (format, progression,
  exploration, uniqueness) -- no live environment required.

  **Online** (with ``env_url``): Tools connect to an OpenEnv server.
  The metric can also score on flag capture and actual tool output.
  Requires ``open-ctf-env`` server running with challenges loaded.

References:
  - Paper: https://arxiv.org/abs/2507.19457
  - GEPA library: https://github.com/gepa-ai/gepa
  - DSPy integration: dspy.teleprompt.gepa.GEPA
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# BoxPwnr system prompt seed (from references/boxpwnr/prompts/generic_prompt.yaml)
# GEPA evolves this instruction to maximize CTF solve performance.
SEED_PROMPT = """\
You are an autonomous security testing agent authorized to conduct this assessment.

# ENVIRONMENT
You operate in a Kali Linux Docker container with comprehensive security tools.
Install additional tools with apt-get if needed.

# APPROACH
- Break problems into smaller steps
- Be methodical: test one component at a time
- Prefer simple, debuggable commands
- After discovering a vulnerability, exploit it immediately
- When you find credentials, try them right away
- Enumerate systematically (e.g., user IDs, endpoints, parameters)
- Use curl with custom headers for IDOR testing
- Check JavaScript source for hidden API endpoints and headers
- Follow the RECON -> ENUM -> EXPLOIT methodology

# STRATEGY
- Start with reconnaissance: nmap, whatweb, curl
- Enumerate endpoints: gobuster, ffuf, manual discovery
- Test default credentials on login forms
- Look for IDOR, SSRF, SQLi, command injection
- When you find a flag, submit it immediately with flag_found"""


# ---------------------------------------------------------------------------
# Stub tools for offline GEPA (no environment server needed)
# ---------------------------------------------------------------------------


def _make_stub_tools() -> list:
    """Create stub tool callables for offline GEPA.

    These tools have proper signatures and docstrings so DSPy ReAct
    can generate tool-call decisions. Stubs return acknowledgment
    strings -- the metric scores on trajectory structure, not output.
    """

    def shell_command(command: str, timeout: int = 30) -> str:
        """Run a shell command in the CTF attacker container and return output.

        Use this to execute reconnaissance, enumeration, and exploitation
        commands against the target challenge.

        Args:
            command: The shell command to execute (e.g. ``nmap -sV target``).
            timeout: Maximum execution time in seconds. Defaults to 30.

        Returns:
            The combined stdout and stderr output from the command.
        """
        return f"$ {command}\n[command executed, output returned]"

    def exec_command(cmd: str, workdir: str = "", yield_time: int = 5) -> str:
        """Start an interactive process in a PTY session and return its output.

        Args:
            cmd: Shell command to execute (e.g. ``python3``, ``ssh user@host``).
            workdir: Optional working directory.
            yield_time: Seconds to wait for initial output. Defaults to 5.

        Returns:
            Session ID and initial output from the process.
        """
        return f"Session started: {cmd}\n[session_id: 1]"

    def python_code(code: str, timeout: int = 120) -> str:
        """Execute Python code in the CTF attacker container.

        Args:
            code: Python source code to execute.
            timeout: Maximum execution time in seconds. Defaults to 120.

        Returns:
            The output from executing the Python code.
        """
        return f"[Python executed]\n{code[:100]}..."

    def read_file(file_path: str, line_numbers: bool = True) -> str:
        """Read the contents of a file in the CTF environment.

        Args:
            file_path: Path to the file to read.
            line_numbers: Whether to include line numbers. Defaults to True.

        Returns:
            The file contents.
        """
        return f"[Read] {file_path}\n[file contents returned]"

    def grep(pattern: str, path: str = ".", include: str = "") -> str:
        """Search for a regex pattern in files.

        Args:
            pattern: Regex pattern to search for.
            path: Directory to search in. Defaults to current directory.
            include: File pattern to include (e.g. ``*.py``). Optional.

        Returns:
            Grep results with line numbers.
        """
        return f"[Grep] pattern='{pattern}' path='{path}'\n[matches returned]"

    def file_search(pattern: str, path: str = ".") -> str:
        """Find files by name pattern.

        Args:
            pattern: Glob pattern to match file names (e.g. ``*.py``).
            path: Directory to search in. Defaults to current directory.

        Returns:
            List of matching file paths.
        """
        return f"[Find] {pattern} in {path}\n[files listed]"

    def apply_patch(patch: str) -> str:
        """Apply a patch to modify files in the CTF environment.

        Args:
            patch: The patch content to apply.

        Returns:
            Result of applying the patch.
        """
        return "[Patch applied]"

    def flag_found(content: str) -> str:
        """Submit a discovered flag for verification.

        Args:
            content: The flag string you discovered (e.g. ``FLAG{example}``).

        Returns:
            Verification result.
        """
        return f"Flag submitted: {content}"

    def web_search(query: str) -> str:
        """Search the web for information using DuckDuckGo.

        Args:
            query: The search query string.

        Returns:
            Search results with titles, snippets, and URLs.
        """
        return f"[Search] {query}\n[results returned]"

    return [
        shell_command,
        exec_command,
        python_code,
        read_file,
        grep,
        file_search,
        apply_patch,
        flag_found,
        web_search,
    ]


# ---------------------------------------------------------------------------
# GEPA metric (wraps CTFReward for trajectory scoring + textual feedback)
# ---------------------------------------------------------------------------


def _build_metric(reward_fn):
    """Wrap CTFReward as a GEPA feedback metric.

    Returns a callable matching the ``GEPAFeedbackMetric`` protocol::

        (gold, pred, trace, pred_name, pred_trace) -> ScoreWithFeedback
    """
    from dspy.teleprompt.gepa.gepa_utils import ScoreWithFeedback

    def ctf_metric(gold, pred, trace, pred_name, pred_trace):
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
        feedback_parts = []
        if score >= 0.8:
            feedback_parts.append(
                "Strong performance. The agent followed a structured "
                "exploitation methodology and used tools effectively."
            )
        elif score >= 0.5:
            feedback_parts.append(
                "Moderate performance. Some CTF phases were completed "
                "but the exploit was not fully successful."
            )
        elif score >= 0.2:
            feedback_parts.append(
                "Weak performance. The agent attempted some tool calls "
                "but lacked a structured approach."
            )
        else:
            feedback_parts.append(
                "Very weak performance. The agent failed to engage "
                "with the challenge effectively."
            )

        # Phase analysis
        phases_seen = set()
        for tc in tool_calls:
            phase = reward_fn._classify_phase(tc)
            if phase:
                phases_seen.add(phase)

        if not phases_seen:
            feedback_parts.append(
                "No CTF phases detected. The instruction should guide "
                "the agent to follow RECON->ENUM->EXPLOIT methodology."
            )
        else:
            missing = {"recon", "enum", "exploit"} - phases_seen
            if missing:
                feedback_parts.append(
                    f"Missing phases: {', '.join(sorted(missing))}. "
                    f"The instruction should emphasize "
                    f"{'reconnaissance' if 'recon' in missing else ''}"
                    f"{'enumeration' if 'enum' in missing else ''}"
                    f"{'exploitation' if 'exploit' in missing else ''}."
                )
            else:
                feedback_parts.append(
                    "All CTF phases (recon, enum, exploit) were attempted."
                )

        # Tool diversity
        tool_names = {tc["name"] for tc in tool_calls}
        if len(tool_calls) == 0:
            feedback_parts.append(
                "No tool calls made. The instruction should encourage "
                "active use of shell_command, python_code, and other tools."
            )
        elif len(tool_names) <= 1:
            feedback_parts.append(
                f"Only used {tool_names}. The instruction should encourage "
                "diverse tool usage across execution tiers."
            )
        elif len(tool_calls) > 25:
            feedback_parts.append(
                f"Used {len(tool_calls)} tool calls. The instruction "
                "should emphasize efficiency and focused exploitation."
            )

        feedback = " ".join(feedback_parts)
        return ScoreWithFeedback(score=score, feedback=feedback)

    return ctf_metric


# ---------------------------------------------------------------------------
# Challenge data loader
# ---------------------------------------------------------------------------


def _load_challenges(data_path: str, max_samples: Optional[int] = None) -> list:
    """Load challenges from GRPO JSONL as DSPy Examples.

    Each example contains:
    - ``challenge``: The CTF challenge description (from user message)
    - ``ground_truth_flag``: The expected flag (for scoring)
    - ``optimal_steps``: Minimum steps to solve (for efficiency scoring)
    """
    import dspy

    examples = []
    with open(data_path) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages", [])

            # Extract challenge description from the user message
            challenge_text = ""
            for msg in messages:
                if msg.get("role") == "user":
                    challenge_text = msg.get("content", "")
                    break

            if not challenge_text:
                continue

            ex = dspy.Example(
                challenge=challenge_text,
                ground_truth_flag=row.get("ground_truth_flag", ""),
                optimal_steps=row.get("optimal_steps"),
            ).with_inputs("challenge")

            examples.append(ex)

            if max_samples and len(examples) >= max_samples:
                break

    return examples


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
    env_url: Optional[str] = None,
) -> str:
    """Run GEPA prompt optimization.

    Evolves the CTF agent's system prompt by reflecting on execution
    traces. Uses DSPy ReAct with BoxPwnr tool schemas and the
    CTFReward metric for scoring.

    Args:
        model_id: LLM model identifier for ``dspy.LM``
            (e.g. ``openai/ctf-agent`` for local vLLM, or
            ``anthropic/claude-sonnet-4-20250514`` for API).
        data_path: Path to GRPO JSONL data (challenges with flags).
        output_dir: Directory for optimized prompts and logs.
        config: Merged config dict (may contain ``gepa:`` section).
        reflection_model: Strong LLM for GEPA reflection. Defaults
            to ``model_id``. Use a frontier model for best results
            (e.g. ``openai/gpt-5``).
        budget: GEPA budget preset (``light`` / ``medium`` / ``heavy``).
        val_data_path: Optional separate validation data path.
        max_samples: Maximum number of training examples to load.
        env_url: OpenEnv server URL for online mode. If not set,
            falls back to ``OPEN_CTF_ENV_URL`` env var, then offline
            mode with stub tools.

    Returns:
        Path to saved optimized prompt file.
    """
    import dspy
    from dspy.teleprompt.gepa import GEPA

    from open_ctf.rewards import CTFReward

    logger.info("=" * 60)
    logger.info("GEPA PROMPT OPTIMIZATION (Stage 3)")
    logger.info("  Model:      %s", model_id)
    logger.info("  Reflection: %s", reflection_model or model_id)
    logger.info("  Data:       %s", data_path)
    logger.info("  Budget:     %s", budget)
    logger.info("  Output:     %s", output_dir)
    logger.info("=" * 60)

    gepa_cfg = config.get("gepa", {})
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Configure DSPy LM ------------------------------------------------
    lm = dspy.LM(model=model_id, temperature=0.7, max_tokens=4096)
    dspy.configure(lm=lm)

    # Reflection LM (should be strong -- paper recommends frontier models)
    ref_model = reflection_model or model_id
    reflection_lm = dspy.LM(model=ref_model, temperature=1.0, max_tokens=32000)

    # --- Load challenge data -----------------------------------------------
    max_n = max_samples or gepa_cfg.get("max_samples")
    trainset = _load_challenges(data_path, max_samples=max_n)
    valset = None
    if val_data_path and Path(val_data_path).exists():
        valset = _load_challenges(val_data_path)

    logger.info("Loaded %d training examples", len(trainset))
    if valset:
        logger.info("Loaded %d validation examples", len(valset))

    # --- Build CTF agent with tools ----------------------------------------
    # Determine online vs offline mode
    env = env_url or os.environ.get("OPEN_CTF_ENV_URL")
    if env:
        from open_ctf.training.tools import get_all_tools, init_env
        init_env(env)
        tools = get_all_tools()
        logger.info("Online mode: tools connected to %s", env)
    else:
        tools = _make_stub_tools()
        logger.info("Offline mode: using stub tools (structural scoring only)")

    seed = gepa_cfg.get("seed_prompt", SEED_PROMPT)
    agent = dspy.ReAct(
        signature=dspy.Signature(
            {
                "challenge": dspy.InputField(
                    desc="CTF challenge description and target information",
                ),
            },
            seed,
        ).append(
            "answer",
            dspy.OutputField(desc="The captured flag or final answer"),
            type_=str,
        ),
        tools=tools,
        max_iters=gepa_cfg.get("max_iters", 20),
    )

    # --- Build metric ------------------------------------------------------
    reward_fn = CTFReward()
    metric = _build_metric(reward_fn)

    # --- Run GEPA ----------------------------------------------------------
    optimizer = GEPA(
        metric=metric,
        auto=budget,
        reflection_lm=reflection_lm,
        reflection_minibatch_size=gepa_cfg.get("reflection_minibatch_size", 3),
        log_dir=str(out_dir / "gepa_logs"),
        track_stats=True,
        seed=gepa_cfg.get("seed", 42),
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

    prompt_path.write_text(optimized_instruction)
    logger.info("Optimized prompt saved to %s", prompt_path)

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
