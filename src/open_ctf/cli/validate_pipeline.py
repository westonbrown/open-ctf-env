#!/usr/bin/env python3
"""Pipeline validation for Open CTF Environment.

Validates data format, reward functions, training scripts, tool registry,
model formatters, and reference projects WITHOUT requiring GPU or model weights.

Usage:
    open-ctf-validate
"""

import json
import py_compile
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def _ok(msg):
    print(f"  {GREEN}+{RESET} {msg}")


def _fail(msg, errors):
    print(f"  {RED}x{RESET} {msg}")
    errors.append(msg)


def _warn(msg, warnings):
    print(f"  {YELLOW}?{RESET} {msg}")
    warnings.append(msg)


def _section(msg):
    print(f"\n{BOLD}{'_'*60}\n  {msg}\n{'_'*60}{RESET}")


def main() -> None:
    errors = []
    warnings = []

    # Resolve paths: cli/ -> open_ctf/ -> src/ -> project root
    OCE_ROOT = Path(__file__).resolve().parent.parent.parent.parent
    SRC_DIR = OCE_ROOT / "src"
    DATA_DIR = OCE_ROOT / "data"

    # Also check sample data if main data doesn't exist
    SAMPLE_DIR = DATA_DIR / "sample"

    # -------------------------------------------------------------------
    # 1. Data files
    # -------------------------------------------------------------------
    _section("1. DATA FILES")

    data_checks = [
        ("SFT BoxPwnr", DATA_DIR / "sft_boxpwnr.jsonl"),
        ("GRPO BoxPwnr", DATA_DIR / "grpo_boxpwnr.jsonl"),
        ("SFT Sample", SAMPLE_DIR / "sft_sample.jsonl"),
        ("GRPO Sample", SAMPLE_DIR / "grpo_sample.jsonl"),
    ]

    for label, fpath in data_checks:
        if not fpath.exists():
            if "Sample" in label:
                _warn(f"{label}: Not found at {fpath}", warnings)
            else:
                _warn(f"{label}: Not found at {fpath.name} (expected after data conversion)", warnings)
            continue

        with open(fpath) as f:
            lines = f.readlines()
        count = len(lines)

        if count == 0:
            _fail(f"{label}: Empty file", errors)
            continue

        _ok(f"{label}: {count} samples at {fpath.name}")

        # Validate format of first 5 lines
        format_errors = 0
        for i, line in enumerate(lines[:5]):
            try:
                obj = json.loads(line)
                msgs = obj.get("messages")
                if not msgs or not isinstance(msgs, list):
                    format_errors += 1
                    continue
                roles = {m.get("role") for m in msgs}
                if "assistant" not in roles:
                    _warn(f"{label} line {i}: Missing assistant role", warnings)
            except json.JSONDecodeError:
                format_errors += 1

        if format_errors > 0:
            _fail(f"{label}: {format_errors}/5 lines have invalid JSON", errors)
        else:
            _ok(f"{label}: Format validated (first 5 lines)")

        # Check for ground_truth_flag in GRPO data
        if "grpo" in label.lower():
            try:
                first = json.loads(lines[0])
                if "ground_truth_flag" in first:
                    _ok(f"{label}: ground_truth_flag field present")
                else:
                    _warn(f"{label}: Missing ground_truth_flag field", warnings)
            except json.JSONDecodeError:
                pass

    # -------------------------------------------------------------------
    # 2. Reward function
    # -------------------------------------------------------------------
    _section("2. REWARD FUNCTION")

    try:
        from open_ctf.rewards.ctf_reward import CTFReward, SKILL_PATTERNS

        _ok("CTFReward imported successfully")

        reward = CTFReward()
        _ok(f"CTFReward instantiated (weights: flag={reward.flag_weight}, "
           f"grammar={reward.grammar_weight}, efficiency={reward.efficiency_weight}, "
           f"format={reward.format_weight})")

        mock_completions = [
            '{"name": "shell_command", "arguments": {"command": "nmap 10.10.10.1"}}\n'
            '{"name": "shell_command", "arguments": {"command": "gobuster dir"}}\n'
            '{"name": "shell_command", "arguments": {"command": "sqlmap -u target"}}\n'
            'FLAG{test_flag_12345}',
            "I cannot help with that request.",
        ]

        rewards = reward(
            completions=mock_completions,
            ground_truth_flag=["FLAG{test_flag_12345}", None],
            optimal_steps=[3, None],
        )

        if len(rewards) == 2:
            _ok(f"Reward batch scoring works: [{rewards[0]:.3f}, {rewards[1]:.3f}]")
        else:
            _fail(f"Expected 2 rewards, got {len(rewards)}", errors)

        if rewards[0] > rewards[1]:
            _ok(f"Success reward ({rewards[0]:.3f}) > failure reward ({rewards[1]:.3f})")
        else:
            _fail("Success reward should exceed failure reward", errors)

        for phase in ["recon", "enum", "exploit"]:
            if phase in SKILL_PATTERNS and len(SKILL_PATTERNS[phase]) > 0:
                _ok(f"SKILL_PATTERNS['{phase}']: {len(SKILL_PATTERNS[phase])} patterns")
            else:
                _fail(f"SKILL_PATTERNS missing or empty for '{phase}'", errors)

    except ImportError as e:
        _fail(f"CTFReward import failed: {e}", errors)
    except Exception as e:
        _fail(f"CTFReward test failed: {e}", errors)

    # -------------------------------------------------------------------
    # 3. Training scripts
    # -------------------------------------------------------------------
    _section("3. TRAINING SCRIPTS")

    train_files = {
        "src/open_ctf/cli/train.py": SRC_DIR / "open_ctf" / "cli" / "train.py",
        "src/open_ctf/training/sft.py": SRC_DIR / "open_ctf" / "training" / "sft.py",
        "src/open_ctf/training/grpo.py": SRC_DIR / "open_ctf" / "training" / "grpo.py",
    }

    for label, fpath in train_files.items():
        if not fpath.exists():
            _fail(f"{label}: Not found", errors)
            continue
        try:
            py_compile.compile(str(fpath), doraise=True)
            _ok(f"{label}: Syntax OK ({fpath.stat().st_size} bytes)")
        except py_compile.PyCompileError as e:
            _fail(f"{label}: Syntax error -> {e}", errors)

    config_file = OCE_ROOT / "configs" / "training.yaml"
    if config_file.exists():
        import yaml

        with open(config_file) as f:
            cfg = yaml.safe_load(f) or {}

        required_sections = ["model", "lora", "sft", "grpo", "output"]
        for s in required_sections:
            if s in cfg:
                _ok(f"configs/training.yaml: '{s}' section present")
            else:
                _fail(f"configs/training.yaml: Missing '{s}' section", errors)

        grpo = cfg.get("grpo", {})
        if grpo.get("beta", 1.0) <= 0.01:
            _ok(f"GRPO beta={grpo['beta']} (correctly low)")
        else:
            _warn(f"GRPO beta={grpo.get('beta', 'missing')} (should be <= 0.01)", warnings)

        if grpo.get("loss_type") == "dapo":
            _ok("GRPO loss_type=dapo")
        else:
            _warn(f"GRPO loss_type={grpo.get('loss_type', 'missing')} (should be dapo)", warnings)
    else:
        _fail("configs/training.yaml: Not found", errors)

    launch_script = OCE_ROOT / "scripts" / "launch_training.sh"
    if launch_script.exists():
        _ok(f"scripts/launch_training.sh: Found ({launch_script.stat().st_size} bytes)")
    else:
        _warn("scripts/launch_training.sh: Not found", warnings)

    # -------------------------------------------------------------------
    # 4. Tool registry
    # -------------------------------------------------------------------
    _section("4. TOOL REGISTRY")

    try:
        from open_ctf.formatters.tool_registry import BOXPWNR_TOOLS

        _ok(f"BOXPWNR_TOOLS imported: {len(BOXPWNR_TOOLS)} tools")

        tool_names = {t["function"]["name"] for t in BOXPWNR_TOOLS}
        required_tools = ["shell_command", "exec_command", "write_stdin", "flag_found"]
        for tool in required_tools:
            if tool in tool_names:
                _ok(f"  Tool '{tool}' present")
            else:
                _fail(f"  Tool '{tool}' missing from registry", errors)
    except ImportError as e:
        _fail(f"Tool registry import failed: {e}", errors)
    except Exception as e:
        _fail(f"Tool registry validation failed: {e}", errors)

    # -------------------------------------------------------------------
    # 5. Model formatters
    # -------------------------------------------------------------------
    _section("5. MODEL FORMATTERS")

    formatters_dir = SRC_DIR / "open_ctf" / "formatters"
    if formatters_dir.exists():
        formatter_files = list(formatters_dir.glob("*.py"))
        _ok(f"Formatters directory: {len(formatter_files)} files")

        try:
            from open_ctf.formatters import get_formatter

            test_models = [
                ("Qwen/Qwen3-8B", "Qwen3Formatter"),
                ("THUDM/glm-4-9b", "GLM4Formatter"),
                ("mistralai/Devstral-Small-2", "DevstralFormatter"),
            ]
            for model_id, expected_cls in test_models:
                f = get_formatter(model_id)
                cls_name = type(f).__name__
                if cls_name == expected_cls:
                    _ok(f"  {model_id} -> {cls_name}")
                else:
                    _fail(f"  {model_id} -> {cls_name} (expected {expected_cls})", errors)
        except ImportError as e:
            _warn(f"Formatters import failed: {e}", warnings)
        except Exception as e:
            _warn(f"Formatters validation failed: {e}", warnings)
    else:
        _warn("Formatters directory not found", warnings)

    # -------------------------------------------------------------------
    # 6. BoxPwnr reference
    # -------------------------------------------------------------------
    _section("6. BOXPWNR REFERENCE")

    boxpwnr_ref = OCE_ROOT / "references" / "boxpwnr"
    if boxpwnr_ref.exists():
        if (boxpwnr_ref / ".git").exists():
            _ok("BoxPwnr reference: Valid git repo")
        else:
            _warn("BoxPwnr reference exists but no .git directory", warnings)

        tools_file = boxpwnr_ref / "src" / "boxpwnr" / "tools" / "tools.py"
        if tools_file.exists():
            _ok(f"BoxPwnr tools.py: Found ({tools_file.stat().st_size} bytes)")
        else:
            _warn("BoxPwnr tools.py not found", warnings)
    else:
        _warn("BoxPwnr reference not found at references/boxpwnr/", warnings)

    # -------------------------------------------------------------------
    # 7. Evaluation harness
    # -------------------------------------------------------------------
    _section("7. EVALUATION HARNESS")

    eval_files = {
        "src/open_ctf/eval/evaluator.py": SRC_DIR / "open_ctf" / "eval" / "evaluator.py",
        "src/open_ctf/cli/evaluate.py": SRC_DIR / "open_ctf" / "cli" / "evaluate.py",
    }

    for label, fpath in eval_files.items():
        if not fpath.exists():
            _fail(f"{label}: Not found", errors)
            continue
        try:
            py_compile.compile(str(fpath), doraise=True)
            _ok(f"{label}: Syntax OK ({fpath.stat().st_size} bytes)")
        except py_compile.PyCompileError as e:
            _fail(f"{label}: Syntax error -> {e}", errors)

    challenges_file = OCE_ROOT / "configs" / "challenges.yaml"
    if challenges_file.exists():
        import yaml
        with open(challenges_file) as f:
            data = yaml.safe_load(f) or {}
        challenges = data.get("challenges", [])
        if challenges:
            _ok(f"configs/challenges.yaml: {len(challenges)} challenges defined")
        else:
            _warn("configs/challenges.yaml: No challenges defined", warnings)
    else:
        _warn("configs/challenges.yaml: Not found", warnings)

    # -------------------------------------------------------------------
    # 8. Agent runner
    # -------------------------------------------------------------------
    _section("8. AGENT RUNNER")

    runner_file = SRC_DIR / "open_ctf" / "agent" / "runner.py"
    if runner_file.exists():
        try:
            py_compile.compile(str(runner_file), doraise=True)
            _ok(f"agent/runner.py: Syntax OK ({runner_file.stat().st_size} bytes)")
        except py_compile.PyCompileError as e:
            _fail(f"agent/runner.py: Syntax error -> {e}", errors)

        try:
            from open_ctf.agent import AgentRunner
            _ok("AgentRunner imported successfully")
        except ImportError as e:
            _warn(f"AgentRunner import failed (may need BoxPwnr): {e}", warnings)
    else:
        _warn("agent/runner.py: Not found", warnings)

    # -------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------
    _section("SUMMARY")

    if not errors:
        print(f"\n  {GREEN}{BOLD}ALL CHECKS PASSED{RESET}")
        print(f"  Pipeline is ready for training.\n")
    else:
        print(f"\n  {RED}{BOLD}{len(errors)} ERROR(S) FOUND:{RESET}")
        for e in errors:
            print(f"    {RED}-{RESET} {e}")
        print()

    if warnings:
        print(f"  {YELLOW}{len(warnings)} warning(s){RESET}\n")

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
