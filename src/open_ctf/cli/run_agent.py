#!/usr/bin/env python3
"""Open CTF Agent Runner (BoxPwnr-based).

Runs BoxPwnr's Solver against CTF challenges from the command line.

Usage:
    open-ctf agent --platform xbow --target XBEN-003-24
    open-ctf agent --platform xbow --target XBEN-003-24 --model ollama/glm-4.7-flash
    open-ctf agent --check
"""

import argparse
import sys

from open_ctf.agent.runner import AgentRunner


def main():
    parser = argparse.ArgumentParser(
        description="Run BoxPwnr agent against CTF challenges"
    )

    parser.add_argument(
        "--platform", "-p",
        default="xbow",
        choices=["xbow", "local", "htb", "portswigger", "cybench"],
        help="Target platform (default: xbow)",
    )
    parser.add_argument(
        "--target", "-t",
        help="Target identifier (e.g. XBEN-003-24 for xbow)",
    )
    parser.add_argument(
        "--model", "-m",
        default="openrouter/openai/gpt-oss-120b",
        help="LLM model (default: openrouter/openai/gpt-oss-120b)",
    )
    parser.add_argument(
        "--strategy", "-s",
        default="chat_tools",
        choices=["chat", "chat_tools"],
        help="LLM strategy (default: chat_tools)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=50,
        help="Maximum conversation turns (default: 50)",
    )
    parser.add_argument(
        "--max-time",
        type=int,
        default=30,
        help="Maximum time in minutes (default: 30)",
    )
    parser.add_argument(
        "--max-cost",
        type=float,
        default=None,
        help="Maximum cost in USD",
    )
    parser.add_argument(
        "--traces-dir",
        default="./targets",
        help="Directory to store traces (default: ./targets)",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        choices=["minimal", "low", "medium", "high", "enabled", "disabled"],
        help="Reasoning effort for supported models (default: medium)",
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Number of solve attempts (default: 1)",
    )
    parser.add_argument(
        "--keep-container",
        action="store_true",
        help="Keep Docker container after completion",
    )
    parser.add_argument(
        "--keep-target",
        action="store_true",
        help="Keep target running after completion",
    )
    parser.add_argument(
        "--custom-instructions",
        type=str,
        default=None,
        help="Additional instructions appended to system prompt",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check that BoxPwnr components can be imported",
    )

    args = parser.parse_args()

    runner = AgentRunner(
        platform=args.platform,
        model=args.model,
        strategy=args.strategy,
        max_turns=args.max_turns,
        max_time=args.max_time,
        max_cost=args.max_cost,
        traces_dir=args.traces_dir,
        debug=args.debug,
        keep_container=args.keep_container,
        keep_target=args.keep_target,
        reasoning_effort=args.reasoning_effort,
        attempts=args.attempts,
        custom_instructions=args.custom_instructions,
    )

    if args.check:
        sys.exit(0 if runner.check_setup() else 1)

    if not args.target:
        parser.error("--target is required (use --check to verify setup)")

    try:
        runner.run(target=args.target)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Error: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
