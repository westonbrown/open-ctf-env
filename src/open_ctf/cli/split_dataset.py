#!/usr/bin/env python3
"""Split converted BoxPwnr traces into SFT and GRPO datasets.

Usage:
    open-ctf split --input data/converted.jsonl
    open-ctf split --input data/converted.jsonl \\
        --sft-output data/sft.jsonl \\
        --grpo-output data/grpo.jsonl \\
        --max-grpo-tokens 32768
"""

import argparse
import logging
import sys
from pathlib import Path

from open_ctf.data.splitter import DatasetSplitter


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split converted BoxPwnr traces into SFT and GRPO datasets"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL from the converter",
    )
    parser.add_argument(
        "--sft-output",
        default="data/sft.jsonl",
        help="Output path for SFT dataset (default: data/sft.jsonl)",
    )
    parser.add_argument(
        "--grpo-output",
        default="data/grpo.jsonl",
        help="Output path for GRPO dataset (default: data/grpo.jsonl)",
    )
    parser.add_argument(
        "--max-grpo-tokens",
        type=int,
        default=32768,
        help="Max estimated tokens per GRPO trace (default: 32768)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not Path(args.input).exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    splitter = DatasetSplitter(max_grpo_tokens=args.max_grpo_tokens)
    stats = splitter.split(args.input, args.sft_output, args.grpo_output)

    # Print summary
    print("\n=== Dataset Split Summary ===\n")
    print(f"  Input traces:              {stats['total_input']}")
    print(f"  SFT output:                {stats['sft_count']}")
    print(f"  GRPO output:               {stats['grpo_count']}")
    print(f"  GRPO filtered (too long):  {stats['grpo_filtered']}")
    print(f"  GRPO missing flag:         {stats['grpo_missing_flag']}")
    print(f"  Avg turns (SFT):           {stats['avg_turns_sft']}")
    print(f"  Avg turns (GRPO):          {stats['avg_turns_grpo']}")

    print("\n  Tool distribution:")
    for tool, count in stats["tool_distribution"].items():
        print(f"    {tool:<25s} {count:>6d}")

    print("\n  Platform distribution:")
    for platform, count in stats["platform_distribution"].items():
        print(f"    {platform:<25s} {count:>6d}")

    print(f"\n  Written: {args.sft_output}")
    print(f"  Written: {args.grpo_output}")
    print()


if __name__ == "__main__":
    main()
