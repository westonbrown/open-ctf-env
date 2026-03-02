#!/usr/bin/env python3
"""Curate SFT JSONL data with benchmark-aware filters.

This script is benchmark-agnostic: if a challenge registry is provided, samples
are matched via registry.resolve_id() and only resolved samples are retained.
Additional filters can remove oversized trajectories and malformed rows.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CurationStats:
    total: int = 0
    kept: int = 0
    dropped_invalid_json: int = 0
    dropped_missing_messages: int = 0
    dropped_unresolved_registry: int = 0
    dropped_over_max_chars: int = 0
    dropped_platform_mismatch: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input SFT JSONL path")
    parser.add_argument("--output", required=True, help="Output curated JSONL path")
    parser.add_argument(
        "--registry",
        default=None,
        help="Optional challenge registry YAML. If set, unresolved samples are dropped.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=220_000,
        help="Drop samples whose concatenated message content exceeds this size.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats only; do not write output file.",
    )
    parser.add_argument(
        "--platform",
        action="append",
        default=[],
        help="Keep only samples whose metadata.platform matches this value (repeatable).",
    )
    return parser.parse_args()


def _text_size(messages: Iterable[dict[str, Any]]) -> int:
    return sum(len(str(msg.get("content", ""))) for msg in messages if isinstance(msg, dict))


def _challenge_key(sample: dict[str, Any]) -> str:
    metadata = sample.get("metadata", {}) or {}
    return str(metadata.get("challenge") or metadata.get("challenge_id") or "").strip()


def _resolve_registry_challenge(sample: dict[str, Any], registry: Any) -> str | None:
    """Resolve sample challenge name/id to a canonical registry id."""
    metadata = sample.get("metadata", {}) or {}
    candidates = [
        metadata.get("challenge_id"),
        metadata.get("challenge"),
        metadata.get("challenge_name"),
        metadata.get("task_name"),
    ]
    for raw in candidates:
        value = str(raw or "").strip()
        if not value:
            continue
        resolved = registry.resolve_id(value)
        if resolved:
            return resolved
    return None


def curate_dataset(
    input_path: Path,
    output_path: Path,
    max_chars: int,
    registry_path: str | None = None,
    dry_run: bool = False,
    platforms: list[str] | None = None,
) -> tuple[CurationStats, Counter]:
    stats = CurationStats()
    challenge_counts: Counter = Counter()
    registry = None
    if registry_path:
        from open_ctf.challenges.registry import ChallengeRegistry
        registry = ChallengeRegistry(registry_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = None if dry_run else output_path.open("w", encoding="utf-8")
    platform_filter = {p.strip().lower() for p in (platforms or []) if p.strip()}
    try:
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                stats.total += 1

                try:
                    sample = json.loads(line)
                except json.JSONDecodeError:
                    stats.dropped_invalid_json += 1
                    continue

                messages = sample.get("messages", [])
                if not isinstance(messages, list) or not messages:
                    stats.dropped_missing_messages += 1
                    continue

                if platform_filter:
                    metadata = sample.get("metadata", {}) or {}
                    platform = str(metadata.get("platform", "")).strip().lower()
                    if platform not in platform_filter:
                        stats.dropped_platform_mismatch += 1
                        continue

                challenge = _challenge_key(sample)
                if registry is not None:
                    resolved = _resolve_registry_challenge(sample, registry)
                    if resolved is None:
                        stats.dropped_unresolved_registry += 1
                        continue
                    metadata = sample.get("metadata") or {}
                    metadata["challenge_id"] = resolved
                    sample["metadata"] = metadata
                    challenge = resolved

                if _text_size(messages) > max_chars:
                    stats.dropped_over_max_chars += 1
                    continue

                stats.kept += 1
                challenge_counts[challenge or "<unknown>"] += 1
                if out_f is not None:
                    out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    finally:
        if out_f is not None:
            out_f.close()

    return stats, challenge_counts


def main() -> None:
    args = parse_args()
    stats, challenge_counts = curate_dataset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        max_chars=args.max_chars,
        registry_path=args.registry,
        dry_run=args.dry_run,
        platforms=args.platform,
    )

    print(f"input: {args.input}")
    print(f"output: {args.output}")
    print(f"total={stats.total} kept={stats.kept}")
    print(
        "dropped="
        f"invalid_json:{stats.dropped_invalid_json}, "
        f"missing_messages:{stats.dropped_missing_messages}, "
        f"unresolved_registry:{stats.dropped_unresolved_registry}, "
        f"platform_mismatch:{stats.dropped_platform_mismatch}, "
        f"over_max_chars:{stats.dropped_over_max_chars}"
    )
    print(f"top_challenges={challenge_counts.most_common(20)}")


if __name__ == "__main__":
    main()
