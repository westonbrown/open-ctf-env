#!/usr/bin/env python3
"""
Fix missing, CHECK, and generic ground_truth_flag values in GRPO and SFT data.

Strategy:
1. Cross-reference SFT data: copy valid flags from SFT for same challenge
2. Cross-reference GRPO self: copy valid flags from other GRPO traces of same challenge
3. Extract from trace content: search tool responses for real flag patterns
4. PortSwigger "CHECK" traces with success=True: set to "Congratulations, you solved the lab!"
5. PortSwigger "Complete" traces with success=True: set to "Congratulations, you solved the lab!"
6. Still missing after above: log them but keep (useful for non-flag reward components)
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# --- Configuration ---
DATA_DIR = Path(__file__).parent.parent / "data"
GRPO_PATH = DATA_DIR / "grpo.jsonl"
SFT_PATH = DATA_DIR / "sft.jsonl"

# Flags considered invalid/needing fix
INVALID_FLAGS = {None, "", "CHECK", "Complete", "FLAG{XSS}"}

# Known fake/placeholder flags to exclude when extracting from content
FAKE_FLAGS = {
    "FLAG{...}", "flag{...}", "HTB{...}", "picoCTF{...}",
    "FLAG{XSS}", "HTB{f4k3_fl4g_4_t3st1ng}",
    "picoCTF{f4k3_fl4g_f0r_t3st1ng}", "CTF{f4k3_fl4g_f0r_t3st1ng}",
    "picoCTF{sample_flag_for_testing}", "CTF{sample_flag_for_testing}",
    "FLAG{flag}", "flag{flag}", "FLAG{test}", "flag{test}",
}

# Real flag patterns (must have 3+ chars of actual content)
FLAG_REGEXES = [
    re.compile(r'picoCTF\{[a-zA-Z0-9_\-!@#$%^&*()+=,./?]{3,}\}'),
    re.compile(r'HTB\{[a-zA-Z0-9_\-!@#$%^&*()+=,./?]{3,}\}'),
    re.compile(r'FLAG\{[a-f0-9]{32,128}\}'),  # hex hash flags (XBOW style)
    re.compile(r'FLAG\{[a-zA-Z0-9_\-!@#$%^&*()+=,./?]{3,}\}'),
    re.compile(r'flag\{[a-zA-Z0-9_\-!@#$%^&*()+=,./?]{3,}\}'),
    re.compile(r'gctf\{[a-zA-Z0-9_\-!@#$%^&*()+=,./?]{3,}\}'),
    re.compile(r'hkcert22\{[a-zA-Z0-9_\-!@#$%^&*()+=,./?]{3,}\}'),
    re.compile(r'SEKAI\{[a-zA-Z0-9_\-!@#$%^&*()+=,./?]{3,}\}'),
    re.compile(r'THM\{[a-zA-Z0-9_\-!@#$%^&*()+=,./?]{3,}\}'),
    re.compile(r'CTF\{[a-zA-Z0-9_\-!@#$%^&*()+=,./?]{3,}\}'),
]


def load_jsonl(path):
    """Load all records from a JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(records, path):
    """Save records to a JSONL file."""
    with open(path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_key(record):
    """Create a platform/challenge key from a record's metadata."""
    meta = record.get("metadata", {})
    platform = meta.get("platform", "")
    challenge = meta.get("challenge", "")
    return f"{platform}/{challenge}"


def is_invalid_flag(flag):
    """Check if a flag value needs fixing."""
    return flag in INVALID_FLAGS


def is_sane_flag(flag):
    """Check if a flag value looks like a real flag (not garbage/truncated data)."""
    if not flag or not isinstance(flag, str):
        return False
    # Too long to be a real flag (most flags < 200 chars, Oracle version string is ~180)
    if len(flag) > 250:
        return False
    # Contains newlines -> probably garbage/code snippet
    if "\n" in flag:
        return False
    # Contains common code artifacts
    if any(x in flag for x in ["import ", "def ", "subprocess", "<COMMAND>"]):
        return False
    return True


def build_flag_lookup(records):
    """Build a challenge -> set(flags) lookup from records with valid flags."""
    lookup = defaultdict(set)
    for record in records:
        flag = record.get("ground_truth_flag")
        if flag and not is_invalid_flag(flag) and flag not in FAKE_FLAGS and is_sane_flag(flag):
            key = make_key(record)
            lookup[key].add(flag)
    return lookup


def extract_flag_from_content(record):
    """
    Search tool/assistant message content for real flag patterns.
    Returns the best flag found, or None.
    Skips system messages (which contain placeholder patterns).
    """
    candidates = set()

    for msg in record.get("messages", []):
        role = msg.get("role", "")
        # Only search tool responses and assistant messages
        if role not in ("tool", "assistant"):
            continue

        content = str(msg.get("content", "") or "")
        if not content:
            continue

        for pattern in FLAG_REGEXES:
            for match in pattern.finditer(content):
                flag_val = match.group()
                if flag_val not in FAKE_FLAGS:
                    candidates.add(flag_val)

    if not candidates:
        return None

    # Prefer longer hex-hash flags (more likely to be real) over short ones
    # Also prefer picoCTF{}, HTB{}, etc. over generic FLAG{}
    def flag_quality(f):
        # Hex hash flags are highest quality
        if re.match(r'FLAG\{[a-f0-9]{32,128}\}', f):
            return (3, len(f))
        # Platform-specific flags
        for prefix in ("picoCTF{", "HTB{", "hkcert22{", "SEKAI{", "THM{", "gctf{"):
            if f.startswith(prefix):
                return (2, len(f))
        # Generic CTF/FLAG
        return (1, len(f))

    return max(candidates, key=flag_quality)


def pick_best_flag(flags_set):
    """From a set of flags for the same challenge, pick the best one."""
    if not flags_set:
        return None
    if len(flags_set) == 1:
        return next(iter(flags_set))

    # Prefer hex hash flags, then longer flags
    def quality(f):
        if re.match(r'FLAG\{[a-f0-9]{32,128}\}', f):
            return (3, len(f))
        if f.startswith(("picoCTF{", "HTB{", "hkcert22{", "SEKAI{", "THM{", "gctf{")):
            return (2, len(f))
        if "Congratulations" in f or "Lab solved" in f:
            return (1, len(f))
        return (0, len(f))

    return max(flags_set, key=quality)


def fix_records(records, sft_lookup, grpo_lookup, label="GRPO"):
    """Fix missing/invalid flags in a list of records. Returns fix statistics."""
    stats = {
        "total": len(records),
        "already_valid": 0,
        "fixed_sft_crossref": 0,
        "fixed_grpo_crossref": 0,
        "fixed_content_extract": 0,
        "fixed_portswigger_lab": 0,
        "still_missing": 0,
        "details": [],
    }

    for i, record in enumerate(records):
        flag = record.get("ground_truth_flag")
        if not is_invalid_flag(flag):
            stats["already_valid"] += 1
            continue

        key = make_key(record)
        meta = record.get("metadata", {})
        platform = meta.get("platform", "")
        challenge = meta.get("challenge", "")
        success = meta.get("success", False)
        old_flag = flag

        new_flag = None
        method = None

        # Method 1: Cross-reference from SFT
        if key in sft_lookup:
            new_flag = pick_best_flag(sft_lookup[key])
            method = "sft_crossref"

        # Method 2: Cross-reference from GRPO (other traces of same challenge)
        if new_flag is None and key in grpo_lookup:
            new_flag = pick_best_flag(grpo_lookup[key])
            method = "grpo_crossref"

        # Method 3: Extract from trace content
        if new_flag is None:
            extracted = extract_flag_from_content(record)
            if extracted:
                new_flag = extracted
                method = "content_extract"

        # Method 4: PortSwigger CHECK/Complete with success=True -> "Congratulations, you solved the lab!"
        if new_flag is None and platform == "portswigger" and success:
            new_flag = "Congratulations, you solved the lab!"
            method = "portswigger_lab"

        # Apply fix
        if new_flag is not None:
            record["ground_truth_flag"] = new_flag
            stats[f"fixed_{method}"] += 1
            stats["details"].append({
                "index": i,
                "key": key,
                "old_flag": old_flag,
                "new_flag": new_flag,
                "method": method,
            })
        else:
            stats["still_missing"] += 1
            stats["details"].append({
                "index": i,
                "key": key,
                "old_flag": old_flag,
                "new_flag": None,
                "method": "UNFIXED",
            })

    return records, stats


def print_report(stats, label):
    """Print a human-readable report of fixes applied."""
    print(f"\n{'='*60}")
    print(f"  {label} Flag Fix Report")
    print(f"{'='*60}")
    print(f"  Total traces:           {stats['total']}")
    print(f"  Already valid:          {stats['already_valid']}")
    print(f"  Fixed (SFT cross-ref):  {stats['fixed_sft_crossref']}")
    print(f"  Fixed (GRPO cross-ref): {stats['fixed_grpo_crossref']}")
    print(f"  Fixed (content extract):{stats['fixed_content_extract']}")
    print(f"  Fixed (PortSwigger lab):{stats['fixed_portswigger_lab']}")
    total_fixed = (
        stats["fixed_sft_crossref"]
        + stats["fixed_grpo_crossref"]
        + stats["fixed_content_extract"]
        + stats["fixed_portswigger_lab"]
    )
    print(f"  --------------------------------")
    print(f"  Total fixed:            {total_fixed}")
    print(f"  Still missing:          {stats['still_missing']}")
    print()

    # Show details
    if stats["details"]:
        print("  --- Fixes Applied ---")
        for d in stats["details"]:
            if d["method"] != "UNFIXED":
                print(f"    [{d['index']:>3}] {d['key']}")
                print(f"          old: {repr(d['old_flag'])}")
                print(f"          new: {repr(d['new_flag'][:80])}{'...' if d['new_flag'] and len(d['new_flag']) > 80 else ''}")
                print(f"          method: {d['method']}")
        print()

        unfixed = [d for d in stats["details"] if d["method"] == "UNFIXED"]
        if unfixed:
            print(f"  --- Still Missing ({len(unfixed)}) ---")
            for d in unfixed:
                print(f"    [{d['index']:>3}] {d['key']} (old: {repr(d['old_flag'])})")
            print()


def main():
    dry_run = "--dry-run" in sys.argv

    print("Loading data...")
    grpo_records = load_jsonl(GRPO_PATH)
    sft_records = load_jsonl(SFT_PATH)
    print(f"  GRPO: {len(grpo_records)} traces")
    print(f"  SFT:  {len(sft_records)} traces")

    # Build lookups
    print("\nBuilding flag lookups...")
    sft_lookup = build_flag_lookup(sft_records)
    grpo_lookup = build_flag_lookup(grpo_records)
    print(f"  SFT lookup:  {len(sft_lookup)} challenges with valid flags")
    print(f"  GRPO lookup: {len(grpo_lookup)} challenges with valid flags")

    # Fix GRPO
    print("\n--- Fixing GRPO ---")
    grpo_records, grpo_stats = fix_records(
        grpo_records, sft_lookup, grpo_lookup, label="GRPO"
    )
    print_report(grpo_stats, "GRPO")

    # Fix SFT (same logic)
    print("\n--- Fixing SFT ---")
    # Rebuild GRPO lookup after fixes (may have new flags)
    grpo_lookup_updated = build_flag_lookup(grpo_records)
    sft_records, sft_stats = fix_records(
        sft_records, sft_lookup, grpo_lookup_updated, label="SFT"
    )
    print_report(sft_stats, "SFT")

    # Write results
    if dry_run:
        print("\n*** DRY RUN - no files written ***")
    else:
        print(f"\nWriting fixed GRPO to {GRPO_PATH}...")
        save_jsonl(grpo_records, GRPO_PATH)
        print(f"Writing fixed SFT to {SFT_PATH}...")
        save_jsonl(sft_records, SFT_PATH)
        print("Done!")

    # Summary
    grpo_fixed = (
        grpo_stats["fixed_sft_crossref"]
        + grpo_stats["fixed_grpo_crossref"]
        + grpo_stats["fixed_content_extract"]
        + grpo_stats["fixed_portswigger_lab"]
    )
    sft_fixed = (
        sft_stats["fixed_sft_crossref"]
        + sft_stats["fixed_grpo_crossref"]
        + sft_stats["fixed_content_extract"]
        + sft_stats["fixed_portswigger_lab"]
    )
    print(f"\n{'='*60}")
    print(f"  TOTAL: Fixed {grpo_fixed} GRPO + {sft_fixed} SFT = {grpo_fixed + sft_fixed} traces")
    print(f"  Remaining: {grpo_stats['still_missing']} GRPO + {sft_stats['still_missing']} SFT")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
