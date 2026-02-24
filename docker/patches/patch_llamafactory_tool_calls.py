"""Patch LlamaFactory converter.py to handle None tool_calls.

When HuggingFace datasets loads JSONL with mixed schemas (some messages have
tool_calls, some don't), Arrow normalizes all records to include the key with
None for missing values. LlamaFactory's converter does `len(message["tool_calls"])`
without checking for None first, causing TypeError.

This patch adds a None guard to the tool_calls check in OpenAIDatasetConverter.
"""

import importlib.util
import sys

def apply():
    spec = importlib.util.find_spec("llamafactory")
    if spec is None:
        print("   Patch (LlamaFactory tool_calls): SKIPPED (llamafactory not installed)")
        return

    import llamafactory
    base = llamafactory.__path__[0]
    converter_path = f"{base}/data/converter.py"

    try:
        with open(converter_path, "r") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"   Patch (LlamaFactory tool_calls): SKIPPED ({converter_path} not found)")
        return

    old = 'if "tool_calls" in message and len(message["tool_calls"]) > 0:'
    new = 'if "tool_calls" in message and message["tool_calls"] is not None and len(message["tool_calls"]) > 0:'

    if new in content:
        print("   Patch (LlamaFactory tool_calls): already applied")
        return

    if old not in content:
        print("   Patch (LlamaFactory tool_calls): WARNING - pattern not found, skipping")
        return

    content = content.replace(old, new)
    with open(converter_path, "w") as f:
        f.write(content)

    # Clear bytecode cache
    import pathlib
    cache_dir = pathlib.Path(f"{base}/data/__pycache__")
    if cache_dir.exists():
        for pyc in cache_dir.glob("converter*.pyc"):
            pyc.unlink()

    print("   Patch (LlamaFactory tool_calls): APPLIED")

if __name__ == "__main__":
    apply()
