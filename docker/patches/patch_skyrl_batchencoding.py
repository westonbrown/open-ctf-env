#!/usr/bin/env python3
"""Patch apply_chat_template calls in skyrl_gym_generator.py to wrap with list().

Some tokenizers (e.g. Nanbeige) return BatchEncoding from apply_chat_template()
instead of List[int]. SkyRL concatenates token ID lists with `+`, which fails
for BatchEncoding objects.

Fix: Wrap apply_chat_template(..., tokenize=True) calls with list() to ensure
a plain List[int] is always returned.

Also patches remote_inference_client.py to wrap prompt_token_ids concatenation.
"""
import pathlib
import re
import sys

GEN_PATH = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/skyrl_train/generators/skyrl_gym_generator.py"
)

CLIENT_PATH = pathlib.Path(
    "/usr/local/lib/python3.12/dist-packages/skyrl_train/inference_engines/remote_inference_client.py"
)


def _add_list_wrapper(content: str, var_pattern: str) -> tuple[str, int]:
    """Wrap `var = self.tokenizer.apply_chat_template(...)` with list(...)."""
    changes = 0
    target = f"{var_pattern} = self.tokenizer.apply_chat_template("
    wrapped = f"{var_pattern} = list(self.tokenizer.apply_chat_template("

    if target in content and wrapped not in content:
        content = content.replace(target, wrapped, 1)
        # Find closing paren of the apply_chat_template call and add extra )
        idx = content.index(wrapped) + len(wrapped)
        depth = 1
        while depth > 0 and idx < len(content):
            if content[idx] == '(':
                depth += 1
            elif content[idx] == ')':
                depth -= 1
            idx += 1
        content = content[:idx] + ")" + content[idx:]
        changes = 1
    return content, changes


def patch_generator():
    if not GEN_PATH.exists():
        print(f"   skyrl_train generators not found, skipping BatchEncoding patch")
        return False

    content = GEN_PATH.read_text()

    # Check if already patched
    if "list(self.tokenizer.apply_chat_template(" in content:
        print("   Patch (BatchEncoding wrapping): already applied")
        return True

    changes = 0

    # Wrap 5 apply_chat_template call sites (skip return_dict=True ones)
    for var in [
        "self.base_conversation_token_ids",
        "initial_input_ids",
        "agent_loop_state.input_ids",
        "obs_ids_to_add",
        "prompt_token_ids",
    ]:
        content, n = _add_list_wrapper(content, var)
        if n:
            changes += n
            print(f"   Wrapped {var}")

    if changes > 0:
        GEN_PATH.write_text(content)
        print(f"   Patch (BatchEncoding wrapping): APPLIED ({changes} call sites)")
    else:
        print("   Patch (BatchEncoding wrapping): no patterns found (may be different version)")

    return True


def patch_client():
    """Patch remote_inference_client.py to wrap prompt_token_ids in list()."""
    if not CLIENT_PATH.exists():
        print("   remote_inference_client not found, skipping")
        return

    content = CLIENT_PATH.read_text()

    # Pattern: prompt_token_ids + accum_token_ids
    old = "prompt_token_ids + accum_token_ids"
    new = "list(prompt_token_ids) + accum_token_ids"

    if new in content:
        print("   Patch (client BatchEncoding): already applied")
        return

    if old in content:
        content = content.replace(old, new, 1)
        CLIENT_PATH.write_text(content)
        print("   Patch (client BatchEncoding): APPLIED")
    else:
        print("   Patch (client BatchEncoding): pattern not found")


if __name__ == "__main__":
    patch_generator()
    patch_client()
