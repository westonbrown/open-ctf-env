#!/usr/bin/env python3
"""Patch SkyRL's string-based torch version comparison.

SkyRL 0.3.1 `distributed/utils.py` line 85 uses:
    str(torch.__version__) >= "2.6"

This is a STRING comparison, so "2.10" < "2.6" lexicographically.
PyTorch 2.10+ gets the WRONG parameter name (pg_options instead of
backend_options), causing TypeError in _new_process_group_helper().

Fix: Replace string comparison with proper tuple comparison.
"""

import re
import sys

def apply():
    try:
        import skyrl_train.distributed.utils as du
        source_file = du.__file__
    except ImportError:
        print("   skyrl_train not installed, skipping version comparison patch")
        return

    with open(source_file) as f:
        content = f.read()

    # Check if already patched
    if "int(major), int(minor)" in content or "tuple(" in content:
        print("   Patch (version comparison): already applied")
        return

    # Find and replace the string comparison
    old_pattern = r'pg_options_param_name = "backend_options" if str\(torch\.__version__\) >= "2\.6" else "pg_options"'
    new_code = (
        '# Patched: use tuple comparison (string comparison fails for "2.10" < "2.6")\n'
        '    _torch_ver_parts = torch.__version__.split(".")[:2]\n'
        '    _torch_ver_tuple = (int(_torch_ver_parts[0]), int(re.sub(r"[^0-9]", "", _torch_ver_parts[1])))\n'
        '    pg_options_param_name = "backend_options" if _torch_ver_tuple >= (2, 6) else "pg_options"'
    )

    if re.search(old_pattern, content):
        content = re.sub(old_pattern, new_code, content)
        # Make sure 'import re' is present
        if 'import re' not in content:
            content = 'import re\n' + content
        with open(source_file, 'w') as f:
            f.write(content)
        print("   Patch (version comparison): applied successfully")
    else:
        # Try simpler pattern match
        old_simple = 'str(torch.__version__) >= "2.6"'
        if old_simple in content:
            replacement = (
                '(lambda v: (int(v[0]), int(re.sub(r"[^0-9]", "", v[1]))) >= (2, 6))'
                '(torch.__version__.split(".")[:2])'
            )
            content = content.replace(old_simple, replacement)
            if 'import re' not in content:
                content = 'import re\n' + content
            with open(source_file, 'w') as f:
                f.write(content)
            print("   Patch (version comparison): applied (simple replacement)")
        else:
            print("   Patch (version comparison): pattern not found, may already be fixed")


if __name__ == "__main__":
    apply()
