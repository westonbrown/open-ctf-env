"""Create a minimal torchaudio stub for text-only training.

LlamaFactory unconditionally imports torchaudio in data/mm_plugin.py for audio
processing. When using NGC PyTorch containers, the PyPI torchaudio package has
ABI incompatibilities (different libtorch symbols). Since we only do text SFT,
we can replace torchaudio with a stub that satisfies the import.

This patch creates a minimal torchaudio package at the Python site-packages
level if torchaudio is not already importable.
"""

import importlib.util
import os
import site
import sys

def apply():
    # Check if torchaudio already works
    try:
        import torchaudio
        print(f"   Patch (torchaudio stub): SKIPPED (torchaudio {torchaudio.__version__} already works)")
        return
    except (ImportError, OSError):
        pass  # Need to install stub

    # Find site-packages directory
    site_packages = None
    for p in site.getsitepackages():
        if os.path.isdir(p):
            site_packages = p
            break

    if site_packages is None:
        # Fallback
        site_packages = os.path.dirname(os.path.dirname(importlib.util.find_spec("torch").origin))

    torchaudio_dir = os.path.join(site_packages, "torchaudio")
    functional_dir = os.path.join(torchaudio_dir, "functional")

    os.makedirs(functional_dir, exist_ok=True)

    # Main __init__.py
    init_content = '''\
"""Minimal torchaudio stub for LlamaFactory text-only training.
Only satisfies import; actual audio functions raise NotImplementedError."""

__version__ = "0.0.0+stub"

def load(*args, **kwargs):
    raise NotImplementedError("torchaudio.load not available (stub module for text-only training)")
'''

    # functional/__init__.py
    functional_content = '''\
def resample(*args, **kwargs):
    raise NotImplementedError("torchaudio.functional.resample not available (stub module)")
'''

    with open(os.path.join(torchaudio_dir, "__init__.py"), "w") as f:
        f.write(init_content)

    with open(os.path.join(functional_dir, "__init__.py"), "w") as f:
        f.write(functional_content)

    print(f"   Patch (torchaudio stub): CREATED at {torchaudio_dir}")

if __name__ == "__main__":
    apply()
