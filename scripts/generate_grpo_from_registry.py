#!/usr/bin/env python3
"""Backward-compatible wrapper for generate_online_rl_from_registry.py."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("generate_online_rl_from_registry.py")),
        run_name="__main__",
    )

