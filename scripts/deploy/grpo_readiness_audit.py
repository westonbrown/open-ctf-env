#!/usr/bin/env python3
"""Backward-compatible wrapper for online_rl_readiness_audit.py."""

from __future__ import annotations

import runpy
from pathlib import Path


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("online_rl_readiness_audit.py")),
        run_name="__main__",
    )

