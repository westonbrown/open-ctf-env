#!/usr/bin/env python3
"""Compatibility wrapper for BoxPwnr profile on generic LangGraph adapter.

This wrapper preserves the old command path while delegating all runtime logic
to ``langgraph_native_runtime_adapter.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# Preserve historical defaults for existing configs/scripts.
os.environ.setdefault("OPEN_CTF_AGENT_FRAMEWORK", "boxpwnr_langgraph")
os.environ.setdefault("OPEN_CTF_AGENT_ADAPTER", "boxpwnr_native_runtime_adapter")

from langgraph_native_runtime_adapter import main


if __name__ == "__main__":
    raise SystemExit(main())
