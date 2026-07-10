"""pytest bootstrap: make `import oicc` work without needing PYTHONPATH=src.

Adds <project-root>/src to sys.path so `pytest tests_oicc` runs on a fresh
machine (A100/Linux) with no environment setup. Also exposes the experiment
helper dir for the real-data tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "experiments/oicc_runs"):
    p = str(_ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)
