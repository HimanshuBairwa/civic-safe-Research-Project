"""Path resolution for OICC experiments -- portable across Windows/Linux/A100.

The India NCRB dataset ("crime-detection-ai") is external to this repo. We locate
it, in order of precedence:

  1. the OICC_INDIA_DATA environment variable (explicit override), then
  2. a `data/ncrb` folder inside this project (if the user copies it in), then
  3. sibling folders of the project root named "crime-detection-ai/data" or
     "PCC best"-style layouts (the original dev machine).

If none resolve, callers get None and skip gracefully -- never an error. This is
the single source of truth so no experiment or test hardcodes an absolute path.
"""
from __future__ import annotations

import os
from pathlib import Path

# project root = two levels up from this file (experiments/oicc_runs/paths.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def find_india_data() -> Path | None:
    """Return the India NCRB `data` directory, or None if it cannot be found."""
    # 1. explicit env override
    env = os.environ.get("OICC_INDIA_DATA")
    if env:
        p = Path(env).expanduser()
        if p.exists():
            return p

    # 2. in-repo copy
    candidates = [
        PROJECT_ROOT / "data" / "ncrb",
        PROJECT_ROOT / "data" / "crime-detection-ai" / "data",
    ]
    # 3. sibling of the project root (original dev layout)
    parent = PROJECT_ROOT.parent
    candidates += [
        parent / "crime-detection-ai" / "data",
        parent / "crime-detection-ai",
    ]
    for c in candidates:
        # a valid NCRB data dir has the IPC panel under crime/
        if (c / "crime" / "01_District_wise_crimes_committed_IPC_2001_2012.csv").exists():
            return c
    return None


def find_us_panel(city: str) -> Path | None:
    """Return the processed US panel .pt for a city, or None if absent."""
    env = os.environ.get("OICC_US_PANELS")
    roots = []
    if env:
        roots.append(Path(env).expanduser())
    roots.append(PROJECT_ROOT / "data" / "processed")
    for r in roots:
        p = r / f"{city}_panel.pt"
        if p.exists():
            return p
    return None
