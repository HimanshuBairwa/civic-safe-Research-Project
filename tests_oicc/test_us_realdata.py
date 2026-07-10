"""US real-data testbed tests (skip gracefully if panels are absent)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "experiments" / "oicc_runs"))

_CHI = _ROOT / "data" / "processed" / "chicago_panel.pt"
_HAVE_TORCH = __import__("importlib").util.find_spec("torch") is not None

pytestmark = pytest.mark.skipif(
    not _CHI.exists() or not _HAVE_TORCH,
    reason="US panels absent or torch not installed",
)


def test_us_loader_builds_channels():
    from us_loader import build_us_channels
    d = build_us_channels(_CHI, period_weeks=4)
    Y = d["log_channels"]
    assert Y.shape[0] == 3            # violent / property / drug
    assert Y.shape[1] > 1000          # many area-period cells
    assert np.all(np.isfinite(Y))


def test_us_categories_reject_one_factor():
    """Same-filter crime categories SHOULD trip the over-ID test (a good sign):
    they are not mechanism-independent, and the test detects that."""
    from us_loader import build_us_channels
    from oicc.spec_test import overid_wald_test
    d = build_us_channels(_CHI, period_weeks=4)
    spec = overid_wald_test(d["log_channels"], seed=0)
    # strong dependence among categories -> the test should reject
    assert spec.pvalue < 0.05
