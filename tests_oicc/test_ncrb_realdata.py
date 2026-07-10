"""Real-data loader tests (skip gracefully if the NCRB dataset is absent)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT / "experiments" / "oicc_runs"))

from paths import find_india_data  # noqa: E402

_DATA = find_india_data()

pytestmark = pytest.mark.skipif(
    _DATA is None, reason="India NCRB dataset not found (set OICC_INDIA_DATA)"
)


def test_ncrb_loader_shapes_and_finiteness():
    from ncrb_loader import load_ncrb_channels

    d = load_ncrb_channels(_DATA)
    Y = d["log_channels"]
    assert Y.ndim == 2 and Y.shape[0] == 4
    assert Y.shape[1] >= 100           # enough aligned state-year cells
    assert np.all(np.isfinite(Y))
    assert len(d["states"]) == Y.shape[1]
    assert len(d["years"]) == Y.shape[1]


def test_ncrb_channels_share_a_positive_factor():
    """All four channels should be positively correlated (shared latent)."""
    from ncrb_loader import load_ncrb_channels

    d = load_ncrb_channels(_DATA)
    C = np.corrcoef(d["log_channels"])
    off = C[np.triu_indices(4, 1)]
    assert np.all(off > 0.1)           # genuinely co-vary


def test_ncrb_pipeline_runs_end_to_end():
    from ncrb_loader import load_ncrb_channels
    from oicc.moments import estimate_factor_moments
    from oicc.spec_test import overid_wald_test
    from oicc.conformal import leave_pivot_out_conformal

    d = load_ncrb_channels(_DATA)
    Y = d["log_channels"]
    fm = estimate_factor_moments(Y)
    assert fm.beta[0] == 1.0 and fm.var_theta > 0
    spec = overid_wald_test(Y, seed=0)
    assert 0.0 <= spec.pvalue <= 1.0
    res = leave_pivot_out_conformal(Y, alpha=0.1, gamma_cm=0.5)
    assert np.all(res.upper >= res.lower)
