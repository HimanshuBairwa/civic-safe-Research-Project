"""Test the two-control NCRB loader + real-data point-ID path (skips if absent)."""
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


def test_two_control_loader_shapes():
    from ncrb_loader import load_ncrb_two_control
    d = load_ncrb_two_control(_DATA)
    assert d["signal_channels"].shape[0] == 3
    assert d["controls"].shape[0] == 2
    assert d["signal_channels"].shape[1] == d["controls"].shape[1]
    assert np.all(np.isfinite(d["signal_channels"]))
    assert np.all(np.isfinite(d["controls"]))


def test_two_control_point_id_runs_and_detects_common_mode():
    from ncrb_loader import load_ncrb_two_control
    from oicc.proximal import point_identify
    d = load_ncrb_two_control(_DATA)
    r = point_identify(d["signal_channels"], d["controls"])
    assert r.identified is True
    assert np.isfinite(r.var_theta_clean) and np.isfinite(r.var_theta_naive)
    # the controls reveal a common mode -> naive exceeds clean
    assert r.var_theta_naive >= r.var_theta_clean
    assert r.var_W >= 0.0
