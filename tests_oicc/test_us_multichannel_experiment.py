"""Test the ISOLATED US multichannel experiment runner (demo mode, no real data).

Confirms the optional (b) experiment runs standalone and never breaks the build.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "experiments" / "oicc_runs"))


def test_us_multichannel_experiment_demo_runs():
    from run_us_multichannel_experiment import run
    res = run()                       # no data dir -> demo mode
    assert res["is_demo"] is True
    assert "lines" in res and len(res["lines"]) > 5
    assert res["moments"].var_theta > 0


def test_us_multichannel_experiment_real_flag_falls_back(tmp_path):
    """--real with no files must NOT crash; it falls back to demo."""
    from run_us_multichannel_experiment import run
    res = run(data_dir=tmp_path, force_real=True)
    assert res["is_demo"] is True     # cleanly fell back


def test_us_multichannel_experiment_uses_real_when_present(tmp_path):
    import numpy as np
    from run_us_multichannel_experiment import run
    rng = np.random.default_rng(0)
    # aligned synthetic "real" files
    theta = rng.normal(0, 1, 300)
    for name, bias in [("records", -0.5), ("ncvs", 0.0), ("cfs", -0.2)]:
        vals = np.clip(np.exp(theta + bias + rng.normal(0, 0.3, 300)) * 10, 0, None)
        np.save(tmp_path / f"us_{name}.npy", vals)
    res = run(data_dir=tmp_path, force_real=True)
    assert res["is_demo"] is False    # picked up the real files
