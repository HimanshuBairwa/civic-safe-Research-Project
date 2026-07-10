"""Tests for the US multi-channel loader scaffold (records + NCVS + 911)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "experiments" / "oicc_runs"))

from us_multichannel_loader import (  # noqa: E402
    build_us_multichannel,
    load_real_if_available,
)


def test_demo_runs_and_has_three_channels():
    d = build_us_multichannel(demo=True, n=2000, seed=0)
    assert d.log_channels.shape == (3, 2000)
    assert d.is_demo is True
    assert len(d.channel_names) == 3
    assert np.all(np.isfinite(d.log_channels))


def test_real_array_path():
    rng = np.random.default_rng(0)
    rec = rng.poisson(30, 400)
    ncvs = rng.poisson(50, 400)
    cfs = rng.poisson(40, 400)
    d = build_us_multichannel(rec, ncvs, cfs)
    assert d.is_demo is False
    assert d.log_channels.shape == (3, 400)
    assert d.channel_names == ["records", "ncvs", "cfs"]


def test_misaligned_channels_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        build_us_multichannel(rng.poisson(10, 100), rng.poisson(10, 90),
                              rng.poisson(10, 100))


def test_negative_input_rejected():
    with pytest.raises(ValueError):
        build_us_multichannel(np.array([1.0, np.nan, 2.0]),
                              np.array([1.0, 2.0, 3.0]),
                              np.array([1.0, 2.0, 3.0]))


def test_load_real_returns_none_when_absent(tmp_path):
    assert load_real_if_available(tmp_path) is None


def test_load_real_reads_aligned_npy(tmp_path):
    rng = np.random.default_rng(0)
    for name in ("records", "ncvs", "cfs"):
        np.save(tmp_path / f"us_{name}.npy", rng.poisson(20, 200))
    d = load_real_if_available(tmp_path)
    assert d is not None and d.log_channels.shape == (3, 200) and not d.is_demo
