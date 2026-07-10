"""Tests for the third-cumulant over-identification test.

Locks in: correct size under H0; GENUINE power at K=3 (where the second-moment
tetrad test has df=0); blindness to a common-mode confounder (the all-order
impossibility); and the non-Gaussianity gate.
"""
from __future__ import annotations

import numpy as np
import pytest

from oicc.spec_test import overid_cumulant_test


def _gen(n=4000, seed=0, K=3, confound_pair=0.0, common_mode=0.0, nongauss=True):
    rng = np.random.default_rng(seed)
    theta = rng.standard_exponential(n) if nongauss else rng.normal(0, 1, n)
    theta = theta - theta.mean()
    beta = np.linspace(1.0, 1.4, K)
    pair = rng.standard_exponential(n) * confound_pair
    cm = rng.standard_exponential(n) * common_mode
    Y = np.empty((K, n))
    for c in range(K):
        shared = beta[c] * cm + (pair if c < 2 else 0.0)
        Y[c] = beta[c] * theta + shared + rng.normal(0, 0.4, n)
    return Y


def test_cumulant_size_under_null():
    rej = [overid_cumulant_test(_gen(seed=s), seed=s).pvalue < 0.05
           for s in range(25)]
    assert np.mean(rej) <= 0.15


def test_cumulant_power_at_K3_where_second_moment_has_no_df():
    """The KEY gain: at K=3 the tetrad test has df=0 (no power); the third-cumulant
    test detects a detectable confounder."""
    rej = [overid_cumulant_test(_gen(seed=s, confound_pair=0.5), seed=s).pvalue < 0.05
           for s in range(20)]
    assert np.mean(rej) >= 0.8


def test_cumulant_blind_to_common_mode_all_order_impossibility():
    """The third-cumulant test is ALSO blind to a common-mode confounder ->
    the impossibility holds beyond second moments."""
    rej = [overid_cumulant_test(_gen(seed=s, common_mode=1.2), seed=s).pvalue < 0.05
           for s in range(20)]
    assert np.mean(rej) <= 0.15


def test_cumulant_gate_flags_gaussian_theta():
    usable_g = [overid_cumulant_test(_gen(seed=s, nongauss=False), seed=s).usable
                for s in range(15)]
    usable_ng = [overid_cumulant_test(_gen(seed=s, nongauss=True), seed=s).usable
                 for s in range(15)]
    assert np.mean(usable_g) <= 0.2       # Gaussian theta -> not usable
    assert np.mean(usable_ng) >= 0.8      # non-Gaussian theta -> usable


def test_cumulant_requires_three_channels():
    Y = _gen(K=3)[:2]
    with pytest.raises(ValueError):
        overid_cumulant_test(Y)


def test_cumulant_block_bootstrap_runs():
    Y = _gen(seed=0)
    r = overid_cumulant_test(Y, seed=0, block=10)
    assert 0.0 <= r.pvalue <= 1.0 and np.isfinite(r.theta_skew)
