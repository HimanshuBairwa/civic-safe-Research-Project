"""Stress, property, and robustness tests for OICC flawlessness.

Goal: guarantee the package NEVER raises an uncaught runtime error on adversarial
-but-shaped input, is deterministic, and validates bad input with clear errors.
Covers uncertainty (bootstrap) coverage too.
"""
from __future__ import annotations

import numpy as np
import pytest

import oicc
from oicc.uncertainty import bootstrap_moments, bootstrap_point_id


# --------------------------------------------------------------------------- #
# Uncertainty / bootstrap coverage
# --------------------------------------------------------------------------- #
def test_bootstrap_moments_covers_truth():
    hits = 0
    trials = 20
    for s in range(trials):
        c = oicc.generate(n=4000, seed=s, K=4)
        bm = bootstrap_moments(c.log_channels, n_boot=200, level=0.9, seed=s)
        ci = bm["var_theta"]
        hits += ci.lower <= np.var(c.theta) <= ci.upper
    # ~90% nominal; allow honest slack for a 20-trial estimate
    assert hits >= 14


def test_bootstrap_point_id_separates_clean_from_naive():
    c = oicc.generate_proximal(n=5000, seed=0, K=4, Q=2, cm_strength=1.5)
    bp = bootstrap_point_id(c.signal_channels, c.controls, n_boot=200, seed=0)
    true = np.var(c.theta)
    # clean CI contains truth; naive CI lies strictly above it
    assert bp["var_theta_clean"].lower <= true <= bp["var_theta_clean"].upper
    assert bp["var_theta_naive"].lower > true


def test_bootstrap_block_option_runs():
    c = oicc.generate(n=2000, seed=0, K=4)
    bm = bootstrap_moments(c.log_channels, n_boot=100, block=10, seed=0)
    assert np.isfinite(bm["var_theta"].se)


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_all_estimators_are_deterministic():
    c = oicc.generate(n=1500, seed=3, K=4)
    a = oicc.estimate_factor_moments(c.log_channels)
    b = oicc.estimate_factor_moments(c.log_channels)
    assert np.array_equal(a.beta, b.beta) and a.var_theta == b.var_theta
    s1 = oicc.overid_wald_test(c.log_channels, seed=7).pvalue
    s2 = oicc.overid_wald_test(c.log_channels, seed=7).pvalue
    assert s1 == s2
    r1 = oicc.split_conformal_latent(c.log_channels, seed=1)
    r2 = oicc.split_conformal_latent(c.log_channels, seed=1)
    assert np.array_equal(r1.lat_lower, r2.lat_lower)


# --------------------------------------------------------------------------- #
# Degenerate / adversarial inputs: must not raise uncaught errors
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("K", [3, 4, 5, 6, 8])
def test_pipeline_across_K_no_crash(K):
    c = oicc.generate(n=1200, seed=0, K=K)
    oicc.estimate_factor_moments(c.log_channels)
    oicc.overid_wald_test(c.log_channels, seed=0)
    r = oicc.split_conformal_latent(c.log_channels, alpha=0.1, seed=0)
    assert np.all(np.isfinite(r.lat_lower)) and np.all(np.isfinite(r.lat_upper))


def test_near_constant_channel_is_handled():
    """A channel with almost no variance must not produce NaNs/inf downstream."""
    c = oicc.generate(n=1000, seed=0, K=4)
    Y = c.log_channels.copy()
    Y[3] = Y[3].mean() + 1e-9 * np.random.default_rng(0).standard_normal(Y.shape[1])
    fm = oicc.estimate_factor_moments(Y)
    assert np.all(np.isfinite(fm.beta)) and np.isfinite(fm.var_theta)
    r = oicc.split_conformal_latent(Y, alpha=0.1, seed=0)
    assert np.all(np.isfinite(r.lat_upper))


def test_highly_correlated_channels_handled():
    """Nearly collinear channels (tiny idiosyncratic noise) must stay finite."""
    rng = np.random.default_rng(0)
    n = 2000
    theta = rng.normal(0, 1, n)
    Y = np.vstack([theta + 0.01 * rng.standard_normal(n) for _ in range(4)])
    fm = oicc.estimate_factor_moments(Y)
    assert np.all(np.isfinite(fm.beta))
    r = oicc.split_conformal_latent(Y, alpha=0.1, seed=0)
    assert np.all(np.isfinite(r.lat_lower))


def test_extreme_alpha_values():
    c = oicc.generate(n=1500, seed=0, K=4)
    for alpha in (0.01, 0.5, 0.99):
        r = oicc.split_conformal_latent(c.log_channels, alpha=alpha, seed=0)
        assert np.all(r.lat_upper >= r.lat_lower)


def test_minimum_sample_size_boundary():
    c = oicc.generate(n=40, seed=0, K=4)   # small but valid
    r = oicc.split_conformal_latent(c.log_channels, alpha=0.1, seed=0)
    assert np.all(np.isfinite(r.lat_lower))


def test_proximal_many_controls():
    d = oicc.generate_proximal(n=3000, seed=0, K=4, Q=3, cm_strength=1.0)
    r = oicc.point_identify(d.signal_channels, d.controls)
    assert r.identified and np.isfinite(r.var_theta_clean)


def test_monitor_handles_extreme_pvalues():
    m = oicc.EProcessMonitor(alpha=0.05)
    for p in (0.0, 1.0, 1e-12, 0.999999):
        m.update(p)
    assert np.isfinite(m.wealth)


# --------------------------------------------------------------------------- #
# Input validation raises (not silently wrong)
# --------------------------------------------------------------------------- #
def test_input_validation_raises_clearly():
    with pytest.raises(ValueError):
        oicc.estimate_factor_moments(np.zeros((2, 3)))          # n too small
    with pytest.raises(ValueError):
        oicc.generate(n=100, K=1)                              # K too small
    with pytest.raises(ValueError):
        oicc.split_conformal_latent(oicc.generate(n=500, K=4).log_channels,
                                    alpha=0.0)                  # bad alpha
    with pytest.raises(ValueError):
        oicc.point_identify(np.zeros((4, 100)), np.zeros((2, 50)))  # n mismatch


def test_nan_input_rejected():
    Y = oicc.generate(n=500, seed=0, K=4).log_channels.copy()
    Y[0, 0] = np.nan
    with pytest.raises(ValueError):
        oicc.estimate_factor_moments(Y)
