"""Tests for the SOTA additions: split-conformal two-interval, CF deconvolution,
and the proximal / negative-control common-mode correction.
"""
from __future__ import annotations

import numpy as np
import pytest

import oicc
from oicc.conformal_split import split_conformal_latent
from oicc.cf_deconv import deconvolve_error_law
from oicc.measurement import generate_proximal
from oicc.moments import estimate_factor_moments
from oicc.deconvolve import blup_from_subset
from oicc.proximal import proximal_deconfound


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _test_fold(n: int, seed: int) -> np.ndarray:
    """Reproduce the test-fold indices used by split_conformal_latent."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_cal = int(round(0.5 * (n // 2)))
    n_cal = min(max(n_cal, 4), (n // 2) - 2)
    n_train = n - 2 * n_cal
    if n_train < 8:
        n_cal = max(4, (n - 8) // 2)
        n_train = n - 2 * n_cal
    return perm[n_train + n_cal:]


def _theta_rmse(Y: np.ndarray, theta: np.ndarray) -> float:
    fm = estimate_factor_moments(Y, pivot=0)
    others = [i for i in range(Y.shape[0]) if i != 0]
    th = blup_from_subset(Y, fm, others, float(Y[0].mean())).theta_hat
    A = np.vstack([np.ones_like(th), th]).T
    c, *_ = np.linalg.lstsq(A, theta, rcond=None)
    return float(np.sqrt(np.mean((A @ c - theta) ** 2)))


# --------------------------------------------------------------------------- #
# CF deconvolution
# --------------------------------------------------------------------------- #
def test_cf_deconv_recovers_variance():
    rng = np.random.default_rng(0)
    n = 20000
    S = rng.standard_gamma(2.0, n) * 0.4 - 0.8
    s1 = 0.4
    R = S + rng.normal(0.0, np.sqrt(s1), n)
    dd = deconvolve_error_law(R, s1)
    assert abs(dd.var_s - np.var(S)) / np.var(S) < 0.1
    # quantiles are monotone and finite
    qs = [dd.quantile(p) for p in (0.05, 0.25, 0.5, 0.75, 0.95)]
    assert all(np.isfinite(qs)) and np.all(np.diff(qs) > 0)


def test_cf_deconv_fallback_is_safe_on_tiny_sample():
    rng = np.random.default_rng(1)
    R = rng.normal(0.0, 1.0, 12)
    dd = deconvolve_error_law(R, 0.3)
    assert dd.method in ("cf", "gaussian")
    assert np.isfinite(dd.quantile(0.5))


def test_cf_deconv_validates_input():
    with pytest.raises(ValueError):
        deconvolve_error_law(np.zeros(3), 0.1)   # too few points


# --------------------------------------------------------------------------- #
# split conformal: exact observed-value + latent
# --------------------------------------------------------------------------- #
def test_split_conformal_exact_observed_coverage():
    """The observed-pivot interval has finite-sample >= 1-alpha coverage."""
    covs = []
    for s in range(30):
        ch = oicc.generate(n=4000, seed=s, K=4)
        r = split_conformal_latent(ch.log_channels, alpha=0.1, seed=s,
                                   use_spec_test=False)
        ti = _test_fold(ch.log_channels.shape[1], s)
        y = ch.log_channels[0, ti]
        covs.append(np.mean((y >= r.obs_lower) & (y <= r.obs_upper)))
    # finite-sample guarantee: must not under-cover
    assert np.mean(covs) >= 0.88


def test_split_conformal_latent_coverage_near_nominal():
    covs = []
    for s in range(30):
        ch = oicc.generate(n=4000, seed=s, K=4)
        r = split_conformal_latent(ch.log_channels, alpha=0.1, seed=s,
                                   use_spec_test=False, latent_method="gaussian")
        ti = _test_fold(ch.log_channels.shape[1], s)
        covs.append(np.mean((ch.theta[ti] >= r.lat_lower)
                            & (ch.theta[ti] <= r.lat_upper)))
    assert 0.85 <= np.mean(covs) <= 0.96


def test_split_conformal_latent_narrower_than_observed():
    """Removing pivot noise makes the latent band tighter than the observed one."""
    ch = oicc.generate(n=4000, seed=3, K=4)
    r = split_conformal_latent(ch.log_channels, alpha=0.1, seed=3,
                               use_spec_test=False)
    assert np.mean(r.lat_upper - r.lat_lower) < np.mean(r.obs_upper - r.obs_lower)


def test_split_conformal_validates_inputs():
    ch = oicc.generate(n=1000, seed=0, K=4)
    with pytest.raises(ValueError):
        split_conformal_latent(ch.log_channels[:2], alpha=0.1)  # K < 3
    with pytest.raises(ValueError):
        split_conformal_latent(ch.log_channels, alpha=1.5)      # bad alpha
    with pytest.raises(ValueError):
        split_conformal_latent(ch.log_channels, latent_method="bogus")


# --------------------------------------------------------------------------- #
# proximal / negative-control common-mode correction (the ceiling-lifter)
# --------------------------------------------------------------------------- #
def test_proximal_no_harm_without_confounder():
    """With no common mode, proximal correction must not hurt recovery."""
    naive, prox = [], []
    for s in range(20):
        d = generate_proximal(n=4000, seed=s, K=4, Q=2, cm_strength=0.0)
        naive.append(_theta_rmse(d.signal_channels, d.theta))
        pc = proximal_deconfound(d.signal_channels, d.controls)
        prox.append(_theta_rmse(pc.deconfounded, d.theta))
    # within noise of each other
    assert np.mean(prox) <= np.mean(naive) * 1.15


def test_proximal_fixes_common_mode_confounding():
    """THE KEY RESULT: proximal correction rescues recovery under a common mode
    that the over-identification test is provably blind to."""
    naive, prox = [], []
    for s in range(20):
        d = generate_proximal(n=4000, seed=s, K=4, Q=2, cm_strength=1.0)
        naive.append(_theta_rmse(d.signal_channels, d.theta))
        pc = proximal_deconfound(d.signal_channels, d.controls)
        prox.append(_theta_rmse(pc.deconfounded, d.theta))
    # proximal must be substantially better under strong common-mode confounding
    assert np.mean(prox) < 0.7 * np.mean(naive)


def test_proximal_diagnostic_and_flags():
    d = generate_proximal(n=3000, seed=0, K=4, Q=2, cm_strength=1.0)
    pc = proximal_deconfound(d.signal_channels, d.controls)
    assert pc.identified is True and pc.n_controls == 2
    assert np.all((pc.what_explained >= 0) & (pc.what_explained <= 1))
    # under real confounding the controls explain a non-trivial share
    assert np.mean(pc.what_explained) > 0.1


def test_proximal_single_control_is_partial():
    d = generate_proximal(n=2000, seed=0, K=4, Q=1, cm_strength=1.0)
    pc = proximal_deconfound(d.signal_channels, d.controls)
    assert pc.identified is False and pc.n_controls == 1


def test_proximal_validates_shape():
    d = generate_proximal(n=1000, seed=0, K=4, Q=2)
    with pytest.raises(ValueError):
        proximal_deconfound(d.signal_channels, d.controls[:, :500])  # n mismatch
