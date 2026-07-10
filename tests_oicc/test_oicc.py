"""Test suite for the OICC package.

These tests lock in the SCIENTIFIC behavior, not just "it runs":
  * moment estimation recovers the true loadings and latent variance,
  * multi-channel deconvolution beats the best single channel,
  * the leave-pivot-out conformal predictor covers the LATENT target at nominal,
  * the over-identification test has correct size, real power against detectable
    (Delta-perp) violations, and is PROVABLY BLIND to common-mode (Delta-parallel),
  * every public function validates its inputs.

Run:  pytest tests_oicc -q         (from the project root, with src on the path)
"""
from __future__ import annotations

import numpy as np
import pytest

import oicc
from oicc.deconvolve import blup_from_subset
from oicc.spec_test import overid_wald_test


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _single_channel_rmse(ch) -> float:
    """Best achievable RMSE from any single bias-centered channel."""
    best = np.inf
    for k in range(ch.K):
        g = ch.log_channels[k] - ch.log_channels[k].mean() + ch.log_channels[0].mean()
        best = min(best, float(np.sqrt(np.mean((g - ch.theta) ** 2))))
    return best


# --------------------------------------------------------------------------- #
# Measurement / generation
# --------------------------------------------------------------------------- #
def test_generate_shapes_and_determinism():
    a = oicc.generate(n=500, seed=7, K=4)
    b = oicc.generate(n=500, seed=7, K=4)
    assert a.log_channels.shape == (4, 500)
    assert np.allclose(a.log_channels, b.log_channels)  # deterministic
    assert a.beta[0] == 1.0 and a.alpha[0] == 0.0        # pivot normalization


def test_generate_rejects_bad_inputs():
    with pytest.raises(ValueError):
        oicc.generate(n=4, seed=0)          # n too small
    with pytest.raises(ValueError):
        oicc.generate(n=100, seed=0, K=1)   # K too small
    with pytest.raises(ValueError):
        oicc.generate(n=100, seed=0, common_mode=-1.0)  # negative shock


def test_to_log_rate_roundtrip_and_guards():
    counts = np.array([0.0, 5.0, 100.0])
    pop = np.array([1000.0, 1000.0, 1000.0])
    lr = oicc.to_log_rate(counts, pop)
    assert lr[0] == 0.0 and np.all(np.isfinite(lr))
    with pytest.raises(ValueError):
        oicc.to_log_rate(counts, np.array([0.0, 1.0, 1.0]))  # zero population


# --------------------------------------------------------------------------- #
# Moment estimation
# --------------------------------------------------------------------------- #
def test_moments_recover_loadings_and_variance():
    ch = oicc.generate(n=20000, seed=3, K=4)
    fm = oicc.estimate_factor_moments(ch.log_channels)
    # loadings within 5% of truth
    assert np.allclose(fm.beta, ch.beta, rtol=0.05, atol=0.05)
    # latent variance within 8%
    assert abs(fm.var_theta - np.var(ch.theta)) / np.var(ch.theta) < 0.08
    # noise variances positive and ordered roughly like the truth
    assert np.all(fm.noise_var > 0)


def test_moments_input_validation():
    with pytest.raises(ValueError):
        oicc.estimate_factor_moments(np.zeros((1, 100)))     # <2 channels
    with pytest.raises(ValueError):
        oicc.estimate_factor_moments(np.zeros((3, 4)))       # n too small
    with pytest.raises(ValueError):
        oicc.estimate_factor_moments(np.full((3, 50), np.nan))  # non-finite


# --------------------------------------------------------------------------- #
# Deconvolution beats single channels
# --------------------------------------------------------------------------- #
def test_deconvolution_beats_best_single_channel():
    wins = 0
    trials = 15
    for s in range(trials):
        ch = oicc.generate(n=4000, seed=s, K=4)
        est = oicc.deconvolve_blup(ch.log_channels)
        rmse = float(np.sqrt(np.mean((est.theta_hat - ch.theta) ** 2)))
        if rmse < _single_channel_rmse(ch):
            wins += 1
    # should win the large majority of the time under valid assumptions
    assert wins >= trials - 1


def test_blup_from_subset_scale_matches_pivot():
    ch = oicc.generate(n=8000, seed=11, K=4)
    fm = oicc.estimate_factor_moments(ch.log_channels, pivot=0)
    est = blup_from_subset(ch.log_channels, fm, subset=[1, 2, 3],
                           anchor_mean=float(ch.log_channels[0].mean()))
    # recovered theta_hat is on the latent scale: correlates strongly with truth
    r = np.corrcoef(est.theta_hat, ch.theta)[0, 1]
    assert r > 0.8


# --------------------------------------------------------------------------- #
# Conformal coverage of the LATENT target
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("K", [3, 4, 5])
def test_latent_coverage_at_nominal(K):
    covs = []
    for s in range(25):
        ch = oicc.generate(n=3000, seed=s, K=K)
        res = oicc.leave_pivot_out_conformal(ch.log_channels, alpha=0.1,
                                             use_spec_test=True, spec_seed=s)
        covs.append(np.mean((ch.theta >= res.lower) & (ch.theta <= res.upper)))
    mean_cov = float(np.mean(covs))
    # nominal is 0.90; allow honest slack (recovery + deconvolution error).
    # Must not badly under-cover, and must not wildly over-cover.
    assert 0.86 <= mean_cov <= 0.99


def test_conformal_requires_three_channels():
    ch = oicc.generate(n=1000, seed=0, K=4)
    with pytest.raises(ValueError):
        oicc.leave_pivot_out_conformal(ch.log_channels[:2], alpha=0.1)


def test_gamma_cm_monotonically_widens_intervals():
    ch = oicc.generate(n=3000, seed=2, K=4)
    w0 = oicc.leave_pivot_out_conformal(ch.log_channels, gamma_cm=0.0,
                                        use_spec_test=False)
    w1 = oicc.leave_pivot_out_conformal(ch.log_channels, gamma_cm=0.5,
                                        use_spec_test=False)
    w2 = oicc.leave_pivot_out_conformal(ch.log_channels, gamma_cm=1.0,
                                        use_spec_test=False)
    width0 = float(np.mean(w0.upper - w0.lower))
    width1 = float(np.mean(w1.upper - w1.lower))
    width2 = float(np.mean(w2.upper - w2.lower))
    assert width0 < width1 < width2  # visible degradation in the user knob


# --------------------------------------------------------------------------- #
# Over-identification test: size, power, and the honest blind spot
# --------------------------------------------------------------------------- #
def test_overid_size_under_null():
    rej = [overid_wald_test(oicc.generate(n=3000, seed=s, K=4).log_channels,
                            seed=s).pvalue < 0.05 for s in range(40)]
    # correct size: should rarely fire under H0
    assert np.mean(rej) <= 0.15


def test_overid_power_against_detectable_confounder():
    rej = [overid_wald_test(
        oicc.generate(n=3000, seed=s, K=4, confound_pair=0.5).log_channels,
        seed=s).pvalue < 0.05 for s in range(20)]
    # real power against a Delta-perp violation
    assert np.mean(rej) >= 0.8


def test_overid_is_blind_to_common_mode():
    """THE HONEST LIMIT: a common-mode (Delta-parallel) shock is invisible."""
    rej = [overid_wald_test(
        oicc.generate(n=3000, seed=s, K=4, common_mode=1.5).log_channels,
        seed=s).pvalue < 0.05 for s in range(20)]
    # must NOT fire: this documents the irreducible untestable direction
    assert np.mean(rej) <= 0.15


def test_overid_flags_k3_as_underpowered():
    res = overid_wald_test(oicc.generate(n=2000, seed=0, K=3).log_channels)
    assert res.kind == "cumulant" and res.underpowered is True


def test_delta_perp_grows_with_violation():
    """The data-driven radius should be ~0 under H0 and grow with the violation."""
    d0 = np.mean([overid_wald_test(oicc.generate(n=3000, seed=s, K=4).log_channels,
                                   seed=s).delta_perp_hat for s in range(10)])
    d1 = np.mean([overid_wald_test(
        oicc.generate(n=3000, seed=s, K=4, confound_pair=0.6).log_channels,
        seed=s).delta_perp_hat for s in range(10)])
    assert d1 > d0
