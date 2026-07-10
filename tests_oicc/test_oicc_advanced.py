"""Tests for point-identification of the common mode and the anytime-valid monitor."""
from __future__ import annotations

import numpy as np
import pytest

from oicc.measurement import generate_proximal
from oicc.proximal import point_identify, exclusion_sensitivity
from oicc.monitor import EProcessMonitor


# --------------------------------------------------------------------------- #
# Point-identification of Var(theta) under a common-mode confounder
# --------------------------------------------------------------------------- #
def test_point_id_recovers_true_variance_under_confounding():
    """The naive estimate inflates with the common mode; point-ID stays at truth."""
    for cm in (0.5, 1.0, 2.0):
        true_v, naive_v, clean_v = [], [], []
        for s in range(20):
            d = generate_proximal(n=6000, seed=s, K=4, Q=2, cm_strength=cm)
            r = point_identify(d.signal_channels, d.controls)
            true_v.append(np.var(d.theta))
            naive_v.append(r.var_theta_naive)
            clean_v.append(r.var_theta_clean)
        tv, nv, cv = np.mean(true_v), np.mean(naive_v), np.mean(clean_v)
        # naive must be inflated; point-ID must be close to the truth
        assert nv > tv * 1.1                        # confounder inflates naive
        assert abs(cv - tv) / tv < 0.12             # point-ID ~ truth


def test_point_id_gate_no_false_correction_without_confounder():
    """With no common mode, point-ID must NOT subtract noise (detection gate)."""
    clean_v, true_v = [], []
    for s in range(20):
        d = generate_proximal(n=6000, seed=s, K=4, Q=2, cm_strength=0.0)
        r = point_identify(d.signal_channels, d.controls)
        clean_v.append(r.var_theta_clean)
        true_v.append(np.var(d.theta))
    assert abs(np.mean(clean_v) - np.mean(true_v)) / np.mean(true_v) < 0.1


def test_point_id_estimates_confounder_variance():
    d = generate_proximal(n=6000, seed=0, K=4, Q=2, cm_strength=1.5)
    r = point_identify(d.signal_channels, d.controls)
    assert r.identified is True
    assert r.var_W > 0.5                             # recovers a sizeable Var(W)


def test_point_id_single_control_is_partial():
    d = generate_proximal(n=3000, seed=0, K=4, Q=1, cm_strength=1.0)
    r = point_identify(d.signal_channels, d.controls)
    assert r.identified is False


def test_point_id_validates_shape():
    d = generate_proximal(n=1000, seed=0, K=4, Q=2)
    with pytest.raises(ValueError):
        point_identify(d.signal_channels, d.controls[:, :500])


# --------------------------------------------------------------------------- #
# Anytime-valid e-process monitor
# --------------------------------------------------------------------------- #
def test_monitor_anytime_false_alarm_controlled():
    """Under H0 (uniform p-values) the anytime false-alarm rate is <= alpha."""
    rng = np.random.default_rng(0)
    alpha = 0.05
    T, trials = 150, 800
    fa = 0
    for _ in range(trials):
        p = rng.uniform(0.0, 1.0, T)
        m = EProcessMonitor(alpha=alpha).run(p)
        fa += m.alarm
    # Ville guarantee: must not exceed alpha (allow a hair of Monte-Carlo slack)
    assert fa / trials <= alpha + 0.02


def test_monitor_detects_drift_with_power():
    rng = np.random.default_rng(1)
    fired = 0
    for _ in range(300):
        p_ok = rng.uniform(0.0, 1.0, 40)
        p_drift = rng.beta(0.3, 3.0, 120)          # small p-values (rejecting)
        m = EProcessMonitor(alpha=0.05).run(np.concatenate([p_ok, p_drift]))
        fired += m.alarm
    assert fired / 300 > 0.9                         # high power under drift


def test_monitor_wealth_starts_at_one_and_is_finite():
    m = EProcessMonitor(alpha=0.05)
    assert m.wealth == pytest.approx(1.0)
    m.update(0.5)
    assert np.isfinite(m.wealth)


def test_monitor_validates_inputs():
    with pytest.raises(ValueError):
        EProcessMonitor(alpha=1.5)
    with pytest.raises(ValueError):
        EProcessMonitor(kappas=np.array([0.0, 0.5]))  # kappa not in (0,1)
    m = EProcessMonitor()
    with pytest.raises(ValueError):
        m.update(1.5)                                 # p out of [0,1]


# --------------------------------------------------------------------------- #
# Exclusion-sensitivity analysis (answers the reviewer's #1 objection)
# --------------------------------------------------------------------------- #
def test_exclusion_sensitivity_band_collapses_at_eps0():
    """At eps=0 (valid exclusion) the band is exactly the point estimate."""
    d = generate_proximal(n=6000, seed=0, K=4, Q=2, cm_strength=1.0,
                          ctrl_theta_load=0.0)
    es = exclusion_sensitivity(d.signal_channels, d.controls, eps_max=0.3)
    assert es.var_theta_lo[0] == pytest.approx(es.var_theta_ref, rel=1e-6)
    assert es.var_theta_hi[0] == pytest.approx(es.var_theta_ref, rel=1e-6)


def test_exclusion_sensitivity_band_widens_with_eps():
    d = generate_proximal(n=6000, seed=0, K=4, Q=2, cm_strength=1.0)
    es = exclusion_sensitivity(d.signal_channels, d.controls, eps_max=0.3)
    width0 = es.var_theta_hi[0] - es.var_theta_lo[0]
    width_end = es.var_theta_hi[-1] - es.var_theta_lo[-1]
    assert width_end > width0
    assert 0.0 <= es.robustness_eps <= 1.0


def test_exclusion_sensitivity_band_contains_truth_under_violation():
    """KEY: with a KNOWN exclusion violation, the swept band brackets the true
    Var(theta) that the naive point-ID misses."""
    for dl in (0.0, 0.15, 0.30):
        d = generate_proximal(n=8000, seed=1, K=4, Q=2, cm_strength=1.2,
                              ctrl_theta_load=dl)
        true = np.var(d.theta)
        es = exclusion_sensitivity(d.signal_channels, d.controls,
                                   eps_max=0.4, n_grid=17)
        lo = float(es.var_theta_lo.min())
        hi = float(es.var_theta_hi.max())
        assert lo <= true <= hi, f"band [{lo},{hi}] missed true {true} at delta={dl}"


def test_exclusion_sensitivity_requires_two_controls():
    d = generate_proximal(n=2000, seed=0, K=4, Q=1, cm_strength=1.0)
    with pytest.raises(ValueError):
        exclusion_sensitivity(d.signal_channels, d.controls)
