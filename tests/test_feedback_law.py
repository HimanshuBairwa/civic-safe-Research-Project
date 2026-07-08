"""Tests for the Feedback Amplification Law (the theoretical core).

Every theorem is checked numerically against the actual iterated feedback
dynamics. These are the tests that turn the claimed contribution into a
verified one.
"""

from __future__ import annotations

import numpy as np
import pytest

from civicsafe.theory.feedback_law import (
    amplification_exponent,
    disparity_ratio,
    general_fixed_point,
    identify_kappa_did,
    local_feedback_gain,
    power_law_fixed_point,
)


def test_amplification_law_matches_iterated_dynamics() -> None:
    """Theorem 1: recorded ratio == (true ratio) ** (1/(1-kappa)), power-law case."""
    rng = np.random.default_rng(0)
    lam = np.sort(rng.gamma(2.0, 2.0, 40) + 0.3)
    beta, rho = 1.0, 0.0
    true_ratio = lam[-1] / lam[len(lam) // 2]
    for kappa in [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]:
        rho = kappa  # beta = 1
        mu = power_law_fixed_point(lam, beta, rho)
        assert mu is not None
        empirical = mu[-1] / mu[len(lam) // 2]
        predicted = true_ratio ** amplification_exponent(kappa)
        assert empirical == pytest.approx(predicted, rel=1e-6)


def test_pole_at_kappa_one() -> None:
    """The amplification exponent diverges at kappa* = 1."""
    assert amplification_exponent(0.99) > 50
    assert amplification_exponent(1.0) == float("inf")
    assert amplification_exponent(1.5) == float("inf")


def test_universal_law_non_power_law() -> None:
    """Theorem 1': the pole is coordinate-free — holds for tanh policy + exp detection."""
    rng = np.random.default_rng(0)
    lam = rng.gamma(2.0, 2.0, 4000) + 0.3

    def policy(mu: np.ndarray, m: float) -> np.ndarray:
        return 1.0 + np.tanh(1.3 * (mu / m - 1.0)) * 0.9  # saturating, non-power-law

    def detection(a: np.ndarray) -> np.ndarray:
        return np.exp(0.7 * (a - 1.0))  # exponential, non-power-law

    mu = general_fixed_point(lam, policy, detection)
    kappa_local = local_feedback_gain(mu, policy, detection)

    # Verify slope d log mu / d log lam == 1/(1-kappa_local) at a few cells.
    def single_cell(lam_s: float, m: float) -> float:
        x = lam_s
        for _ in range(4000):
            nxt = lam_s * detection(policy(np.array([x]), m))[0]
            if abs(nxt - x) < 1e-13:
                break
            x = 0.5 * x + 0.5 * nxt
        return x

    m = mu.mean()
    idx = rng.choice(len(lam), 6, replace=False)
    for i in idx:
        l0 = lam[i]
        up = single_cell(l0 * (1 + 1e-4), m)
        dn = single_cell(l0 * (1 - 1e-4), m)
        slope = (np.log(up) - np.log(dn)) / (2e-4)
        predicted = amplification_exponent(float(kappa_local[i]))
        assert slope == pytest.approx(predicted, rel=1e-4)


def test_disparity_corollary() -> None:
    """Corollary: equal-truth groups diverge as (initial_bias) ** (1/(1-kappa))."""
    rng = np.random.default_rng(1)
    S = 2000
    lam = np.full(S, 4.0)  # identical true intensity for both groups
    bias = np.ones(S)
    bias[S // 2:] = 1.5  # group B historically over-recorded by 1.5x

    for kappa in [0.0, 0.3, 0.6, 0.8]:
        mu = lam.copy()
        for _ in range(3000):
            m = mu.mean()
            nxt = lam * bias * (mu / m) ** kappa
            if np.max(np.abs(nxt - mu)) < 1e-9 * m:
                break
            mu = nxt
        empirical = mu[S // 2:].mean() / mu[:S // 2].mean()
        assert empirical == pytest.approx(disparity_ratio(1.5, kappa), rel=1e-3)


def test_passive_impossibility() -> None:
    """Theorem 2: biased and honest worlds produce statistically identical observables."""
    rng = np.random.default_rng(2)
    lam = np.sort(rng.gamma(2.0, 2.0, 30) + 0.3)
    kappa = 0.8
    mu = power_law_fixed_point(lam, 1.0, kappa)
    assert mu is not None
    # World A: true intensity lam, amplification present -> Poisson(mu).
    # World B: true intensity mu, no amplification         -> Poisson(mu).
    yA = rng.poisson(mu, size=(4000, len(mu)))
    yB = rng.poisson(mu, size=(4000, len(mu)))
    # Observable moments coincide; the latent truth (lam vs mu) differs.
    assert yA.mean() == pytest.approx(yB.mean(), rel=0.02)
    assert yA.var() == pytest.approx(yB.var(), rel=0.05)
    assert not np.allclose(lam, mu)  # the worlds genuinely differ in truth


def test_active_identification_recovers_kappa() -> None:
    """Theorem 3: kappa is point-identified by a DiD on log recorded rates."""
    rng = np.random.default_rng(3)
    S = 6000
    lam = rng.gamma(2.0, 2.0, S) + 0.3
    beta, rho, delta = 1.0, 0.5, 0.6  # true kappa = 0.5
    treated = rng.random(S) < 0.5

    mu_pre = power_law_fixed_point(lam, beta, rho)
    rho_vec = np.where(treated, rho * (1 + delta), rho)
    from civicsafe.theory.feedback_law import _hetero_fixed_point

    mu_post = _hetero_fixed_point(lam, beta, rho_vec)
    assert mu_pre is not None and mu_post is not None

    result = identify_kappa_did(
        mu_pre_treated=mu_pre[treated],
        mu_pre_control=mu_pre[~treated],
        mu_post_treated=mu_post[treated],
        mu_post_control=mu_post[~treated],
        delta=delta,
        beta=beta,
        lam=lam,
        treated_mask=treated,
    )
    assert result.recovered
    assert result.kappa_hat == pytest.approx(beta * rho, abs=0.02)
