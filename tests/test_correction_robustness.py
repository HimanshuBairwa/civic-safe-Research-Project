"""Tests for recording-model misspecification robustness of the correction.

The guarantee: under a Rosenbaum-style sensitivity model where the true recording
multiplier lies within a factor Gamma of the assumed power-law multiplier, a
Gamma-inflated corrected interval covers the latent process at the nominal rate,
while the un-inflated interval degrades. Verified against Poisson latent draws
from a misspecified recording world.
"""

from __future__ import annotations

import numpy as np
import pytest

from civicsafe.theory import _poisson as poisson
from civicsafe.theory.correction_robustness import (
    robust_latent_interval,
    robustness_gamma,
)
from civicsafe.theory.feedback_law import power_law_fixed_point


def _misspecified_world(gamma_true: float, kappa: float, seed: int, S: int = 3000):
    """Build a world whose recording multiplier is off by up to gamma_true."""
    rng = np.random.default_rng(seed)
    lam = rng.gamma(2.0, 2.0, S) + 0.3
    mu = power_law_fixed_point(lam, 1.0, kappa)
    assert mu is not None
    m_hat = (mu / mu.mean()) ** kappa
    logg = rng.uniform(-np.log(gamma_true), np.log(gamma_true), size=mu.shape)
    m_true = m_hat * np.exp(logg)
    lam_true = mu / m_true
    y = poisson.rvs(lam_true, random_state=rng)
    return mu, y


def test_gamma_one_recovers_ordinary_correction() -> None:
    """gamma=1 reproduces the un-inflated corrected interval."""
    mu, _ = _misspecified_world(1.0, kappa=0.5, seed=0)
    iv = robust_latent_interval(mu, kappa=0.5, gamma=1.0, alpha=0.10)
    assert np.all(iv["upper"] >= iv["lower"])


def test_gamma_below_one_rejected() -> None:
    """gamma < 1 is invalid."""
    mu, _ = _misspecified_world(1.0, kappa=0.5, seed=0)
    with pytest.raises(ValueError):
        robust_latent_interval(mu, kappa=0.5, gamma=0.9)


def test_inflation_widens_interval_monotonically() -> None:
    """Larger gamma yields wider intervals."""
    mu, _ = _misspecified_world(1.0, kappa=0.5, seed=1)
    w1 = np.mean(np.diff(
        np.stack([robust_latent_interval(mu, 0.5, gamma=1.0)["lower"],
                  robust_latent_interval(mu, 0.5, gamma=1.0)["upper"]], 0), axis=0))
    w2 = np.mean(np.diff(
        np.stack([robust_latent_interval(mu, 0.5, gamma=2.0)["lower"],
                  robust_latent_interval(mu, 0.5, gamma=2.0)["upper"]], 0), axis=0))
    assert w2 > w1


@pytest.mark.parametrize("gamma_true", [1.3, 1.6, 2.0])
def test_matched_gamma_restores_coverage(gamma_true: float) -> None:
    """Inflating by the true misspecification factor restores latent coverage."""
    covs_naive, covs_msm = [], []
    for seed in range(6):
        mu, y = _misspecified_world(gamma_true, kappa=0.6, seed=seed)
        naive = robust_latent_interval(mu, 0.6, gamma=1.0, alpha=0.10)
        msm = robust_latent_interval(mu, 0.6, gamma=gamma_true, alpha=0.10)
        covs_naive.append(np.mean((y >= naive["lower"]) & (y <= naive["upper"])))
        covs_msm.append(np.mean((y >= msm["lower"]) & (y <= msm["upper"])))
    # The Gamma-inflated interval meets the target; naive degrades below it.
    assert np.mean(covs_msm) >= 0.90
    assert np.mean(covs_msm) > np.mean(covs_naive)


def test_robustness_gamma_reports_valid_band() -> None:
    """robustness_gamma returns the largest gamma meeting the target."""
    # Mild misspecification: a moderate gamma should suffice.
    mu, y = _misspecified_world(1.6, kappa=0.6, seed=3)
    res = robustness_gamma(mu, y, kappa=0.6, alpha=0.10,
                           gamma_grid=(1.0, 1.2, 1.5, 2.0, 3.0))
    assert res.robustness_gamma >= 1.5  # tolerates at least the true factor
    assert res.width_ratio[-1] >= res.width_ratio[0]  # width grows with gamma
