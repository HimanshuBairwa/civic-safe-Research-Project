"""Tests for feedback-corrected latent prediction intervals.

The decisive experiment: in a closed observation-biased feedback loop, a naive
interval calibrated on the recorded process loses coverage of the *true latent*
process as the feedback gain rises, while the feedback-corrected interval — using
kappa point-identified by the difference-in-differences design — recovers it.
"""

from __future__ import annotations

import numpy as np
import pytest

from civicsafe.theory import _poisson as poisson
from civicsafe.theory.feedback_law import (
    _hetero_fixed_point,
    identify_kappa_did,
    power_law_fixed_point,
)
from civicsafe.theory.latent_correction import (
    deflate_latent_rate,
    latent_prediction_interval,
    recording_multiplier,
    should_abstain,
)


def test_deflation_recovers_latent_rate_exactly() -> None:
    """With the true kappa, deflating the fixed-point record returns lambda."""
    rng = np.random.default_rng(0)
    lam = rng.gamma(2.0, 2.0, 500) + 0.3
    for kappa in [0.0, 0.3, 0.6, 0.8]:
        mu = power_law_fixed_point(lam, 1.0, kappa)
        assert mu is not None
        lam_hat = deflate_latent_rate(mu, kappa)
        # Deflation recovers latent rate up to the shared mean-field constant.
        ratio = lam_hat / lam
        assert ratio.std() / ratio.mean() < 1e-6  # constant ratio => exact up to scale


def test_multiplier_identity() -> None:
    """recording_multiplier * deflated_rate == mu (definitional round-trip)."""
    rng = np.random.default_rng(1)
    mu = rng.gamma(3.0, 2.0, 200) + 0.5
    for kappa in [0.0, 0.4, 0.7]:
        m = recording_multiplier(mu, kappa)
        assert np.allclose(m * deflate_latent_rate(mu, kappa), mu)


def test_abstention_triggers_near_runaway() -> None:
    """Abstain everywhere as kappa approaches the runaway threshold."""
    mu = np.linspace(1.0, 10.0, 50)
    assert should_abstain(mu, kappa=0.95).all()
    assert not should_abstain(mu, kappa=0.0).any()


@pytest.mark.parametrize("kappa_true", [0.0, 0.3, 0.5, 0.7])
def test_correction_restores_latent_coverage(kappa_true: float) -> None:
    """CORE RESULT: correction recovers ~nominal latent coverage where naive fails."""
    rng = np.random.default_rng(123 + int(kappa_true * 100))
    S = 3000
    beta = 1.0
    rho = kappa_true

    naive_cov, corr_cov = [], []
    for _ in range(8):
        lam = rng.gamma(2.0, 2.0, S) + 0.3
        mu = power_law_fixed_point(lam, beta, rho)
        assert mu is not None

        # Identify kappa via DiD (method never observes lam directly for this).
        # The identifying shock must keep TREATED cells sub-runaway:
        # kappa*(1+delta) < 1. A fixed large delta drives high-kappa treated
        # cells past the runaway pole and breaks identification, so scale it.
        delta = min(0.6, max(0.05, 0.9 / rho - 1.0)) if rho > 0 else 0.6
        treated = rng.random(S) < 0.5
        rho_vec = np.where(treated, rho * (1 + delta), rho)
        mu_post = _hetero_fixed_point(lam, beta, rho_vec)
        res = identify_kappa_did(
            mu[treated], mu[~treated], mu_post[treated], mu_post[~treated],
            delta=delta, beta=beta, lam=lam, treated_mask=treated,
        )
        khat = res.kappa_hat if res.recovered else 0.0

        y_latent = poisson.rvs(lam, random_state=rng)
        # Naive interval from the recorded rate mu.
        lo_n, hi_n = poisson.ppf(0.05, mu), poisson.ppf(0.95, mu)
        naive_cov.append(float(np.mean((y_latent >= lo_n) & (y_latent <= hi_n))))
        # Corrected interval on the deflated latent rate.
        pi = latent_prediction_interval(mu, khat, alpha=0.10)
        keep = ~should_abstain(mu, khat)
        if keep.sum() > 0:
            corr_cov.append(float(np.mean(
                (y_latent[keep] >= pi["lower"][keep]) & (y_latent[keep] <= pi["upper"][keep])
            )))

    mean_corr = float(np.mean(corr_cov))
    mean_naive = float(np.mean(naive_cov))
    # Corrected coverage is near nominal across the whole feedback range...
    assert mean_corr >= 0.85
    # ...and strictly better than naive once feedback meaningfully degrades the
    # naive interval. At mild gain (kappa <= 0.3) the naive record is only
    # slightly biased, so the honest claim is "no worse than naive"; the strict
    # improvement is asserted where naive actually breaks (kappa >= 0.5).
    if kappa_true >= 0.5:
        assert mean_corr > mean_naive + 0.05
    else:
        assert mean_corr >= mean_naive - 0.02
