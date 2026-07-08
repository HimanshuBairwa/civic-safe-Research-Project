"""Feedback-corrected latent prediction intervals.

The prior literature (Ensign et al. 2018; van Amsterdam et al. 2025; Algometrics
2026) *diagnoses* observation-biased feedback but does not *correct* it. This
module provides the constructive step: given the feedback gain ``kappa`` (which
is point-identified by the difference-in-differences design of
:func:`civicsafe.theory.feedback_law.identify_kappa_did`), it deflates the
recorded-scale forecast back to the latent scale and issues a prediction
interval that recovers coverage of the *true* latent process rather than the
biased record.

Mechanism
---------
At the feedback fixed point ``mu_s = lambda_s * m_s`` with recording multiplier
``m_s = (mu_s / M) ** kappa`` (``M = mean(mu)``). Hence the latent rate is
recovered by deflation,

    lambda_hat_s = mu_s / m_s = mu_s ** (1 - kappa) * M ** kappa,

and a latent-valid prediction interval is formed from the count-distribution
quantiles evaluated at ``lambda_hat_s`` (optionally widened by a conformal
margin calibrated on a gold-standard anchor). When ``kappa`` approaches the
runaway threshold ``1`` — or the deflation is too uncertain — the interval is
flagged for **abstention** rather than issued with false confidence.

This is the deployable counterpart to the impossibility result: the
confidently-wrong state cannot be detected from passive data, but once ``kappa``
is measured by intervention, the correction below restores honest coverage. See
``docs/RESULTS_latent_correction.md`` for the verified coverage table and the
non-obvious operating condition (the identifying shock must keep treated cells
sub-runaway, ``kappa*(1+delta) < 1``).
"""

from __future__ import annotations

import numpy as np

from civicsafe.theory import _poisson as poisson

__all__ = [
    "recording_multiplier",
    "deflate_latent_rate",
    "latent_prediction_interval",
    "should_abstain",
]


def recording_multiplier(mu: np.ndarray, kappa: float) -> np.ndarray:
    """Estimated per-cell recording inflation ``m_s = (mu_s / mean(mu)) ** kappa``.

    Args:
        mu: Tracked recorded-rate estimates, shape ``(S,)``, positive.
        kappa: Feedback gain in ``[0, 1)`` (e.g. from the DiD identification).

    Returns:
        The recording multiplier per cell; ``> 1`` means over-recorded.
    """
    mu = np.asarray(mu, dtype=float)
    m = mu.mean()
    if m <= 0:
        raise ValueError("mean(mu) must be positive")
    return (mu / m) ** kappa


def deflate_latent_rate(mu: np.ndarray, kappa: float) -> np.ndarray:
    """Recover the latent rate ``lambda_hat = mu / m`` by deflating the record.

    Args:
        mu: Recorded-rate estimates, shape ``(S,)``.
        kappa: Feedback gain in ``[0, 1)``.

    Returns:
        Deflated latent-rate estimates; equals ``mu`` when ``kappa == 0``.
    """
    return np.asarray(mu, dtype=float) / recording_multiplier(mu, kappa)


def latent_prediction_interval(
    mu: np.ndarray,
    kappa: float,
    alpha: float = 0.1,
    conformal_margin: np.ndarray | float = 0.0,
) -> dict[str, np.ndarray]:
    """Feedback-corrected prediction interval for the *latent* count process.

    Forms Poisson quantiles at the deflated rate ``lambda_hat`` (a valid
    prediction interval for a ``Poisson(lambda_s)`` draw when ``kappa`` is
    correct), optionally widened by a conformal margin calibrated on an anchor.

    Args:
        mu: Recorded-rate estimates, shape ``(S,)``.
        kappa: Feedback gain in ``[0, 1)``.
        alpha: Target miscoverage (coverage ``1 - alpha``).
        conformal_margin: Extra half-width (scalar or per-cell) from anchor
            calibration; ``0`` gives the pure distributional interval.

    Returns:
        Dict with ``lower``, ``upper`` (latent-scale interval) and
        ``lambda_hat`` (the deflated rate).
    """
    lam_hat = deflate_latent_rate(mu, kappa)
    lam_hat = np.clip(lam_hat, 1e-6, None)
    lo = poisson.ppf(alpha / 2.0, lam_hat).astype(float)
    hi = poisson.ppf(1.0 - alpha / 2.0, lam_hat).astype(float)
    margin = np.asarray(conformal_margin, dtype=float)
    lower = np.clip(lo - margin, 0.0, None)
    upper = hi + margin
    return {"lower": lower, "upper": upper, "lambda_hat": lam_hat}


def should_abstain(
    mu: np.ndarray,
    kappa: float,
    kappa_runaway: float = 0.9,
    max_multiplier: float = 5.0,
) -> np.ndarray:
    """Flag cells where correction is untrustworthy and the system should abstain.

    Abstention triggers when the feedback gain is near the runaway threshold
    (the correction's variance explodes as ``kappa -> 1``) or the required
    deflation is extreme (a cell recorded far above the mean, where a small
    ``kappa`` error causes a large latent error).

    Args:
        mu: Recorded-rate estimates, shape ``(S,)``.
        kappa: Estimated feedback gain.
        kappa_runaway: Global abstention threshold on ``kappa``.
        max_multiplier: Per-cell abstention threshold on the recording multiplier.

    Returns:
        Boolean mask, ``True`` where the system should abstain.
    """
    if kappa >= kappa_runaway:
        return np.ones(np.shape(mu), dtype=bool)
    m = recording_multiplier(mu, kappa)
    return (m > max_multiplier) | (m < 1.0 / max_multiplier)
