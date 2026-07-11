"""Feedback-corrected latent prediction intervals (kappa-sensitivity model).

The prior literature (Ensign et al. 2018; and later feedback-diagnosis work)
*diagnoses* observation-biased feedback but does not *correct* it. This module
provides a constructive correction *conditional on an assumed feedback gain*
``kappa``: it deflates the recorded-scale forecast back to a latent scale and
issues a prediction interval for that latent process.

HONESTY NOTE (retraction): ``kappa`` is **NOT point-identified from passive
data** — an earlier claim that a difference-in-differences design point-identifies
it was over-stated and is retracted (see ``docs/AUDIT_2026-07.md``). Treat
``kappa`` as a **sensitivity parameter**: sweep it (see
:mod:`civicsafe.theory.sensitivity`) and report how conclusions move. The
*identified* latent field for this project is the OICC estimate
(:func:`oicc.leave_pivot_out_conformal`), which needs no feedback-gain
assumption; routing should prefer that field via
:func:`civicsafe.routing.feedback_aware.oicc_routing_field`.

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

This is a sensitivity-model counterpart to the impossibility result: the
confidently-wrong state cannot be detected from passive data, so rather than
claim a single identified ``kappa``, we expose it as a knob and lean on OICC for
the identified estimate. See ``docs/RESULTS_latent_correction.md`` for the
coverage table across the ``kappa`` sweep.
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
