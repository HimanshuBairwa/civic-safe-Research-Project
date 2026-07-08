"""Robustness of the feedback correction to a misspecified recording model.

The correction (:mod:`civicsafe.theory.latent_correction`) deflates the record
by an *assumed* recording multiplier ``m_hat`` (the power-law
``(mu/mean(mu))**kappa``). A referee's sharpest objection is: *what if the true
recording mechanism is not that power law?* This module answers it with a
distribution-free sensitivity guarantee in the spirit of Rosenbaum's marginal
sensitivity model (and Jin--Ren--Candès 2023 for conformal).

Sensitivity model
-----------------
Assume the true recording multiplier ``m_true`` lies within a factor ``Gamma``
of the assumed one:

    m_true(s) / m_hat(s) in [1/Gamma, Gamma]   for every cell s,   Gamma >= 1.

Then the true latent rate ``lambda = mu / m_true`` and the estimated latent rate
``lambda_hat = mu / m_hat`` satisfy ``lambda / lambda_hat in [1/Gamma, Gamma]``.
A ``Gamma``-inflated interval --- lower quantile evaluated at
``lambda_hat / Gamma``, upper quantile at ``lambda_hat * Gamma`` --- therefore
covers a ``Poisson(lambda)`` draw for *any* admissible ``m_true``.

**Proposition (verified numerically, `tests/test_correction_robustness.py`).**
Under the sensitivity model above, the ``Gamma``-inflated corrected interval has
latent coverage at least the nominal ``1 - alpha`` for every recording model in
the ``Gamma``-band; ``Gamma = 1`` recovers the ordinary corrected interval. The
price is a bounded increase in interval width that grows smoothly with
``Gamma`` --- the quantified cost of not knowing the recording model.

The ``robustness_gamma`` helper reports the largest ``Gamma`` under which a target
coverage is still met on given data: an interpretable *robustness value* stating
how badly the recording model may be misspecified before the guarantee lapses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from civicsafe.theory import _poisson as poisson
from civicsafe.theory.latent_correction import deflate_latent_rate

__all__ = [
    "robust_latent_interval",
    "robustness_gamma",
    "GammaRobustnessResult",
]


def robust_latent_interval(
    mu: np.ndarray,
    kappa: float,
    gamma: float = 1.0,
    alpha: float = 0.10,
) -> dict[str, np.ndarray]:
    """Gamma-inflated feedback-corrected interval for the latent process.

    Deflates ``mu`` to ``lambda_hat`` by the assumed power-law multiplier, then
    inflates the interval to cover any true recording multiplier within a factor
    ``gamma``: lower quantile at ``lambda_hat / gamma``, upper at
    ``lambda_hat * gamma``.

    Args:
        mu: Recorded rates (feedback fixed point), shape ``(S,)``.
        kappa: Identified feedback gain used to form the assumed multiplier.
        gamma: Sensitivity factor ``>= 1``; ``1`` gives the ordinary correction.
        alpha: Target miscoverage (coverage ``1 - alpha``).

    Returns:
        Dict with ``lower``, ``upper`` (latent-scale interval) and
        ``lambda_hat`` (the deflated point estimate).

    Raises:
        ValueError: if ``gamma < 1``.
    """
    if gamma < 1.0:
        raise ValueError(f"gamma must be >= 1, got {gamma}")
    lam_hat = np.clip(deflate_latent_rate(np.asarray(mu, dtype=float), kappa), 1e-6, None)
    lo_rate = np.clip(lam_hat / gamma, 1e-6, None)
    hi_rate = lam_hat * gamma
    lower = poisson.ppf(alpha / 2.0, lo_rate)
    upper = poisson.ppf(1.0 - alpha / 2.0, hi_rate)
    return {
        "lower": np.asarray(lower, dtype=float),
        "upper": np.asarray(upper, dtype=float),
        "lambda_hat": lam_hat,
    }


@dataclass
class GammaRobustnessResult:
    """Result of a recording-model robustness search.

    Attributes:
        gamma_grid: The sensitivity factors evaluated.
        coverage: Latent coverage achieved at each ``gamma``.
        width_ratio: Mean interval width relative to the ``gamma = 1`` interval.
        robustness_gamma: Largest ``gamma`` whose coverage meets the target
            (``1.0`` if even the un-inflated interval already misses).
    """

    gamma_grid: list[float]
    coverage: list[float]
    width_ratio: list[float]
    robustness_gamma: float


def robustness_gamma(
    mu: np.ndarray,
    y_latent: np.ndarray,
    kappa: float,
    alpha: float = 0.10,
    gamma_grid: tuple[float, ...] = (1.0, 1.2, 1.5, 2.0, 3.0),
) -> GammaRobustnessResult:
    """Largest misspecification factor under which target coverage still holds.

    Sweeps ``gamma`` and reports latent coverage and relative width at each,
    plus the largest ``gamma`` meeting the ``1 - alpha`` target — an
    interpretable statement of how badly the recording model may be misspecified
    before the guarantee lapses.

    Args:
        mu: Recorded rates, shape ``(S,)``.
        y_latent: Realized latent counts to score coverage against, shape ``(S,)``.
        kappa: Identified feedback gain.
        alpha: Target miscoverage.
        gamma_grid: Sensitivity factors to evaluate (ascending, first is ``1.0``).

    Returns:
        A :class:`GammaRobustnessResult`.
    """
    target = 1.0 - alpha
    y = np.asarray(y_latent, dtype=float)
    coverage: list[float] = []
    width_ratio: list[float] = []
    base_width: float | None = None
    best_gamma = 1.0

    for g in gamma_grid:
        iv = robust_latent_interval(mu, kappa, gamma=g, alpha=alpha)
        cov = float(np.mean((y >= iv["lower"]) & (y <= iv["upper"])))
        width = float(np.mean(iv["upper"] - iv["lower"] + 1.0))
        if base_width is None:
            base_width = width
        coverage.append(cov)
        width_ratio.append(width / base_width if base_width > 0 else float("nan"))
        if cov >= target:
            best_gamma = max(best_gamma, g)

    return GammaRobustnessResult(
        gamma_grid=list(gamma_grid),
        coverage=coverage,
        width_ratio=width_ratio,
        robustness_gamma=best_gamma,
    )
