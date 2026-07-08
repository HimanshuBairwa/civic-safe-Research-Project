"""Sensitivity analysis for the feedback correction — how much can we trust it?

The correction deflates the record by an *estimated* feedback gain ``kappa_hat``.
If that estimate is wrong, the deflation is wrong and latent coverage degrades.
This module quantifies that dependence so the guarantee is reported honestly:

* :func:`latent_coverage_at_kappa` — the latent coverage achieved when the
  correction uses ``kappa_used`` while the world runs at ``kappa_true``.
* :func:`sensitivity_curve` — latent coverage as a function of the used gain,
  tracing how gracefully (or not) the correction degrades under misspecification.
* :func:`robustness_value` — the largest gain-misspecification the coverage claim
  tolerates before falling below a floor (a Cinelli--Hazlett-style robustness
  value). It answers: *how precisely must the natural experiment identify*
  ``kappa`` *for the correction to hold?*

Together these turn "we assume kappa is known" into a reported, bounded operating
envelope — the difference between a fragile method and a deployable one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from civicsafe.theory import _poisson as poisson
from civicsafe.theory.feedback_law import power_law_fixed_point
from civicsafe.theory.latent_correction import latent_prediction_interval, should_abstain

__all__ = [
    "latent_coverage_at_kappa",
    "sensitivity_curve",
    "robustness_value",
    "RobustnessResult",
]


def latent_coverage_at_kappa(
    mu: np.ndarray,
    y_latent: np.ndarray,
    kappa_used: float,
    alpha: float = 0.10,
    respect_abstention: bool = True,
) -> float:
    """Latent coverage when the corrector uses ``kappa_used``.

    Args:
        mu: Recorded rates (the feedback fixed point of the true world), ``(S,)``.
        y_latent: Realized latent counts to score coverage against, ``(S,)``.
        kappa_used: The (possibly misspecified) gain fed to the correction.
        alpha: Target miscoverage.
        respect_abstention: If True, coverage is measured only on non-abstained
            cells (the honest operating set).

    Returns:
        Empirical coverage of ``y_latent`` by the corrected interval.
    """
    interval = latent_prediction_interval(mu, kappa_used, alpha=alpha)
    lo, hi = interval["lower"], interval["upper"]
    covered = (y_latent >= lo) & (y_latent <= hi)
    if respect_abstention:
        keep = ~should_abstain(mu, kappa_used)
        if keep.sum() == 0:
            return float("nan")
        return float(np.mean(covered[keep]))
    return float(np.mean(covered))


def sensitivity_curve(
    kappa_true: float,
    used_grid: np.ndarray | None = None,
    num_cells: int = 4000,
    trials: int = 8,
    alpha: float = 0.10,
    seed: int = 0,
) -> list[dict[str, float]]:
    """Trace latent coverage vs. the gain used by the correction.

    Simulates the true world at ``kappa_true`` (never observed by the corrector),
    then applies the correction with each ``kappa_used`` on the grid.

    Args:
        kappa_true: The world's true feedback gain.
        used_grid: Grid of gains the correction might use. Defaults to a band
            around ``kappa_true``.
        num_cells: Cells per trial.
        trials: Trials averaged per grid point.
        alpha: Target miscoverage.
        seed: RNG seed.

    Returns:
        List of ``{"kappa_used", "coverage"}`` dicts.
    """
    if used_grid is None:
        used_grid = np.clip(np.linspace(kappa_true - 0.3, kappa_true + 0.3, 13), 0.0, 0.98)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []
    for kappa_used in used_grid:
        covs = []
        for _ in range(trials):
            lam = rng.gamma(2.0, 2.0, num_cells) + 0.3
            mu = power_law_fixed_point(lam, 1.0, kappa_true)
            if mu is None:
                continue
            y_latent = poisson.rvs(lam, random_state=rng)
            covs.append(latent_coverage_at_kappa(mu, y_latent, float(kappa_used), alpha))
        rows.append({
            "kappa_used": float(kappa_used),
            "coverage": float(np.nanmean(covs)) if np.any(np.isfinite(covs)) else float("nan"),
        })
    return rows


@dataclass
class RobustnessResult:
    """Result of a robustness-value computation.

    Attributes:
        kappa_true: The world's true gain.
        coverage_floor: The minimum acceptable latent coverage.
        safe_low: Smallest used-gain keeping coverage >= floor.
        safe_high: Largest used-gain keeping coverage >= floor.
        robustness_value: Half-width of the safe band around ``kappa_true`` — the
            largest gain error tolerated. ``0`` if even the exact gain fails.
    """

    kappa_true: float
    coverage_floor: float
    safe_low: float
    safe_high: float
    robustness_value: float


def robustness_value(
    kappa_true: float,
    coverage_floor: float = 0.85,
    alpha: float = 0.10,
    num_cells: int = 4000,
    trials: int = 8,
    resolution: int = 25,
    seed: int = 0,
) -> RobustnessResult:
    """Largest gain-misspecification the coverage claim tolerates.

    Sweeps ``kappa_used`` around ``kappa_true`` and finds the contiguous band
    where mean latent coverage stays at or above ``coverage_floor``. The
    robustness value is the smaller distance from ``kappa_true`` to a band edge —
    i.e. how far the identified gain may be off before the guarantee breaks.

    Args:
        kappa_true: The world's true feedback gain.
        coverage_floor: Minimum acceptable latent coverage.
        alpha: Target miscoverage.
        num_cells: Cells per trial.
        trials: Trials averaged per grid point.
        resolution: Number of grid points in the sweep.
        seed: RNG seed.

    Returns:
        A :class:`RobustnessResult`.
    """
    grid = np.clip(np.linspace(kappa_true - 0.4, kappa_true + 0.4, resolution), 0.0, 0.98)
    curve = sensitivity_curve(
        kappa_true, used_grid=grid, num_cells=num_cells, trials=trials,
        alpha=alpha, seed=seed,
    )
    ok = np.array([r["coverage"] >= coverage_floor for r in curve])
    used = np.array([r["kappa_used"] for r in curve])

    # Find the contiguous OK band containing kappa_true (nearest grid point).
    i0 = int(np.argmin(np.abs(used - kappa_true)))
    if not ok[i0]:
        return RobustnessResult(kappa_true, coverage_floor, kappa_true, kappa_true, 0.0)
    lo_i = i0
    while lo_i - 1 >= 0 and ok[lo_i - 1]:
        lo_i -= 1
    hi_i = i0
    while hi_i + 1 < len(ok) and ok[hi_i + 1]:
        hi_i += 1
    safe_low, safe_high = float(used[lo_i]), float(used[hi_i])
    rv = float(min(kappa_true - safe_low, safe_high - kappa_true))
    return RobustnessResult(kappa_true, coverage_floor, safe_low, safe_high, rv)
