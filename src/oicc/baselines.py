"""Baseline comparison for latent-rate recovery (empirical defensibility).

A reviewer's first question is "does the deconvolution actually beat the obvious
alternatives?" This module answers it on synthetic ground truth, where the true
latent theta is known, comparing OICC's multi-channel BLUP against:

  * best single channel        (bias-centered, re-anchored)  -- naive record use
  * naive average of channels  (equal weight)                -- the folk method
  * reporting-rate scale-up     (divide the pivot by an assumed reporting rate q)
                                                              -- criminology incumbent

OICC (BLUP) is expected to win on RMSE under the maintained assumptions, and --
the honest point -- the reporting-rate baseline hits the SAME wall OICC does
(an assumed nuisance you cannot test), which is exactly OICC's framing: same
ceiling, but with a *testable* over-identification restriction and finite-sample
conformal coverage that the incumbents lack.

`compare_baselines` returns a table of mean RMSE(log-rate) per method; the OICC
row should be lowest. Used by the reproduction script and the paper's baseline
table.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from oicc.measurement import _as_2d_channels
from oicc.deconvolve import deconvolve_blup

ArrayF = np.ndarray


@dataclass
class BaselineComparison:
    """Mean RMSE(log-rate) of each method vs known ground truth.

    methods : ordered list of method names.
    rmse : dict name -> mean RMSE over the trials.
    n_trials : int.
    winner : name of the lowest-RMSE method (should be 'oicc_blup').
    """

    methods: list[str]
    rmse: dict[str, float]
    n_trials: int
    winner: str = field(default="")

    def __post_init__(self) -> None:
        if self.rmse:
            self.winner = min(self.rmse, key=self.rmse.get)


def _affine_align(estimate: ArrayF, truth: ArrayF) -> float:
    """RMSE of the best affine rescaling of `estimate` onto `truth` (the latent
    scale is only identified up to an affine map, so align before scoring)."""
    A = np.vstack([np.ones_like(estimate), estimate]).T
    coef, *_ = np.linalg.lstsq(A, truth, rcond=None)
    return float(np.sqrt(np.mean((A @ coef - truth) ** 2)))


def compare_baselines(
    n: int = 4000,
    K: int = 4,
    n_trials: int = 20,
    seed0: int = 0,
    reporting_rate: float = 0.6,
) -> BaselineComparison:
    """Compare OICC BLUP against single-channel, naive-average, and reporting-rate.

    Uses `oicc.generate` (valid one-factor data) with known latent theta.
    Returns mean RMSE(log-rate) per method; OICC should win.
    """
    from oicc.measurement import generate  # local import to avoid cycle

    names = ["best_single", "naive_average", "reporting_rate_scaleup", "oicc_blup"]
    acc = {nm: [] for nm in names}

    for t in range(n_trials):
        ch = generate(n=n, seed=seed0 + t, K=K)
        Y = _as_2d_channels(ch.log_channels)
        theta = ch.theta

        # best single channel (choose the one with lowest aligned RMSE)
        best = min(_affine_align(Y[c], theta) for c in range(K))
        acc["best_single"].append(best)

        # naive equal-weight average of channels
        acc["naive_average"].append(_affine_align(Y.mean(axis=0), theta))

        # reporting-rate scale-up: treat pivot as under-reported by factor q,
        # so latent ~ pivot / q on the RATE scale. On the log scale this is a
        # constant shift (absorbed by the affine alignment), so it is exactly the
        # single pivot channel -- demonstrating the incumbent cannot separate the
        # nuisance from the level. We score it honestly as such.
        pivot = Y[0]
        scaled = pivot - np.log(max(reporting_rate, 1e-6))
        acc["reporting_rate_scaleup"].append(_affine_align(scaled, theta))

        # OICC multi-channel BLUP
        est = deconvolve_blup(Y)
        acc["oicc_blup"].append(_affine_align(est.theta_hat, theta))

    rmse = {nm: float(np.mean(v)) for nm, v in acc.items()}
    return BaselineComparison(methods=names, rmse=rmse, n_trials=n_trials)


def compare_baselines_confounded(
    n: int = 6000,
    K: int = 4,
    Q: int = 2,
    n_trials: int = 20,
    seed0: int = 0,
    cm_strength: float = 1.0,
) -> BaselineComparison:
    """Compare under a COMMON-MODE confounder, where naive methods break.

    Adds OICC's proximal point-identification (uses Q>=2 negative controls) to the
    comparison. Under confounding the single-channel / naive-average / reporting-
    rate baselines are all biased by the confounder; only proximal point-ID
    recovers the true latent-variance scale. We score RMSE of the recovered
    per-area latent against ground truth (all affine-aligned).
    """
    from oicc.measurement import generate_proximal
    from oicc.proximal import proximal_deconfound

    names = ["best_single", "naive_average", "oicc_blup_naive",
             "oicc_proximal"]
    acc = {nm: [] for nm in names}

    for t in range(n_trials):
        d = generate_proximal(n=n, seed=seed0 + t, K=K, Q=Q,
                              cm_strength=cm_strength)
        Y = _as_2d_channels(d.signal_channels)
        theta = d.theta

        acc["best_single"].append(min(_affine_align(Y[c], theta)
                                      for c in range(K)))
        acc["naive_average"].append(_affine_align(Y.mean(axis=0), theta))
        acc["oicc_blup_naive"].append(
            _affine_align(deconvolve_blup(Y).theta_hat, theta))
        # proximal correction: residualize signals on the controls, then BLUP
        pc = proximal_deconfound(Y, d.controls)
        acc["oicc_proximal"].append(
            _affine_align(deconvolve_blup(pc.deconfounded).theta_hat, theta))

    rmse = {nm: float(np.mean(v)) for nm, v in acc.items()}
    return BaselineComparison(methods=names, rmse=rmse, n_trials=n_trials)
