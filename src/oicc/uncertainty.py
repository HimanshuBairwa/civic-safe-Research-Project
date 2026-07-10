"""Uncertainty quantification for OICC estimators via the (block) bootstrap.

Every point estimate the package produces -- factor loadings, Var(theta), the
over-identification statistic, and the proximal point-identified variances -- is
a functional of the sample. We attach nonparametric bootstrap confidence
intervals so that reported numbers come with honest uncertainty, not bare points.

Two resampling schemes:
  * i.i.d. bootstrap over units (default), and
  * MOVING-BLOCK bootstrap (`block>1`) for panels with serial dependence
    (e.g. state-year data), which preserves within-block correlation.

All intervals are percentile bootstrap CIs; they are finite-sample honest about
sampling variability (they do NOT capture model misspecification -- that is what
the over-ID test and the proximal/sensitivity machinery are for).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oicc.measurement import _as_2d_channels
from oicc.moments import estimate_factor_moments

ArrayF = np.ndarray


@dataclass
class BootstrapCI:
    """A bootstrap point estimate with a percentile confidence interval.

    estimate : float, the point estimate on the full sample.
    lower, upper : float, the (level)-percentile CI bounds.
    se : float, bootstrap standard error.
    level : float, nominal coverage of the interval (e.g. 0.9).
    n_boot : int, number of bootstrap replications used.
    """

    estimate: float
    lower: float
    upper: float
    se: float
    level: float
    n_boot: int


def _resample_indices(n: int, block: int, rng: np.random.Generator) -> ArrayF:
    """Return a length-n resample: i.i.d. if block<=1, else moving-block."""
    if block <= 1:
        return rng.integers(0, n, n)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, max(n - block + 1, 1), n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
    return np.clip(idx, 0, n - 1)


def _percentile_ci(
    point: float, boot: ArrayF, level: float, n_boot: int
) -> BootstrapCI:
    boot = boot[np.isfinite(boot)]
    if boot.size < 2:
        return BootstrapCI(float(point), float(point), float(point), 0.0,
                           level, n_boot)
    a = (1.0 - level) / 2.0
    lo, hi = np.quantile(boot, [a, 1.0 - a])
    return BootstrapCI(float(point), float(lo), float(hi),
                       float(np.std(boot, ddof=1)), level, n_boot)


def bootstrap_moments(
    log_channels: ArrayF,
    *,
    pivot: int = 0,
    n_boot: int = 400,
    block: int = 1,
    level: float = 0.9,
    seed: int = 0,
) -> dict[str, object]:
    """Bootstrap CIs for Var(theta) and each loading beta_c.

    Returns
    -------
    dict with keys:
      "var_theta" : BootstrapCI
      "beta"      : list[BootstrapCI], one per channel
    """
    Y = _as_2d_channels(log_channels)
    K, n = Y.shape
    rng = np.random.default_rng(seed)

    fm0 = estimate_factor_moments(Y, pivot=pivot)
    vboot = np.empty(n_boot)
    bboot = np.empty((n_boot, K))
    for b in range(n_boot):
        idx = _resample_indices(n, block, rng)
        try:
            fm = estimate_factor_moments(Y[:, idx], pivot=pivot)
            vboot[b] = fm.var_theta
            bboot[b] = fm.beta
        except Exception:
            vboot[b] = np.nan
            bboot[b] = np.nan
    return {
        "var_theta": _percentile_ci(fm0.var_theta, vboot, level, n_boot),
        "beta": [_percentile_ci(fm0.beta[c], bboot[:, c], level, n_boot)
                 for c in range(K)],
    }


def bootstrap_point_id(
    signal_channels: ArrayF,
    controls: ArrayF,
    *,
    pivot: int = 0,
    n_boot: int = 400,
    block: int = 1,
    level: float = 0.9,
    seed: int = 0,
) -> dict[str, BootstrapCI]:
    """Bootstrap CIs for the proximal point-ID: Var(theta)_clean, Var(theta)_naive,
    and Var(W). Uses the same resampling on units for signals and controls jointly.
    """
    from oicc.proximal import point_identify  # local import (avoid cycle)

    Y = _as_2d_channels(signal_channels)
    N = np.asarray(controls, dtype=float)
    if N.ndim != 2 or N.shape[1] != Y.shape[1]:
        raise ValueError("controls must be (Q, n) with n matching the channels")
    K, n = Y.shape
    rng = np.random.default_rng(seed)

    r0 = point_identify(Y, N, pivot=pivot)
    clean = np.empty(n_boot)
    naive = np.empty(n_boot)
    varw = np.empty(n_boot)
    for b in range(n_boot):
        idx = _resample_indices(n, block, rng)
        try:
            r = point_identify(Y[:, idx], N[:, idx], pivot=pivot)
            clean[b] = r.var_theta_clean
            naive[b] = r.var_theta_naive
            varw[b] = r.var_W
        except Exception:
            clean[b] = naive[b] = varw[b] = np.nan
    return {
        "var_theta_clean": _percentile_ci(r0.var_theta_clean, clean, level, n_boot),
        "var_theta_naive": _percentile_ci(r0.var_theta_naive, naive, level, n_boot),
        "var_W": _percentile_ci(r0.var_W, varw, level, n_boot),
    }
