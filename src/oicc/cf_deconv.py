"""Nonparametric characteristic-function deconvolution for OICC.

We observe the leave-pivot-out residual  R = S + eps_pivot, where
  S = theta - theta_hat   (the infeasible latent-recovery error), and
  eps_pivot               (the pivot channel's idiosyncratic noise),
independent of S under the model.  We want the QUANTILES of S (to build a latent
prediction interval), not just its variance.

Characteristic functions multiply under convolution:

    phi_R(t) = phi_S(t) * phi_eps(t)   =>   phi_S(t) = phi_R(t) / phi_eps(t).

phi_R is estimated by the empirical CF of the R-sample.  phi_eps is the CF of the
pivot noise; we model it as N(0, sigma1^2) with sigma1^2 estimated from the
one-factor moments, so phi_eps(t) = exp(-sigma1^2 t^2 / 2) (a *supersmooth*
error, which makes the division ill-posed at high frequency).

Regularization (Neumann 1997 / Delaigle-Hall-Meister): multiply by a smooth
flat-top spectral kernel that is 1 for |t| <= T and tapers to 0, with the
cutoff T chosen where |phi_eps(t)| falls below a floor so the noise cannot
explode.  The density of S is recovered by numerical Fourier inversion on a grid,
clipped to be non-negative and renormalized; quantiles come from its CDF.

Robustness: if the deconvolved density is degenerate (all mass in one bin, NaNs,
etc.) we FALL BACK to a moment-matched Gaussian with Var(S)=Var(R)-sigma1^2 so the
caller always gets finite, sensible quantiles.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ArrayF = np.ndarray

_VAR_FLOOR = 1e-6

# numpy 2.0 renamed np.trapz -> np.trapezoid (np.trapz deprecated, slated for
# removal). Bind whichever exists so the module runs identically on numpy 1.x
# and 2.x without a DeprecationWarning (and without breaking `python -W error`).
_trapz = getattr(np, "trapezoid", getattr(np, "trapz"))


@dataclass
class DeconvDensity:
    """Recovered density/quantiles of the latent-recovery error S.

    grid : (M,) support grid for S.
    density : (M,) non-negative, integrates to 1 over `grid`.
    cdf : (M,) cumulative distribution on `grid`.
    var_s : float, Var(S) used (moment target).
    method : "cf" (deconvolved) or "gaussian" (fallback).
    """

    grid: ArrayF
    density: ArrayF
    cdf: ArrayF
    var_s: float
    method: str

    def quantile(self, p: float | ArrayF) -> ArrayF:
        """Interpolated quantile(s) of S from the recovered CDF."""
        p = np.atleast_1d(np.asarray(p, dtype=float))
        if np.any((p < 0) | (p > 1)):
            raise ValueError("quantile probabilities must be in [0, 1]")
        q = np.interp(p, self.cdf, self.grid)
        return q if q.size > 1 else float(q[0])


def _empirical_cf(sample: ArrayF, t: ArrayF) -> ArrayF:
    """Empirical characteristic function phi(t) = mean(exp(i t X)), vectorized."""
    # shape (len(t), n) -> mean over n
    return np.exp(1j * np.outer(t, sample)).mean(axis=1)


def deconvolve_error_law(
    residual: ArrayF,
    pivot_noise_var: float,
    *,
    grid_size: int = 512,
    grid_span: float = 8.0,
    cf_floor: float = 0.01,
) -> DeconvDensity:
    """Recover the law of S from R = S + eps_pivot by CF deconvolution.

    Parameters
    ----------
    residual : (n,) array, the observed leave-pivot-out residual R.
    pivot_noise_var : float, Var(eps_pivot) (>= small epsilon).
    grid_size : int, number of support points for S (power of 2 is efficient).
    grid_span : float, half-width of the S-grid in units of sd(R).
    cf_floor : float, spectral cutoff: drop frequencies where |phi_eps(t)| < floor.

    Returns
    -------
    DeconvDensity
    """
    R = np.asarray(residual, dtype=float)
    if R.ndim != 1 or R.size < 8:
        raise ValueError(f"residual must be 1-D with >= 8 points; got {R.shape}")
    sigma1_sq = max(float(pivot_noise_var), _VAR_FLOOR)

    var_r = float(np.var(R))
    var_s = max(var_r - sigma1_sq, _VAR_FLOOR)
    sd_r = np.sqrt(max(var_r, _VAR_FLOOR))
    center = float(np.median(R))

    def _gaussian_fallback() -> DeconvDensity:
        grid = np.linspace(center - grid_span * sd_r,
                           center + grid_span * sd_r, grid_size)
        sd_s = np.sqrt(var_s)
        dens = np.exp(-0.5 * ((grid - center) / sd_s) ** 2)
        dens = dens / _trapz(dens, grid)
        cdf = np.concatenate([[0.0], np.cumsum((dens[1:] + dens[:-1]) / 2
                                               * np.diff(grid))])
        cdf = cdf / cdf[-1]
        return DeconvDensity(grid, dens, cdf, var_s, "gaussian")

    # frequency grid; cut off where the Gaussian error CF gets too small.
    # |phi_eps(t)| = exp(-sigma1^2 t^2/2) = cf_floor  =>  t_max.
    t_max = np.sqrt(max(-2.0 * np.log(cf_floor) / sigma1_sq, 1e-6))
    t = np.linspace(-t_max, t_max, grid_size)
    dt = t[1] - t[0]

    phi_R = _empirical_cf(R - center, t)          # center for numerical stability
    phi_eps = np.exp(-0.5 * sigma1_sq * t**2)
    # flat-top taper: stays ~1 across the passband, smooth cosine rolloff only in
    # the outer 30% of the band (preserves distributional shape, kills ringing).
    rel = np.abs(t) / t_max
    roll_start = 0.7
    taper = np.ones_like(t)
    mask = rel > roll_start
    taper[mask] = 0.5 * (1.0 + np.cos(
        np.pi * (rel[mask] - roll_start) / (1.0 - roll_start)))
    phi_S = phi_R / phi_eps * taper

    # Fourier inversion onto the S-grid.
    grid = np.linspace(center - grid_span * sd_r,
                       center + grid_span * sd_r, grid_size)
    # f_S(x) = 1/(2 pi) integral exp(-i t x) phi_S(t) dt
    # numerical: sum over t of exp(-i t (x-center)) phi_S(t) dt / (2 pi)
    E = np.exp(-1j * np.outer(grid - center, t))   # (M, len t)
    dens = (E @ phi_S).real * dt / (2.0 * np.pi)

    # sanitize: clip negatives, handle degeneracy, renormalize.
    if not np.all(np.isfinite(dens)):
        return _gaussian_fallback()
    dens = np.clip(dens, 0.0, None)
    area = _trapz(dens, grid)
    if area <= 1e-8 or np.count_nonzero(dens) < 3:
        return _gaussian_fallback()
    dens = dens / area

    cdf = np.concatenate([[0.0], np.cumsum((dens[1:] + dens[:-1]) / 2
                                           * np.diff(grid))])
    if cdf[-1] <= 0:
        return _gaussian_fallback()
    cdf = cdf / cdf[-1]

    dd = DeconvDensity(grid, dens, cdf, var_s, "cf")
    # Variance calibration: Var(S) = Var(R) - sigma1^2 is known EXACTLY from the
    # moment identity, but CF deconvolution's spread is biased in hard (low-SNR)
    # regimes. Rescale the recovered law about its median to match the known
    # variance, preserving SHAPE (skew/tails) while fixing SPREAD. This is a
    # standard, principled moment correction.
    return _match_variance(dd, var_s)


def _match_variance(dd: DeconvDensity, target_var: float) -> DeconvDensity:
    """Rescale a DeconvDensity about its median so Var == target_var (shape kept)."""
    grid, dens = dd.grid, dd.density
    m = float(_trapz(grid * dens, grid))
    cur_var = float(_trapz((grid - m) ** 2 * dens, grid))
    if cur_var <= _VAR_FLOOR or not np.isfinite(cur_var):
        return dd
    scale = float(np.sqrt(max(target_var, _VAR_FLOOR) / cur_var))
    new_grid = m + (grid - m) * scale
    # density transforms as f_new(new_grid) = f_old(grid)/scale (Jacobian)
    new_dens = dens / scale
    area = _trapz(new_dens, new_grid)
    if area <= 1e-8:
        return dd
    new_dens = new_dens / area
    new_cdf = np.concatenate([[0.0], np.cumsum((new_dens[1:] + new_dens[:-1]) / 2
                                               * np.diff(new_grid))])
    new_cdf = new_cdf / new_cdf[-1]
    return DeconvDensity(new_grid, new_dens, new_cdf, target_var, dd.method)
