"""Minimal Poisson quantile/sampling helpers with an optional SciPy backend.

The theory modules only need two Poisson operations: the inverse CDF (``ppf``)
and random sampling (``rvs``). SciPy provides both, but it is a heavy optional
dependency and is not always installed in lightweight or offline environments.

This module exposes ``ppf`` and ``rvs`` that use SciPy when it is available and
fall back to exact NumPy implementations otherwise. The NumPy ``ppf`` is exact
(not a normal approximation): it accumulates the Poisson CDF via the pmf
recurrence ``p_k = p_{k-1} * lam / k`` and returns the smallest ``k`` with
``CDF(k) >= q`` — identical semantics to ``scipy.stats.poisson.ppf``.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised implicitly by whichever backend is present
    from scipy.stats import poisson as _scipy_poisson

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _scipy_poisson = None
    _HAVE_SCIPY = False

__all__ = ["ppf", "rvs", "HAVE_SCIPY"]

HAVE_SCIPY = _HAVE_SCIPY


def _ppf_numpy(q: float, lam: np.ndarray) -> np.ndarray:
    """Exact Poisson inverse CDF via the pmf recurrence (vectorised over ``lam``)."""
    lam_arr = np.atleast_1d(np.asarray(lam, dtype=float))
    if np.any(lam_arr < 0):
        raise ValueError("lambda must be non-negative")
    if not (0.0 <= q <= 1.0):
        raise ValueError("q must lie in [0, 1]")

    result = np.full(lam_arr.shape, -1.0)
    if q <= 0.0:
        result[:] = 0.0
        return result

    # k=0 term.
    pmf = np.exp(-lam_arr)
    cdf = pmf.copy()
    result[(cdf >= q) & (result < 0)] = 0

    # Upper bound on the search: mean + generous tail (Poisson ~ Normal for
    # large lambda) plus a floor for tiny lambda. Guarantees termination.
    kmax = int(np.max(lam_arr + 12.0 * np.sqrt(lam_arr + 1.0)) + 40.0)
    k = 0
    while np.any(result < 0) and k < kmax:
        k += 1
        pmf = pmf * lam_arr / k
        cdf += pmf
        newly = (cdf >= q) & (result < 0)
        result[newly] = k

    # Numerical safety: anything still unresolved is at the tail bound.
    result[result < 0] = float(kmax)
    return result


def ppf(q: float, lam) -> np.ndarray:
    """Poisson inverse CDF; smallest integer ``k`` with ``CDF(k) >= q``."""
    scalar = np.isscalar(lam) or np.ndim(lam) == 0
    if _HAVE_SCIPY:
        out = np.asarray(_scipy_poisson.ppf(q, lam), dtype=float)
    else:
        out = _ppf_numpy(q, lam)
    if scalar:
        return float(np.atleast_1d(out)[0])
    return np.asarray(out, dtype=float)


def rvs(lam, random_state=None) -> np.ndarray:
    """Draw Poisson samples elementwise from rate array ``lam``."""
    if _HAVE_SCIPY:
        return _scipy_poisson.rvs(lam, random_state=random_state)
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    return rng.poisson(np.asarray(lam, dtype=float))
