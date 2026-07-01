"""
Statistical significance tests for comparing probabilistic forecasts.

This module provides two complementary hypothesis tests for comparing the
predictive accuracy of competing probabilistic crime-rate forecasts, measured
by the Continuous Ranked Probability Score (CRPS).

Implemented tests
-----------------
1. **Diebold–Mariano test** (Diebold & Mariano, 1995) with Newey–West
   heteroscedasticity-and-autocorrelation-consistent (HAC) standard errors
   (Newey & West, 1987).
2. **Temporal block bootstrap** (Politis & Romano, 1994) for paired CRPS
   comparison under temporal dependence.

References
----------
.. [1] Diebold, F. X. & Mariano, R. S. (1995).  "Comparing Predictive
       Accuracy."  *Journal of Business & Economic Statistics*, 13(3),
       253–263.  https://doi.org/10.1080/07350015.1995.10524599
.. [2] Newey, W. K. & West, K. D. (1987).  "A Simple, Positive
       Semi-definite, Heteroskedasticity and Autocorrelation Consistent
       Covariance Matrix."  *Econometrica*, 55(3), 703–708.
       https://doi.org/10.2307/1913610
.. [3] Politis, D. N. & Romano, J. P. (1994).  "The Stationary Bootstrap."
       *Journal of the American Statistical Association*, 89(428),
       1303–1313.  https://doi.org/10.1080/01621459.1994.10476870
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import Tensor

__all__ = [
    "diebold_mariano_test",
    "block_bootstrap_test",
    "compare_forecasts",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MIN_SAMPLE_SIZE: int = 10
"""Minimum number of time-steps required for either test."""


def _validate_crps_pair(crps_1: Tensor, crps_2: Tensor) -> None:
    """Validate that two CRPS tensors are compatible 1-D vectors.

    Parameters
    ----------
    crps_1 : Tensor
        Per-timestep CRPS values for model 1.  Shape ``(T,)``.
    crps_2 : Tensor
        Per-timestep CRPS values for model 2.  Shape ``(T,)``.

    Raises
    ------
    TypeError
        If inputs are not ``torch.Tensor`` instances.
    ValueError
        If tensors are not 1-D, have different lengths, contain NaN/Inf,
        or have fewer than ``_MIN_SAMPLE_SIZE`` elements.
    """
    if not isinstance(crps_1, Tensor) or not isinstance(crps_2, Tensor):
        raise TypeError(
            f"Both inputs must be torch.Tensor, got "
            f"{type(crps_1).__name__} and {type(crps_2).__name__}."
        )
    if crps_1.ndim != 1 or crps_2.ndim != 1:
        raise ValueError(
            f"CRPS tensors must be 1-D, got shapes "
            f"{tuple(crps_1.shape)} and {tuple(crps_2.shape)}."
        )
    if crps_1.shape[0] != crps_2.shape[0]:
        raise ValueError(
            f"CRPS tensors must have the same length, got "
            f"{crps_1.shape[0]} and {crps_2.shape[0]}."
        )
    T = crps_1.shape[0]
    if T < _MIN_SAMPLE_SIZE:
        raise ValueError(
            f"Need at least {_MIN_SAMPLE_SIZE} observations, got T={T}."
        )
    if torch.isnan(crps_1).any() or torch.isnan(crps_2).any():
        raise ValueError("CRPS tensors must not contain NaN values.")
    if torch.isinf(crps_1).any() or torch.isinf(crps_2).any():
        raise ValueError("CRPS tensors must not contain Inf values.")


def _normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function.

    Computed via the complementary error function for numerical stability:

    .. math::

        \\Phi(x) = \\tfrac{1}{2}\\,\\mathrm{erfc}\\!\\bigl(-x / \\sqrt{2}\\bigr)

    Parameters
    ----------
    x : float
        Quantile value.

    Returns
    -------
    float
        Pr(Z <= x) where Z ~ N(0, 1).
    """
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _newey_west_variance(d: np.ndarray, h: int) -> float:
    """Newey–West (1987) HAC estimate of the long-run variance.

    .. math::

        \\hat\\sigma_d^2
        = \\hat\\gamma_0
          + 2 \\sum_{j=1}^{h}
              \\Bigl(1 - \\frac{j}{h+1}\\Bigr)\\,\\hat\\gamma_j

    where :math:`\\hat\\gamma_j = \\frac{1}{T}\\sum_{t=j+1}^{T}
    (d_t - \\bar d)(d_{t-j} - \\bar d)` is the sample auto-covariance
    at lag *j*, and the Bartlett kernel weight
    :math:`w_j = 1 - j/(h+1)` guarantees positive semi-definiteness.

    Parameters
    ----------
    d : np.ndarray
        Loss differential series of length *T*.
    h : int
        Truncation lag (bandwidth).  Typically ``floor(T^{1/3})``.

    Returns
    -------
    float
        The HAC variance estimate :math:`\\hat\\sigma_d^2`.
        Guaranteed non-negative by construction of the Bartlett kernel.
    """
    T = len(d)
    d_demean = d - d.mean()

    # gamma_0 — variance at lag 0
    gamma_0 = float(np.dot(d_demean, d_demean) / T)

    # Accumulate weighted auto-covariances for lags 1..h
    nw_var = gamma_0
    for j in range(1, h + 1):
        # gamma_j = (1/T) * sum_{t=j}^{T-1} d_demean[t] * d_demean[t-j]
        gamma_j = float(np.dot(d_demean[j:], d_demean[:-j]) / T)
        bartlett_weight = 1.0 - j / (h + 1)
        nw_var += 2.0 * bartlett_weight * gamma_j

    # Clamp to a tiny positive value to avoid division by zero
    return max(nw_var, 1e-15)


# ---------------------------------------------------------------------------
# 1. Diebold–Mariano test
# ---------------------------------------------------------------------------


def diebold_mariano_test(
    crps_1: Tensor,
    crps_2: Tensor,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Diebold–Mariano test for equal predictive ability.

    Compares two sets of per-timestep CRPS values using the DM test
    statistic with Newey–West HAC standard errors.

    **Hypotheses**

    * H₀ : E[d_t] = 0  — the two forecasts have equal predictive ability.
    * H₁ depends on ``alternative``:

      - ``'two-sided'``: E[d_t] ≠ 0
      - ``'less'``     : E[d_t] < 0  (model 1 is better)
      - ``'greater'``  : E[d_t] > 0  (model 2 is better)

    **Test statistic** (Diebold & Mariano, 1995):

    .. math::

        d_t       &= \\text{CRPS}_1(t) - \\text{CRPS}_2(t) \\\\
        \\bar{d}  &= \\frac{1}{T} \\sum_{t=1}^{T} d_t \\\\
        DM        &= \\frac{\\bar{d}}{\\sqrt{\\hat\\sigma_d^2 / T}}
                     \\;\\xrightarrow{d}\\; \\mathcal{N}(0, 1)

    where :math:`\\hat\\sigma_d^2` is the Newey–West HAC variance with
    truncation lag :math:`h = \\lfloor T^{1/3} \\rfloor`.

    Parameters
    ----------
    crps_1 : Tensor
        Shape ``(T,)``.  Per-timestep CRPS values for model 1.
    crps_2 : Tensor
        Shape ``(T,)``.  Per-timestep CRPS values for model 2.
    alternative : {'two-sided', 'less', 'greater'}, default ``'two-sided'``
        Direction of the alternative hypothesis.

    Returns
    -------
    dict[str, float]
        ``dm_stat``   — the DM test statistic.
        ``p_value``   — p-value under the chosen alternative.
        ``mean_diff`` — mean loss differential :math:`\\bar d`.
        ``ci_lower``  — lower bound of the 95 % confidence interval for
                        :math:`E[d_t]`.
        ``ci_upper``  — upper bound of the 95 % confidence interval for
                        :math:`E[d_t]`.

    Raises
    ------
    TypeError
        If inputs are not ``torch.Tensor``.
    ValueError
        If inputs fail shape / finiteness / minimum-length checks, or
        if ``alternative`` is not one of the recognised strings.

    Notes
    -----
    If the loss differentials are identically zero (constant forecasts),
    the test statistic is set to 0.0 and the p-value to 1.0, because the
    null of equal predictive ability holds trivially.

    Examples
    --------
    >>> import torch
    >>> crps_a = torch.rand(200)
    >>> crps_b = crps_a + 0.05 * torch.randn(200)
    >>> result = diebold_mariano_test(crps_a, crps_b)
    >>> result['p_value'] < 0.05  # may or may not reject at 5 %
    True
    """
    # --- input validation ---------------------------------------------------
    _validate_crps_pair(crps_1, crps_2)
    _ALTERNATIVES = {"two-sided", "less", "greater"}
    if alternative not in _ALTERNATIVES:
        raise ValueError(
            f"alternative must be one of {_ALTERNATIVES}, got '{alternative}'."
        )

    # --- compute loss differentials -----------------------------------------
    d = (crps_1 - crps_2).detach().cpu().double().numpy()
    T = len(d)
    d_bar: float = float(d.mean())

    # --- edge case: identically zero differentials --------------------------
    if np.allclose(d, 0.0, atol=1e-12):
        return {
            "dm_stat": 0.0,
            "p_value": 1.0,
            "mean_diff": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
        }

    # --- Newey–West HAC variance --------------------------------------------
    h = int(math.floor(T ** (1.0 / 3.0)))
    sigma2_d = _newey_west_variance(d, h)
    se = math.sqrt(sigma2_d / T)

    # --- DM statistic -------------------------------------------------------
    dm_stat = d_bar / se

    # --- p-value under chosen alternative -----------------------------------
    if alternative == "two-sided":
        p_value = 2.0 * (1.0 - _normal_cdf(abs(dm_stat)))
    elif alternative == "less":
        # H1: E[d_t] < 0  ⟹  reject for large negative DM
        p_value = _normal_cdf(dm_stat)
    else:  # 'greater'
        # H1: E[d_t] > 0  ⟹  reject for large positive DM
        p_value = 1.0 - _normal_cdf(dm_stat)

    # --- 95 % confidence interval for E[d_t] --------------------------------
    z_alpha_half = 1.959964  # z_{0.025} for 95 % CI
    ci_lower = d_bar - z_alpha_half * se
    ci_upper = d_bar + z_alpha_half * se

    return {
        "dm_stat": float(dm_stat),
        "p_value": float(p_value),
        "mean_diff": float(d_bar),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


# ---------------------------------------------------------------------------
# 2. Temporal block bootstrap
# ---------------------------------------------------------------------------


def block_bootstrap_test(
    crps_1: Tensor,
    crps_2: Tensor,
    n_bootstrap: int = 10_000,
    block_length: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Temporal block bootstrap test for paired CRPS comparison.

    Uses the non-overlapping block bootstrap of Politis & Romano (1994)
    to construct a null distribution that preserves temporal dependence
    in the loss differentials.

    **Algorithm**

    1. Compute loss differentials :math:`d_t = \\text{CRPS}_1(t) -
       \\text{CRPS}_2(t)`.
    2. Choose block length :math:`l = \\lceil T^{1/3} \\rceil` (or
       user-supplied).
    3. For each bootstrap replicate *b* = 1, …, *B*:

       a. Draw :math:`\\lceil T/l \\rceil` block starting indices
          uniformly with replacement from
          :math:`\\{0, 1, \\ldots, T - l\\}`.
       b. Concatenate the corresponding blocks and truncate to length
          *T* to form :math:`d^{*}`.
       c. Centre the bootstrap sample:
          :math:`d^{*}_{\\text{centred}} = d^{*} - \\bar{d}^{*}
          + \\bar{d}`.  (*Not* needed for the mean-under-H₀ approach
          below; we instead re-centre under H₀.)
       d. Compute :math:`\\bar{d}^{*}_b = \\text{mean}(d^{*})`.

    4. Compute the two-sided bootstrap p-value as the proportion of
       :math:`|\\bar{d}^{*}_b|` that exceed :math:`|\\bar{d}|`.

    Parameters
    ----------
    crps_1 : Tensor
        Shape ``(T,)``.  Per-timestep CRPS values for model 1.
    crps_2 : Tensor
        Shape ``(T,)``.  Per-timestep CRPS values for model 2.
    n_bootstrap : int, default 10 000
        Number of bootstrap replicates.
    block_length : int or None, default None
        Block length *l*.  If ``None``, computed automatically as
        :math:`\\lceil T^{1/3} \\rceil`.
    seed : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    dict[str, float]
        ``mean_diff``    — observed mean loss differential :math:`\\bar d`.
        ``p_value``      — two-sided bootstrap p-value.
        ``ci_lower``     — lower bound of the 95 % bootstrap percentile
                           confidence interval.
        ``ci_upper``     — upper bound of the 95 % bootstrap percentile
                           confidence interval.
        ``block_length`` — block length actually used.

    Raises
    ------
    TypeError
        If inputs are not ``torch.Tensor``.
    ValueError
        If inputs fail validation, or ``n_bootstrap < 1``,
        or ``block_length < 1``.

    Notes
    -----
    The p-value is computed under the centred bootstrap principle: each
    bootstrap mean is centred by subtracting :math:`\\bar d` so that the
    resampled distribution is centred at zero (the null).  Then:

    .. math::

        \\hat p = \\frac{1}{B} \\sum_{b=1}^{B}
                  \\mathbf{1}\\bigl(|\\bar{d}^{*}_b - \\bar d|
                  \\ge |\\bar d|\\bigr)

    A small correction of :math:`1 / (B + 1)` is *not* applied; the
    raw proportion is returned.

    Examples
    --------
    >>> import torch
    >>> crps_a = torch.rand(200)
    >>> crps_b = crps_a + 0.05 * torch.randn(200)
    >>> result = block_bootstrap_test(crps_a, crps_b)
    >>> 0.0 <= result['p_value'] <= 1.0
    True
    """
    # --- input validation ---------------------------------------------------
    _validate_crps_pair(crps_1, crps_2)
    if n_bootstrap < 1:
        raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}.")

    d = (crps_1 - crps_2).detach().cpu().double().numpy()
    T = len(d)

    # --- block length -------------------------------------------------------
    if block_length is None:
        block_length = int(math.ceil(T ** (1.0 / 3.0)))
    if block_length < 1:
        raise ValueError(f"block_length must be >= 1, got {block_length}.")
    # Clamp block_length to T so that at least one full block exists.
    block_length = min(block_length, T)

    d_bar: float = float(d.mean())

    # --- edge case: identically zero differentials --------------------------
    if np.allclose(d, 0.0, atol=1e-12):
        return {
            "mean_diff": 0.0,
            "p_value": 1.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "block_length": float(block_length),
        }

    # --- bootstrap ----------------------------------------------------------
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(T / block_length))
    max_start = T - block_length  # last valid block start index

    # Pre-compute all block starting indices: shape (B, n_blocks)
    starts = rng.integers(0, max_start + 1, size=(n_bootstrap, n_blocks))

    boot_means = np.empty(n_bootstrap, dtype=np.float64)

    for b in range(n_bootstrap):
        # Build bootstrap sample by concatenating blocks
        blocks = [d[s : s + block_length] for s in starts[b]]
        boot_sample = np.concatenate(blocks)[:T]
        boot_means[b] = boot_sample.mean()

    # --- centred bootstrap p-value (two-sided) ------------------------------
    # Under H0, the centred bootstrap distribution is shifted to mean zero.
    centred_boot_means = boot_means - d_bar
    p_value = float(np.mean(np.abs(centred_boot_means) >= abs(d_bar)))

    # --- 95 % bootstrap percentile CI --------------------------------------
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return {
        "mean_diff": float(d_bar),
        "p_value": float(p_value),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "block_length": float(block_length),
    }


# ---------------------------------------------------------------------------
# 3. Convenience wrapper
# ---------------------------------------------------------------------------


def compare_forecasts(
    crps_model: Tensor,
    crps_baseline: Tensor,
    baseline_name: str = "baseline",
) -> dict[str, Any]:
    """Run both DM test and block bootstrap, return combined results.

    This is the primary entry-point for CIVIC-SAFE forecast evaluation.
    A negative ``mean_diff`` indicates that **our model** (``crps_model``)
    outperforms the baseline on average.

    Parameters
    ----------
    crps_model : Tensor
        Shape ``(T,)``.  Per-timestep CRPS for our model.
    crps_baseline : Tensor
        Shape ``(T,)``.  Per-timestep CRPS for the baseline.
    baseline_name : str, default ``'baseline'``
        Human-readable label for the baseline model (used in the
        summary string).

    Returns
    -------
    dict[str, Any]
        ``dm``            — full result dict from :func:`diebold_mariano_test`.
        ``bootstrap``     — full result dict from :func:`block_bootstrap_test`.
        ``summary``       — human-readable summary string.
        ``baseline_name`` — echo of the baseline label.
        ``T``             — number of time-steps compared.

    Examples
    --------
    >>> import torch
    >>> crps_ours = torch.rand(200) * 0.8
    >>> crps_bl   = torch.rand(200)
    >>> out = compare_forecasts(crps_ours, crps_bl, baseline_name='Poisson')
    >>> print(out['summary'])  # doctest: +SKIP
    """
    # Validation is performed inside each sub-function.
    dm_result = diebold_mariano_test(
        crps_model, crps_baseline, alternative="two-sided"
    )
    boot_result = block_bootstrap_test(crps_model, crps_baseline)

    T = int(crps_model.shape[0])
    mean_diff = dm_result["mean_diff"]

    # --- human-readable summary -------------------------------------------
    if mean_diff < 0:
        direction = "LOWER (better)"
        winner = "our model"
    elif mean_diff > 0:
        direction = "HIGHER (worse)"
        winner = baseline_name
    else:
        direction = "EQUAL"
        winner = "neither"

    dm_sig = "YES" if dm_result["p_value"] < 0.05 else "NO"
    boot_sig = "YES" if boot_result["p_value"] < 0.05 else "NO"

    summary = (
        f"Forecast comparison vs '{baseline_name}' (T={T})\n"
        f"  Mean CRPS diff (ours − baseline): {mean_diff:+.6f} [{direction}]\n"
        f"  Diebold–Mariano: DM={dm_result['dm_stat']:.4f}, "
        f"p={dm_result['p_value']:.4f} (significant at 5%: {dm_sig})\n"
        f"    95% CI for E[d_t]: [{dm_result['ci_lower']:.6f}, "
        f"{dm_result['ci_upper']:.6f}]\n"
        f"  Block bootstrap:  p={boot_result['p_value']:.4f} "
        f"(significant at 5%: {boot_sig})\n"
        f"    95% CI (percentile): [{boot_result['ci_lower']:.6f}, "
        f"{boot_result['ci_upper']:.6f}]\n"
        f"  Winner: {winner}"
    )

    return {
        "dm": dm_result,
        "bootstrap": boot_result,
        "summary": summary,
        "baseline_name": baseline_name,
        "T": T,
    }
