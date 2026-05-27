"""ZINB distribution utilities: CDF, PPF (quantile), and PMF.

Provides GPU-accelerated, batch-compatible functions for computing the
cumulative distribution function, probability mass function, and quantile
function of Zero-Inflated Negative Binomial distributions.

These are used by the conformal calibration module to compute non-conformity
scores and construct prediction intervals.

Parameterisation throughout:
    pi: zero-inflation probability P(structural zero) in [0, 1]
    mu: NB mean in (0, ∞)
    r:  NB dispersion (total_count) in (0, ∞)
    Mapping to scipy: n = r, p = r / (r + mu)
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Internal: Negative Binomial PMF/CDF (log-space for stability)
# ---------------------------------------------------------------------------

def _nb_log_pmf(ks: Tensor, mu: Tensor, r: Tensor) -> Tensor:
    """Compute NB log-PMF at integer points.

    Args:
        ks: Integer points. Shape: (K,) or (1, K)
        mu: NB mean. Shape: (B, 1)
        r: NB dispersion. Shape: (B, 1)

    Returns:
        Log-PMF values. Shape: (B, K)
    """
    log_p = torch.log(r / (r + mu))       # log(r / (r + μ))
    log_1mp = torch.log(mu / (r + mu))    # log(μ / (r + μ))

    return (
        torch.lgamma(ks + r)
        - torch.lgamma(ks + 1.0)
        - torch.lgamma(r)
        + r * log_p
        + ks * log_1mp
    )


def _nb_cdf(ks: Tensor, mu: Tensor, r: Tensor) -> Tensor:
    """Compute NB CDF via cumulative PMF summation.

    Args:
        ks: Integer grid. Shape: (K,)
        mu: NB mean. Shape: (B,)
        r: NB dispersion. Shape: (B,)

    Returns:
        CDF values at each k. Shape: (B, K)
    """
    mu_exp = mu.unsqueeze(-1)   # (B, 1)
    r_exp = r.unsqueeze(-1)     # (B, 1)
    ks_exp = ks.unsqueeze(0)    # (1, K)

    log_pmf = _nb_log_pmf(ks_exp, mu_exp, r_exp)
    pmf = torch.exp(log_pmf).clamp(min=0.0)
    return pmf.cumsum(dim=-1).clamp(max=1.0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def zinb_cdf(
    k: Tensor | int,
    pi: Tensor,
    mu: Tensor,
    r: Tensor,
    k_max: int | None = None,
) -> Tensor:
    """Compute the ZINB CDF: F_ZINB(k) = π + (1-π)·F_NB(k; μ, r).

    Handles both single-point evaluation and full CDF curve computation.

    Args:
        k: Points at which to evaluate the CDF.
           - int: returns F(k) for all elements. Shape out: (B,)
           - Tensor of shape (B,): per-element k values. Shape out: (B,)
        pi: Zero-inflation probability. Shape: (B,)
        mu: NB mean. Shape: (B,)
        r: NB dispersion. Shape: (B,)
        k_max: Truncation point for CDF grid. Auto-computed if None.

    Returns:
        CDF values. Shape: (B,)
    """
    pi = pi.float().clamp(0.0, 1.0)
    mu = mu.float().clamp(min=1e-6)
    r = r.float().clamp(min=0.1)

    if k_max is None:
        max_mu = mu.max().item()
        max_r = r.max().item()
        variance = max_mu + max_mu**2 / max(max_r, 0.1)
        k_max = min(int(max_mu + 10.0 * math.sqrt(variance)) + 1, 500)
        k_max = max(k_max, 50)

    device = mu.device
    ks = torch.arange(0, k_max + 1, device=device, dtype=torch.float32)
    F_nb = _nb_cdf(ks, mu, r)  # (B, K)

    # ZINB CDF = π + (1-π)·F_NB
    pi_exp = pi.unsqueeze(-1)
    F_zinb = pi_exp + (1.0 - pi_exp) * F_nb  # (B, K)

    # Gather at the requested k values
    if isinstance(k, int):
        idx = min(k, k_max)
        return F_zinb[:, idx]  # type: ignore[no-any-return]
    else:
        k_idx = k.long().clamp(0, k_max)
        return F_zinb.gather(1, k_idx.unsqueeze(-1)).squeeze(-1)  # type: ignore[no-any-return]


def zinb_cdf_full(
    pi: Tensor,
    mu: Tensor,
    r: Tensor,
    k_max: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute the full ZINB CDF curve from k=0 to k_max.

    Args:
        pi, mu, r: ZINB parameters. Shape: (B,)
        k_max: Truncation point. Auto if None.

    Returns:
        Tuple of (ks, F_zinb):
            ks: Integer grid. Shape: (K,)
            F_zinb: CDF values. Shape: (B, K)
    """
    pi = pi.float().clamp(0.0, 1.0)
    mu = mu.float().clamp(min=1e-6)
    r = r.float().clamp(min=0.1)

    if k_max is None:
        max_mu = mu.max().item()
        max_r = r.max().item()
        variance = max_mu + max_mu**2 / max(max_r, 0.1)
        k_max = min(int(max_mu + 10.0 * math.sqrt(variance)) + 1, 500)
        k_max = max(k_max, 50)

    device = mu.device
    ks = torch.arange(0, k_max + 1, device=device, dtype=torch.float32)
    F_nb = _nb_cdf(ks, mu, r)

    pi_exp = pi.unsqueeze(-1)
    F_zinb = pi_exp + (1.0 - pi_exp) * F_nb

    return ks, F_zinb


def zinb_ppf(
    q: Tensor,
    pi: Tensor,
    mu: Tensor,
    r: Tensor,
    k_max: int | None = None,
) -> Tensor:
    """Quantile function (inverse CDF / PPF) for the ZINB distribution.

    Uses binary search via torch.searchsorted on the CDF curve.

    For ZINB: PPF(q) = 0 if q <= F_ZINB(0), else smallest k where F_ZINB(k) >= q.

    Args:
        q: Quantile levels in [0, 1]. Shape: (B,)
        pi, mu, r: ZINB parameters. Shape: (B,)
        k_max: CDF truncation. Auto if None.

    Returns:
        Integer quantile values. Shape: (B,) as float tensor.
    """
    q = q.float().clamp(0.0, 1.0)

    ks, F_zinb = zinb_cdf_full(pi, mu, r, k_max=k_max)

    # searchsorted: find smallest k where F_zinb(k) >= q
    # searchsorted with 'right' would give insertion point after equal values
    # We want the leftmost k where CDF >= q, so use 'left' side.
    indices = torch.searchsorted(F_zinb, q.unsqueeze(-1))  # (B, 1)
    indices = indices.squeeze(-1)  # (B,)

    # Clamp to valid range
    K = ks.shape[0]
    indices = indices.clamp(0, K - 1)

    return indices.float()


def zinb_ppf_pair(
    alpha: float,
    pi: Tensor,
    mu: Tensor,
    r: Tensor,
    k_max: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute lower and upper heuristic quantiles for CQR.

    Returns PPF(α/2) and PPF(1 - α/2) — the uncalibrated prediction bounds.

    Args:
        alpha: Miscoverage level (e.g., 0.1 for 90% coverage).
        pi, mu, r: ZINB parameters. Shape: (B,)
        k_max: CDF truncation.

    Returns:
        Tuple of (q_low, q_high) each of shape (B,).
    """
    B = pi.shape[0]
    device = pi.device

    q_lo_level = torch.full((B,), alpha / 2.0, device=device)
    q_hi_level = torch.full((B,), 1.0 - alpha / 2.0, device=device)

    q_low = zinb_ppf(q_lo_level, pi, mu, r, k_max=k_max)
    q_high = zinb_ppf(q_hi_level, pi, mu, r, k_max=k_max)

    return q_low, q_high
