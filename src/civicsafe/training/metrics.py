"""Evaluation metrics for ZINB count-data forecasting.

Implements proper scoring rules and point-forecast metrics for evaluating
Zero-Inflated Negative Binomial predictive distributions.

Key metrics:
  - CRPS: Continuous Ranked Probability Score via CDF summation (primary)
  - MAE / RMSE: Point-forecast accuracy from E[Y] = (1-π)·μ
  - Brier Score: Zero-inflation calibration on P(Y=0)
  - PIT: Probability Integral Transform for calibration diagnostics

References:
  - Gneiting & Raftery (2007): Strictly Proper Scoring Rules
  - R scoringRules package: crps_nbinom parametric formula
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from civicsafe.utils.numerics import nb_k_max


def crps_zinb(
    y: Tensor,
    pi: Tensor,
    mu: Tensor,
    r: Tensor,
    k_max: int | None = None,
) -> Tensor:
    """Continuous Ranked Probability Score for ZINB distributions.

    Computed via the CDF summation formula — exact up to truncation.
    This is the *primary* evaluation metric for probabilistic forecasting.

    CRPS(F, y) = Σ_k [F_ZINB(k) - I(k ≥ y)]²

    where F_ZINB(k) = π·I(k≥0) + (1-π)·F_NB(k; μ, r).

    Args:
        y: Observed counts. Shape: (B,) or (B, C)
        pi: Zero-inflation probabilities. Shape: same as y
        mu: NB mean parameters. Shape: same as y
        r: NB dispersion parameters. Shape: same as y
        k_max: Truncation point. Auto-computed if None.

    Returns:
        CRPS values. Shape: same as y input.
    """
    # Flatten to 1-D for uniform computation
    orig_shape = y.shape
    y = y.reshape(-1).float()
    pi = pi.reshape(-1).float().clamp(0.0, 1.0)
    mu = mu.reshape(-1).float().clamp(min=1e-6)
    r = r.reshape(-1).float().clamp(min=0.1)

    B = y.shape[0]

    # --- Truncation point ---------------------------------------------------
    # The sum only needs to run far enough for F_ZINB to saturate at 1. It does
    # NOT need to reach y: for k in (k_max, y) the summand is exactly
    # (1 - 0)^2 = 1, and that stretch is added analytically after the loop. So
    # the tensor stays small even when a single observation is enormous.
    if k_max is None:
        # Shared per-observation rule (see nb_k_max). It must be per-observation
        # and then maxed: pairing a batch-wide max_mu with a batch-wide max_r
        # understates the spread of the cell that actually has large mu and
        # small r -- the overdispersed cell whose CDF saturates slowest -- and
        # the analytic tail below then charges that cell a full 1.0 per step
        # while its true F is still well under 1.
        k_max = nb_k_max(mu, r)

    device = y.device
    ks = torch.arange(0, k_max + 1, device=device, dtype=torch.float32)  # (K,)

    # Chunk over observations so the (chunk, K) intermediates stay bounded no
    # matter how large the batch or the truncation point is.
    max_elems = 4_000_000
    chunk = max(1, min(B, max_elems // (k_max + 1)))
    summed = torch.empty(B, device=device, dtype=torch.float32)
    F_edge = torch.empty(B, device=device, dtype=torch.float32)

    for lo in range(0, B, chunk):
        hi = min(lo + chunk, B)

        # --- Compute NB PMF in log-space for numerical stability ---
        # NB parameterization: total_count=r, probs=r/(r+μ) (success probability)
        # PMF: P(X=k) = Γ(k+r)/(k!·Γ(r)) · p^r · (1-p)^k
        # where p = r/(r+μ)
        r_expanded = r[lo:hi].unsqueeze(-1)  # (b, 1)
        mu_expanded = mu[lo:hi].unsqueeze(-1)  # (b, 1)
        ks_expanded = ks.unsqueeze(0)  # (1, K)

        # Log-space NB PMF
        log_p = torch.log(r_expanded / (r_expanded + mu_expanded))  # log(r/(r+μ))
        log_1mp = torch.log(mu_expanded / (r_expanded + mu_expanded))  # log(μ/(r+μ))

        log_pmf = (
            torch.lgamma(ks_expanded + r_expanded)
            - torch.lgamma(ks_expanded + 1.0)
            - torch.lgamma(r_expanded)
            + r_expanded * log_p
            + ks_expanded * log_1mp
        )  # (b, K)

        # CDF via cumulative sum of PMF
        pmf = torch.exp(log_pmf)
        # Numerical safety: ensure PMF sums to ≤1 and is non-negative
        pmf = pmf.clamp(min=0.0)
        F_nb = pmf.cumsum(dim=-1).clamp(max=1.0)  # (b, K)

        # --- ZINB CDF ---
        # F_ZINB(k) = π + (1-π)·F_NB(k) for k ≥ 0
        pi_expanded = pi[lo:hi].unsqueeze(-1)  # (b, 1)
        F_zinb = pi_expanded + (1.0 - pi_expanded) * F_nb  # (b, K)

        # --- CRPS via CDF summation ---
        indicator = (ks_expanded >= y[lo:hi].unsqueeze(-1)).float()  # (b, K)
        summed[lo:hi] = ((F_zinb - indicator) ** 2).sum(dim=-1)  # (b,)

        # CDF level at the truncation boundary, for the analytic tail below.
        F_edge[lo:hi] = F_zinb[:, -1]

    # --- Analytic tail beyond the truncation point --------------------------
    # For k_max < k < y the indicator is still 0 while F_ZINB has (very nearly)
    # saturated, so each of those ceil(y) - 1 - k_max terms contributes
    # F_ZINB(k)^2. Dropping them understates CRPS by up to ~y whenever a
    # forecast is confidently wrong on a large count -- exactly the error a
    # proper scoring rule exists to punish. Without this term the metric
    # systematically flatters whichever model predicts the smallest counts.
    #
    # F_edge is used rather than a hard 1.0 so that an explicitly supplied
    # k_max below the saturation point degrades gracefully instead of
    # over-charging. With the auto k_max above, F_edge ~ 1 and this is exact.
    n_tail = (torch.ceil(y) - 1.0 - float(k_max)).clamp(min=0.0)
    tail = n_tail * F_edge**2

    return (summed + tail).reshape(orig_shape)  # type: ignore[no-any-return]


def mae_zinb(y: Tensor, pi: Tensor, mu: Tensor) -> Tensor:
    """Mean Absolute Error using the ZINB point estimate.

    Point estimate: E[Y_ZINB] = (1 - π) · μ

    Args:
        y: Observed counts. Shape: (B,) or (B, C)
        pi: Zero-inflation probabilities.
        mu: NB mean parameters.

    Returns:
        Scalar MAE.
    """
    y_hat = (1.0 - pi.clamp(0.0, 1.0)) * mu.clamp(min=0.0)
    return (y.float() - y_hat).abs().mean()  # type: ignore[no-any-return]


def rmse_zinb(y: Tensor, pi: Tensor, mu: Tensor) -> Tensor:
    """Root Mean Squared Error using the ZINB point estimate.

    Args:
        y: Observed counts.
        pi: Zero-inflation probabilities.
        mu: NB mean parameters.

    Returns:
        Scalar RMSE.
    """
    y_hat = (1.0 - pi.clamp(0.0, 1.0)) * mu.clamp(min=0.0)
    return torch.sqrt(((y.float() - y_hat) ** 2).mean())


def brier_zero_inflation(y: Tensor, pi: Tensor, mu: Tensor = None, r: Tensor = None) -> Tensor:
    """Brier Score for zero-probability calibration.

    Evaluates how well the model predicts P(Y=0).
    For ZINB: P(Y=0) = π + (1-π)·(r/(r+μ))^r
    Brier = mean((P_pred(Y=0) - I(y=0))²)

    Perfect score = 0, worst = 1.

    Args:
        y: Observed counts.
        pi: Zero-inflation probability.
        mu: NB mean (optional, for full ZINB P(Y=0)).
        r: NB dispersion (optional, for full ZINB P(Y=0)).

    Returns:
        Scalar Brier score.
    """
    is_zero = (y == 0).float()
    pi_clamped = pi.clamp(0.0, 1.0)

    if mu is not None and r is not None:
        # Full ZINB P(Y=0) = pi + (1-pi) * (r/(r+mu))^r
        mu_safe = mu.clamp(min=1e-6)
        r_safe = r.clamp(min=0.1)
        nb_zero_prob = (r_safe / (r_safe + mu_safe)).pow(r_safe)
        p_zero = pi_clamped + (1.0 - pi_clamped) * nb_zero_prob
    else:
        # Fallback: use pi only (backward compatible)
        p_zero = pi_clamped

    return ((p_zero - is_zero) ** 2).mean()


def pit_values(
    y: Tensor,
    pi: Tensor,
    mu: Tensor,
    r: Tensor,
    k_max: int = 200,
) -> Tensor:
    """Randomized Probability Integral Transform for calibration checks.

    For discrete distributions:
      PIT = F(y-1) + U · [F(y) - F(y-1)]
    where U ~ Uniform(0, 1).

    A well-calibrated model produces PIT values uniformly distributed on [0,1].

    Args:
        y: Observed counts. Shape: (B,)
        pi, mu, r: ZINB parameters.
        k_max: CDF truncation.

    Returns:
        PIT values. Shape: (B,) in [0, 1].
    """
    y = y.reshape(-1).float()
    pi = pi.reshape(-1).float().clamp(0.0, 1.0)
    mu = mu.reshape(-1).float().clamp(min=1e-6)
    r = r.reshape(-1).float().clamp(min=0.1)

    device = y.device
    ks = torch.arange(0, k_max + 1, device=device, dtype=torch.float32)

    # NB PMF in log-space
    r_exp = r.unsqueeze(-1)
    mu_exp = mu.unsqueeze(-1)
    ks_exp = ks.unsqueeze(0)

    log_p = torch.log(r_exp / (r_exp + mu_exp))
    log_1mp = torch.log(mu_exp / (r_exp + mu_exp))

    log_pmf = (
        torch.lgamma(ks_exp + r_exp)
        - torch.lgamma(ks_exp + 1.0)
        - torch.lgamma(r_exp)
        + r_exp * log_p
        + ks_exp * log_1mp
    )
    pmf = torch.exp(log_pmf).clamp(min=0.0)
    F_nb = pmf.cumsum(dim=-1).clamp(max=1.0)

    # ZINB CDF
    pi_exp = pi.unsqueeze(-1)
    F_zinb = pi_exp + (1.0 - pi_exp) * F_nb  # (B, K)

    # Gather F(y) and F(y-1)
    y_idx = y.long().clamp(0, k_max)
    F_at_y = F_zinb.gather(1, y_idx.unsqueeze(-1)).squeeze(-1)
    y_minus_1_idx = (y_idx - 1).clamp(min=0)
    F_at_ym1 = F_zinb.gather(1, y_minus_1_idx.unsqueeze(-1)).squeeze(-1)
    # For y=0, F(y-1) = 0
    F_at_ym1 = torch.where(y_idx > 0, F_at_ym1, torch.zeros_like(F_at_ym1))

    # Randomized PIT
    U = torch.rand_like(y)
    pit = F_at_ym1 + U * (F_at_y - F_at_ym1)

    return pit.clamp(0.0, 1.0)  # type: ignore[no-any-return]


def compute_all_metrics(
    y: Tensor,
    pi: Tensor,
    mu: Tensor,
    r: Tensor,
) -> dict[str, float]:
    """Compute all evaluation metrics in a single pass.

    Args:
        y: Observed counts. Shape: (B,) or (B, C)
        pi, mu, r: ZINB parameters. Same shape as y.

    Returns:
        Dictionary with metric names as keys and scalar values.
    """
    with torch.no_grad():
        crps = crps_zinb(y, pi, mu, r).mean().item()
        mae = mae_zinb(y, pi, mu).item()
        rmse = rmse_zinb(y, pi, mu).item()
        brier = brier_zero_inflation(y, pi, mu, r).item()

    return {
        "crps": crps,
        "mae": mae,
        "rmse": rmse,
        "brier_zero": brier,
    }
