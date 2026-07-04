"""Sharpness-Aware Calibration (SAC) Loss for ZINB models.

A novel unified training objective that simultaneously optimizes:
  1. Distributional calibration (via CRPS)
  2. Distributional sharpness (via ZINB variance penalty)
  3. Distributional health (via r-collapse regularization)

The key insight: CRPS alone rewards calibrated distributions but doesn't
explicitly penalize unnecessarily wide distributions. SAC adds a sharpness
term that encourages tight, well-calibrated distributions — the exact
property needed for useful prediction intervals.

This is the training-time analog of what conformal prediction does at
inference time: trade off coverage for width. By optimizing sharpness
during training, the model produces better base predictions, which
downstream conformal calibration can then adjust with tighter intervals.

Mathematical formulation:
    L_SAC = CRPS(F_ZINB, y) + lambda_s * Sharpness(pi, mu, r) + lambda_r * R_penalty(r)

where Sharpness = log(1 + Var[Y_ZINB]) and
      Var[Y_ZINB] = (1-pi)*mu + (1-pi)*mu^2/r + pi*(1-pi)*mu^2

References:
  - Gneiting & Raftery (2007): "Maximizing sharpness subject to calibration"
  - This module implements that principle as a differentiable training loss.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from civicsafe.training.metrics import crps_zinb


def zinb_variance(pi: Tensor, mu: Tensor, r: Tensor) -> Tensor:
    """Compute the variance of a ZINB distribution (fully differentiable).

    Var[Y_ZINB] = (1-pi)*mu + (1-pi)*mu^2/r + pi*(1-pi)*mu^2

    Decomposition:
      - (1-pi)*mu: Poisson-like variance contribution
      - (1-pi)*mu^2/r: Overdispersion from the NB component
      - pi*(1-pi)*mu^2: Zero-inflation variance contribution

    Args:
        pi: Zero-inflation probability. Shape: (B,)
        mu: NB mean. Shape: (B,)
        r: NB dispersion. Shape: (B,)

    Returns:
        Variance values. Shape: (B,)
    """
    one_minus_pi = 1.0 - pi.clamp(0.0, 1.0)
    mu_safe = mu.clamp(min=1e-6)
    r_safe = r.clamp(min=0.1)

    # NB variance: mu + mu^2/r
    nb_var = mu_safe + mu_safe ** 2 / r_safe

    # ZINB variance: (1-pi)*NB_var + pi*(1-pi)*mu^2
    zinb_var = one_minus_pi * nb_var + pi.clamp(0.0, 1.0) * one_minus_pi * mu_safe ** 2

    return zinb_var


def sharpness_loss(pi: Tensor, mu: Tensor, r: Tensor) -> Tensor:
    """Compute the sharpness penalty (log-variance) for ZINB distributions.

    We use log(1 + Var) instead of Var directly because:
      1. Crime count variances span many orders of magnitude
      2. log(1+x) prevents gradient explosion for high-variance cells
      3. It's scale-invariant: a 10% reduction in variance contributes
         equally whether the base variance is 5 or 5000

    Args:
        pi, mu, r: ZINB parameters. Shape: (B,) each.

    Returns:
        Scalar mean sharpness loss.
    """
    var = zinb_variance(pi, mu, r)
    return torch.log1p(var).mean()


def sac_loss(
    y: Tensor,
    pi: Tensor,
    mu: Tensor,
    r: Tensor,
    lambda_sharpness: float = 0.0,
    lambda_r_reg: float = 0.1,
    r_reg_floor: float = 0.5,
) -> tuple[Tensor, dict[str, float]]:
    """Sharpness-Aware Calibration (SAC) loss.

    L_SAC = CRPS + lambda_s * log(1 + Var[ZINB]) + lambda_r * relu(r_floor - r).mean()

    This unified objective implements Gneiting & Raftery's principle of
    "maximizing sharpness subject to calibration" as a differentiable
    training loss for zero-inflated count distributions.

    Args:
        y: Observed counts. Shape: (B,)
        pi, mu, r: ZINB parameters. Shape: (B,)
        lambda_sharpness: Weight for sharpness penalty.
        lambda_r_reg: Weight for r-collapse regularization.
        r_reg_floor: Floor value for r-regularization.

    Returns:
        Tuple of (total_loss, diagnostics_dict).
        diagnostics_dict contains individual loss components for logging.
    """
    # Component 1: CRPS (distributional calibration)
    crps = crps_zinb(y, pi, mu, r).mean()

    # Component 2: Sharpness (distributional tightness)
    sharp = sharpness_loss(pi, mu, r)

    # Component 3: r-collapse regularization
    r_penalty = torch.nn.functional.relu(r_reg_floor - r).mean()

    # Combined loss
    total = crps + lambda_sharpness * sharp + lambda_r_reg * r_penalty

    # Diagnostics for logging
    diagnostics = {
        "sac/crps": crps.detach().item(),
        "sac/sharpness": sharp.detach().item(),
        "sac/r_penalty": r_penalty.detach().item(),
        "sac/r_mean": r.mean().detach().item(),
        "sac/r_min": r.min().detach().item(),
        "sac/mu_mean": mu.mean().detach().item(),
        "sac/pi_mean": pi.mean().detach().item(),
        "sac/zinb_var_mean": zinb_variance(pi, mu, r).mean().detach().item(),
    }

    return total, diagnostics
