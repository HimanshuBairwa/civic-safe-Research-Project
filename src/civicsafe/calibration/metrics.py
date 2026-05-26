"""Evaluation metrics for conformal prediction intervals.

Metrics for assessing the quality of calibrated prediction intervals
on discrete count data (crime counts):

- **PICP**: Prediction Interval Coverage Probability (primary — must be ≥ 1-α)
- **AIW**: Average Interval Width (efficiency — lower is better)
- **Winkler Score**: Proper scoring rule balancing width + undercoverage penalty
- **Conditional Coverage**: Per-group coverage for fairness assessment

References:
    - Winkler (1972): "A Decision-Theoretic Approach to Interval Estimation"
    - Gneiting & Raftery (2007): "Strictly Proper Scoring Rules"
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor


def picp(
    y: Tensor,
    lower: Tensor,
    upper: Tensor,
) -> float:
    """Prediction Interval Coverage Probability.

    The fraction of observations falling inside [lower, upper].
    Must be ≥ (1 - α) for a valid conformal interval.

    Args:
        y: Observed counts. Shape: (N,) or any shape (flattened internally).
        lower: Lower bounds. Same shape as y.
        upper: Upper bounds. Same shape as y.

    Returns:
        Scalar coverage in [0, 1].
    """
    y = y.reshape(-1).float()
    lower = lower.reshape(-1).float()
    upper = upper.reshape(-1).float()

    covered = ((y >= lower) & (y <= upper)).float()
    return covered.mean().item()


def average_interval_width(
    lower: Tensor,
    upper: Tensor,
) -> float:
    """Average Interval Width (AIW).

    Mean of (upper - lower) across all predictions.
    Lower is better, provided coverage is maintained.

    Args:
        lower: Lower bounds. Shape: (N,)
        upper: Upper bounds. Shape: (N,)

    Returns:
        Scalar mean width.
    """
    lower = lower.reshape(-1).float()
    upper = upper.reshape(-1).float()
    return (upper - lower).mean().item()


def winkler_score(
    y: Tensor,
    lower: Tensor,
    upper: Tensor,
    alpha: float = 0.1,
) -> float:
    """Winkler Interval Score (proper scoring rule).

    For a (1-α)×100% prediction interval [ℓ, u] and observation y:

        W = (u - ℓ)                                  if ℓ ≤ y ≤ u
        W = (u - ℓ) + (2/α)·(ℓ - y)                 if y < ℓ
        W = (u - ℓ) + (2/α)·(y - u)                 if y > u

    Lower is better. The penalty multiplier 2/α is very large for small
    alpha (e.g., 20× for α=0.1), heavily penalising undercoverage.

    Args:
        y: Observed counts. Shape: (N,)
        lower: Lower bounds. Shape: (N,)
        upper: Upper bounds. Shape: (N,)
        alpha: Nominal miscoverage level.

    Returns:
        Scalar mean Winkler score.
    """
    y = y.reshape(-1).float()
    lower = lower.reshape(-1).float()
    upper = upper.reshape(-1).float()

    width = upper - lower
    penalty_below = (2.0 / alpha) * (lower - y).clamp(min=0.0)
    penalty_above = (2.0 / alpha) * (y - upper).clamp(min=0.0)

    scores = width + penalty_below + penalty_above
    return scores.mean().item()


def conditional_coverage(
    y: Tensor,
    lower: Tensor,
    upper: Tensor,
    groups: Tensor,
) -> dict[int, float]:
    """Per-group (conditional) coverage.

    Computes coverage separately for each group to detect fairness issues
    where certain groups (e.g., disadvantaged neighbourhoods) may have
    systematically worse coverage.

    Args:
        y: Observed counts. Shape: (N,)
        lower: Lower bounds. Shape: (N,)
        upper: Upper bounds. Shape: (N,)
        groups: Integer group labels. Shape: (N,)

    Returns:
        Dictionary mapping group_id → coverage (float).
    """
    y = y.reshape(-1).float()
    lower = lower.reshape(-1).float()
    upper = upper.reshape(-1).float()
    groups = groups.reshape(-1)

    covered = ((y >= lower) & (y <= upper)).float()
    result: dict[int, float] = {}

    for g in groups.unique().tolist():  # type: ignore[no-untyped-call]
        mask = groups == g
        if mask.sum() > 0:
            result[int(g)] = covered[mask].mean().item()

    return result


def coverage_gap(
    y: Tensor,
    lower: Tensor,
    upper: Tensor,
    groups: Tensor,
    alpha: float = 0.1,
) -> float:
    """Maximum coverage gap across groups.

    max_g |coverage(g) - (1-α)|

    Measures the worst-case deviation from target coverage across all
    groups. A key fairness metric — lower is better.

    Args:
        y, lower, upper: Predictions and observations.
        groups: Group labels.
        alpha: Nominal miscoverage level.

    Returns:
        Scalar maximum coverage gap.
    """
    target = 1.0 - alpha
    cond_cov = conditional_coverage(y, lower, upper, groups)

    if not cond_cov:
        return 0.0

    return max(abs(cov - target) for cov in cond_cov.values())


def compute_all_calibration_metrics(
    y: Tensor,
    lower: Tensor,
    upper: Tensor,
    alpha: float = 0.1,
    groups: Tensor | None = None,
) -> dict[str, Any]:
    """Compute all calibration evaluation metrics in one call.

    Args:
        y: Observed counts. Shape: (N,)
        lower: Lower interval bounds. Shape: (N,)
        upper: Upper interval bounds. Shape: (N,)
        alpha: Nominal miscoverage level.
        groups: Optional group labels for fairness metrics. Shape: (N,)

    Returns:
        Dictionary with:
            picp: Empirical coverage (must be ≥ 1-α)
            aiw: Average interval width
            winkler: Mean Winkler score
            coverage_valid: Boolean — is PICP ≥ 1-α?
            coverage_gap: Max per-group coverage deviation (if groups given)
            conditional_coverage: Per-group dict (if groups given)
    """
    result: dict[str, Any] = {
        "picp": picp(y, lower, upper),
        "aiw": average_interval_width(lower, upper),
        "winkler": winkler_score(y, lower, upper, alpha=alpha),
    }
    result["coverage_valid"] = result["picp"] >= (1.0 - alpha - 1e-6)

    if groups is not None:
        result["coverage_gap"] = coverage_gap(y, lower, upper, groups, alpha)
        result["conditional_coverage"] = conditional_coverage(
            y, lower, upper, groups
        )

    return result
