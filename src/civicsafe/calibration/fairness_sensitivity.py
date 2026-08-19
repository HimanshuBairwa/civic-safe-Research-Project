"""Demographic-imputation robustness diagnostics.

These helpers evaluate fairness metrics after changing only the demographic
group construction.  Forecasts and conformal intervals are treated as frozen
post-training artifacts.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

ImputationStrategy = Literal["median", "lower_quartile", "complete_case"]


def impute_demographic(
    values: np.ndarray,
    strategy: ImputationStrategy = "median",
) -> tuple[np.ndarray, np.ndarray]:
    """Impute missing demographic values and return ``(values, keep_mask)``."""
    raw = np.asarray(values, dtype=float).reshape(-1)
    finite = np.isfinite(raw)
    if not finite.any():
        raise ValueError("demographic values contain no observed entries")
    if strategy == "complete_case":
        return raw[finite], finite
    if strategy == "median":
        fill = float(np.nanmedian(raw))
    elif strategy == "lower_quartile":
        fill = float(np.nanquantile(raw, 0.25))
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")
    return np.where(finite, raw, fill), np.ones(raw.size, dtype=bool)


def _quartile_groups(values: np.ndarray) -> np.ndarray:
    """Rank values into four stable groups, handling ties and tiny samples."""
    if values.size == 0:
        return np.empty(0, dtype=int)
    ranks = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    return np.minimum(3, (4 * ranks) // max(values.size, 1)).astype(int)


def coverage_disparity(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    groups: np.ndarray,
) -> float:
    """Maximum-minus-minimum interval coverage across non-empty groups."""
    y = np.asarray(y).reshape(-1)
    lower = np.asarray(lower).reshape(-1)
    upper = np.asarray(upper).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    if not (y.size == lower.size == upper.size == groups.size):
        raise ValueError("y, lower, upper, and groups must have the same size")
    covered = (y >= lower) & (y <= upper)
    rates = [float(covered[groups == group].mean()) for group in np.unique(groups) if np.any(groups == group)]
    return float(max(rates) - min(rates)) if len(rates) >= 2 else 0.0


def evaluate_imputation_sensitivity(
    y: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    demographic_values: np.ndarray,
    *,
    strategies: tuple[ImputationStrategy, ...] = ("median", "lower_quartile", "complete_case"),
    max_disparity: float = 0.030,
) -> dict[str, Any]:
    """Evaluate conformal coverage disparity under three imputations.

    For complete-case analysis all aligned arrays are restricted to observed
    demographic rows before quartile construction.  The output includes both
    the disparity and a machine-readable pass/fail flag for preregistered
    fairness checks.
    """
    raw = np.asarray(demographic_values, dtype=float).reshape(-1)
    y_arr = np.asarray(y).reshape(-1)
    lo_arr = np.asarray(lower).reshape(-1)
    hi_arr = np.asarray(upper).reshape(-1)
    if not (raw.size == y_arr.size == lo_arr.size == hi_arr.size):
        raise ValueError("demographic_values and interval arrays must have equal size")
    results: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        values, keep = impute_demographic(raw, strategy)
        y_use, lo_use, hi_use = y_arr[keep], lo_arr[keep], hi_arr[keep]
        groups = _quartile_groups(values)
        disparity = coverage_disparity(y_use, lo_use, hi_use, groups)
        results[strategy] = {
            "disparity": disparity,
            "n": int(keep.sum()),
            "passes": bool(disparity <= max_disparity + 1e-12),
            "max_disparity": float(max_disparity),
        }
    return {
        "strategies": results,
        "all_pass": all(item["passes"] for item in results.values()),
        "max_observed_disparity": max((item["disparity"] for item in results.values()), default=0.0),
    }


__all__ = [
    "ImputationStrategy",
    "coverage_disparity",
    "evaluate_imputation_sensitivity",
    "impute_demographic",
]
