"""Tests for the feedback-correction sensitivity analysis."""

from __future__ import annotations

import numpy as np

from civicsafe.calibration.fairness_sensitivity import (
    evaluate_imputation_sensitivity,
    impute_demographic,
)
from civicsafe.theory.sensitivity import (
    robustness_value,
    sensitivity_curve,
)


def test_coverage_peaks_near_true_kappa() -> None:
    """Latent coverage is (near-)maximal when the used gain matches the truth."""
    curve = sensitivity_curve(kappa_true=0.5, num_cells=3000, trials=6, seed=1)
    used = np.array([r["kappa_used"] for r in curve])
    cov = np.array([r["coverage"] for r in curve])
    best = used[int(np.nanargmax(cov))]
    # The best-coverage gain is within a reasonable band of the truth.
    assert abs(best - 0.5) <= 0.15


def test_robustness_value_nonnegative_and_finite() -> None:
    """Robustness value is finite and non-negative for a moderate gain."""
    res = robustness_value(kappa_true=0.5, coverage_floor=0.85, num_cells=3000, trials=6)
    assert res.robustness_value >= 0.0
    assert res.safe_low <= 0.5 <= res.safe_high


def test_higher_kappa_is_more_fragile() -> None:
    """Correction near the runaway threshold tolerates less gain error."""
    rv_mid = robustness_value(kappa_true=0.4, num_cells=2500, trials=6, seed=2).robustness_value
    rv_high = robustness_value(kappa_true=0.8, num_cells=2500, trials=6, seed=2).robustness_value
    # Near runaway the safe band is no wider than in the mild regime.
    assert rv_high <= rv_mid + 0.05


def test_demographic_imputation_strategies_are_explicit() -> None:
    values = np.array([10.0, np.nan, 20.0, 30.0])
    median, keep_median = impute_demographic(values, "median")
    lower, keep_lower = impute_demographic(values, "lower_quartile")
    complete, keep_complete = impute_demographic(values, "complete_case")
    assert keep_median.all() and keep_lower.all()
    assert keep_complete.tolist() == [True, False, True, True]
    assert median[1] == np.nanmedian(values)
    assert lower[1] == np.nanquantile(values, 0.25)
    assert complete.size == 3


def test_conformal_fairness_is_robust_to_demographic_imputation() -> None:
    rng = np.random.default_rng(7)
    n = 160
    income = rng.normal(50_000.0, 10_000.0, size=n)
    income[rng.choice(n, size=20, replace=False)] = np.nan
    y = rng.poisson(4.0, size=n)
    lower = np.zeros(n)
    upper = y + 3.0  # Deliberately conservative, with uniform coverage.
    result = evaluate_imputation_sensitivity(y, lower, upper, income)
    assert result["all_pass"]
    assert set(result["strategies"]) == {"median", "lower_quartile", "complete_case"}
    assert result["max_observed_disparity"] <= 0.030
