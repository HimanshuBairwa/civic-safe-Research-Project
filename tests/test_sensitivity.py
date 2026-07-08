"""Tests for the feedback-correction sensitivity analysis."""

from __future__ import annotations

import numpy as np

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
