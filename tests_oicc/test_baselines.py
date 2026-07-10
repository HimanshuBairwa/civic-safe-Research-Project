"""Tests for the baseline comparison (empirical defensibility)."""
from __future__ import annotations

import numpy as np

from oicc.baselines import compare_baselines, compare_baselines_confounded


def test_oicc_wins_under_valid_assumptions():
    r = compare_baselines(n=4000, K=4, n_trials=12)
    # OICC BLUP should be the (weak) winner or within noise of the best baseline
    assert r.winner == "oicc_blup"
    assert r.rmse["oicc_blup"] <= min(
        r.rmse["best_single"], r.rmse["reporting_rate_scaleup"]) + 1e-6


def test_reporting_rate_baseline_equals_single_channel():
    """Honest demonstration: a constant reporting-rate scale-up cannot beat the
    single pivot channel (the nuisance is a level shift, absorbed by alignment)."""
    r = compare_baselines(n=4000, K=4, n_trials=12)
    assert abs(r.rmse["reporting_rate_scaleup"] - r.rmse["best_single"]) < 1e-6


def test_proximal_wins_under_confounding():
    """THE headline empirical result: under a common-mode confounder every naive
    method fails and only proximal correction recovers the latent."""
    r = compare_baselines_confounded(n=6000, K=4, Q=2, n_trials=12,
                                     cm_strength=1.0)
    assert r.winner == "oicc_proximal"
    # proximal must be substantially better than every naive method
    naive_best = min(r.rmse["best_single"], r.rmse["naive_average"],
                     r.rmse["oicc_blup_naive"])
    assert r.rmse["oicc_proximal"] < 0.7 * naive_best


def test_comparison_fields():
    r = compare_baselines(n=2000, K=4, n_trials=5)
    assert set(r.methods) == set(r.rmse)
    assert r.n_trials == 5
    assert all(v > 0 for v in r.rmse.values())
