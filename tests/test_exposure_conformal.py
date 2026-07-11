"""Tests for conformal exposure certificates (the novel routing primitive).

The headline test is EMPIRICAL COVERAGE: over many random calibration/test
splits, the fraction of test scenarios whose realized exposure exceeds the
certified bound must be <= alpha (up to Monte-Carlo error). This is the honest
proof the guarantee holds, not just an assertion.
"""
from __future__ import annotations

import numpy as np
import pytest

from civicsafe.routing.exposure_conformal import (
    Scenario,
    certify_route_exposure,
    conformal_upper_quantile,
    route_exposure,
    select_risk_budget,
)


def _make_scenarios(n, N=40, seed=0):
    """Random scenarios: predicted ~ latent + noise; realized = latent draw."""
    rng = np.random.default_rng(seed)
    scs = []
    for _ in range(n):
        latent = rng.gamma(2.0, 1.0, size=N)
        predicted = latent + rng.normal(0, 0.5, size=N)  # imperfect prediction
        realized = rng.gamma(2.0, 1.0, size=N) * 0.1 + latent * 0.9  # correlated truth
        scs.append(Scenario(predicted=predicted, realized=realized))
    return scs


def _greedy_low_risk_policy(k=8):
    """Policy: 'route' = the k lowest-predicted-risk nodes (a stand-in path)."""
    def policy(predicted):
        return list(np.argsort(predicted)[:k])
    return policy


def test_route_exposure_basic():
    realized = np.array([1.0, 2.0, 3.0, 4.0])
    assert route_exposure([0, 2], realized) == pytest.approx(4.0)
    assert route_exposure([], realized) == 0.0


def test_conformal_quantile_rank_and_infinite():
    scores = np.arange(1, 11, dtype=float)  # 1..10
    q, k, finite = conformal_upper_quantile(scores, alpha=0.1)
    # k = ceil(11 * 0.9) = 10 -> 10th smallest = 10, finite
    assert finite and k == 10 and q == pytest.approx(10.0)
    # tiny n cannot certify at small alpha -> +inf, not finite
    q2, k2, finite2 = conformal_upper_quantile(np.array([5.0, 6.0]), alpha=0.1)
    assert not finite2 and np.isinf(q2)


def test_certificate_fields():
    scs = _make_scenarios(60, seed=1)
    cert = certify_route_exposure(_greedy_low_risk_policy(), scs, alpha=0.1)
    assert cert.finite
    assert cert.n_cal == 60
    assert np.isfinite(cert.q_upper)
    assert cert.q_upper >= cert.mean_exposure  # upper bound above the mean


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_empirical_coverage_holds(alpha):
    """Over many splits, exceedance rate <= alpha (+ MC slack). The real proof."""
    policy = _greedy_low_risk_policy()
    n_cal, n_trials = 200, 400
    exceed = 0
    for t in range(n_trials):
        pool = _make_scenarios(n_cal + 1, seed=1000 + t)
        cal, test = pool[:n_cal], pool[n_cal]
        cert = certify_route_exposure(policy, cal, alpha=alpha)
        e_test = route_exposure(policy(test.predicted), test.realized)
        if e_test > cert.q_upper:
            exceed += 1
    rate = exceed / n_trials
    # split-conformal guarantee: exceedance <= alpha; allow 3*SE MC slack
    se = (alpha * (1 - alpha) / n_trials) ** 0.5
    assert rate <= alpha + 3 * se, f"coverage violated: exceed={rate:.3f} > {alpha}"


def test_select_risk_budget_valid_on_disjoint_fold():
    """Budget selection meets the target and stays valid on the certify fold."""
    # policy family: k = number of low-risk nodes; higher knob = more averse
    def family(knob):
        k = int(knob)
        return lambda predicted: list(np.argsort(predicted)[:k])

    sel = _make_scenarios(150, seed=7)
    cert = _make_scenarios(150, seed=8)
    # exposures grow with k, so a mid budget should pick a small-ish k
    out = select_risk_budget(
        family, knobs=[4, 8, 12, 16, 20],
        select_scenarios=sel, certify_scenarios=cert,
        target_exposure=1e9, alpha=0.1,  # generous budget -> picks smallest knob
    )
    assert out["met_budget"] is True
    assert out["chosen_knob"] == 4.0          # least-aversion that meets budget
    assert out["certificate"].finite
    assert len(out["selection_curve"]) == 5
