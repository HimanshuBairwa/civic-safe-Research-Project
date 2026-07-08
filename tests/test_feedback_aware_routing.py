"""Tests for feedback-aware routing (routing over latent-corrected risk).

The headline test: when observation-biased over-recording concentrates in one
demographic group, risk-aware routing on the raw record over/under-exposes that
group; deflating the risk by the identified feedback gain shrinks that exposure
disparity. This is the routing analogue of undoing algorithmic redlining.
"""

from __future__ import annotations

import numpy as np

from civicsafe.routing.feedback_aware import (
    ExposureDisparityAudit,
    LatentCVaRCost,
    correct_node_intervals,
    correct_node_risk,
)
from civicsafe.theory.feedback_law import power_law_fixed_point


class _FakeEdge:
    """Minimal stand-in for routing.graph.Edge (duck-typed for the cost fn)."""

    def __init__(self, distance: float, risk_upper: float, interval_width: float) -> None:
        self.distance = distance
        self.risk_upper = risk_upper
        self.interval_width = interval_width


def test_correct_node_risk_identity_at_zero_gain() -> None:
    """With kappa=0 the correction is a no-op."""
    mu = np.array([1.0, 5.0, 20.0, 3.0])
    assert np.allclose(correct_node_risk(mu, 0.0), mu)


def test_correct_node_risk_recovers_latent() -> None:
    """Deflating the feedback fixed point recovers the latent rate (up to scale)."""
    rng = np.random.default_rng(0)
    lam = rng.gamma(2.0, 2.0, 400) + 0.3
    for kappa in [0.3, 0.6, 0.8]:
        mu = power_law_fixed_point(lam, 1.0, kappa)
        assert mu is not None
        lam_hat = correct_node_risk(mu, kappa)
        ratio = lam_hat / lam
        assert ratio.std() / ratio.mean() < 1e-6  # constant ratio => exact up to scale


def test_interval_deflation_orders_and_shrinks_high_cells() -> None:
    """Corrected bounds stay ordered; over-recorded cells are deflated more."""
    mu = np.array([2.0, 2.0, 50.0])  # third cell far above the mean
    lower = mu * 0.5
    upper = mu * 1.5
    lo_c, hi_c = correct_node_intervals(lower, upper, mu, kappa=0.6)
    assert np.all(hi_c >= lo_c)
    # The high cell is deflated (its corrected upper < its recorded upper).
    assert hi_c[2] < upper[2]


def test_cvar_cost_between_lo_and_hi() -> None:
    """CVaR tail-risk lies within the interval and rises with beta."""
    cost = LatentCVaRCost(w_dist=0.0, w_risk=1.0, beta=0.9)
    edge = _FakeEdge(distance=1.0, risk_upper=10.0, interval_width=4.0)  # [6, 10]
    c = cost(edge)
    assert 6.0 <= c <= 10.0
    # Higher beta -> closer to the worst case (hi).
    cost_hi = LatentCVaRCost(w_dist=0.0, w_risk=1.0, beta=0.99)
    assert cost_hi(edge) >= c


def test_cvar_abstains_on_nonfinite_width() -> None:
    """An abstained node (NaN width) incurs the abstention penalty."""
    cost = LatentCVaRCost(w_dist=0.3, abstain_penalty=1e6)
    edge = _FakeEdge(distance=1.0, risk_upper=float("nan"), interval_width=float("nan"))
    assert cost(edge) >= 1e6


def test_correction_reduces_exposure_disparity() -> None:
    """CORE: feedback correction shrinks the worst-group exposure disparity."""
    rng = np.random.default_rng(7)
    S = 2000
    # Two groups with the SAME latent incidence distribution...
    lam = rng.gamma(2.0, 2.0, S) + 0.3
    groups = (np.arange(S) >= S // 2).astype(int)
    # ...but group 1 is historically over-policed: attention concentrates there,
    # so the feedback loop inflates its recorded rate. Emulate via a group-biased
    # fixed point (group 1 gets a higher effective detection).
    kappa = 0.6
    mu = power_law_fixed_point(lam, 1.0, kappa)
    assert mu is not None
    bias = np.where(groups == 1, 1.8, 1.0)  # structural over-recording in group 1
    recorded = mu * bias

    audit = ExposureDisparityAudit()
    res = audit.correction_reduces_disparity(recorded, lam, groups, kappa=kappa)
    # Correction reduces the worst-group exposure disparity.
    assert res["reduction"] > 0.0
    assert res["corrected_max_disparity"] < res["biased_max_disparity"]
