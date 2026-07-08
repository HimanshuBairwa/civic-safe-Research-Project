"""Tests for the anytime-valid feedback tripwire.

Two guarantees matter: (1) low false-alarm rate under the null (a calibrated
stream rarely fires), and (2) power — it fires, and fires quickly, when
miscoverage drifts above nominal (the feedback regime).
"""

from __future__ import annotations

import numpy as np

from civicsafe.theory.feedback_tripwire import FeedbackTripwire


def test_false_alarm_rate_under_null() -> None:
    """Under a calibrated stream, false-alarm rate stays within the budget."""
    rng = np.random.default_rng(0)
    alpha, alarm = 0.10, 0.05
    fires = 0
    runs = 200
    for _ in range(runs):
        covered = (rng.random(500) > alpha).astype(int)  # exactly nominal coverage
        tw = FeedbackTripwire(alpha_nominal=alpha, alarm_level=alarm)
        if tw.run(covered).fired:
            fires += 1
    # Ville's inequality guarantees <= alarm in the long run; allow slack for MC.
    assert fires / runs <= 3 * alarm


def test_fires_under_miscoverage_drift() -> None:
    """When true miscoverage rises well above nominal, the tripwire fires."""
    rng = np.random.default_rng(1)
    alpha = 0.10
    # Feedback regime: true miscoverage 0.45 >> nominal 0.10.
    covered = (rng.random(400) > 0.45).astype(int)
    tw = FeedbackTripwire(alpha_nominal=alpha, alarm_level=0.01)
    state = tw.run(covered)
    assert state.fired
    assert 0 < state.fired_at <= 400


def test_over_coverage_does_not_fire() -> None:
    """Over-covering (miscoverage below nominal) must not trigger the alarm."""
    rng = np.random.default_rng(2)
    covered = (rng.random(500) > 0.02).astype(int)  # 98% coverage, nominal 90%
    tw = FeedbackTripwire(alpha_nominal=0.10, alarm_level=0.01)
    assert not tw.run(covered).fired


def test_earlier_detection_for_larger_drift() -> None:
    """Larger miscoverage drift is detected sooner (monotone power)."""
    rng = np.random.default_rng(3)

    def first_fire(miscov: float) -> int:
        hits = []
        for s in range(20):
            r = np.random.default_rng(100 + s)
            covered = (r.random(600) > miscov).astype(int)
            tw = FeedbackTripwire(alpha_nominal=0.10, alarm_level=0.01)
            st = tw.run(covered)
            hits.append(st.fired_at if st.fired else 600)
        return int(np.median(hits))

    assert first_fire(0.50) <= first_fire(0.25)
