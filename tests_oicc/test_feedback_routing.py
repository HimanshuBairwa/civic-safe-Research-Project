"""Test the feedback-loop mitigation experiment (G2 of the routing contribution).

Verifies, across multiple seeds, that OICC-anchored allocation stays calibrated
and cuts exposure disparity relative to the record-only runaway loop. This locks
the headline claim as reproducible, not a lucky draw.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "experiments" / "oicc_runs"))


def test_oicc_breaks_feedback_loop_single():
    from run_feedback_routing_experiment import run
    out = run(seed=0)
    s = out["summary"]
    # OICC stays far better correlated with truth than the runaway record policy
    assert s["final_corr_oicc"] > 0.8
    assert s["final_corr_oicc"] > s["final_corr_record"]
    # and cuts the over-patrolled group's exposure disparity
    assert s["disparity_reduction"] > 0.1


def test_mitigation_robust_across_seeds():
    from run_feedback_routing_experiment import run
    reductions, oicc_corr, rec_corr = [], [], []
    for seed in range(8):
        s = run(seed=seed)["summary"]
        reductions.append(s["disparity_reduction"])
        oicc_corr.append(s["final_corr_oicc"])
        rec_corr.append(s["final_corr_record"])
    reductions = np.array(reductions)
    # disparity reduction is positive in every seed (robust, not a lucky draw)
    assert (reductions > 0).all(), f"reductions: {reductions}"
    # OICC beats record-only calibration on average by a wide margin
    assert np.mean(oicc_corr) > np.mean(rec_corr) + 0.2


def test_no_feedback_baseline_sanity():
    """With zero initial patrol bias, disparity starts small for both."""
    from run_feedback_routing_experiment import run
    out = run(seed=3, init_patrol_bias=0.0)
    # OICC remains well-calibrated regardless
    assert out["summary"]["final_corr_oicc"] > 0.8
