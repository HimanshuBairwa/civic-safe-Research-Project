"""Publication figure for the Conformal Safe Routing contribution.

Three panels, all from LIVE tested computation (no hand-drawn numbers):

  (a) Feedback-loop divergence: belief-vs-truth correlation over rounds, record-
      only (runs away) vs OICC-anchored (stays calibrated).
  (b) Exposure-disparity reduction: over-patrolled group's disparity, record-only
      vs OICC-anchored.
  (c) Conformal exposure coverage: empirical exceedance of the certified bound
      across alphas lies on/under the diagonal (the guarantee holds).

Run:  python experiments/oicc_runs/make_routing_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments" / "oicc_runs"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from oicc_style import apply_style, PALETTE, panel_label  # noqa: E402
from run_feedback_routing_experiment import run as run_feedback  # noqa: E402
from civicsafe.routing.exposure_conformal import (  # noqa: E402
    Scenario, certify_route_exposure, route_exposure,
)

OUT = _ROOT / "paper" / "figures" / "pub"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, VERM, GREEN = PALETTE[0], PALETTE[1], PALETTE[2]


def _coverage_curve(alphas, n_cal=150, n_trials=300, seed=0):
    """Empirical exceedance of the conformal exposure bound vs alpha."""
    rng = np.random.default_rng(seed)

    def make(n, s):
        r = np.random.default_rng(s)
        out = []
        for _ in range(n):
            lat = r.gamma(2.0, 1.0, size=30)
            pred = lat + r.normal(0, 0.5, size=30)
            real = 0.9 * lat + 0.1 * r.gamma(2.0, 1.0, size=30)
            out.append(Scenario(pred, real))
        return out

    policy = lambda pr: list(np.argsort(pr)[:8])
    exceed = []
    for a in alphas:
        cnt = 0
        for t in range(n_trials):
            pool = make(n_cal + 1, seed * 10000 + t)
            cert = certify_route_exposure(policy, pool[:n_cal], alpha=a)
            e = route_exposure(policy(pool[n_cal].predicted), pool[n_cal].realized)
            cnt += int(e > cert.q_upper)
        exceed.append(cnt / n_trials)
    return np.array(exceed)


def main() -> int:
    apply_style()
    fb = run_feedback(seed=0)
    recA, recB = fb["record"], fb["oicc"]
    rounds = np.arange(1, len(recA.belief_corr) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.7))

    # (a) feedback-loop divergence
    ax = axes[0]
    ax.plot(rounds, recA.belief_corr, color=VERM, lw=2, label="record-only (loop)")
    ax.plot(rounds, recB.belief_corr, color=BLUE, lw=2, label="OICC-anchored")
    ax.set_xlabel("feedback round"); ax.set_ylabel("belief–truth correlation")
    ax.set_ylim(0, 1.02); ax.legend(loc="lower left", fontsize=8)
    ax.set_title("Feedback-loop calibration", fontsize=10)
    panel_label(ax, "a")

    # (b) disparity reduction
    ax = axes[1]
    ax.plot(rounds, recA.group_disparity, color=VERM, lw=2, label="record-only")
    ax.plot(rounds, recB.group_disparity, color=BLUE, lw=2, label="OICC-anchored")
    ax.set_xlabel("feedback round")
    ax.set_ylabel("over-patrolled group disparity")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Navigational-redlining disparity", fontsize=10)
    panel_label(ax, "b")

    # (c) conformal exposure coverage
    ax = axes[2]
    alphas = np.array([0.05, 0.1, 0.15, 0.2, 0.3])
    exceed = _coverage_curve(alphas)
    ax.plot([0, 0.32], [0, 0.32], color="#999999", ls="--", lw=1,
            label="nominal = actual")
    ax.plot(alphas, exceed, "o-", color=GREEN, lw=2, ms=5,
            label="empirical exceedance")
    ax.set_xlabel(r"target miscoverage $\alpha$")
    ax.set_ylabel("empirical exceedance")
    ax.set_title("Conformal exposure guarantee", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    panel_label(ax, "c")

    fig.tight_layout()
    fig.savefig(OUT / "pub_fig8_routing.pdf")
    fig.savefig(OUT / "pub_fig8_routing.png", dpi=200)
    plt.close(fig)

    # honest check: exceedance should sit on/under the diagonal
    slack = 0.03
    ok = bool(np.all(exceed <= alphas + slack))
    print(f"  [OK] routing figure -> pub_fig8_routing  "
          f"(coverage holds: {ok}; exceedance={np.round(exceed,3).tolist()})")
    print(f"  disparity reduction (final): "
          f"{fb['summary']['disparity_reduction']:+.3f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
