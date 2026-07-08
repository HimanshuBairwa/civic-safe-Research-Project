"""Experiment: does feedback-correction reduce routing exposure disparity?

Risk-aware routing over the raw (observation-biased) crime record systematically
diverts civilians around historically over-recorded — often over-policed —
neighborhoods, laundering enforcement bias into navigation. This experiment
measures the worst-group exposure disparity of routing on the biased record vs.
on the feedback-corrected latent risk, across the feedback-gain range.

Run:
    python scripts/routing_disparity_experiment.py
    python scripts/routing_disparity_experiment.py --trials 20 --cells 3000

Reports, per feedback gain kappa: the worst-group |exposure disparity| when
routing on the biased record vs. the corrected field, and the reduction. If the
corrected column is consistently lower, feedback-correction demonstrably shrinks
navigational redlining.
"""

from __future__ import annotations

import argparse

import numpy as np

from civicsafe.routing.feedback_aware import ExposureDisparityAudit
from civicsafe.theory.feedback_law import power_law_fixed_point


def run(
    kappa_grid: list[float] | None = None,
    num_cells: int = 2000,
    trials: int = 12,
    group1_overpolicing: float = 1.8,
    seed: int = 11,
) -> list[dict[str, float]]:
    """Measure biased vs. corrected exposure disparity over a kappa grid."""
    if kappa_grid is None:
        kappa_grid = [0.0, 0.3, 0.5, 0.7, 0.85]
    rng = np.random.default_rng(seed)
    audit = ExposureDisparityAudit()
    rows: list[dict[str, float]] = []

    for kappa in kappa_grid:
        biased, corrected, reduction = [], [], []
        for _ in range(trials):
            # Two groups, identical latent incidence distribution.
            lam = rng.gamma(2.0, 2.0, num_cells) + 0.3
            groups = (np.arange(num_cells) >= num_cells // 2).astype(int)
            mu = power_law_fixed_point(lam, 1.0, kappa)
            if mu is None:
                continue
            # Group 1 is structurally over-recorded (historical over-policing).
            bias = np.where(groups == 1, group1_overpolicing, 1.0)
            recorded = mu * bias
            res = audit.correction_reduces_disparity(recorded, lam, groups, kappa=kappa)
            biased.append(res["biased_max_disparity"])
            corrected.append(res["corrected_max_disparity"])
            reduction.append(res["reduction"])
        rows.append({
            "kappa": kappa,
            "biased_disparity": float(np.mean(biased)) if biased else float("nan"),
            "corrected_disparity": float(np.mean(corrected)) if corrected else float("nan"),
            "reduction": float(np.mean(reduction)) if reduction else float("nan"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Routing exposure-disparity experiment")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--cells", type=int, default=2000)
    parser.add_argument("--overpolicing", type=float, default=1.8)
    args = parser.parse_args()

    rows = run(num_cells=args.cells, trials=args.trials, group1_overpolicing=args.overpolicing)
    print(f"Group-1 structural over-recording factor = {args.overpolicing}\n")
    print(f"{'kappa':>6} | {'biased disparity':>16} | {'corrected disparity':>19} | {'reduction':>9}")
    print("-" * 62)
    for r in rows:
        print(f"{r['kappa']:>6.2f} | {r['biased_disparity']:>16.3f} | "
              f"{r['corrected_disparity']:>19.3f} | {r['reduction']:>9.3f}")
    print("-" * 62)
    print("Lower corrected disparity => feedback-correction shrinks navigational")
    print("redlining: routes stop systematically avoiding over-recorded areas.")


if __name__ == "__main__":
    main()
