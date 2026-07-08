"""Decisive experiment: does feedback correction restore LATENT coverage?

In an observation-biased feedback loop, a conformal interval calibrated on the
recorded process keeps coverage of the *record* but loses coverage of the *true
latent* process as the feedback gain rises (the "confidently wrong" regime).
This script measures whether the feedback correction of
:mod:`civicsafe.theory.latent_correction` — using kappa point-identified by the
difference-in-differences design of :mod:`civicsafe.theory.feedback_law` —
recovers latent coverage.

Key design point (verified empirically, see docs/RESULTS_latent_correction.md):
the identifying detection-shock ``delta`` must keep the *treated* cells below the
runaway threshold, i.e. ``kappa * (1 + delta) < 1``. If the shock is too large at
high kappa it drives treated cells into the divergent regime and identification
fails (kappa_hat collapses to the grid floor). The runner therefore chooses
``delta`` adaptively so treated cells stay at gain ``<= safety < 1``.

Run:
    python scripts/latent_correction_experiment.py
    python scripts/latent_correction_experiment.py --trials 12 --cells 4000

Reports, per true feedback gain kappa: naive latent coverage, the DiD estimate
kappa_hat (the method never observes the latent rate to get it), the corrected
latent coverage, and the fraction of cells the corrector keeps (does not abstain
on). If the corrected column holds near the 1-alpha target while the naive column
collapses, the correction recovers honest coverage of true crime from a biased
record.
"""

from __future__ import annotations

import argparse

import numpy as np

from civicsafe.theory import _poisson as poisson
from civicsafe.theory.feedback_law import (
    _hetero_fixed_point,
    identify_kappa_did,
    power_law_fixed_point,
)
from civicsafe.theory.latent_correction import (
    latent_prediction_interval,
    should_abstain,
)


def _adaptive_delta(rho: float, max_delta: float = 0.6, safety: float = 0.9) -> float:
    """Detection shock keeping treated cells sub-runaway: ``rho*(1+delta) <= safety``."""
    if rho <= 0:
        return max_delta
    return float(min(max_delta, max(0.05, safety / rho - 1.0)))


def run(
    kappa_grid: list[float] | None = None,
    num_cells: int = 4000,
    trials: int = 12,
    alpha: float = 0.10,
    grid_points: int = 97,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Run the naive-vs-corrected latent-coverage comparison over a kappa grid."""
    if kappa_grid is None:
        kappa_grid = [0.0, 0.3, 0.5, 0.7, 0.85]
    rng = np.random.default_rng(seed)
    beta = 1.0
    search_grid = np.linspace(0.02, 0.98, grid_points)
    rows: list[dict[str, float]] = []

    for kappa_true in kappa_grid:
        rho = kappa_true  # beta = 1  =>  kappa = rho
        delta = _adaptive_delta(rho)
        naive, corrected, khats, kept_frac = [], [], [], []
        for _ in range(trials):
            lam = rng.gamma(2.0, 2.0, num_cells) + 0.3
            mu = power_law_fixed_point(lam, beta, rho)
            if mu is None:
                continue

            # Identify kappa via a staggered detection-sensitivity shock (DiD).
            treated = rng.random(num_cells) < 0.5
            rho_vec = np.where(treated, rho * (1 + delta), rho)
            mu_post = _hetero_fixed_point(lam, beta, rho_vec)
            if mu_post is None:
                continue
            res = identify_kappa_did(
                mu[treated], mu[~treated], mu_post[treated], mu_post[~treated],
                delta=delta, beta=beta, lam=lam, treated_mask=treated,
                grid=search_grid,
            )
            khat = res.kappa_hat if res.recovered else 0.0
            khats.append(khat)

            # Fresh latent realization to score coverage of the TRUTH.
            y_latent = poisson.rvs(lam, random_state=rng)

            lo_n, hi_n = poisson.ppf(alpha / 2, mu), poisson.ppf(1 - alpha / 2, mu)
            naive.append(float(np.mean((y_latent >= lo_n) & (y_latent <= hi_n))))

            interval = latent_prediction_interval(mu, khat, alpha=alpha)
            keep = ~should_abstain(mu, khat)
            kept_frac.append(float(np.mean(keep)))
            if keep.sum() > 0:
                corrected.append(float(np.mean(
                    (y_latent[keep] >= interval["lower"][keep])
                    & (y_latent[keep] <= interval["upper"][keep])
                )))

        rows.append({
            "kappa": kappa_true,
            "delta": delta,
            "naive_latent_cov": float(np.mean(naive)) if naive else float("nan"),
            "kappa_hat": float(np.mean(khats)) if khats else float("nan"),
            "corrected_latent_cov": float(np.mean(corrected)) if corrected else float("nan"),
            "kept_frac": float(np.mean(kept_frac)) if kept_frac else float("nan"),
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Feedback-correction latent-coverage experiment")
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--cells", type=int, default=4000)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--grid-points", type=int, default=97)
    args = parser.parse_args()

    rows = run(num_cells=args.cells, trials=args.trials, alpha=args.alpha,
               grid_points=args.grid_points)
    target = 1.0 - args.alpha
    print(f"scipy backend = {poisson.HAVE_SCIPY}")
    print(f"Target latent coverage = {target:.2f}\n")
    print(f"{'kappa':>6} | {'delta':>5} | {'naive latent':>12} | {'kappa_hat':>9} | "
          f"{'CORRECTED latent':>16} | {'kept':>6}")
    print("-" * 74)
    for r in rows:
        print(f"{r['kappa']:>6.2f} | {r['delta']:>5.2f} | {r['naive_latent_cov']:>12.3f} | "
              f"{r['kappa_hat']:>9.3f} | {r['corrected_latent_cov']:>16.3f} | {r['kept_frac']:>6.2f}")
    print("-" * 74)
    print("If CORRECTED stays ~target while naive collapses, the correction recovers")
    print("honest coverage of true crime from a feedback-biased record.")


if __name__ == "__main__":
    main()
