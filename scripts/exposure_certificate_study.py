#!/usr/bin/env python
"""Route-exposure certificate study: does the split-conformal bound actually hold?

The routing primitive in src/civicsafe/routing/exposure_conformal.py carries a
finite-sample guarantee, and the unit tests check its algebra. What no persisted
artifact showed until now is the empirical exceedance rate: over many scenarios,
how often does realized route exposure actually breach the certified bound? The
guarantee says at most alpha. This script measures it and writes the numbers so
the supplementary can cite them instead of restating the theorem.

Two policies are certified, because the interesting comparison is not
"certificate holds" but "which risk field should the router plan on":

  record   plans on the RECORDED risk field, which the feedback loop has inflated
           in already-over-recorded areas
  deflated plans on the deflated field lambda_hat = M^kappa mu^(1-kappa), the
           Theorem 2 estimate of latent risk

Both get a valid certificate -- split conformal does not care whether the field
is biased, only that scenarios are exchangeable. The difference shows up in the
realized exposure the certificate bounds, which is the quantity a civilian on
the route actually experiences.

The risk fields here are simulated, for the reason stated in the manuscript's
Limitations: latent incidence is unobservable on real records, so a study that
needs the realized latent field cannot be run on Chicago or New York. This is a
mechanism check on the certificate, not a field result, and the supplementary
says so.

Run:
    python scripts/exposure_certificate_study.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from civicsafe.routing.exposure_conformal import (  # noqa: E402
    Scenario,
    certify_route_exposure,
    route_exposure,
)

OUT = PROJECT_ROOT / "outputs" / "routing"

N_NODES = 40          # nodes in the advisory graph
ROUTE_LEN = 8         # nodes a route visits
N_CAL = 150           # calibration scenarios per trial
N_TRIALS = 400        # independent trials per alpha
KAPPA = 0.6           # feedback gain, matching the manuscript's assumed value
ALPHAS = (0.05, 0.10, 0.20)


def _greedy(field: np.ndarray) -> list[int]:
    """Visit the ROUTE_LEN nodes the planner believes are lowest risk."""
    return list(np.argsort(field)[:ROUTE_LEN])


def _scenarios(n: int, seed: int, deflate: bool) -> list[Scenario]:
    """Build n scenarios whose recorded field is feedback-inflated.

    latent   ~ Gamma(2, 2), the true incidence field
    recorded = latent * (latent/M)^kappa, the fixed point of eq. (1): areas above
               the panel mean are over-recorded, below it under-recorded
    realized = a fresh draw at the latent rate, i.e. what actually happens

    The planner sees either the recorded field or its deflation. It never sees
    `realized`, which is what the certificate bounds.
    """
    rng = np.random.default_rng(seed)
    out: list[Scenario] = []
    for _ in range(n):
        latent = rng.gamma(2.0, 2.0, size=N_NODES) + 0.1
        M = latent.mean()
        recorded = latent * (latent / M) ** KAPPA
        if deflate:
            # Theorem 2(i): lambda_hat = M_rec^kappa * mu^(1-kappa) recovers latent.
            M_rec = recorded.mean()
            planned = M_rec ** KAPPA * recorded ** (1.0 - KAPPA)
        else:
            planned = recorded
        realized = rng.poisson(latent).astype(float)
        out.append(Scenario(planned, realized))
    return out


def main() -> None:
    rows = []
    print(f"nodes={N_NODES} route_len={ROUTE_LEN} n_cal={N_CAL} "
          f"trials={N_TRIALS} kappa={KAPPA}")
    for label, deflate in (("record", False), ("deflated", True)):
        for alpha in ALPHAS:
            breaches = 0
            q_upper: list[float] = []
            realized: list[float] = []
            for t in range(N_TRIALS):
                pool = _scenarios(N_CAL + 1, seed=hash((label, alpha, t)) % (2**31),
                                  deflate=deflate)
                cert = certify_route_exposure(_greedy, pool[:N_CAL], alpha=alpha)
                test = pool[N_CAL]
                e = route_exposure(_greedy(test.predicted), test.realized)
                breaches += int(e > cert.q_upper)
                q_upper.append(cert.q_upper)
                realized.append(e)
            rate = breaches / N_TRIALS
            row = {
                "policy": label,
                "alpha": alpha,
                "target_max_breach": alpha,
                "empirical_breach_rate": rate,
                "holds": bool(rate <= alpha),
                "mean_certified_bound": float(np.mean(q_upper)),
                "mean_realized_exposure": float(np.mean(realized)),
                "n_cal": N_CAL,
                "n_trials": N_TRIALS,
            }
            rows.append(row)
            print("  %-8s alpha=%.2f  breach %.4f (<= %.2f? %s)  "
                  "bound %6.2f  realized %6.2f"
                  % (label, alpha, rate, alpha, "yes" if row["holds"] else "NO",
                     row["mean_certified_bound"], row["mean_realized_exposure"]))

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_nodes": N_NODES,
        "route_len": ROUTE_LEN,
        "n_cal": N_CAL,
        "n_trials": N_TRIALS,
        "kappa": KAPPA,
        "note": (
            "Simulated risk fields. Latent incidence is unobservable on real "
            "records, so a study needing the realized latent field cannot be run "
            "on Chicago or New York. This is a mechanism check on the certificate."
        ),
        "rows": rows,
    }
    path = OUT / "exposure_certificates.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
