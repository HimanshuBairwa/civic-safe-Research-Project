"""Q7 evidence: time the post-hoc calibration and ensemble layers.

Honest scope. The trained checkpoints live on the GPU server and are not present
here, so the model forward pass is NOT timed by this script and no end-to-end
dispatch latency is claimed. What is measured is everything downstream of the
forward pass -- the layers this paper actually contributes -- at the true panel
sizes reported in the manuscript metadata: 6006 calibration cells and 4081 test
cells for Chicago.

Reported as median over repeats on CPU, which is a conservative stand-in for the
deployment target: an operational system would run the same code on the same
hardware that served the forward pass.
"""

from __future__ import annotations

import json
import statistics
import time

import torch

from civicsafe.calibration.conformal import create_calibrator
from civicsafe.calibration.emos import apply_emos_weights, learn_emos_weights

N_CAL = 6006
N_TEST = 4081
N_SEEDS = 5
REPEATS = 7
ALPHA = 0.1


def synth(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    pi = torch.rand(n, generator=g) * 0.4
    mu = torch.rand(n, generator=g) * 30 + 0.5
    r = torch.rand(n, generator=g) * 3 + 0.5
    nb = torch.distributions.NegativeBinomial(
        total_count=r, probs=mu / (mu + r)
    ).sample()
    y = torch.where(torch.rand(n, generator=g) < pi, torch.zeros(n), nb)
    return y, pi, mu, r


def timeit(fn, repeats: int = REPEATS) -> float:
    """Median wall time in milliseconds."""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ts)


def main() -> None:
    y_cal, pi_cal, mu_cal, r_cal = synth(N_CAL, 1)
    y_te, pi_te, mu_te, r_te = synth(N_TEST, 2)
    groups_cal = torch.randint(0, 5, (N_CAL,))
    groups_te = torch.randint(0, 5, (N_TEST,))

    members = [synth(N_CAL, 10 + k) for k in range(N_SEEDS)]
    all_pi = [m[1] for m in members]
    all_mu = [m[2] for m in members]
    all_r = [m[3] for m in members]

    print(f"panel sizes: n_cal = {N_CAL}, n_test = {N_TEST}, "
          f"{N_SEEDS} ensemble members")
    print(f"median of {REPEATS} repeats, CPU\n")
    results: dict[str, float] = {}

    # --- one-off, offline: fitting ---
    results["EMOS weight learning (offline, once)"] = timeit(
        lambda: learn_emos_weights(y_cal, all_pi, all_mu, all_r), repeats=3
    )

    w = [1.0 / N_SEEDS] * N_SEEDS
    results["EMOS weight application (per batch)"] = timeit(
        lambda: apply_emos_weights(w, all_pi, all_mu, all_r)
    )

    for method in ("split_cp", "randomized_split_cp", "ecrc"):
        cfg = {"method": method, "alpha": ALPHA}
        if method == "ecrc":
            cfg["group_type"] = "demographic"

        def fit() -> None:
            c = create_calibrator(cfg)
            kw = {"groups": groups_cal} if method == "ecrc" else {}
            c.fit(y_cal, pi_cal, mu_cal, r_cal, **kw)

        results[f"{method}: fit on {N_CAL} cal cells (offline)"] = timeit(fit, 3)

        c = create_calibrator(cfg)
        kw = {"groups": groups_cal} if method == "ecrc" else {}
        c.fit(y_cal, pi_cal, mu_cal, r_cal, **kw)
        pkw = {"groups": groups_te} if method == "ecrc" else {}
        results[f"{method}: predict {N_TEST} intervals (online)"] = timeit(
            lambda: c.predict(pi_te, mu_te, r_te, **pkw)
        )

    width = max(len(k) for k in results)
    for k, v in results.items():
        per_cell = v / N_TEST * 1000.0 if "predict" in k else float("nan")
        extra = f"   ({per_cell:7.2f} us/cell)" if "predict" in k else ""
        print(f"  {k:<{width}}  {v:9.2f} ms{extra}")

    predict_ms = [v for k, v in results.items() if "predict" in k]
    online_total = results["EMOS weight application (per batch)"] + min(predict_ms)
    print(f"\n  online path per test panel (ensemble weighting + one calibrator "
          f"predict): ~{online_total:.1f} ms for {N_TEST} cells, "
          f"~{online_total / N_TEST * 1000:.1f} us/cell")
    print(f"  offline path, run once per campaign: EMOS learning "
          f"{results['EMOS weight learning (offline, once)'] / 1000:.1f} s "
          f"plus calibrator fit under 0.2 s")
    print("\nNOT MEASURED: the model forward pass. Checkpoints are on the GPU")
    print("server and absent here, so no end-to-end dispatch latency is claimed.")

    with open("outputs/calibration_latency.json", "w", encoding="utf-8") as f:
        json.dump({"n_cal": N_CAL, "n_test": N_TEST, "n_seeds": N_SEEDS,
                   "repeats": REPEATS, "device": "cpu",
                   "median_ms": results,
                   "forward_pass_measured": False}, f, indent=2)
    print("\nwrote outputs/calibration_latency.json")


if __name__ == "__main__":
    main()
