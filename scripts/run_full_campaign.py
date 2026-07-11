#!/usr/bin/env python
"""Full A100 publication campaign -- one command, runs for hours/days, logs all.

Runs the COMPLETE pipeline end-to-end and writes everything under a single
timestamped results dir so there is zero confusion with old outputs:

  results_campaign_<timestamp>/
    00_env.txt                 environment + GPU report
    oicc/                      the contribution (fast): reproduce + India + US + figures
    train_chicago/ train_nyc/  15-seed GNN baseline training (the slow part)
    eval_chicago/  eval_nyc/   conformal evaluation of the trained models
    figures/                   publication figures
    campaign.log               full log

Usage on the A100:
    python scripts/run_full_campaign.py                 # everything (OICC + 15-seed train)
    python scripts/run_full_campaign.py --oicc-only      # just the contribution (minutes)
    python scripts/run_full_campaign.py --seeds 15       # override seed count
    python scripts/run_full_campaign.py --skip-train      # OICC + figures, no GPU training

Honest note: the OICC block is the research contribution and is what matters. The
train block is the applied BASELINE (does not beat seasonal-naive); it is run at
15 seeds only to report publication-grade mean +/- std. More epochs do not help
(early-stop plateau ~epoch 52).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oicc-only", action="store_true",
                    help="run only the OICC contribution (fast, no GPU)")
    ap.add_argument("--skip-train", action="store_true",
                    help="run OICC + figures but skip GNN training")
    ap.add_argument("--seeds", type=int, default=15,
                    help="GNN training seeds (default 15 for publication CIs)")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--smoke-first", action="store_true",
                    help="run a 2-epoch/1-seed training check per city BEFORE the "
                         "full run; abort the full train if the smoke fails")
    ap.add_argument("--india-data", type=str,
                    default=os.environ.get("OICC_INDIA_DATA", ""),
                    help="path to crime-detection-ai/data (India NCRB)")
    args = ap.parse_args()

    # single timestamped campaign dir (no collision with old outputs)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    camp = ROOT / f"results_campaign_{stamp}"
    camp.mkdir(parents=True, exist_ok=True)
    log = (camp / "campaign.log").open("w", encoding="utf-8")

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"),
           "MPLBACKEND": "Agg", "WANDB_MODE": "disabled",
           # force child Python to flush stdout line-by-line so `tail -f`
           # shows live per-epoch training progress instead of nothing for days.
           "PYTHONUNBUFFERED": "1"}
    if args.india_data:
        env["OICC_INDIA_DATA"] = args.india_data

    def run(cmd: list[str], label: str, outfile: Path | None = None) -> bool:
        banner = f"\n{'='*70}\n>>> {label}\n{'='*70}"
        print(banner, flush=True); log.write(banner + "\n"); log.flush()
        t = time.time()
        # Stream child output line-by-line so `tail -f campaign.log` shows live
        # progress (e.g. per-epoch training) instead of nothing until the step
        # finishes. We still keep the full text for the per-step outfile.
        captured: list[str] = []
        proc = subprocess.Popen(
            cmd, cwd=str(ROOT), env=env, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            captured.append(line)
            log.write(line); log.flush()   # live to campaign.log
        proc.wait()
        dt = time.time() - t
        full = "".join(captured)
        # echo a short tail to the console summary
        print("\n".join(full.splitlines()[-12:]), flush=True)
        log.write(f"\n[{label}: rc={proc.returncode} in {dt:.0f}s]\n")
        log.flush()
        if outfile:
            outfile.write_text(full, encoding="utf-8")
        status = "OK" if proc.returncode == 0 else f"FAIL(rc={proc.returncode})"
        print(f"<<< {label}: {status} in {dt:.0f}s", flush=True)
        return proc.returncode == 0

    py = sys.executable
    results = []

    # 0. environment
    run([py, "run_all.py", "--env-only"], "environment report",
        camp / "00_env.txt")

    # 1. OICC -- the contribution (fast, no GPU) -----------------------------
    oicc = camp / "oicc"; oicc.mkdir(exist_ok=True)
    results.append(("oicc reproduce (rigorous)",
        run([py, "experiments/oicc_runs/reproduce_all.py", "--rigorous"],
            "OICC reproduction (rigorous, tight CIs)",
            oicc / "reproduce.txt")))
    if args.india_data:
        results.append(("india ncrb",
            run([py, "experiments/oicc_runs/run_ncrb_experiment.py"],
                "OICC on real India NCRB", oicc / "india_ncrb.txt")))
    else:
        print("[skip] India NCRB: set --india-data or OICC_INDIA_DATA")
    results.append(("us contrast",
        run([py, "experiments/oicc_runs/run_us_experiment.py"],
            "OICC US cross-national contrast", oicc / "us_contrast.txt")))
    results.append(("feedback routing",
        run([py, "experiments/oicc_runs/run_feedback_routing_experiment.py"],
            "conformal safe routing: feedback-loop mitigation",
            oicc / "feedback_routing.txt")))
    results.append(("pub figures",
        run([py, "experiments/oicc_runs/make_pub_figures.py"],
            "publication figures (heatmaps, choropleth)")))
    results.append(("routing figure",
        run([py, "experiments/oicc_runs/make_routing_figure.py"],
            "routing contribution figure (feedback + exposure coverage)")))
    run([py, "experiments/oicc_runs/make_figures.py"], "core figures")

    if not args.oicc_only and not args.skip_train:
        # 2. GNN baseline training (slow, GPU) -------------------------------
        if args.smoke_first:
            # cheap 2-epoch/1-seed check per city: catch any training error in
            # ~2 min instead of hours into the real run.
            for city in ("chicago", "nyc"):
                ok = run([py, "scripts/train.py", f"data={city}",
                          "training.num_seeds=1", "training.epochs=2"],
                         f"SMOKE training {city} (2 epochs, 1 seed)")
                results.append((f"smoke {city}", ok))
                if not ok:
                    print(f"\n[ABORT] smoke training for {city} FAILED -- fix "
                          f"before the multi-day run. See campaign.log.")
                    log.write(f"[ABORT] smoke {city} failed\n"); log.close()
                    return 1
        for city in ("chicago", "nyc"):
            ok = run([py, "scripts/train.py", f"data={city}",
                      f"training.num_seeds={args.seeds}",
                      f"training.epochs={args.epochs}"],
                     f"GNN training {city} ({args.seeds} seeds x {args.epochs} ep)")
            results.append((f"train {city}", ok))

    # summary
    print("\n" + "=" * 70 + "\nCAMPAIGN SUMMARY\n" + "=" * 70)
    for name, ok in results:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}")
    print(f"\nAll results in: {camp}")
    log.write(f"\nCAMPAIGN DONE. results in {camp}\n"); log.close()
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
