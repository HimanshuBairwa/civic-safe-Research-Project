"""Helper: scaffold the records channel .npy from the existing US panel.

Builds `data/processed/us_records.npy` (community-area x month, flattened) from
the shipped chicago/nyc panel, and prints the exact length + ordering so you can
build `us_ncvs.npy` and `us_cfs.npy` on the SAME grid. This only prepares the
records channel; NCVS and 911 are external (see REAL_US_DATA.md).

Run:
    python experiments/oicc_runs/make_us_npy.py --city chicago
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "experiments" / "oicc_runs"))
from paths import find_us_panel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", default="chicago", choices=["chicago", "nyc"])
    ap.add_argument("--period-weeks", type=int, default=4,
                    help="weeks per period (4 ~ monthly)")
    ap.add_argument("--category", type=int, default=0,
                    help="crime category index (0=violent,1=property,2=drug)")
    args = ap.parse_args()

    panel = find_us_panel(args.city)
    if panel is None:
        print(f"panel for {args.city} not found under data/processed/.")
        return 1
    try:
        import torch
    except ImportError:
        print("torch is required to read the .pt panel; pip install torch")
        return 1

    p = torch.load(panel, weights_only=False)
    counts = np.asarray(p["counts"], dtype=float)          # (S, T, C)
    S, T, C = counts.shape
    P = T // args.period_weeks
    agg = counts[:, : P * args.period_weeks, :].reshape(
        S, P, args.period_weeks, C).sum(axis=2)             # (S, P, C)
    records = agg[:, :, args.category].reshape(-1)          # (S*P,), area-major

    outdir = _ROOT / "data" / "processed"
    outpath = outdir / "us_records.npy"
    np.save(outpath, records)
    print(f"wrote {outpath}  (length {records.size} = {S} areas x {P} periods)")
    print("Ordering: area 0 periods 0..P-1, area 1 periods 0..P-1, ...")
    print("Build us_ncvs.npy and us_cfs.npy with the SAME length and ordering,")
    print("then: python experiments/oicc_runs/run_us_multichannel_experiment.py --real")
    return 0


if __name__ == "__main__":
    sys.exit(main())
