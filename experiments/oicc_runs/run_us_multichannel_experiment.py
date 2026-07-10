"""OPTIONAL, ISOLATED experiment: OICC on US records + NCVS + 911 channels.

*** This file is self-contained and NOT imported by any other module, test, or
the reproduction script. The core project runs identically whether or not this
file exists or whether or not you ever obtain the real data. ***

It has two modes:
  * demo (default, runs today): a synthetic 3-channel US panel, so you can see
    the exact end-to-end output shape the real run will produce;
  * real: if aligned files `us_records.npy`, `us_ncvs.npy`, `us_cfs.npy` exist
    under data/processed/ (drop them there -- see docs/data_access/ and
    experiments/oicc_runs/REAL_US_DATA.md), it runs the FIRST genuine 3-channel
    latent over-identification + conformal analysis on real US data.

Run:
    python experiments/oicc_runs/run_us_multichannel_experiment.py          # demo
    python experiments/oicc_runs/run_us_multichannel_experiment.py --real   # if data present
    python experiments/oicc_runs/run_us_multichannel_experiment.py --data /path/to/dir

Nothing here changes the rating or the core method; it is the drop-in slot for
the one real ceiling-lever (an independent-channel real-data latent run).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

from oicc.moments import estimate_factor_moments          # noqa: E402
from oicc.spec_test import overid_wald_test, overid_cumulant_test  # noqa: E402
from oicc.conformal_split import split_conformal_latent   # noqa: E402
from oicc.uncertainty import bootstrap_moments            # noqa: E402
from us_multichannel_loader import (                      # noqa: E402
    build_us_multichannel,
    load_real_if_available,
)


def run(data_dir: Path | None = None, force_real: bool = False) -> dict:
    lines: list[str] = []

    def out(s: str = "") -> None:
        lines.append(s)
        print(s)

    data_dir = Path(data_dir) if data_dir else (_ROOT / "data" / "processed")

    mc = load_real_if_available(data_dir)
    if mc is None:
        if force_real:
            out("[--real requested but no aligned files found]")
            out(f"    expected {data_dir}/us_records.npy, us_ncvs.npy, us_cfs.npy")
            out("    See experiments/oicc_runs/REAL_US_DATA.md. Falling back to demo.")
        mc = build_us_multichannel(demo=True, n=3000, seed=0)

    tag = "REAL" if not mc.is_demo else "DEMO (synthetic)"
    out("=" * 72)
    out(f"OICC on US records + NCVS + 911 channels  [{tag}]")
    out("=" * 72)
    out(f"channels: {mc.channel_names}   units N={mc.n_units}")
    if mc.is_demo:
        out("NOTE: demo data. Provide real NCVS+911 to run the genuine analysis;")
        out("      see docs/data_access/ for the request templates.")
    out("")

    Y = mc.log_channels
    K = Y.shape[0]

    # 1. one-factor structure + bootstrap CI on Var(theta)
    fm = estimate_factor_moments(Y, pivot=0)
    bm = bootstrap_moments(Y, n_boot=300, block=8, level=0.9, seed=0)
    vt = bm["var_theta"]
    out("[1] One-factor structure (pivot = police records)")
    for i, nm in enumerate(mc.channel_names):
        out(f"    loading beta[{nm:12s}] = {fm.beta[i]:+.3f}")
    out(f"    Var(latent) = {fm.var_theta:.3f}  "
        f"90% CI [{vt.lower:.3f}, {vt.upper:.3f}] (block bootstrap)")
    out("")

    # 2. over-ID specification test (i.i.d. + dependence-robust block bootstrap)
    sp = overid_wald_test(Y, seed=0)
    sp_blk = overid_wald_test(Y, seed=0, block=8, bootstrap_pvalue=True)
    out("[2] Over-identification test (do the channels share ONE latent?)")
    out(f"    2nd-moment p:  iid={sp.pvalue:.4f}  block(b=8)={sp_blk.pvalue:.4f}")
    if K == 3:
        cm = overid_cumulant_test(Y, seed=0, block=8)
        out(f"    3rd-cumulant p={cm.pvalue:.4f}  (usable={cm.usable}, "
            f"factor skew={cm.theta_skew:+.3f}) -- adds power at K=3")
    verdict = ("REJECT one latent (channels not mutually consistent)"
               if sp_blk.pvalue < 0.05 else
               "do NOT reject one latent in detectable directions")
    out(f"    verdict: {verdict}")
    out("")

    # 3. latent prediction intervals
    if mc.n_units >= 24:
        res = split_conformal_latent(Y, alpha=0.1, seed=0, use_spec_test=True)
        out("[3] Latent prediction intervals (leave-pivot-out conformal)")
        out(f"    exact observed-value band mean width = "
            f"{float(np.mean(res.obs_upper - res.obs_lower)):.3f}")
        out(f"    model-assisted latent band mean width = "
            f"{float(np.mean(res.lat_upper - res.lat_lower)):.3f}")
        out("")

    out("[interpretation]")
    if mc.is_demo:
        out("    This is the exact analysis that will run on REAL data. On real")
        out("    records+NCVS+911 the over-ID test becomes a genuine check of")
        out("    whether the three independent channels measure one latent rate --")
        out("    the first real-data latent validation OICC can offer.")
    else:
        out("    REAL 3-channel run. If the over-ID test does not reject, the three")
        out("    independent channels are mutually consistent with a single latent")
        out("    rate -- partial real-data support for the latent target. It still")
        out("    cannot see a common-mode confounder shared by all three (the proved")
        out("    impossibility); report that limitation honestly.")

    return {"lines": lines, "is_demo": mc.is_demo, "moments": fm, "spec": sp_blk}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="require real data files (falls back to demo if absent)")
    ap.add_argument("--data", type=str, default=None,
                    help="directory holding us_{records,ncvs,cfs}.npy")
    args = ap.parse_args()
    run(data_dir=args.data, force_real=args.real)
    return 0


if __name__ == "__main__":
    sys.exit(main())
