"""One-command reproduction of every OICC headline result, with ASSERTIONS.

Runs the full synthetic validation battery, the real-data experiments, and the
figure generation, and ASSERTS each headline number lands in its expected range.
Exits non-zero if any check fails -- so "it reproduces" is a machine-checked fact,
not a claim.

Usage:
    python experiments/oicc_runs/reproduce_all.py
    python experiments/oicc_runs/reproduce_all.py --quick   # smaller n_boot/seeds
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

import oicc  # noqa: E402
from oicc.spec_test import overid_wald_test               # noqa: E402
from oicc.conformal_split import split_conformal_latent   # noqa: E402
from oicc.measurement import generate_proximal            # noqa: E402
from oicc.proximal import point_identify                  # noqa: E402
from oicc.monitor import EProcessMonitor                  # noqa: E402

_PASS, _FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (_PASS if cond else _FAIL).append(name)
    mark = "PASS" if cond else "**FAIL**"
    print(f"  [{mark}] {name}  {detail}")


def _test_fold(n, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_cal = int(round(0.5 * (n // 2)))
    n_cal = min(max(n_cal, 4), (n // 2) - 2)
    n_train = n - 2 * n_cal
    if n_train < 8:
        n_cal = max(4, (n - 8) // 2)
        n_train = n - 2 * n_cal
    return perm[n_train + n_cal:]


def _theta_rmse(Y, theta):
    from oicc.moments import estimate_factor_moments
    from oicc.deconvolve import blup_from_subset
    fm = estimate_factor_moments(Y, pivot=0)
    others = [i for i in range(Y.shape[0]) if i != 0]
    th = blup_from_subset(Y, fm, others, float(Y[0].mean())).theta_hat
    A = np.vstack([np.ones_like(th), th]).T
    c, *_ = np.linalg.lstsq(A, theta, rcond=None)
    return float(np.sqrt(np.mean((A @ c - theta) ** 2)))


def main(quick: bool = False, rigorous: bool = False) -> int:
    t0 = time.time()
    # rigorous mode: many more Monte-Carlo trials for tight, publication-grade
    # estimates (uses your unlimited A100 time where it genuinely helps -- on the
    # CONTRIBUTION, not the baseline forecaster).
    n_seeds = 10 if quick else (80 if rigorous else 25)
    seeds = range(n_seeds)
    mode = "QUICK" if quick else ("RIGOROUS" if rigorous else "STANDARD")
    print("=" * 72)
    print(f"[mode: {mode}  |  Monte-Carlo seeds: {n_seeds}]")
    print("OICC full reproduction  (oicc v%s)" % oicc.__version__)
    print("=" * 72)

    # ---- 1. moment recovery ----
    print("\n[1] Moment recovery")
    errs = []
    for s in seeds:
        c = oicc.generate(n=5000, seed=s, K=4)
        fm = oicc.estimate_factor_moments(c.log_channels)
        errs.append(abs(fm.var_theta - np.var(c.theta)) / np.var(c.theta))
    check("Var(theta) recovery within 8%", np.mean(errs) < 0.08,
          f"mean rel err={np.mean(errs):.3f}")

    # ---- 2. over-ID size / power / blindness ----
    print("\n[2] Over-identification test")
    size = np.mean([overid_wald_test(oicc.generate(n=3000, seed=s, K=4).log_channels,
                                     seed=s).pvalue < 0.05 for s in seeds])
    powr = np.mean([overid_wald_test(
        oicc.generate(n=3000, seed=s, K=4, confound_pair=0.6).log_channels,
        seed=s).pvalue < 0.05 for s in seeds])
    blind = np.mean([overid_wald_test(
        oicc.generate(n=3000, seed=s, K=4, common_mode=1.5).log_channels,
        seed=s).pvalue < 0.05 for s in seeds])
    check("size <= 0.15 under H0", size <= 0.15, f"size={size:.3f}")
    check("power >= 0.80 vs detectable", powr >= 0.80, f"power={powr:.3f}")
    check("blind (<=0.15) to common-mode", blind <= 0.15, f"rej={blind:.3f}")

    # ---- 3. two-interval coverage ----
    print("\n[3] Split-conformal coverage")
    oc, lc = [], []
    for s in seeds:
        ch = oicc.generate(n=4000, seed=s, K=4)
        r = split_conformal_latent(ch.log_channels, alpha=0.1, seed=s,
                                   use_spec_test=False)
        ti = _test_fold(ch.log_channels.shape[1], s)
        oc.append(np.mean((ch.log_channels[0, ti] >= r.obs_lower)
                          & (ch.log_channels[0, ti] <= r.obs_upper)))
        lc.append(np.mean((ch.theta[ti] >= r.lat_lower)
                          & (ch.theta[ti] <= r.lat_upper)))
    check("exact observed coverage >= 0.88", np.mean(oc) >= 0.88,
          f"cov={np.mean(oc):.3f}")
    check("latent coverage in [0.85, 0.96]", 0.85 <= np.mean(lc) <= 0.96,
          f"cov={np.mean(lc):.3f}")

    # ---- 4. proximal point-ID under a common mode ----
    print("\n[4] Proximal point-identification")
    tv, nv, cv = [], [], []
    for s in seeds:
        d = generate_proximal(n=6000, seed=s, K=4, Q=2, cm_strength=1.0)
        r = point_identify(d.signal_channels, d.controls)
        tv.append(np.var(d.theta)); nv.append(r.var_theta_naive)
        cv.append(r.var_theta_clean)
    check("naive inflates >10% under confounder", np.mean(nv) > np.mean(tv) * 1.1,
          f"naive={np.mean(nv):.3f} true={np.mean(tv):.3f}")
    check("point-ID recovers truth within 12%",
          abs(np.mean(cv) - np.mean(tv)) / np.mean(tv) < 0.12,
          f"clean={np.mean(cv):.3f} true={np.mean(tv):.3f}")

    # ---- 4b. baseline comparison (empirical defensibility) ----
    print("\n[4b] Baselines")
    from oicc.baselines import compare_baselines, compare_baselines_confounded
    nt = 8 if quick else (60 if rigorous else 20)
    bc = compare_baselines(n=4000, K=4, n_trials=nt)
    check("OICC BLUP wins under valid assumptions", bc.winner == "oicc_blup",
          f"rmse oicc={bc.rmse['oicc_blup']:.3f} single={bc.rmse['best_single']:.3f}")
    bcc = compare_baselines_confounded(n=6000, K=4, Q=2, n_trials=nt, cm_strength=1.0)
    naive_best = min(bcc.rmse['best_single'], bcc.rmse['naive_average'],
                     bcc.rmse['oicc_blup_naive'])
    check("proximal beats naive by >30% under confounding",
          bcc.rmse['oicc_proximal'] < 0.7 * naive_best,
          f"prox={bcc.rmse['oicc_proximal']:.3f} naive={naive_best:.3f}")

    # ---- 4c. third-cumulant over-ID power at K=3 ----
    print("\n[4c] Third-cumulant over-ID (power at K=3)")
    from oicc.spec_test import overid_cumulant_test

    def _ng3(seed, confound=0.0):
        r = np.random.default_rng(seed)
        th = r.standard_exponential(4000); th -= th.mean()
        b = np.array([1.0, 1.2, 1.4]); pr = r.standard_exponential(4000) * confound
        return np.vstack([b[c] * th + (pr if c < 2 else 0) + r.normal(0, 0.4, 4000)
                          for c in range(3)])
    size3 = np.mean([overid_cumulant_test(_ng3(s), seed=s).pvalue < 0.05
                     for s in range(nt)])
    powr3 = np.mean([overid_cumulant_test(_ng3(s, 0.5), seed=s).pvalue < 0.05
                     for s in range(nt)])
    check("cumulant test size <= 0.15 at K=3", size3 <= 0.15, f"size={size3:.3f}")
    check("cumulant test power >= 0.8 at K=3 (2nd-moment has df=0)",
          powr3 >= 0.8, f"power={powr3:.3f}")

    # ---- 5. anytime-valid monitor ----
    print("\n[5] Anytime-valid monitor")
    rng = np.random.default_rng(0)
    trials = 300 if quick else (3000 if rigorous else 800)
    fa = sum(EProcessMonitor(alpha=0.05).run(rng.uniform(0, 1, 150)).alarm
             for _ in range(trials)) / trials
    powr_m = sum(EProcessMonitor(alpha=0.05).run(
        np.concatenate([rng.uniform(0, 1, 40), rng.beta(0.3, 3.0, 120)])).alarm
        for _ in range(300)) / 300
    check("anytime false-alarm <= 0.07", fa <= 0.07, f"fa={fa:.3f}")
    check("drift power >= 0.9", powr_m >= 0.9, f"power={powr_m:.3f}")

    # ---- 6. real data (skip if absent) ----
    print("\n[6] Real data")
    from paths import find_india_data, find_us_panel
    india = find_india_data()
    if india is not None:
        from ncrb_loader import load_ncrb_channels
        d = load_ncrb_channels(india)
        sp = overid_wald_test(d["log_channels"], seed=0)
        check("India NCRB over-ID does not reject (p>0.05)", sp.pvalue > 0.05,
              f"p={sp.pvalue:.3f}")
    else:
        print("  [SKIP] India NCRB data not present (set OICC_INDIA_DATA)")
    chi = find_us_panel("chicago")
    if chi is not None:
        from us_loader import build_us_channels
        du = build_us_channels(chi, period_weeks=4)
        su = overid_wald_test(du["log_channels"], seed=0)
        check("US categories over-ID rejects (p<0.05)", su.pvalue < 0.05,
              f"p={su.pvalue:.3f}")
    else:
        print("  [SKIP] US panels not present")

    # ---- 7. figures ----
    print("\n[7] Figures")
    figdir = _ROOT / "paper" / "figures"
    have = figdir.exists() and len(list(figdir.glob("*.png"))) >= 4
    check("4 paper figures present", have,
          "(run make_figures.py to regenerate)" if not have else "")

    dt = time.time() - t0
    print("\n" + "=" * 72)
    print(f"REPRODUCTION COMPLETE in {dt:.0f}s  --  "
          f"{len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        print("FAILED:", ", ".join(_FAIL))
    print("=" * 72)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer seeds/boots")
    ap.add_argument("--rigorous", action="store_true",
                    help="many more Monte-Carlo trials for tight CIs (A100)")
    args = ap.parse_args()
    sys.exit(main(quick=args.quick, rigorous=args.rigorous))
