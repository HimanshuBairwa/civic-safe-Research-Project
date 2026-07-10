"""Generate all OICC paper figures from the actual (verified) experiments.

Produces PNGs in paper/figures/:
  fig1_overid_size_power.png    over-ID test: size, power vs Delta-perp,
                                blindness to Delta-parallel (the honest limit)
  fig2_coverage.png             exact observed vs latent coverage across K
  fig3_proximal_rescue.png      naive vs point-ID latent variance under a
                                common-mode confounder (the ceiling-lifter)
  fig4_monitor.png              anytime-valid e-process wealth: H0 vs drift

Run:  python experiments/oicc_runs/make_figures.py
Everything here re-runs the verified computations; no numbers are hard-coded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import oicc  # noqa: E402
from oicc.spec_test import overid_wald_test          # noqa: E402
from oicc.measurement import generate_proximal        # noqa: E402
from oicc.proximal import point_identify              # noqa: E402
from oicc.conformal_split import split_conformal_latent  # noqa: E402
from oicc.monitor import EProcessMonitor              # noqa: E402

OUT = _ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# a clean, accessible style
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25,
})
BLUE, ORANGE, GREEN, RED = "#2b6cb0", "#dd6b20", "#2f855a", "#c53030"


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


def fig1_overid():
    strengths = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    perp_power, para_power = [], []
    for s in strengths:
        rp = np.mean([overid_wald_test(
            oicc.generate(n=3000, seed=i, K=4, confound_pair=s).log_channels,
            seed=i).pvalue < 0.05 for i in range(20)])
        pp = np.mean([overid_wald_test(
            oicc.generate(n=3000, seed=i, K=4, common_mode=s).log_channels,
            seed=i).pvalue < 0.05 for i in range(20)])
        perp_power.append(rp)
        para_power.append(pp)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(strengths, perp_power, "-o", color=BLUE,
            label=r"detectable $\Delta_\perp$ confounder")
    ax.plot(strengths, para_power, "-s", color=RED,
            label=r"common-mode $\Delta_\parallel$ confounder")
    ax.axhline(0.05, ls="--", color="gray", lw=1, label=r"nominal size $\alpha=0.05$")
    ax.set_xlabel("confounder strength")
    ax.set_ylabel("over-ID rejection rate")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("Over-identification test: powerful vs detectable,\n"
                 "provably blind to the common mode (the honest limit)")
    ax.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_overid_size_power.png")
    plt.close(fig)


def fig2_coverage():
    Ks = [3, 4, 5]
    obs_c, lat_c = [], []
    for K in Ks:
        oc, lc = [], []
        for s in range(25):
            ch = oicc.generate(n=4000, seed=s, K=K)
            r = split_conformal_latent(ch.log_channels, alpha=0.1, seed=s,
                                        use_spec_test=False)
            ti = _test_fold(ch.log_channels.shape[1], s)
            oc.append(np.mean((ch.log_channels[0, ti] >= r.obs_lower)
                              & (ch.log_channels[0, ti] <= r.obs_upper)))
            lc.append(np.mean((ch.theta[ti] >= r.lat_lower)
                              & (ch.theta[ti] <= r.lat_upper)))
        obs_c.append(np.mean(oc))
        lat_c.append(np.mean(lc))
    x = np.arange(len(Ks))
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.bar(x - 0.19, obs_c, 0.36, color=BLUE,
           label="exact observed (finite-sample)")
    ax.bar(x + 0.19, lat_c, 0.36, color=GREEN,
           label="latent (asymptotic)")
    ax.axhline(0.90, ls="--", color="gray", lw=1, label="target 0.90")
    ax.set_xticks(x)
    ax.set_xticklabels([f"K={k}" for k in Ks])
    ax.set_ylim(0.80, 1.0)
    ax.set_ylabel("empirical coverage")
    ax.set_title("Two-interval coverage on synthetic ground truth")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_coverage.png")
    plt.close(fig)


def fig3_proximal():
    cms = [0.0, 0.5, 1.0, 1.5, 2.0]
    true_v, naive_v, clean_v = [], [], []
    for cm in cms:
        tv, nv, cv = [], [], []
        for s in range(20):
            d = generate_proximal(n=6000, seed=s, K=4, Q=2, cm_strength=cm)
            r = point_identify(d.signal_channels, d.controls)
            tv.append(np.var(d.theta))
            nv.append(r.var_theta_naive)
            cv.append(r.var_theta_clean)
        true_v.append(np.mean(tv))
        naive_v.append(np.mean(nv))
        clean_v.append(np.mean(cv))
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(cms, true_v, "-k", lw=2, label=r"true $\mathrm{Var}(\theta)$")
    ax.plot(cms, naive_v, "-o", color=RED,
            label="naive (confounded, invisible to over-ID)")
    ax.plot(cms, clean_v, "-s", color=GREEN,
            label="proximal point-ID (2 controls)")
    ax.set_xlabel("common-mode confounder strength")
    ax.set_ylabel(r"estimated $\mathrm{Var}(\theta)$")
    ax.set_title("Negative-control point-ID recovers the truth\n"
                 "where the over-ID test is blind")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig3_proximal_rescue.png")
    plt.close(fig)


def fig4_monitor():
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    # H0 streams (uniform p): wealth stays low
    for i in range(8):
        p = rng.uniform(0, 1, 160)
        m = EProcessMonitor(alpha=0.05)
        w = [m.update(pi) or m.wealth for pi in p]
        ax.plot(np.log10(np.maximum(w, 1e-3)), color=BLUE, alpha=0.35,
                lw=0.9, label="H0 (structure holds)" if i == 0 else None)
    # drift at t=60: p-values become small
    for i in range(8):
        p = np.concatenate([rng.uniform(0, 1, 60), rng.beta(0.3, 3.0, 100)])
        m = EProcessMonitor(alpha=0.05)
        w = [m.update(pi) or m.wealth for pi in p]
        ax.plot(np.log10(np.maximum(w, 1e-3)), color=RED, alpha=0.5,
                lw=0.9, label="drift at t=60" if i == 0 else None)
    ax.axhline(np.log10(1 / 0.05), ls="--", color="k", lw=1,
               label=r"alarm $\log_{10}(1/\alpha)$")
    ax.axvline(60, ls=":", color="gray", lw=1)
    ax.set_xlabel("monitoring window t")
    ax.set_ylabel(r"$\log_{10}$ e-process wealth $M_t$")
    ax.set_title("Anytime-valid monitor: quiet under H0,\nfires after drift")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_monitor.png")
    plt.close(fig)


def fig5_exclusion_sensitivity():
    from oicc.proximal import exclusion_sensitivity
    d = generate_proximal(n=8000, seed=1, K=4, Q=2, cm_strength=1.2,
                          ctrl_theta_load=0.0)
    true = float(np.var(d.theta))
    es = exclusion_sensitivity(d.signal_channels, d.controls,
                               eps_max=0.4, n_grid=17)
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.fill_between(es.eps_grid, es.var_theta_lo, es.var_theta_hi,
                    color=BLUE, alpha=0.25,
                    label=r"implied $\mathrm{Var}(\theta)$ band (both signs)")
    ax.plot(es.eps_grid, es.var_theta_lo, color=BLUE, lw=1)
    ax.plot(es.eps_grid, es.var_theta_hi, color=BLUE, lw=1)
    ax.axhline(es.var_theta_ref, ls="-", color=GREEN, lw=2,
               label=r"point estimate ($\epsilon=0$)")
    ax.axhline(true, ls="--", color="k", lw=1.3,
               label=r"true $\mathrm{Var}(\theta)$")
    if es.robustness_eps < 1.0:
        ax.axvline(es.robustness_eps, ls=":", color=RED, lw=1.5,
                   label=fr"robustness $\epsilon^*={es.robustness_eps:.2f}$")
    ax.set_xlabel(r"exclusion violation $\epsilon$ "
                  r"(fraction of control variance driven by $\theta$)")
    ax.set_ylabel(r"$\mathrm{Var}(\theta)$")
    ax.set_title("Exclusion-sensitivity: how the point estimate could move\n"
                 "if the negative controls secretly carry the latent")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_exclusion_sensitivity.png")
    plt.close(fig)


def main():
    print("generating figures ->", OUT)
    fig1_overid(); print("  fig1 over-ID size/power/blindness")
    fig2_coverage(); print("  fig2 two-interval coverage")
    fig3_proximal(); print("  fig3 proximal point-ID rescue")
    fig4_monitor(); print("  fig4 anytime-valid monitor")
    fig5_exclusion_sensitivity(); print("  fig5 exclusion-sensitivity band")
    print("done:", sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
