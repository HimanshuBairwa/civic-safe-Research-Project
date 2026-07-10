"""Publication-grade figure suite for OICC (NeurIPS / Nature quality).

Regenerates paper/figures/pub/*.pdf|png from LIVE, tested computation using the
validated Okabe-Ito palette (oicc_style). Includes the heatmaps and the method
schematic requested for a top-tier submission:

  pub_fig1_method.(pdf|png)         method schematic: channels -> factor -> tests
  pub_fig2_overid_heatmap.(...)     over-ID power HEATMAP over (confounder type x
                                    strength): powerful vs detectable, blind to
                                    common-mode -- the honest limit as a surface
  pub_fig3_coverage.(...)           two-interval coverage across K (grouped bars)
  pub_fig4_pointid_band.(...)       point-ID vs naive + exclusion-sensitivity band
  pub_fig5_channel_corr.(...)       channel correlation HEATMAP (India NCRB real)
  pub_fig6_latent_field.(...)       recovered latent CHOROPLETH (Chicago, real geo)
  pub_fig7_monitor.(...)            anytime-valid e-process wealth trajectories

Run:  python experiments/oicc_runs/make_pub_figures.py
Vector PDF for the paper + PNG preview. Every number is computed here, not hardcoded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_HERE))

import oicc  # noqa: E402
from oicc.spec_test import overid_wald_test                # noqa: E402
from oicc.measurement import generate_proximal             # noqa: E402
from oicc.proximal import point_identify, exclusion_sensitivity  # noqa: E402
from oicc.conformal_split import split_conformal_latent    # noqa: E402
from oicc.monitor import EProcessMonitor                   # noqa: E402
from oicc_style import (apply_style, PALETTE, BLUE, VERMILLION, GREEN, PURPLE,
                        ORANGE, MUTED, INK, SEQ_CMAP, panel_label)  # noqa: E402
from paths import find_india_data, find_us_panel           # noqa: E402

apply_style()
OUT = _ROOT / "paper" / "figures" / "pub"
OUT.mkdir(parents=True, exist_ok=True)


def _save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


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


# --------------------------------------------------------------------------- #
def fig1_method():
    """Schematic: K biased channels -> one-factor -> over-ID test + conformal +
    proximal escape. Clean boxes-and-arrows, publication style."""
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    def box(x, y, w, h, text, fc, tc="#ffffff", fs=8.5):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=fc, ec="none"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                color=tc, fontsize=fs, fontweight="bold", wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=12, lw=1.4, color=MUTED))

    # channels
    chans = ["records", "911 calls", "survey", "..."]
    for i, c in enumerate(chans):
        box(0.2, 4.0 - i * 1.05, 1.7, 0.75, c, PALETTE[i % 6])
        arrow(1.95, 4.37 - i * 1.05, 3.0, 2.5)
    # latent factor
    box(3.05, 2.0, 1.9, 1.0, "one-factor\nmodel\n$Y^c=\\beta_c\\theta+\\varepsilon^c$",
        INK, fs=8)
    arrow(4.95, 2.5, 5.6, 3.7)
    arrow(4.95, 2.5, 5.6, 2.2)
    arrow(4.95, 2.5, 5.6, 0.7)
    # outputs
    box(5.65, 3.35, 2.1, 0.85, "over-ID test\n(spec check)", GREEN, fs=8)
    box(5.65, 1.85, 2.1, 0.85, "leave-pivot-out\nconformal", BLUE, fs=8)
    box(5.65, 0.35, 2.1, 0.85, "proximal escape\n(neg. controls)", PURPLE, fs=8)
    arrow(7.75, 3.77, 8.4, 2.7)
    arrow(7.75, 2.27, 8.4, 2.6)
    arrow(7.75, 0.77, 8.4, 2.4)
    box(8.45, 2.1, 1.4, 1.0, "latent\nrate $\\hat\\theta$\n+ honesty", VERMILLION, fs=8)
    ax.set_title("OICC: honest latent-rate estimation from biased channels", pad=6)
    _save(fig, "pub_fig1_method")


# --------------------------------------------------------------------------- #
def fig2_overid_heatmap():
    """Over-ID rejection rate as a HEATMAP over (confounder type x strength).
    Shows powerful-vs-detectable and blind-to-common-mode as a surface."""
    strengths = np.linspace(0.0, 1.0, 9)
    kinds = ["detectable", "common-mode"]
    grid = np.zeros((len(kinds), len(strengths)))
    for j, s in enumerate(strengths):
        grid[0, j] = np.mean([overid_wald_test(
            oicc.generate(n=2500, seed=i, K=4, confound_pair=s).log_channels,
            seed=i).pvalue < 0.05 for i in range(12)])
        grid[1, j] = np.mean([overid_wald_test(
            oicc.generate(n=2500, seed=i, K=4, common_mode=s).log_channels,
            seed=i).pvalue < 0.05 for i in range(12)])
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    im = ax.imshow(grid, aspect="auto", cmap=SEQ_CMAP, vmin=0, vmax=1,
                   extent=[strengths[0], strengths[-1], 0, 2], origin="lower")
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(
        ["detectable\n$\\Delta_\\perp$", "common-mode\n$\\Delta_\\parallel$"])
    ax.set_xlabel("confounder strength")
    ax.set_title("Over-identification test: powerful vs detectable,\n"
                 "provably blind to the common mode")
    cb = fig.colorbar(im, ax=ax, pad=0.02); cb.set_label("rejection rate")
    ax.grid(False)
    _save(fig, "pub_fig2_overid_heatmap")


# --------------------------------------------------------------------------- #
def fig3_coverage():
    Ks = [3, 4, 5]
    obs, lat = [], []
    for K in Ks:
        oc, lc = [], []
        for s in range(20):
            ch = oicc.generate(n=4000, seed=s, K=K)
            r = split_conformal_latent(ch.log_channels, alpha=0.1, seed=s,
                                       use_spec_test=False)
            ti = _test_fold(ch.log_channels.shape[1], s)
            oc.append(np.mean((ch.log_channels[0, ti] >= r.obs_lower)
                              & (ch.log_channels[0, ti] <= r.obs_upper)))
            lc.append(np.mean((ch.theta[ti] >= r.lat_lower)
                              & (ch.theta[ti] <= r.lat_upper)))
        obs.append(np.mean(oc)); lat.append(np.mean(lc))
    x = np.arange(len(Ks))
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.bar(x - 0.2, obs, 0.38, color=BLUE, label="observed (exact, finite-sample)")
    ax.bar(x + 0.2, lat, 0.38, color=GREEN, label="latent (asymptotic)")
    ax.axhline(0.90, ls="--", color=MUTED, lw=1)
    ax.text(len(Ks) - 0.5, 0.905, "target 0.90", color=MUTED, fontsize=8, ha="right")
    for xi, (o, l) in enumerate(zip(obs, lat)):
        ax.text(xi - 0.2, o + 0.004, f"{o:.2f}", ha="center", fontsize=7.5, color=INK)
        ax.text(xi + 0.2, l + 0.004, f"{l:.2f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([f"K={k}" for k in Ks])
    ax.set_ylim(0.80, 1.0); ax.set_ylabel("empirical coverage")
    ax.set_title("Two-interval conformal coverage")
    ax.legend(loc="lower center")
    _save(fig, "pub_fig3_coverage")


# --------------------------------------------------------------------------- #
def fig4_pointid_band():
    cms = np.linspace(0, 2, 9)
    tv, nv, cv = [], [], []
    for cm in cms:
        a, b, c = [], [], []
        for s in range(15):
            d = generate_proximal(n=6000, seed=s, K=4, Q=2, cm_strength=cm)
            r = point_identify(d.signal_channels, d.controls)
            a.append(np.var(d.theta)); b.append(r.var_theta_naive)
            c.append(r.var_theta_clean)
        tv.append(np.mean(a)); nv.append(np.mean(b)); cv.append(np.mean(c))
    d0 = generate_proximal(n=8000, seed=1, K=4, Q=2, cm_strength=1.2)
    es = exclusion_sensitivity(d0.signal_channels, d0.controls, eps_max=0.4, n_grid=17)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    ax = axes[0]
    ax.plot(cms, tv, "-", color="#000000", lw=2, label=r"true $\mathrm{Var}(\theta)$")
    ax.plot(cms, nv, "-o", color=VERMILLION, label="naive (confounded)")
    ax.plot(cms, cv, "-s", color=GREEN, label="proximal point-ID")
    ax.set_xlabel("common-mode strength"); ax.set_ylabel(r"$\mathrm{Var}(\theta)$")
    ax.set_title("Point-ID recovers the truth"); ax.legend(loc="upper left")
    panel_label(ax, "(a)")
    ax = axes[1]
    ax.fill_between(es.eps_grid, es.var_theta_lo, es.var_theta_hi, color=BLUE,
                    alpha=0.25, label="implied band (both signs)")
    ax.plot(es.eps_grid, es.var_theta_lo, color=BLUE, lw=1)
    ax.plot(es.eps_grid, es.var_theta_hi, color=BLUE, lw=1)
    ax.axhline(es.var_theta_ref, color=GREEN, lw=2, label=r"estimate ($\epsilon=0$)")
    ax.axhline(float(np.var(d0.theta)), ls="--", color="#000000", lw=1.2,
               label=r"true $\mathrm{Var}(\theta)$")
    if es.robustness_eps < 1:
        ax.axvline(es.robustness_eps, ls=":", color=VERMILLION, lw=1.5,
                   label=fr"$\epsilon^*={es.robustness_eps:.2f}$")
    ax.set_xlabel(r"exclusion violation $\epsilon$")
    ax.set_ylabel(r"$\mathrm{Var}(\theta)$")
    ax.set_title("Exclusion-sensitivity band"); ax.legend(loc="upper right", fontsize=7.5)
    panel_label(ax, "(b)")
    fig.tight_layout()
    _save(fig, "pub_fig4_pointid_band")


# --------------------------------------------------------------------------- #
def fig5_channel_corr():
    india = find_india_data()
    if india is None:
        return
    from ncrb_loader import load_ncrb_channels
    d = load_ncrb_channels(india)
    C = np.corrcoef(d["log_channels"])
    names = d["channel_names"]
    fig, ax = plt.subplots(figsize=(4.4, 3.8))
    im = ax.imshow(C, cmap=SEQ_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right"); ax.set_yticklabels(names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center",
                    color="white" if C[i, j] < 0.7 else "#000000", fontsize=8)
    ax.set_title("India NCRB channel correlation\n(all positive = shared latent)")
    fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    ax.grid(False)
    _save(fig, "pub_fig5_channel_corr")


# --------------------------------------------------------------------------- #
def fig6_latent_field():
    """Recovered latent crime field as a real Chicago choropleth."""
    panel = find_us_panel("chicago")
    geo = _ROOT / "data" / "raw" / "shapefiles" / "chicago_community_areas.geojson"
    if panel is None or not geo.exists():
        return
    try:
        import geopandas as gpd
        import torch
    except ImportError:
        return
    p = torch.load(panel, weights_only=False)
    counts = np.asarray(p["counts"], dtype=float)          # (S, T, C)
    S, T, C = counts.shape
    # 3 category channels aggregated over time -> per-area log-rate, then OICC BLUP
    per = 4
    P = T // per
    agg = counts[:, : P * per, :].reshape(S, P, per, C).sum(axis=2)  # (S,P,C)
    chan = np.vstack([np.log1p(agg[:, :, c].reshape(-1)) for c in range(C)])  # (3, S*P)
    est = oicc.deconvolve_blup(chan)
    theta = est.theta_hat.reshape(S, P).mean(axis=1)       # per-area mean latent

    gdf = gpd.read_file(geo)
    # align by area_number/community index (1..S)
    key = "area_number" if "area_number" in gdf.columns else gdf.columns[0]
    try:
        gdf = gdf.sort_values(key).reset_index(drop=True)
    except Exception:
        pass
    m = min(len(gdf), S)
    gdf = gdf.iloc[:m].copy()
    gdf["latent"] = theta[:m]

    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    gdf.plot(column="latent", ax=ax, cmap=SEQ_CMAP, edgecolor="white",
             linewidth=0.4, legend=True,
             legend_kwds={"label": "recovered latent log-rate", "shrink": 0.6})
    ax.set_title("OICC recovered latent crime field\n(Chicago community areas)")
    ax.axis("off")
    _save(fig, "pub_fig6_latent_field")


# --------------------------------------------------------------------------- #
def fig7_monitor():
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    for i in range(10):
        p = rng.uniform(0, 1, 160)
        m = EProcessMonitor(alpha=0.05)
        w = [m.update(pi) or m.wealth for pi in p]
        ax.plot(np.log10(np.maximum(w, 1e-3)), color=BLUE, alpha=0.3, lw=0.8,
                label="H0 (structure holds)" if i == 0 else None)
    for i in range(10):
        p = np.concatenate([rng.uniform(0, 1, 60), rng.beta(0.3, 3.0, 100)])
        m = EProcessMonitor(alpha=0.05)
        w = [m.update(pi) or m.wealth for pi in p]
        ax.plot(np.log10(np.maximum(w, 1e-3)), color=VERMILLION, alpha=0.45, lw=0.8,
                label="drift at t=60" if i == 0 else None)
    ax.axhline(np.log10(1 / 0.05), ls="--", color="#000000", lw=1,
               label=r"alarm $\log_{10}(1/\alpha)$")
    ax.axvline(60, ls=":", color=MUTED, lw=1)
    ax.set_xlabel("monitoring window $t$")
    ax.set_ylabel(r"$\log_{10}$ e-process wealth")
    ax.set_title("Anytime-valid monitor: quiet under H0, fires after drift")
    ax.legend(loc="upper left")
    _save(fig, "pub_fig7_monitor")


def main():
    print("generating publication figures ->", OUT)
    for fn, nm in [(fig1_method, "method schematic"),
                   (fig2_overid_heatmap, "over-ID power heatmap"),
                   (fig3_coverage, "two-interval coverage"),
                   (fig4_pointid_band, "point-ID + exclusion band"),
                   (fig5_channel_corr, "channel-correlation heatmap (real)"),
                   (fig6_latent_field, "latent choropleth (real Chicago geo)"),
                   (fig7_monitor, "anytime-valid monitor")]:
        try:
            fn(); print(f"  [OK] {nm}")
        except Exception as e:  # never let one figure kill the batch
            print(f"  [SKIP] {nm}: {type(e).__name__}: {e}")
    print("done:", sorted(p.name for p in OUT.glob("*.pdf")))


if __name__ == "__main__":
    main()
