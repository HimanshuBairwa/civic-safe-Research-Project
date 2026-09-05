#!/usr/bin/env python
"""Generate the four supplementary figures by computing them, not by transcribing.

Every value plotted here is recomputed at figure-generation time from raw data or
from a persisted result file. Nothing is hardcoded from the manuscript, so if an
input changes the figure changes with it, and if an input is missing the script
fails loudly rather than drawing a stale curve.

Figures:
  figS_count_distributions   weekly count distribution per crime category against
                              the best cell-heterogeneous Poisson null. Justifies
                              both halves of ZINB: the zero atom and the negative
                              binomial overdispersion.
  figS_graph_degree          in-degree distributions for queen contiguity against
                              centroid k-NN (k=8), both cities. Shows the isolated
                              New York precinct that motivates the dual graph.
  figS_gamma_frontier        the Gamma-sensitivity frontier: coverage and width
                              cost against the assumed misspecification factor.

  figS_ensemble_uncertainty  aleatoric/epistemic variance split and per-member CRPS
                             for both cities. Supersedes fig6_uncertainty_decomposition
                             for the supplementary: that one is a two-wedge donut for
                             New York alone and carries a "CIVIC-SAFE" watermark.

A note on the weekly bin anchor, because it is the one place this script departs
from the training pipeline. The descriptive sparsity audit reported in the
supplementary and in docs/REVIEWER_DEFENSE_DOSSIER.md was computed on
Monday-anchored weeks, and this script reproduces that anchor exactly so the
figure agrees with the prose (drug 50.89% zero cells, max 68; violent 1.09%, 169;
property 0.20%, 532). The model panel built by
civicsafe.data.panel.build_spatiotemporal_panel is Sunday-anchored, which gives
51.00% / 1.13% / 0.23% on the same 313 weeks and the same 1,326,056 incidents.
Totals and per-cell means are identical under either anchor -- only the zero
fraction and the per-cell maximum move, by about a tenth of a percentage point --
so no reported model metric depends on the choice. Stated here rather than left
for a reader to rediscover.

Run:
    python scripts/make_supplementary_figures.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
DATA_DIR = PROJECT_ROOT / "outputs" / "figure_data"

CATEGORIES = ("violent", "property", "drug")
WEEK_ANCHOR = "W-MON"
PANEL_START, PANEL_END = "2018-01-01", "2023-12-31"

# Okabe-Ito, extending the palette already used by make_paper_figures.py. Verified
# colour-blind safe: worst adjacent pair separates at deutan Delta E 11.0.
C_CAT = {"violent": "#0072b2", "property": "#d55e00", "drug": "#009e73"}
C_QUEEN = "#c1272d"
C_KNN = "#0072b2"
C_PLAIN = "#c1272d"
C_INFLATED = "#0072b2"
C_REF = "#555555"

TARGET_COVERAGE = 0.90


def _style() -> None:
    plt.rcParams.update({
        # Serif, matching scripts/generate_figures.py. That script produced
        # fig6_uncertainty_decomposition, which appears in the same supplementary
        # document as these three, and it matches IEEEtran's serif body text.
        "font.family": "serif",
        # Match math glyphs to the serif text rather than leaving mathtext on its
        # sans default, which would set $\Gamma$ and $\times$ in a different face.
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "lines.linewidth": 1.6,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        # Embed TrueType rather than Type 3. IEEE PDF eXpress flags Type 3 fonts,
        # which is matplotlib's default for PDF output.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _save(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path)
        print(f"  wrote {path.relative_to(PROJECT_ROOT)}")
    plt.close(fig)


def _dump(rows: object, name: str) -> None:
    """Persist the plotted numbers so the figure is auditable after the fact."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"  wrote {path.relative_to(PROJECT_ROOT)}")


# ===================================================================
# Fig S1 -- count distributions and the case for ZINB
# ===================================================================
def _chicago_panel() -> tuple[np.ndarray, int]:
    """Chicago (S, T, C) weekly counts on Monday-anchored weeks.

    Returns the counts array and the number of incidents binned into it.
    """
    files = sorted(glob.glob(str(PROJECT_ROOT / "data" / "raw" / "chicago" / "*.parquet")))
    if not files:
        raise SystemExit("no Chicago parquet files under data/raw/chicago/")
    crime = pd.concat(
        [pd.read_parquet(f, columns=["date", "spatial_unit", "category"]) for f in files],
        ignore_index=True,
    )
    crime["date"] = pd.to_datetime(crime["date"], errors="coerce")
    crime = crime.dropna(subset=["date"])

    acs = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "chicago_demographics.csv")
    units = sorted(acs["spatial_unit"].unique())

    weeks = pd.date_range(start=PANEL_START, end=PANEL_END, freq=WEEK_ANCHOR)
    s_idx = crime["spatial_unit"].map({u: i for i, u in enumerate(units)})
    c_idx = crime["category"].map({c: i for i, c in enumerate(CATEGORIES)})
    t_idx = np.searchsorted(
        weeks.values.astype("datetime64[ns]"),
        crime["date"].values.astype("datetime64[ns]"),
        side="right",
    ) - 1

    keep = (s_idx.notna() & c_idx.notna()).values
    keep &= (t_idx >= 0) & (t_idx < len(weeks))
    counts = np.zeros((len(units), len(weeks), len(CATEGORIES)), dtype=np.int64)
    np.add.at(
        counts,
        (s_idx[keep].astype(int).values, t_idx[keep], c_idx[keep].astype(int).values),
        1,
    )
    return counts, int(keep.sum())


def _poisson_pmf(k: np.ndarray, lam: np.ndarray) -> np.ndarray:
    """Poisson pmf, cells on axis 0 and support on axis 1, via log-gamma."""
    from scipy.special import gammaln

    lam = np.clip(lam[:, None], 1e-12, None)
    return np.exp(k[None, :] * np.log(lam) - lam - gammaln(k[None, :] + 1.0))


def figS1() -> None:
    print("[figS1] building Chicago weekly panel from raw parquet...")
    counts, n_incidents = _chicago_panel()
    S, T, C = counts.shape
    print(f"  panel S={S} T={T} C={C}, {n_incidents:,} incidents binned")

    stats: dict[str, dict[str, float]] = {}
    for c, name in enumerate(CATEGORIES):
        x = counts[:, :, c]
        cell_means = x.mean(axis=1)                      # (S,) per-area weekly mean
        # The fair null is not one Poisson at the pooled mean -- areas differ by an
        # order of magnitude. It is a mixture of cell-specific Poissons, i.e. the
        # best a model can do knowing each area's level but allowing neither
        # overdispersion nor a zero atom. That is precisely what ZINB adds.
        implied_zero = float(np.mean(np.exp(-np.clip(cell_means, 0, None))))
        flat = x.ravel()
        stats[name] = {
            "mean": float(flat.mean()),
            "zero_frac": float((flat == 0).mean()),
            "poisson_zero_frac_cellwise": implied_zero,
            "poisson_zero_frac_pooled": float(np.exp(-flat.mean())),
            "max": int(flat.max()),
            "median": float(np.median(flat)),
            "variance": float(flat.var()),
            "var_over_mean": float(flat.var() / max(flat.mean(), 1e-12)),
        }
        print(
            "  %-9s mean %6.2f  zeros %6.2f%%  cellwise-Poisson zeros %6.2f%%  "
            "max %4d  var/mean %6.2f"
            % (name, stats[name]["mean"], 100 * stats[name]["zero_frac"],
               100 * implied_zero, stats[name]["max"], stats[name]["var_over_mean"])
        )

    _style()
    fig = plt.figure(figsize=(7.0, 4.9))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.85], hspace=0.55, wspace=0.32)

    # --- Row 1: empirical distribution against the cell-heterogeneous Poisson ---
    for c, name in enumerate(CATEGORIES):
        ax = fig.add_subplot(gs[0, c])
        x = counts[:, :, c]
        flat = x.ravel()
        cap = int(np.percentile(flat, 99.5))
        cap = max(cap, 12)
        k = np.arange(0, cap + 1)

        emp = np.array([(flat == kk).mean() for kk in k])
        mix = _poisson_pmf(k.astype(float), x.mean(axis=1)).mean(axis=0)

        ax.bar(k, emp, width=1.0, color=C_CAT[name], alpha=0.85, linewidth=0,
               label="observed")
        ax.plot(k, mix, color="#222222", linewidth=1.3, linestyle="--",
                label="Poisson null")
        ax.set_yscale("log")
        ax.set_ylim(1e-5, 1.4)
        ax.set_xlim(-0.7, cap + 0.7)
        ax.set_title("(%s) %s" % ("abc"[c], name), fontweight="bold")
        ax.set_xlabel("weekly count per area")
        if c == 0:
            ax.set_ylabel("fraction of cell-weeks")
        ax.annotate(
            "zeros %.2f%%\nnull %.2f%%" % (100 * stats[name]["zero_frac"],
                                           100 * stats[name]["poisson_zero_frac_cellwise"]),
            xy=(0.97, 0.94), xycoords="axes fraction", ha="right", va="top",
            fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="#bbbbbb", lw=0.5),
        )
        if c == 0:
            ax.legend(loc="lower left", frameon=False, fontsize=7.5)

    # --- Row 2: the two things ZINB buys, side by side ---
    ax = fig.add_subplot(gs[1, 0:2])
    idx = np.arange(len(CATEGORIES))
    w = 0.36
    obs = [stats[n]["zero_frac"] for n in CATEGORIES]
    nul = [stats[n]["poisson_zero_frac_cellwise"] for n in CATEGORIES]
    ax.bar(idx - w / 2, obs, w, color=[C_CAT[n] for n in CATEGORIES], linewidth=0,
           label="observed zeros")
    ax.bar(idx + w / 2, nul, w, facecolor="none", edgecolor="#333333", linewidth=1.0,
           hatch="////", label="Poisson null")
    ax.set_yscale("log")
    ax.set_xticks(idx)
    ax.set_xticklabels(CATEGORIES)
    ax.set_ylabel("zero-cell fraction")
    ax.set_title("(d) excess zero mass vs. null", fontweight="bold", loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=7.5, ncol=2)
    for i, n in enumerate(CATEGORIES):
        ratio = stats[n]["zero_frac"] / max(stats[n]["poisson_zero_frac_cellwise"], 1e-30)
        ax.annotate(r"$\times$%.1f" % ratio if ratio < 100 else r"$\times$%.0f" % ratio,
                    xy=(i, max(obs[i], nul[i]) * 1.9), ha="center", fontsize=7.5)
    ax.set_ylim(min(min(nul), min(obs)) * 0.22, 22.0)

    ax = fig.add_subplot(gs[1, 2])
    vom = [stats[n]["var_over_mean"] for n in CATEGORIES]
    ax.bar(idx, vom, 0.6, color=[C_CAT[n] for n in CATEGORIES], linewidth=0)
    ax.axhline(1.0, color=C_REF, linestyle=":", linewidth=1.2)
    ax.annotate("Poisson (=1)", xy=(len(idx) - 0.45, 1.0), xytext=(0, 4),
                textcoords="offset points", fontsize=7, color=C_REF, ha="right")
    ax.set_xticks(idx)
    ax.set_xticklabels(CATEGORIES, rotation=20, ha="right")
    ax.set_ylabel("variance / mean")
    ax.set_title("(e) overdispersion", fontweight="bold", loc="left")

    _save(fig, "figS_count_distributions")
    _dump({
        "city": "chicago",
        "week_anchor": WEEK_ANCHOR,
        "panel": {"S": S, "T": T, "C": C, "incidents": n_incidents},
        "note": (
            "Poisson null is a mixture of cell-specific Poissons at each area's own "
            "weekly mean, not a single Poisson at the pooled mean."
        ),
        "per_category": stats,
    }, "figS_count_distributions")


# ===================================================================
# Fig S2 -- graph topology and the isolated precinct
# ===================================================================
def figS2() -> None:
    import geopandas as gpd

    from civicsafe.models.graph import build_adjacency_from_geodataframe

    shp = PROJECT_ROOT / "data" / "raw" / "shapefiles"
    cities = [
        ("Chicago", shp / "chicago_community_areas.geojson", "EPSG:26971"),
        ("New York", shp / "nyc_police_precincts.geojson", "EPSG:32118"),
    ]

    print("[figS2] building dual adjacency from released shapefiles...")
    result: dict[str, dict[str, dict]] = {}
    degrees: dict[str, dict[str, np.ndarray]] = {}
    for label, path, crs in cities:
        if not path.exists():
            raise SystemExit(f"shapefile missing: {path}")
        gdf = gpd.read_file(path)
        adj = build_adjacency_from_geodataframe(gdf, knn_k=8, meter_crs=crs)
        n = len(gdf)
        result[label], degrees[label] = {}, {}
        for gname in ("queen", "knn"):
            ei = adj[gname].numpy()
            # In-degree: count edges arriving at each node (row 1 is the target).
            deg = np.bincount(ei[1], minlength=n) if ei.size else np.zeros(n, int)
            degrees[label][gname] = deg
            result[label][gname] = {
                "nodes": int(n),
                "edges": int(ei.shape[1]) if ei.size else 0,
                "mean_degree": float(deg.mean()),
                "min_degree": int(deg.min()),
                "max_degree": int(deg.max()),
                "n_isolated": int((deg == 0).sum()),
            }
            r = result[label][gname]
            print("  %-9s %-5s nodes %3d edges %4d mean %5.2f min %2d max %2d isolated %d"
                  % (label, gname, r["nodes"], r["edges"], r["mean_degree"],
                     r["min_degree"], r["max_degree"], r["n_isolated"]))

    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    ceiling = 0
    for ax, (label, _, _) in zip(axes, cities):
        dq, dk = degrees[label]["queen"], degrees[label]["knn"]
        hi = int(max(dq.max(), dk.max()))
        k = np.arange(0, hi + 1)
        hq = np.array([(dq == kk).sum() for kk in k])
        hk = np.array([(dk == kk).sum() for kk in k])
        ceiling = max(ceiling, int(hq.max()), int(hk.max()))
        w = 0.42
        ax.bar(k - w / 2, hq, w, color=C_QUEEN, linewidth=0, label="queen contiguity")
        ax.bar(k + w / 2, hk, w, color=C_KNN, linewidth=0, label=r"centroid $k$-NN ($k$=8)")
        ax.set_xlabel("in-degree")
        ax.set_title(label, fontweight="bold")
        ax.set_xticks(k[::2])
        ax.set_xlim(-0.8, hi + 0.8)
        if result[label]["queen"]["n_isolated"] > 0:
            ax.annotate(
                "%d isolated precinct,\nno spatial messages"
                % result[label]["queen"]["n_isolated"],
                xy=(-w / 2, 1.35), xytext=(2.3, ceiling * 1.12),
                fontsize=7.5, color=C_QUEEN, ha="left", va="bottom",
                arrowprops=dict(arrowstyle="->", color=C_QUEEN, lw=1.0,
                                shrinkA=1, shrinkB=1,
                                connectionstyle="arc3,rad=0.28"),
            )
    # Headroom so the annotation and legend clear the tallest bar.
    axes[0].set_ylim(0, ceiling * 1.42)
    axes[0].set_ylabel("number of units")
    axes[0].legend(loc="upper left", frameon=False, fontsize=7.5)
    fig.tight_layout()

    _save(fig, "figS_graph_degree")
    _dump(result, "figS_graph_degree")


# ===================================================================
# Fig S3 -- the Gamma-sensitivity frontier
# ===================================================================
def figS3() -> None:
    src = PROJECT_ROOT / "outputs" / "gamma_sensitivity_frontier.json"
    if not src.exists():
        raise SystemExit(
            f"{src} missing. Run scripts/misspecification_sensitivity.py first."
        )
    payload = json.loads(src.read_text(encoding="utf-8"))
    rows = payload["rows"]
    g = np.array([r["gamma"] for r in rows], dtype=float)
    plain = np.array([r["plain"] for r in rows], dtype=float)
    infl = np.array([r["inflated"] for r in rows], dtype=float)
    width = np.array([r["width_ratio"] for r in rows], dtype=float)

    below = g[plain < TARGET_COVERAGE]
    breakdown = float(below.min()) if below.size else float("nan")
    last_ok = float(g[plain >= TARGET_COVERAGE].max())
    print(f"[figS3] kappa={payload['kappa']} target={1 - payload['alpha']:.2f} "
          f"cells={payload['cells']} trials={payload['trials']}")
    print(f"  plain last meets target at Gamma={last_ok}, first fails at {breakdown}")

    _style()
    # Two stacked panels sharing the Gamma axis rather than one panel with two
    # y-scales. Coverage and width ratio are different units, and a twin axis lets
    # the apparent crossing point be placed anywhere by choice of scaling.
    fig, (ax0, ax1) = plt.subplots(
        2, 1, figsize=(5.6, 4.3), sharex=True,
        gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.16},
    )

    ax0.axvspan(breakdown, g.max() + 0.1, color="#c1272d", alpha=0.06, linewidth=0)
    ax0.axhline(TARGET_COVERAGE, color=C_REF, linestyle=":", linewidth=1.2)
    ax0.plot(g, plain, "o-", color=C_PLAIN, markersize=4,
             label="plain corrected interval")
    ax0.plot(g, infl, "s-", color=C_INFLATED, markersize=4,
             label=r"$\Gamma$-inflated interval")
    ax0.annotate("target %.2f" % TARGET_COVERAGE, xy=(g.max(), TARGET_COVERAGE),
                 xytext=(-2, -11), textcoords="offset points", fontsize=7.5,
                 color=C_REF, ha="right")
    ax0.annotate("plain fails\nfrom $\\Gamma=%.1f$" % breakdown,
                 xy=(breakdown, 0.845), xytext=(breakdown + 0.16, 0.755),
                 fontsize=7.5, color=C_PLAIN,
                 arrowprops=dict(arrowstyle="->", color=C_PLAIN, lw=1.0,
                                 shrinkA=0, shrinkB=2))
    ax0.set_ylabel("latent coverage")
    ax0.set_ylim(0.63, 1.02)
    ax0.legend(loc="center left", frameon=False, fontsize=7.5)

    ax1.plot(g, width, "^-", color=C_INFLATED, markersize=4)
    ax1.axhline(1.0, color=C_REF, linestyle=":", linewidth=1.2)
    ax1.set_ylabel(r"width $\times$ vs. $\Gamma=1$")
    ax1.set_xlabel(r"assumed misspecification factor $\Gamma$")
    ax1.set_xlim(g.min() - 0.08, g.max() + 0.08)
    for gi, wi in zip(g, width):
        if gi in (1.0, 1.6, 3.0):
            ax1.annotate("%.2f$\\times$" % wi, xy=(gi, wi), xytext=(0, 6),
                         textcoords="offset points", ha="center", fontsize=7.5,
                         color=C_INFLATED)

    _save(fig, "figS_gamma_frontier")
    _dump({
        "kappa": payload["kappa"],
        "alpha": payload["alpha"],
        "cells": payload["cells"],
        "trials": payload["trials"],
        "target_coverage": TARGET_COVERAGE,
        "plain_last_meets_target_at_gamma": last_ok,
        "plain_first_fails_at_gamma": breakdown,
        "rows": rows,
    }, "figS_gamma_frontier")


# ===================================================================
# Fig S4 -- ensemble uncertainty split and what the ensemble buys
# ===================================================================
def figS4() -> None:
    """Aleatoric/epistemic split and per-member CRPS, both cities.

    This supersedes outputs/figures/fig6_uncertainty_decomposition.pdf for the
    supplementary. That figure is a two-wedge donut for New York alone and carries
    a "CIVIC-SAFE" watermark drawn by scripts/generate_figures.py, neither of which
    belongs in a journal submission. The numbers here are read from the same
    persisted field, for both cities.
    """
    cities = [("Chicago", "chicago"), ("New York", "nyc")]
    data: dict[str, dict] = {}
    for label, key in cities:
        src = (PROJECT_ROOT / "outputs" / "conformal_evaluation"
               / f"{key}_conformal_results.json")
        if not src.exists():
            raise SystemExit(f"{src} missing. Run scripts/run_conformal_evaluation.py.")
        ens = json.loads(src.read_text(encoding="utf-8"))["ensemble"]
        for field in ("aleatoric_uncertainty", "epistemic_uncertainty",
                      "epistemic_fraction", "per_seed_test_crps",
                      "learned_weight_test_crps"):
            if field not in ens:
                raise SystemExit(f"{src} has no ensemble.{field}")
        data[label] = {
            "aleatoric": float(ens["aleatoric_uncertainty"]),
            "epistemic": float(ens["epistemic_uncertainty"]),
            "epistemic_fraction": float(ens["epistemic_fraction"]),
            "per_seed_test_crps": [float(v) for v in ens["per_seed_test_crps"]],
            "ensemble_test_crps": float(ens["learned_weight_test_crps"]),
            "num_seeds": int(ens["num_seeds"]),
        }
        d = data[label]
        print("  %-9s aleatoric %7.3f epistemic %6.3f epistemic frac %6.2f%%  "
              "seeds %d, ensemble CRPS %.4f (best member %.4f)"
              % (label, d["aleatoric"], d["epistemic"],
                 100 * d["epistemic_fraction"], d["num_seeds"],
                 d["ensemble_test_crps"], min(d["per_seed_test_crps"])))

    _style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(7.0, 2.6))

    # --- (a) variance decomposition, stacked, absolute units -----------------
    labels = [lab for lab, _ in cities]
    y = np.arange(len(labels))
    alea = np.array([data[k]["aleatoric"] for k in labels])
    epis = np.array([data[k]["epistemic"] for k in labels])
    ax0.barh(y, alea, 0.5, color=C_INFLATED, linewidth=0, label="aleatoric")
    ax0.barh(y, epis, 0.5, left=alea, color=C_PLAIN, edgecolor="white",
             linewidth=0.8, label="epistemic")
    for i, k in enumerate(labels):
        ax0.annotate("epis. %.1f%%" % (100 * data[k]["epistemic_fraction"]),
                     xy=(alea[i] + epis[i], i), xytext=(5, 0),
                     textcoords="offset points", va="center", fontsize=7.5,
                     color=C_PLAIN)
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels)
    ax0.set_xlim(0, (alea + epis).max() * 1.42)
    ax0.set_xlabel("predictive variance (counts$^2$)")
    ax0.set_title("(a) variance decomposition", fontweight="bold", loc="left")
    ax0.set_ylim(-0.62, 1.55)
    ax0.legend(loc="lower right", frameon=False, fontsize=7.5, ncol=2)
    ax0.grid(axis="y", visible=False)

    # --- (b) the combination against its own members --------------------------
    # Note what this does and does not show. The EMOS combination beats the MEAN
    # member in both cities (2.8730 -> 2.8267 Chicago, 3.2033 -> 3.1401 New York),
    # but in Chicago one member reaches 2.8182 and so beats the combination. That
    # member is only identifiable in hindsight, which is why the ensemble is still
    # the right choice, but the figure must not claim a clean sweep.
    for i, k in enumerate(labels):
        seeds = np.array(data[k]["per_seed_test_crps"])
        ax1.plot(np.full_like(seeds, i, dtype=float), seeds, "o",
                 color=C_CAT["violent"], markersize=5, alpha=0.75,
                 label="ensemble member" if i == 0 else None)
        ax1.plot([i - 0.22, i + 0.22],
                 [data[k]["ensemble_test_crps"]] * 2, "-",
                 color=C_PLAIN, linewidth=2.0,
                 label="5-seed EMOS" if i == 0 else None)
        ax1.annotate("%.4f" % data[k]["ensemble_test_crps"],
                     xy=(i + 0.24, data[k]["ensemble_test_crps"]),
                     fontsize=7.5, color=C_PLAIN, va="center")
    ax1.set_xticks(y)
    ax1.set_xticklabels(labels)
    ax1.set_xlim(-0.5, len(labels) - 0.28)
    ax1.set_ylabel("test CRPS (lower is better)")
    ax1.set_title("(b) ensemble vs. its members", fontweight="bold", loc="left")
    ax1.legend(loc="upper left", frameon=False, fontsize=7.5)
    fig.tight_layout()

    _save(fig, "figS_ensemble_uncertainty")
    _dump(data, "figS_ensemble_uncertainty")


def main() -> None:
    figS1()
    figS2()
    figS3()
    figS4()
    print("\ndone.")


if __name__ == "__main__":
    main()
