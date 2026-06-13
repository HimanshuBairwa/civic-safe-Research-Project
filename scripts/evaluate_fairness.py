#!/usr/bin/env python3
"""Production-grade demographic fairness audit for CIVIC-SAFE.

Evaluates whether crime counts, model prediction errors, or both are
equitably distributed across neighbourhoods with different demographic
compositions.

Modes
-----
**Data-only mode** (``--data``, no ``--checkpoint``):
    Stratifies spatial units by demographic quintiles, computes crime
    distribution statistics per stratum, Kruskal-Wallis tests, Gini
    coefficients, and disparity ratios.  Useful for auditing the *data
    pipeline itself* before a model is trained.

**Model audit mode** (``--checkpoint``):
    Loads a trained ``CivicSafeModel``, generates predictions on the
    test set, and runs the full 7-component equity audit via
    ``AuditHarness``.

Output
------
- Pretty-printed report to stdout.
- JSON results saved to ``outputs/fairness/{city}_audit.json``.

Usage
-----
    # Data-only audit
    python scripts/evaluate_fairness.py --data chicago

    # Model audit (future)
    python scripts/evaluate_fairness.py --data chicago --checkpoint outputs/best.ckpt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats as sp_stats
from torch import Tensor

# -- CIVIC-SAFE audit components -----------------------------------------------
from civicsafe.audit.components import (
    AuditResult,
    PointAccuracyEquityAudit,
    CoverageEquityAudit,
    IntervalWidthEquityAudit,
    CalibrationEquityAudit,
    WinklerEquityAudit,
    AbstentionEquityAudit,
    ReportingBiasSensitivityAudit,
    default_components,
)
from civicsafe.audit.stratification import StratificationEngine, StratConfig

# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fairness_audit")

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "fairness"

DEMOGRAPHIC_COLS = [
    "total_population",
    "median_household_income",
    "poverty_rate",
    "unemployment_rate",
    "pct_black",
    "pct_hispanic",
    "pct_renter_occupied",
]

CRIME_CATEGORIES = ["violent", "property", "drug"]

# Test period: weeks 260-312 (year 2023) — matches paper's chronological split
TEST_START_WEEK = 260
TEST_END_WEEK = 312
HIST_AVG_LOOKBACK = 52  # weeks for historical-average baseline


# =========================================================================== #
#  Utility helpers
# =========================================================================== #


def _gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient of a 1-D array (0 = perfect equality)."""
    v = np.sort(np.asarray(values, dtype=np.float64).ravel())
    v = v[v >= 0]
    n = len(v)
    if n == 0 or v.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2.0 * (index * v).sum() / (n * v.sum())) - (n + 1) / n)


def _disparity_ratio(group_values: dict[int, float]) -> float:
    """max / min of per-group values; returns 1.0 if degenerate."""
    vals = [v for v in group_values.values() if np.isfinite(v) and v > 0]
    if len(vals) < 2:
        return 1.0
    return float(max(vals) / min(vals))


def _cv_across_groups(group_values: dict[int, float]) -> float:
    """Coefficient of variation across groups (lower = more equitable)."""
    arr = np.array([v for v in group_values.values() if np.isfinite(v)])
    if len(arr) < 2 or arr.mean() == 0:
        return 0.0
    return float(arr.std() / arr.mean())


def _format_header(text: str, width: int = 72) -> str:
    """Format a section header for the printed report."""
    return f"\n{'=' * width}\n  {text}\n{'=' * width}"


def _format_subheader(text: str, width: int = 72) -> str:
    return f"\n{'-' * width}\n  {text}\n{'-' * width}"


# =========================================================================== #
#  Data loading
# =========================================================================== #


def load_demographics(city: str) -> pd.DataFrame:
    """Load demographics CSV for *city* and validate columns."""
    path = DATA_DIR / f"{city}_demographics.csv"
    if not path.exists():
        logger.error(
            f"Demographics not found at {path}. "
            "Run `python scripts/build_demographics.py` first."
        )
        sys.exit(1)

    df = pd.read_csv(path)
    df["spatial_unit"] = df["spatial_unit"].astype(str)

    missing = [c for c in DEMOGRAPHIC_COLS if c not in df.columns]
    if missing:
        logger.error(f"Demographics file missing columns: {missing}")
        sys.exit(1)

    logger.info(
        f"Loaded demographics for {city}: {len(df)} spatial units, "
        f"{len(DEMOGRAPHIC_COLS)} demographic features."
    )
    return df


def load_panel(city: str) -> dict[str, Any]:
    """Load the panel .pt file and return its dict.

    Expected keys: ``counts`` (S, T, C), ``features`` (S, T, F),
    ``metadata`` with ``spatial_units`` and ``categories``.
    """
    path = DATA_DIR / f"{city}_panel.pt"
    if not path.exists():
        logger.error(
            f"Panel file not found at {path}. Run `python scripts/fetch_data.py` first."
        )
        sys.exit(1)

    panel = torch.load(path, weights_only=False, map_location="cpu")
    counts = panel["counts"]  # (S, T, C)
    S, T, C = counts.shape
    logger.info(
        f"Loaded panel for {city}: {S} spatial units × {T} weeks × {C} categories."
    )
    return panel


# =========================================================================== #
#  Stratification
# =========================================================================== #


def build_strata(
    demographics: pd.DataFrame,
    n_bins: int = 5,
) -> dict[str, dict[str, Tensor]]:
    """Build quintile-based strata for every demographic dimension.

    Returns
    -------
    dict mapping demographic-column name → dict with keys
    ``'labels'`` (LongTensor of bin assignments) and
    ``'values'`` (FloatTensor of raw demographic values).
    """
    strata: dict[str, dict[str, Tensor]] = {}
    for col in DEMOGRAPHIC_COLS:
        raw = torch.tensor(demographics[col].values, dtype=torch.float32)
        bins = StratificationEngine.quantile_bins(raw, n_bins=n_bins)
        strata[col] = {"labels": bins, "values": raw}
    return strata


# =========================================================================== #
#  Data-only audit
# =========================================================================== #


def _compute_weekly_crime_rates(
    counts: Tensor,
    demographics: pd.DataFrame,
) -> Tensor:
    """Per-spatial-unit mean weekly total crime rate (per 10 000 pop)."""
    total_counts = counts.sum(dim=-1).float().mean(dim=1)  # (S,)
    pop = torch.tensor(
        demographics["total_population"].values, dtype=torch.float32
    )
    pop = pop.clamp(min=1.0)
    return total_counts / pop * 10_000


def run_data_only_audit(
    city: str,
    panel: dict[str, Any],
    demographics: pd.DataFrame,
    n_bins: int = 5,
) -> dict[str, Any]:
    """Data-distribution fairness audit (no model required).

    For each demographic dimension:
      1. Stratify spatial units into quintiles.
      2. Compute per-quintile mean crime counts and rates.
      3. Kruskal-Wallis H-test across quintiles.
      4. Gini coefficient of crime rates.
      5. Disparity ratio (max/min quintile).
    """
    counts = panel["counts"]  # (S, T, C)
    metadata = panel.get("metadata", {})
    spatial_units = metadata.get("spatial_units", list(range(counts.shape[0])))
    categories = metadata.get("categories", CRIME_CATEGORIES)

    S, T, C = counts.shape
    strata = build_strata(demographics, n_bins=n_bins)

    # Mean weekly crime rate per spatial unit (per 10k population)
    crime_rates = _compute_weekly_crime_rates(counts, demographics)  # (S,)
    # Mean weekly total counts per spatial unit (absolute)
    mean_weekly_total = counts.sum(dim=-1).float().mean(dim=1)  # (S,)

    report_sections: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    #  Per-dimension analysis
    # ------------------------------------------------------------------ #
    for dim_name, dim_data in strata.items():
        labels = dim_data["labels"].numpy()
        rates_np = crime_rates.numpy()
        counts_np = mean_weekly_total.numpy()

        # -- Per-quintile stats --
        quintile_stats: dict[int, dict[str, float]] = {}
        groups_for_kw: list[np.ndarray] = []
        for q in range(n_bins):
            mask = labels == q
            if mask.sum() == 0:
                continue
            q_rates = rates_np[mask]
            q_counts = counts_np[mask]
            groups_for_kw.append(q_rates)
            quintile_stats[q] = {
                "n_units": int(mask.sum()),
                "mean_weekly_count": float(np.mean(q_counts)),
                "median_weekly_count": float(np.median(q_counts)),
                "mean_crime_rate_per_10k": float(np.mean(q_rates)),
                "median_crime_rate_per_10k": float(np.median(q_rates)),
                "std_crime_rate": float(np.std(q_rates)),
            }

        # -- Kruskal-Wallis H-test --
        if len(groups_for_kw) >= 2:
            h_stat, p_val = sp_stats.kruskal(*groups_for_kw)
        else:
            h_stat, p_val = 0.0, 1.0

        # -- Disparity ratio --
        mean_rates = {q: s["mean_crime_rate_per_10k"] for q, s in quintile_stats.items()}
        disp_ratio = _disparity_ratio(mean_rates)

        # -- CV --
        cv = _cv_across_groups(mean_rates)

        # -- Gini of the rates across all spatial units --
        gini = _gini_coefficient(rates_np)

        report_sections[dim_name] = {
            "quintile_stats": {str(k): v for k, v in quintile_stats.items()},
            "kruskal_wallis_H": round(float(h_stat), 4),
            "kruskal_wallis_p": round(float(p_val), 6),
            "significant_at_005": bool(p_val < 0.05),
            "disparity_ratio": round(disp_ratio, 4),
            "coefficient_of_variation": round(cv, 4),
            "gini_coefficient": round(gini, 4),
        }

    # ------------------------------------------------------------------ #
    #  Per-category breakdown (violent / property / drug)
    # ------------------------------------------------------------------ #
    per_category: dict[str, dict[str, float]] = {}
    for c_idx, cat_name in enumerate(categories):
        cat_counts = counts[:, :, c_idx].float()  # (S, T)
        mean_per_unit = cat_counts.mean(dim=1)  # (S,)
        per_category[cat_name] = {
            "global_mean_weekly": float(mean_per_unit.mean()),
            "global_std_weekly": float(mean_per_unit.std()),
            "gini_across_units": round(
                _gini_coefficient(mean_per_unit.numpy()), 4
            ),
        }

    # ------------------------------------------------------------------ #
    #  Worst-5 analysis (highest crime rates)
    # ------------------------------------------------------------------ #
    top5_idx = torch.argsort(crime_rates, descending=True)[:5]
    worst5: list[dict[str, Any]] = []
    for idx in top5_idx:
        i = int(idx)
        entry: dict[str, Any] = {
            "spatial_unit": str(spatial_units[i]),
            "crime_rate_per_10k": round(float(crime_rates[i]), 2),
            "mean_weekly_count": round(float(mean_weekly_total[i]), 2),
        }
        for col in DEMOGRAPHIC_COLS:
            entry[col] = round(float(demographics.iloc[i][col]), 2)
        worst5.append(entry)

    # ------------------------------------------------------------------ #
    #  Correlation analysis
    # ------------------------------------------------------------------ #
    correlations: dict[str, dict[str, float]] = {}
    for col in DEMOGRAPHIC_COLS:
        demo_vals = demographics[col].values.astype(np.float64)
        rates_64 = crime_rates.numpy().astype(np.float64)
        mask = np.isfinite(demo_vals) & np.isfinite(rates_64)
        if mask.sum() < 3:
            correlations[col] = {"pearson_r": 0.0, "p_value": 1.0}
            continue
        r, p = sp_stats.pearsonr(demo_vals[mask], rates_64[mask])
        correlations[col] = {
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 6),
        }

    # ------------------------------------------------------------------ #
    #  Assemble full report
    # ------------------------------------------------------------------ #
    full_report = {
        "audit_type": "data_only",
        "city": city,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "panel_shape": {
            "spatial_units": S,
            "weeks": T,
            "categories": C,
        },
        "demographic_dimensions": report_sections,
        "per_category": per_category,
        "worst_5_spatial_units": worst5,
        "correlation_analysis": correlations,
    }
    return full_report


# =========================================================================== #
#  Model audit (with checkpoint)
# =========================================================================== #


def run_model_audit(
    city: str,
    panel: dict[str, Any],
    demographics: pd.DataFrame,
    checkpoint_path: str,
    n_bins: int = 5,
) -> dict[str, Any]:
    """Full 7-component equity audit using a trained model checkpoint.

    Loads the model, generates predictions on the test period, builds
    an ``AuditBundle``, and runs each stratification dimension through
    the ``AuditHarness``.
    """
    from civicsafe.audit.bundle import AuditBundle
    from civicsafe.audit.harness import AuditHarness
    from civicsafe.audit.report import AuditReport

    counts = panel["counts"]  # (S, T, C)
    features = panel["features"]  # (S, T, F)
    metadata = panel.get("metadata", {})
    S, T, C = counts.shape
    categories = metadata.get("categories", CRIME_CATEGORIES)

    # --- Load trained model ---
    try:
        from civicsafe.model import CivicSafeModel  # type: ignore

        model = CivicSafeModel.load_from_checkpoint(checkpoint_path, map_location="cpu")
        model.eval()
        logger.info(f"Loaded model from checkpoint: {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to load model checkpoint: {e}")
        logger.info("Falling back to historical-average baseline predictions.")
        return _run_baseline_audit(city, panel, demographics, n_bins)

    # --- Generate predictions on test period ---
    test_counts = counts[:, TEST_START_WEEK:TEST_END_WEEK, :]  # (S, T_test, C)
    test_features = features[:, TEST_START_WEEK:TEST_END_WEEK, :]

    with torch.no_grad():
        output = model(test_features, test_counts)

    # Extract ZINB parameters from model output
    pi = output.get("pi", torch.zeros(S * (TEST_END_WEEK - TEST_START_WEEK)))
    mu = output.get("mu", torch.ones(S * (TEST_END_WEEK - TEST_START_WEEK)))
    r = output.get("r", torch.ones(S * (TEST_END_WEEK - TEST_START_WEEK)) * 5.0)
    y_pred = mu * (1 - pi)

    # Flatten test ground truth to (N,)
    y_true = test_counts.reshape(-1).float()
    lower = output.get("lower", torch.clamp(y_pred - 5.0, min=0.0))
    upper = output.get("upper", y_pred + 5.0)

    # --- Build strata (tile to match flattened predictions) ---
    strata = build_strata(demographics, n_bins=n_bins)
    T_test = TEST_END_WEEK - TEST_START_WEEK
    tiled_strata: dict[str, Tensor] = {}
    for dim_name, dim_data in strata.items():
        # labels is (S,), tile to (S * T_test * C) or (S * T_test)
        expanded = dim_data["labels"].unsqueeze(1).expand(S, T_test * C).reshape(-1)
        tiled_strata[dim_name] = expanded

    spatial_ids = torch.arange(S).unsqueeze(1).expand(S, T_test * C).reshape(-1)

    # --- Run harness for each demographic dimension ---
    all_results: dict[str, Any] = {}
    alpha = 0.1

    for dim_name in DEMOGRAPHIC_COLS:
        bundle = AuditBundle(
            y_true=y_true,
            y_pred=y_pred.reshape(-1),
            lower=lower.reshape(-1),
            upper=upper.reshape(-1),
            pi=pi.reshape(-1),
            mu=mu.reshape(-1),
            r=r.reshape(-1),
            strata={dim_name: tiled_strata[dim_name]},
            spatial_units=spatial_ids,
            alpha=alpha,
            metadata={"city": city, "strata_dimension": dim_name},
        )
        harness = AuditHarness()
        report = harness.run_full_audit(bundle, strata_key=dim_name)
        all_results[dim_name] = report.to_dict()

    return {
        "audit_type": "model",
        "city": city,
        "checkpoint": checkpoint_path,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "panel_shape": {"spatial_units": S, "weeks": T, "categories": C},
        "alpha": alpha,
        "results_by_dimension": all_results,
    }


def _run_baseline_audit(
    city: str,
    panel: dict[str, Any],
    demographics: pd.DataFrame,
    n_bins: int = 5,
) -> dict[str, Any]:
    """Equity audit using the historical-average baseline.

    Predicts test-period counts as the mean of the preceding 52 weeks.
    Then runs per-dimension disparity analysis on the prediction errors.
    """
    counts = panel["counts"]  # (S, T, C)
    metadata = panel.get("metadata", {})
    S, T, C = counts.shape

    # Clamp test window to available data
    t_start = min(TEST_START_WEEK, T)
    t_end = min(TEST_END_WEEK, T)
    if t_start >= t_end:
        t_end = T
        t_start = max(0, T - 52)

    hist_start = max(0, t_start - HIST_AVG_LOOKBACK)
    hist_counts = counts[:, hist_start:t_start, :].float()  # (S, 52, C)
    hist_avg = hist_counts.mean(dim=1, keepdim=True)  # (S, 1, C)

    test_counts = counts[:, t_start:t_end, :].float()  # (S, T_test, C)
    T_test = test_counts.shape[1]
    predictions = hist_avg.expand(S, T_test, C)  # (S, T_test, C)

    # Per-spatial-unit MAE across test period and categories
    errors = (test_counts - predictions).abs()  # (S, T_test, C)
    mae_per_unit = errors.mean(dim=(1, 2))  # (S,)

    # Strata
    strata = build_strata(demographics, n_bins=n_bins)

    results_by_dim: dict[str, Any] = {}

    for dim_name, dim_data in strata.items():
        labels = dim_data["labels"].numpy()
        mae_np = mae_per_unit.numpy()

        # Per-quintile MAE
        quintile_mae: dict[int, float] = {}
        groups_for_kw: list[np.ndarray] = []
        for q in range(n_bins):
            mask = labels == q
            if mask.sum() == 0:
                continue
            q_mae = mae_np[mask]
            quintile_mae[q] = float(np.mean(q_mae))
            groups_for_kw.append(q_mae)

        # Kruskal-Wallis
        if len(groups_for_kw) >= 2:
            h_stat, p_val = sp_stats.kruskal(*groups_for_kw)
        else:
            h_stat, p_val = 0.0, 1.0

        # Disparity & CV
        disp_ratio = _disparity_ratio(quintile_mae)
        cv = _cv_across_groups(quintile_mae)

        results_by_dim[dim_name] = {
            "per_quintile_mae": {str(k): round(v, 4) for k, v in quintile_mae.items()},
            "disparity_ratio": round(disp_ratio, 4),
            "coefficient_of_variation": round(cv, 4),
            "kruskal_wallis_H": round(float(h_stat), 4),
            "kruskal_wallis_p": round(float(p_val), 6),
            "significant_at_005": bool(p_val < 0.05),
        }

    # Correlation between demographics and MAE
    correlations: dict[str, dict[str, float]] = {}
    for col in DEMOGRAPHIC_COLS:
        demo_vals = demographics[col].values.astype(np.float64)
        mae_64 = mae_per_unit.numpy().astype(np.float64)
        mask = np.isfinite(demo_vals) & np.isfinite(mae_64)
        if mask.sum() < 3:
            correlations[col] = {"pearson_r": 0.0, "p_value": 1.0}
            continue
        r, p = sp_stats.pearsonr(demo_vals[mask], mae_64[mask])
        correlations[col] = {
            "pearson_r": round(float(r), 4),
            "p_value": round(float(p), 6),
        }

    # Worst-5 by MAE
    spatial_units = metadata.get("spatial_units", list(range(S)))
    top5_idx = torch.argsort(mae_per_unit, descending=True)[:5]
    worst5: list[dict[str, Any]] = []
    for idx in top5_idx:
        i = int(idx)
        entry: dict[str, Any] = {
            "spatial_unit": str(spatial_units[i]),
            "mae": round(float(mae_per_unit[i]), 4),
        }
        for col in DEMOGRAPHIC_COLS:
            entry[col] = round(float(demographics.iloc[i][col]), 2)
        worst5.append(entry)

    return {
        "audit_type": "baseline_model",
        "city": city,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "panel_shape": {"spatial_units": S, "weeks": T, "categories": C},
        "test_period": {"start_week": t_start, "end_week": t_end},
        "baseline": "historical_average_52w",
        "overall_mae": round(float(mae_per_unit.mean()), 4),
        "results_by_dimension": results_by_dim,
        "correlation_analysis": correlations,
        "worst_5_spatial_units": worst5,
    }


# =========================================================================== #
#  Pretty-printing
# =========================================================================== #


def print_data_only_report(report: dict[str, Any]) -> None:
    """Print a human-readable data-only audit report."""
    city = report["city"]
    shape = report["panel_shape"]

    print(_format_header(f"CIVIC-SAFE DATA FAIRNESS AUDIT — {city.upper()}"))
    print(f"  Timestamp : {report['timestamp']}")
    print(
        f"  Panel     : {shape['spatial_units']} spatial units × "
        f"{shape['weeks']} weeks × {shape['categories']} categories"
    )

    # -- Per-dimension --
    for dim, data in report["demographic_dimensions"].items():
        print(_format_subheader(f"Dimension: {dim}"))

        kw_sig = "***" if data["significant_at_005"] else ""
        print(
            f"  Kruskal-Wallis H = {data['kruskal_wallis_H']:.3f}  "
            f"(p = {data['kruskal_wallis_p']:.4f}) {kw_sig}"
        )
        print(f"  Disparity ratio  = {data['disparity_ratio']:.3f}")
        print(f"  Coeff. of var.   = {data['coefficient_of_variation']:.3f}")
        print(f"  Gini coefficient = {data['gini_coefficient']:.3f}")

        print(f"\n  {'Quintile':>8}  {'N':>4}  {'Mean Rate/10k':>14}  {'Median':>8}  {'Std':>8}")
        for q, qs in sorted(data["quintile_stats"].items(), key=lambda x: int(x[0])):
            print(
                f"  {'Q' + str(int(q) + 1):>8}  {qs['n_units']:>4}  "
                f"{qs['mean_crime_rate_per_10k']:>14.2f}  "
                f"{qs['median_crime_rate_per_10k']:>8.2f}  "
                f"{qs['std_crime_rate']:>8.2f}"
            )

    # -- Per-category --
    print(_format_subheader("Per-Category Crime Distribution"))
    print(f"  {'Category':>12}  {'Mean/wk':>10}  {'Std/wk':>10}  {'Gini':>6}")
    for cat, cs in report["per_category"].items():
        print(
            f"  {cat:>12}  {cs['global_mean_weekly']:>10.2f}  "
            f"{cs['global_std_weekly']:>10.2f}  {cs['gini_across_units']:>6.3f}"
        )

    # -- Correlations --
    print(_format_subheader("Correlation: Demographics ↔ Crime Rate"))
    print(f"  {'Dimension':>30}  {'Pearson r':>10}  {'p-value':>10}")
    for dim, corr in report["correlation_analysis"].items():
        sig = "***" if corr["p_value"] < 0.05 else ""
        print(f"  {dim:>30}  {corr['pearson_r']:>10.4f}  {corr['p_value']:>10.4f} {sig}")

    # -- Worst 5 --
    print(_format_subheader("Top-5 Highest Crime Rate Spatial Units"))
    for i, w in enumerate(report["worst_5_spatial_units"], 1):
        print(f"\n  #{i}  Unit {w['spatial_unit']}  —  {w['crime_rate_per_10k']:.1f} per 10k/week")
        for col in DEMOGRAPHIC_COLS:
            print(f"       {col}: {w[col]}")

    print(f"\n{'=' * 72}\n  AUDIT COMPLETE\n{'=' * 72}\n")


def print_baseline_report(report: dict[str, Any]) -> None:
    """Print a human-readable baseline-model audit report."""
    city = report["city"]
    shape = report["panel_shape"]

    print(_format_header(f"CIVIC-SAFE BASELINE FAIRNESS AUDIT — {city.upper()}"))
    print(f"  Timestamp : {report['timestamp']}")
    print(f"  Baseline  : {report['baseline']}")
    print(f"  Overall MAE : {report['overall_mae']:.4f}")
    test = report["test_period"]
    print(f"  Test period : weeks {test['start_week']}–{test['end_week']}")

    # -- Per-dimension MAE disparity --
    for dim, data in report["results_by_dimension"].items():
        print(_format_subheader(f"MAE Disparity — {dim}"))
        kw_sig = "***" if data["significant_at_005"] else ""
        print(
            f"  Kruskal-Wallis H = {data['kruskal_wallis_H']:.3f}  "
            f"(p = {data['kruskal_wallis_p']:.4f}) {kw_sig}"
        )
        print(f"  Disparity ratio  = {data['disparity_ratio']:.3f}")
        print(f"  Coeff. of var.   = {data['coefficient_of_variation']:.3f}")

        print(f"\n  {'Quintile':>8}  {'MAE':>10}")
        for q, mae in sorted(
            data["per_quintile_mae"].items(), key=lambda x: int(x[0])
        ):
            print(f"  {'Q' + str(int(q) + 1):>8}  {mae:>10.4f}")

    # -- Correlations --
    print(_format_subheader("Correlation: Demographics ↔ MAE"))
    print(f"  {'Dimension':>30}  {'Pearson r':>10}  {'p-value':>10}")
    for dim, corr in report["correlation_analysis"].items():
        sig = "***" if corr["p_value"] < 0.05 else ""
        print(f"  {dim:>30}  {corr['pearson_r']:>10.4f}  {corr['p_value']:>10.4f} {sig}")

    # -- Worst 5 --
    print(_format_subheader("Top-5 Highest MAE Spatial Units"))
    for i, w in enumerate(report["worst_5_spatial_units"], 1):
        print(f"\n  #{i}  Unit {w['spatial_unit']}  —  MAE = {w['mae']:.4f}")
        for col in DEMOGRAPHIC_COLS:
            print(f"       {col}: {w[col]}")

    print(f"\n{'=' * 72}\n  AUDIT COMPLETE\n{'=' * 72}\n")


# =========================================================================== #
#  Main entry point
# =========================================================================== #


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CIVIC-SAFE Demographic Fairness Audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/evaluate_fairness.py --data chicago\n"
            "  python scripts/evaluate_fairness.py --data nyc --checkpoint outputs/best.ckpt\n"
        ),
    )
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=["chicago", "nyc"],
        help="City dataset to audit (chicago or nyc).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trained model checkpoint (.ckpt). "
        "If omitted, runs in data-only mode.",
    )
    parser.add_argument(
        "--n-bins",
        type=int,
        default=5,
        help="Number of quantile bins for stratification (default: 5 = quintiles).",
    )
    args = parser.parse_args()

    city = args.data

    # ---- Load data ----
    demographics = load_demographics(city)
    panel = load_panel(city)

    # ---- Run audit ----
    if args.checkpoint is not None:
        logger.info("Running MODEL audit with checkpoint: %s", args.checkpoint)
        report = run_model_audit(
            city, panel, demographics, args.checkpoint, n_bins=args.n_bins
        )
        # Model audit produces JSON directly; printing is lighter
        print(_format_header(f"MODEL AUDIT — {city.upper()}"))
        print(json.dumps(report, indent=2, default=str))
    else:
        logger.info("Running DATA-ONLY audit (no checkpoint provided).")
        report = run_data_only_audit(city, panel, demographics, n_bins=args.n_bins)
        print_data_only_report(report)

        # Also run baseline MAE audit if panel has enough time steps
        S, T, C = panel["counts"].shape
        if T > HIST_AVG_LOOKBACK + 10:
            logger.info("Running BASELINE (historical average) audit...")
            baseline_report = run_baseline_audit_standalone(
                city, panel, demographics, n_bins=args.n_bins
            )
            print_baseline_report(baseline_report)
            # Merge into output
            report["baseline_audit"] = baseline_report

    # ---- Save JSON ----
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{city}_audit.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")


def run_baseline_audit_standalone(
    city: str,
    panel: dict[str, Any],
    demographics: pd.DataFrame,
    n_bins: int = 5,
) -> dict[str, Any]:
    """Thin wrapper for baseline audit — callable from main()."""
    return _run_baseline_audit(city, panel, demographics, n_bins)


if __name__ == "__main__":
    main()
