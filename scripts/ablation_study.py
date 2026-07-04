#!/usr/bin/env python
"""CIVIC-SAFE Ablation Study — LaTeX Table Generator.

Reads trained model checkpoints, evaluation results, baseline CSVs,
and conformal calibration JSON to produce publication-ready LaTeX
booktabs tables for the CIVIC-SAFE paper.

Tables generated:
  Table 1: Main results — CIVIC-SAFE vs baselines (per city)
  Table 2: Component ablation — removing GATv2, EMOS, recalibration, r-reg
  Table 3: Conformal method comparison — coverage, width, disparity
  Table 4: Fairness audit — ECRC coverage by demographic group

Additionally computes ensemble size ablation (K = 1, 3, 5 seeds).

Usage:
    python scripts/ablation_study.py --data chicago
    python scripts/ablation_study.py --data chicago --data nyc
    python scripts/ablation_study.py --results-dir outputs/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "outputs"
TABLE_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "tables"

# Metrics columns used across tables
MAIN_METRICS = ["CRPS", "MAE", "RMSE", "Brier"]
CONFORMAL_METRICS = ["Coverage", "Width", "Disparity"]

# Number formatting: 4 decimal places for CRPS/MAE/RMSE, 2 for percentages
FMT_4 = ".4f"
FMT_2 = ".2f"
FMT_PCT = ".2f"  # coverage percentages rendered as e.g. 90.03


# ───────────────────────────────────────────────────────────────────
# Utility helpers
# ───────────────────────────────────────────────────────────────────
def _load_json(path: Path) -> dict[str, Any] | None:
    """Load a JSON file, returning None if it does not exist."""
    if not path.exists():
        logger.warning(f"File not found, skipping: {path}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt(value: float | None, fmt: str = FMT_4, missing: str = "--") -> str:
    """Format a numeric value, returning *missing* sentinel for None / NaN."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return missing
    return f"{value:{fmt}}"


def _bold(text: str) -> str:
    """Wrap text in LaTeX bold."""
    return rf"\textbf{{{text}}}"


def _latex_escape(text: str) -> str:
    """Escape characters for LaTeX."""
    return text.replace("_", r"\_")


def _bold_best_column(
    rows: list[dict[str, str]],
    columns: list[str],
    lower_is_better: dict[str, bool],
) -> list[dict[str, str]]:
    """Bold the best value in each column across rows.

    Modifies rows in-place and returns them for convenience.
    Skips columns where all values are '--' (missing).
    """
    for col in columns:
        # Collect numeric values, ignoring missing
        numeric: list[tuple[int, float]] = []
        for i, row in enumerate(rows):
            val_str = row.get(col, "--")
            if val_str == "--":
                continue
            try:
                numeric.append((i, float(val_str)))
            except ValueError:
                continue

        if not numeric:
            continue

        if lower_is_better.get(col, True):
            best_idx = min(numeric, key=lambda x: x[1])[0]
        else:
            best_idx = max(numeric, key=lambda x: x[1])[0]

        row_val = rows[best_idx][col]
        rows[best_idx][col] = _bold(row_val)

    return rows


def _build_booktabs_table(
    caption: str,
    label: str,
    headers: list[str],
    rows: list[dict[str, str]],
    name_key: str = "name",
) -> str:
    """Assemble a complete LaTeX booktabs table string.

    Args:
        caption: Table caption.
        label: Table label for \\ref{}.
        headers: Column header names (excluding the row-name column).
        rows: List of dicts, each with *name_key* and each header as keys.
        name_key: Key in each row dict that holds the row label text.

    Returns:
        Complete LaTeX table string.
    """
    n_cols = 1 + len(headers)  # name + metrics
    col_spec = "l" + " r" * len(headers)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        rf"  \caption{{{caption}}}",
        rf"  \label{{{label}}}",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
    ]

    # Header row
    header_cells = " & ".join(rf"\textbf{{{h}}}" for h in headers)
    lines.append(rf"    & {header_cells} \\")
    lines.append(r"    \midrule")

    # Data rows
    for row in rows:
        name = row.get(name_key, "")
        cells = " & ".join(row.get(h, "--") for h in headers)
        lines.append(f"    {name} & {cells} \\\\")

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ───────────────────────────────────────────────────────────────────
# Result loaders
# ───────────────────────────────────────────────────────────────────
def load_model_results(results_dir: Path, city: str) -> dict[str, Any] | None:
    """Load CIVIC-SAFE model evaluation results for *city*."""
    # Try the evaluation output from evaluate_trained.py
    path = results_dir / "evaluation" / f"{city}_test_results.json"
    data = _load_json(path)
    if data is not None:
        return data

    # Fallback: eval directory
    path = results_dir / "eval" / "evaluation_results.json"
    return _load_json(path)


def load_baseline_results(results_dir: Path, city: str) -> dict[str, dict[str, float]] | None:
    """Load baseline results CSV as {model_name: {metric: value}}.

    baselines.py writes outputs/baselines/{city}_baselines.csv with columns
    (crps, mae, rmse) indexed by model name.
    """
    import csv

    csv_path = results_dir / "baselines" / f"{city}_baselines.csv"
    if not csv_path.exists():
        logger.warning(f"Baseline CSV not found: {csv_path}")
        return None

    results: dict[str, dict[str, float]] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("Model") or row.get("")  # pandas index col
            if name is None:
                continue
            results[name] = {}
            for key in ("crps", "mae", "rmse"):
                try:
                    results[name][key] = float(row[key])
                except (KeyError, ValueError, TypeError):
                    results[name][key] = float("nan")
    return results if results else None


def load_conformal_results(results_dir: Path, city: str) -> dict[str, Any] | None:
    """Load conformal evaluation results JSON."""
    path = results_dir / "conformal_evaluation" / f"{city}_conformal_results.json"
    return _load_json(path)


def load_fairness_results(results_dir: Path, city: str) -> dict[str, Any] | None:
    """Load fairness audit results JSON."""
    path = results_dir / "fairness" / f"{city}_audit.json"
    return _load_json(path)


# ───────────────────────────────────────────────────────────────────
# Table 1: Main Results (CIVIC-SAFE vs Baselines)
# ───────────────────────────────────────────────────────────────────
def generate_main_results_table(
    chicago_results: dict[str, Any] | None = None,
    nyc_results: dict[str, Any] | None = None,
    chicago_baselines: dict[str, dict[str, float]] | None = None,
    nyc_baselines: dict[str, dict[str, float]] | None = None,
) -> str:
    """Generate Table 1: Main results comparing CIVIC-SAFE vs baselines.

    Returns LaTeX string for a booktabs table.  When both cities are
    provided, a combined table with a \\midrule separator is produced.
    """
    headers = ["CRPS", "MAE", "RMSE", "Brier"]
    lower_is_better = {h: True for h in headers}

    all_rows: list[dict[str, str]] = []

    for city, model_res, base_res in [
        ("Chicago", chicago_results, chicago_baselines),
        ("NYC", nyc_results, nyc_baselines),
    ]:
        if model_res is None and base_res is None:
            continue

        city_rows: list[dict[str, str]] = []

        # Baselines
        if base_res is not None:
            for bname, bmetrics in base_res.items():
                city_rows.append({
                    "name": _latex_escape(bname),
                    "CRPS": _fmt(bmetrics.get("crps")),
                    "MAE": _fmt(bmetrics.get("mae")),
                    "RMSE": _fmt(bmetrics.get("rmse")),
                    "Brier": "--",
                })

        # CIVIC-SAFE model
        if model_res is not None:
            overall = model_res.get("overall", {})
            city_rows.append({
                "name": r"\textsc{Civic-Safe} (Ours)",
                "CRPS": _fmt(overall.get("crps")),
                "MAE": _fmt(overall.get("mae")),
                "RMSE": _fmt(overall.get("rmse")),
                "Brier": _fmt(overall.get("brier_zero")),
            })

        # Bold best per column within this city block
        _bold_best_column(city_rows, headers, lower_is_better)

        # Add city header as a multicolumn separator if we have both cities
        if chicago_results is not None and nyc_results is not None:
            all_rows.append({"name": rf"\multicolumn{{5}}{{l}}{{\textit{{{city}}}}}", "_separator": "true"})

        all_rows.extend(city_rows)

    # Manually build table to support multicolumn separators
    col_spec = "l" + " r" * len(headers)
    header_cells = " & ".join(rf"\textbf{{{h}}}" for h in headers)

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Main results: CIVIC-SAFE vs.\ baselines on test set (2023, rolling one-step-ahead).}",
        r"  \label{tab:main_results}",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        rf"    & {header_cells} \\",
        r"    \midrule",
    ]

    for row in all_rows:
        if "_separator" in row:
            lines.append(r"    \midrule")
            lines.append(f"    {row['name']} \\\\")
            lines.append(r"    \midrule")
        else:
            cells = " & ".join(row.get(h, "--") for h in headers)
            lines.append(f"    {row['name']} & {cells} \\\\")

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ───────────────────────────────────────────────────────────────────
# Table 2: Component Ablation
# ───────────────────────────────────────────────────────────────────
def generate_ablation_table(results: dict[str, Any] | None = None) -> str:
    """Generate Table 2: Component ablation study.

    Expected *results* dict structure::

        {
            "full_model": {"crps": ..., "mae": ..., "rmse": ..., "brier_zero": ...},
            "no_gatv2":   {"crps": ..., "mae": ..., "rmse": ..., "brier_zero": ...},
            "no_emos":    {"crps": ..., "mae": ..., "rmse": ..., "brier_zero": ...},
            "no_recal":   {"crps": ..., "mae": ..., "rmse": ..., "brier_zero": ...},
            "no_r_reg":   {"crps": ..., "mae": ..., "rmse": ..., "brier_zero": ...},
        }

    If *results* is None, a template table with placeholder dashes is
    returned so the paper can compile before all experiments finish.
    """
    headers = ["CRPS", "MAE", "RMSE", "Brier"]
    lower_is_better = {h: True for h in headers}

    ablation_variants = [
        ("full_model", r"\textsc{Civic-Safe} (Full)"),
        ("no_gatv2", r"$-$ Spatial attention (GATv2)"),
        ("no_emos", r"$-$ EMOS weighting"),
        ("no_recal", r"$-$ Recalibration"),
        ("no_r_reg", r"$-$ $r$-floor regularization"),
        ("nb_only", r"$-$ Zero-inflation (NB only)"),
        ("nll_loss", r"$-$ CRPS loss (NLL only)"),
        ("no_sharpness", r"$-$ Sharpness penalty"),
        ("no_grl", r"$-$ GRL (Demographic blindness)"),
    ]

    rows: list[dict[str, str]] = []
    for key, display_name in ablation_variants:
        if results is not None and key in results:
            m = results[key]
            rows.append({
                "name": display_name,
                "CRPS": _fmt(m.get("crps")),
                "MAE": _fmt(m.get("mae")),
                "RMSE": _fmt(m.get("rmse")),
                "Brier": _fmt(m.get("brier_zero")),
            })
        else:
            rows.append({
                "name": display_name,
                "CRPS": "--",
                "MAE": "--",
                "RMSE": "--",
                "Brier": "--",
            })

    _bold_best_column(rows, headers, lower_is_better)

    return _build_booktabs_table(
        caption=(
            r"Ablation study: contribution of each component. "
            r"$-$ denotes removal of the component from the full model."
        ),
        label="tab:ablation",
        headers=headers,
        rows=rows,
    )


# ───────────────────────────────────────────────────────────────────
# Table 3: Conformal Method Comparison
# ───────────────────────────────────────────────────────────────────
def generate_conformal_table(results: dict[str, Any] | None = None) -> str:
    """Generate Table 3: Conformal calibration methods comparison.

    Reads *coverage_results* from the conformal evaluation JSON and
    formats coverage, mean interval width, and coverage disparity.

    Returns LaTeX string for a booktabs table.
    """
    headers = ["Coverage (\\%)", "Width", "Disparity"]
    # Coverage: closer to target is better → we treat higher (≥ target) as better
    # Width: narrower is better → lower is better
    # Disparity: lower is better
    lower_is_better = {"Coverage (\\%)": False, "Width": True, "Disparity": True}

    method_display = {
        "split_cp": "Split CP",
        "weighted_cp": "Weighted CP",
        "mondrian": "Mondrian CP",
        "equalized_coverage": "Equalized CP",
        "ecrc": r"\textsc{ECRC} (Ours)",
        "adaptive_ecrc": "Adaptive ECRC",
    }

    rows: list[dict[str, str]] = []

    if results is not None:
        coverage_data = results.get("coverage_results", {})
        for method_key, display in method_display.items():
            m = coverage_data.get(method_key)
            if m is None:
                rows.append({
                    "name": display,
                    "Coverage (\\%)": "--",
                    "Width": "--",
                    "Disparity": "--",
                })
                continue

            cov = m.get("marginal_coverage")
            width = m.get("mean_width")
            disparity = m.get("coverage_disparity")

            rows.append({
                "name": display,
                "Coverage (\\%)": _fmt(cov * 100 if cov is not None else None, FMT_PCT),
                "Width": _fmt(width, FMT_2),
                "Disparity": _fmt(disparity, FMT_4),
            })
    else:
        for display in method_display.values():
            rows.append({
                "name": display,
                "Coverage (\\%)": "--",
                "Width": "--",
                "Disparity": "--",
            })

    _bold_best_column(rows, headers, lower_is_better)

    return _build_booktabs_table(
        caption=(
            r"Conformal calibration methods: marginal coverage, "
            r"mean interval width, and max group coverage disparity. "
            r"Target coverage is $1-\alpha = 90\%$."
        ),
        label="tab:conformal",
        headers=headers,
        rows=rows,
    )


# ───────────────────────────────────────────────────────────────────
# Table 4: Fairness — ECRC Coverage by Group
# ───────────────────────────────────────────────────────────────────
def generate_fairness_table(results: dict[str, Any] | None = None) -> str:
    """Generate Table 4: Fairness metrics (ECRC coverage by demographic group).

    Expected *results* structure is either:
    - The full conformal JSON with coverage_results.ecrc.per_group
    - A standalone fairness audit JSON with group-level coverage

    Returns LaTeX string for a booktabs table.
    """
    headers = ["Coverage (\\%)", "Width", r"$n$"]
    lower_is_better = {"Coverage (\\%)": False, "Width": True, r"$n$": False}

    rows: list[dict[str, str]] = []

    if results is not None:
        # Try conformal JSON structure first
        ecrc = results.get("coverage_results", {}).get("ecrc", {})
        per_group = ecrc.get("per_group", {})

        if not per_group:
            # Fallback: fairness audit structure
            per_group = results.get("per_group", {})

        if per_group:
            for group_key in sorted(per_group.keys()):
                g = per_group[group_key]
                cov = g.get("coverage")
                width = g.get("mean_width")
                n = g.get("n_samples")

                # More readable group label
                label = group_key.replace("_", " ").replace("group ", "Q")
                if label.startswith("group"):
                    label = label.replace("group", "Q")
                elif label[0].isdigit():
                    label = f"Q{label}"

                rows.append({
                    "name": label,
                    "Coverage (\\%)": _fmt(
                        cov * 100 if cov is not None else None, FMT_PCT
                    ),
                    "Width": _fmt(width, FMT_2),
                    r"$n$": str(n) if n is not None else "--",
                })

            # Add overall row
            marginal = ecrc.get("marginal_coverage")
            mean_w = ecrc.get("mean_width")
            total_n = sum(
                g.get("n_samples", 0) for g in per_group.values()
            )
            rows.append({
                "name": r"\midrule Overall",
                "Coverage (\\%)": _fmt(
                    marginal * 100 if marginal is not None else None, FMT_PCT
                ),
                "Width": _fmt(mean_w, FMT_2),
                r"$n$": str(total_n) if total_n > 0 else "--",
            })

            # Also add per-category if available
            per_cat = ecrc.get("per_category", {})
            if per_cat:
                for cat_name in sorted(per_cat.keys()):
                    c = per_cat[cat_name]
                    cov = c.get("coverage")
                    width = c.get("mean_width")
                    n = c.get("n_samples")
                    rows.append({
                        "name": f"  {cat_name.capitalize()}",
                        "Coverage (\\%)": _fmt(
                            cov * 100 if cov is not None else None, FMT_PCT
                        ),
                        "Width": _fmt(width, FMT_2),
                        r"$n$": str(n) if n is not None else "--",
                    })

    if not rows:
        # Template placeholder
        for q in range(4):
            rows.append({
                "name": f"Q{q}",
                "Coverage (\\%)": "--",
                "Width": "--",
                r"$n$": "--",
            })

    # Do NOT bold best for fairness table — we want to see all values
    return _build_booktabs_table(
        caption=(
            r"ECRC coverage by demographic quartile. "
            r"$Q_0$ = lowest population density, $Q_3$ = highest. "
            r"Target coverage is $1-\alpha = 90\%$."
        ),
        label="tab:fairness",
        headers=headers,
        rows=rows,
    )


# ───────────────────────────────────────────────────────────────────
# Loss Function Ablation
# ───────────────────────────────────────────────────────────────────
def generate_loss_ablation_table(results: dict[str, Any] | None = None) -> str:
    """Generate loss function ablation table: NLL vs CRPS vs SAC.

    Expected *results* dict structure::

        {
            "nll":  {"crps": ..., "mae": ..., "rmse": ..., "brier_zero": ...},
            "crps": {"crps": ..., "mae": ..., "rmse": ..., "brier_zero": ...},
            "sac":  {"crps": ..., "mae": ..., "rmse": ..., "brier_zero": ...},
        }
    """
    headers = ["CRPS", "MAE", "RMSE", "Brier"]
    lower_is_better = {h: True for h in headers}

    loss_variants = [
        ("nll", "NLL (Negative Log-Likelihood)"),
        ("crps", "CRPS (Direct)"),
        ("sac", "SAC (Sharpness-Aware Calibration)"),
    ]

    rows: list[dict[str, str]] = []
    for key, display in loss_variants:
        if results is not None and key in results:
            m = results[key]
            rows.append({
                "name": display,
                "CRPS": _fmt(m.get("crps")),
                "MAE": _fmt(m.get("mae")),
                "RMSE": _fmt(m.get("rmse")),
                "Brier": _fmt(m.get("brier_zero")),
            })
        else:
            rows.append({
                "name": display,
                "CRPS": "--",
                "MAE": "--",
                "RMSE": "--",
                "Brier": "--",
            })

    _bold_best_column(rows, headers, lower_is_better)

    return _build_booktabs_table(
        caption=r"Loss function ablation: comparison of training objectives.",
        label="tab:loss_ablation",
        headers=headers,
        rows=rows,
    )


# ───────────────────────────────────────────────────────────────────
# Ensemble Size Ablation
# ───────────────────────────────────────────────────────────────────
def generate_ensemble_table(results: dict[str, Any] | None = None) -> str:
    """Generate ensemble size (K) ablation table.

    Expected *results* dict structure::

        {
            "K=1": {"crps": ..., "mae": ..., "rmse": ...},
            "K=3": {"crps": ..., "mae": ..., "rmse": ...},
            "K=5": {"crps": ..., "mae": ..., "rmse": ...},
        }

    Also returns plot-ready data for diminishing-returns visualisation.
    """
    headers = ["CRPS", "MAE", "RMSE"]
    lower_is_better = {h: True for h in headers}

    ensemble_sizes = [
        ("K=1", "$K = 1$"),
        ("K=3", "$K = 3$"),
        ("K=5", "$K = 5$"),
    ]

    rows: list[dict[str, str]] = []
    for key, display in ensemble_sizes:
        if results is not None and key in results:
            m = results[key]
            rows.append({
                "name": display,
                "CRPS": _fmt(m.get("crps")),
                "MAE": _fmt(m.get("mae")),
                "RMSE": _fmt(m.get("rmse")),
            })
        else:
            rows.append({
                "name": display,
                "CRPS": "--",
                "MAE": "--",
                "RMSE": "--",
            })

    _bold_best_column(rows, headers, lower_is_better)

    return _build_booktabs_table(
        caption=(
            r"Ensemble size ablation: effect of number of seeds $K$ "
            r"on EMOS-weighted ensemble predictions."
        ),
        label="tab:ensemble",
        headers=headers,
        rows=rows,
    )


def compute_ensemble_plot_data(
    results: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Extract plot-ready data for ensemble size vs. CRPS.

    Returns list of {k, crps, mae, rmse} dicts for plotting.
    """
    plot_data: list[dict[str, Any]] = []
    if results is None:
        return plot_data

    for key in ["K=1", "K=3", "K=5"]:
        if key in results:
            m = results[key]
            k = int(key.split("=")[1])
            plot_data.append({
                "k": k,
                "crps": m.get("crps"),
                "mae": m.get("mae"),
                "rmse": m.get("rmse"),
            })

    return plot_data


# ───────────────────────────────────────────────────────────────────
# Aggregate loader — discovers results across outputs/
# ───────────────────────────────────────────────────────────────────
def discover_ablation_results(results_dir: Path) -> dict[str, Any]:
    """Scan *results_dir* for ablation-variant result files.

    Looks for JSON files matching patterns like:
      - outputs/ablation/no_gatv2_results.json
      - outputs/ablation/full_model_results.json
      - outputs/ablation/loss_nll_results.json
      - outputs/ablation/ensemble_K1_results.json

    Returns structured dict usable by the table generators.
    """
    ablation_dir = results_dir / "ablation"
    out: dict[str, Any] = {
        "component": {},
        "loss": {},
        "ensemble": {},
    }

    if not ablation_dir.exists():
        logger.info(
            f"No ablation directory at {ablation_dir}. "
            f"Tables will be generated with placeholder dashes."
        )
        return out

    # Component ablation files
    for variant in ["full_model", "no_gatv2", "no_emos", "no_recal", "no_r_reg"]:
        path = ablation_dir / f"{variant}_results.json"
        data = _load_json(path)
        if data is not None:
            # Normalise: accept either top-level metrics or nested "overall"
            metrics = data.get("overall", data)
            out["component"][variant] = metrics

    # Loss function ablation
    for loss in ["nll", "crps", "sac"]:
        path = ablation_dir / f"loss_{loss}_results.json"
        data = _load_json(path)
        if data is not None:
            metrics = data.get("overall", data)
            out["loss"][loss] = metrics

    # Ensemble size ablation
    for k in [1, 3, 5]:
        path = ablation_dir / f"ensemble_K{k}_results.json"
        data = _load_json(path)
        if data is not None:
            metrics = data.get("overall", data)
            out["ensemble"][f"K={k}"] = metrics

    return out


# ───────────────────────────────────────────────────────────────────
# Main pipeline
# ───────────────────────────────────────────────────────────────────
def run_ablation_study(args: argparse.Namespace) -> None:
    """Execute the full ablation table generation pipeline."""
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("  CIVIC-SAFE — Ablation Study Table Generator")
    logger.info("=" * 70)

    cities = args.data if args.data else ["chicago"]
    logger.info(f"  Cities: {cities}")
    logger.info(f"  Results dir: {results_dir}")
    logger.info(f"  Output dir: {output_dir}")

    # ── Load results per city ──
    city_model_results: dict[str, Any] = {}
    city_baselines: dict[str, Any] = {}
    city_conformal: dict[str, Any] = {}
    city_fairness: dict[str, Any] = {}

    for city in cities:
        logger.info(f"\n  Loading results for {city}...")
        city_model_results[city] = load_model_results(results_dir, city)
        city_baselines[city] = load_baseline_results(results_dir, city)
        city_conformal[city] = load_conformal_results(results_dir, city)
        city_fairness[city] = load_fairness_results(results_dir, city)

    # Discover ablation-specific results
    logger.info("\n  Discovering ablation-variant results...")
    ablation_data = discover_ablation_results(results_dir)

    # ── Table 1: Main Results ──
    logger.info("\n[1/6] Generating Table 1: Main Results...")
    chicago_res = city_model_results.get("chicago")
    nyc_res = city_model_results.get("nyc")
    chicago_base = city_baselines.get("chicago")
    nyc_base = city_baselines.get("nyc")

    table1 = generate_main_results_table(
        chicago_results=chicago_res,
        nyc_results=nyc_res,
        chicago_baselines=chicago_base,
        nyc_baselines=nyc_base,
    )
    _save_table(output_dir / "table1_main_results.tex", table1, "Table 1: Main Results")

    # ── Table 2: Component Ablation ──
    logger.info("[2/6] Generating Table 2: Component Ablation...")
    comp_results = ablation_data["component"] if ablation_data["component"] else None
    table2 = generate_ablation_table(comp_results)
    _save_table(output_dir / "table2_ablation.tex", table2, "Table 2: Ablation")

    # ── Table 3: Conformal Method Comparison ──
    logger.info("[3/6] Generating Table 3: Conformal Methods...")
    # Use the first city with conformal results
    conformal_res = None
    for city in cities:
        if city_conformal.get(city) is not None:
            conformal_res = city_conformal[city]
            break
    table3 = generate_conformal_table(conformal_res)
    _save_table(output_dir / "table3_conformal.tex", table3, "Table 3: Conformal")

    # ── Table 4: Fairness ──
    logger.info("[4/6] Generating Table 4: Fairness...")
    fairness_res = None
    for city in cities:
        if city_conformal.get(city) is not None:
            fairness_res = city_conformal[city]
            break
        if city_fairness.get(city) is not None:
            fairness_res = city_fairness[city]
            break
    table4 = generate_fairness_table(fairness_res)
    _save_table(output_dir / "table4_fairness.tex", table4, "Table 4: Fairness")

    # ── Table 5: Loss Function Ablation ──
    logger.info("[5/6] Generating Table 5: Loss Function Ablation...")
    loss_results = ablation_data["loss"] if ablation_data["loss"] else None
    table5 = generate_loss_ablation_table(loss_results)
    _save_table(output_dir / "table5_loss_ablation.tex", table5, "Table 5: Loss Ablation")

    # ── Table 6: Ensemble Size Ablation ──
    logger.info("[6/6] Generating Table 6: Ensemble Size...")
    ens_results = ablation_data["ensemble"] if ablation_data["ensemble"] else None
    table6 = generate_ensemble_table(ens_results)
    _save_table(output_dir / "table6_ensemble.tex", table6, "Table 6: Ensemble")

    # Save ensemble plot data as JSON
    plot_data = compute_ensemble_plot_data(ens_results)
    if plot_data:
        plot_path = output_dir / "ensemble_plot_data.json"
        with open(plot_path, "w", encoding="utf-8") as f:
            json.dump(plot_data, f, indent=2)
        logger.info(f"  Ensemble plot data → {plot_path}")

    # ── Summary ──
    logger.info("\n" + "=" * 70)
    logger.info("  ABLATION STUDY COMPLETE")
    logger.info("=" * 70)
    logger.info(f"  Tables written to: {output_dir}")
    n_tables = len(list(output_dir.glob("table*.tex")))
    logger.info(f"  Total tables generated: {n_tables}")
    logger.info("=" * 70)


def _save_table(path: Path, latex: str, name: str) -> None:
    """Write a LaTeX table to disk and log."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(latex)
    logger.info(f"  {name} → {path}")


# ───────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="CIVIC-SAFE ablation study — LaTeX table generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    # Generate tables for Chicago only
    python scripts/ablation_study.py --data chicago

    # Generate tables for both cities
    python scripts/ablation_study.py --data chicago --data nyc

    # Custom results directory
    python scripts/ablation_study.py --data chicago --results-dir outputs/
""",
    )
    parser.add_argument(
        "--data",
        type=str,
        action="append",
        choices=["chicago", "nyc"],
        help="City dataset(s) to include. Can be repeated (default: chicago).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help=f"Root directory for result files (default: {DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(TABLE_OUTPUT_DIR),
        help=f"Directory for output .tex files (default: {TABLE_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    # Default to chicago if no --data flags
    if args.data is None:
        args.data = ["chicago"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_ablation_study(args)


if __name__ == "__main__":
    main()
