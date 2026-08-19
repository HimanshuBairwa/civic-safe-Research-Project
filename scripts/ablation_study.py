#!/usr/bin/env python
"""CIVIC-SAFE Ablation Study — LaTeX Table Generator.

Reads trained model checkpoints, evaluation results, baseline CSVs,
and conformal calibration JSON to produce publication-ready LaTeX
booktabs tables for the CIVIC-SAFE paper.

Tables generated:
  Table 1: Main benchmark — accuracy, CRPSS vs HA, DM significance
  Table 2: Conformal prediction and fairness comparison
  Table 3: Hersbach CRPS decomposition and uncertainty attribution
  Table 4: Component ablation
  Table 5: Loss-function ablation
  Table 6: Ensemble-size ablation

Usage:
    python scripts/ablation_study.py --data chicago
    python scripts/ablation_study.py --data chicago --data nyc
    python scripts/ablation_study.py --results-dir outputs/
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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
MAIN_METRICS = [
    "CRPS",
    "MAE",
    "RMSE",
    "CRPSS vs HA",
    r"DM $p$-value",
    r"Bootstrap $p$-value",
]
CONFORMAL_METRICS = [
    r"Marginal Coverage (\%)",
    "Interval Width",
    r"Demographic Disparity ($\Delta_{\mathrm{dem}}\alpha$)",
    r"Category Disparity ($\Delta_{\mathrm{cat}}\alpha$)",
    r"Abstention Rate (\%)",
]

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
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else None


def _fmt(value: float | None, fmt: str = FMT_4, missing: str = "--") -> str:
    """Format a numeric value, returning *missing* sentinel for None / NaN."""
    if value is None:
        return missing
    try:
        number = float(value)
    except (TypeError, ValueError):
        return missing
    if not math.isfinite(number):
        return missing
    return f"{number:{fmt}}"


def _significance_stars(p_value: float | None) -> str:
    """Return publication significance stars for a two-sided p-value."""
    if p_value is None:
        return ""
    try:
        p = float(p_value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(p):
        return ""
    if p < 0.001:
        return r"^{***}"
    if p < 0.01:
        return r"^{**}"
    if p < 0.05:
        return r"^{*}"
    return ""


def _fmt_p_value(p_value: float | None, missing: str = "--") -> str:
    """Format a p-value and append the conventional significance stars."""
    if p_value is None:
        return missing
    try:
        p = float(p_value)
    except (TypeError, ValueError):
        return missing
    if not math.isfinite(p):
        return missing
    value = "<0.0001" if p < 0.0001 else f"{p:.4f}"
    return rf"${value}{_significance_stars(p)}$"


def _finite_number(value: Any) -> float | None:
    """Coerce nested JSON values to finite floats."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _nested_mapping(value: Any) -> dict[str, Any]:
    """Return a JSON object as a mutable plain mapping, otherwise empty."""
    return value if isinstance(value, dict) else {}


def _first_finite_number(
    values: dict[str, Any], *keys: str
) -> float | None:
    """Return the first finite number available under *keys*."""
    for key in keys:
        number = _finite_number(values.get(key))
        if number is not None:
            return number
    return None


def _is_historical_average(name: str) -> bool:
    """Recognize common labels for the rolling historical-average baseline."""
    normalized = " ".join(name.lower().replace("_", " ").split())
    return normalized == "ha" or "historical average" in normalized


def _category_disparity(metrics: dict[str, Any]) -> float | None:
    """Derive max-minus-min category coverage disparity from saved JSON."""
    explicit = _finite_number(metrics.get("category_disparity"))
    if explicit is not None:
        return explicit
    per_category = _nested_mapping(metrics.get("per_category"))
    coverages = [
        number
        for entry in per_category.values()
        if isinstance(entry, dict)
        for number in [_finite_number(entry.get("coverage"))]
        if number is not None
    ]
    return max(coverages) - min(coverages) if len(coverages) >= 2 else None


def _model_metric_source(
    model_results: dict[str, Any] | None,
    conformal_results: dict[str, Any] | None,
) -> dict[str, Any]:
    """Prefer the conformal JSON metrics, with evaluation JSON as fallback."""
    if conformal_results is not None:
        metrics = conformal_results.get("point_forecast_metrics")
        if isinstance(metrics, dict):
            return metrics
    if model_results is not None:
        overall = model_results.get("overall", model_results)
        if isinstance(overall, dict):
            return overall
    return {}


def _significance_source(conformal_results: dict[str, Any] | None) -> dict[str, Any]:
    """Return the nested forecast-comparison result from either JSON schema."""
    if conformal_results is None:
        return {}
    value = conformal_results.get("statistical_significance")
    if isinstance(value, dict):
        return value
    value = conformal_results.get("significance")
    return value if isinstance(value, dict) else {}


def _crpss_vs_ha(
    metrics: dict[str, Any],
    conformal_results: dict[str, Any] | None,
    ha_crps: float | None,
) -> float | None:
    """Extract or calculate CRPSS against the rolling historical average."""
    if conformal_results is not None:
        skill = conformal_results.get("skill_scores")
        if isinstance(skill, dict):
            direct = _finite_number(skill.get("crpss_vs_ha"))
            if direct is not None:
                return direct
    crps = _finite_number(metrics.get("crps"))
    baseline = _finite_number(ha_crps)
    if crps is None or baseline is None or baseline <= 0.0:
        return None
    return 1.0 - (crps / baseline)


def _fmt_pm(
    value: float | None,
    std: float | None = None,
    fmt: str = FMT_4,
    missing: str = "--",
) -> str:
    """Format ``mean`` with an optional ``\\pm std``.

    Ablation gaps on a 53-week test set are frequently smaller than the spread
    across seeds. Printing the spread next to the mean is what stops a reader
    (or an author) from treating a 0.01 CRPS difference as a finding.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return missing
    if std is None or (isinstance(std, float) and np.isnan(std)) or std == 0:
        return f"{value:{fmt}}"
    return rf"{value:{fmt}} $\pm$ {std:{fmt}}"


def _mean_of(cell: str) -> float | None:
    """Recover the numeric mean from a possibly ``mean $\\pm$ std`` cell.

    ``_bold_best_column`` compares cells numerically; without this it would
    fail to parse any cell carrying a spread and quietly bold nothing.
    """
    if not cell or cell == "--":
        return None
    head = cell.split("$\\pm$")[0].split("±")[0]
    head = head.replace(r"\textbf{", "").strip()
    head = head.replace("$", "").split("^{", maxsplit=1)[0]
    head = head.lstrip("<").rstrip("}").strip()
    try:
        return float(head)
    except ValueError:
        return None


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
            parsed = _mean_of(val_str)
            if parsed is None:
                continue
            numeric.append((i, parsed))

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
    name_header: str = "",
    note: str | None = None,
) -> str:
    """Assemble a complete LaTeX booktabs table string.

    Args:
        caption: Table caption.
        label: Table label for \\ref{}.
        headers: Column header names (excluding the row-name column).
        rows: List of dicts, each with *name_key* and each header as keys.
        name_key: Key in each row dict that holds the row label text.
        name_header: Optional heading for the row-name column.
        note: Optional full-width note rendered beneath the data rows.

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
    name_cell = rf"\textbf{{{name_header}}}" if name_header else ""
    lines.append(rf"    {name_cell} & {header_cells} \\")
    lines.append(r"    \midrule")

    # Data rows
    for row in rows:
        name = row.get(name_key, "")
        cells = " & ".join(row.get(h, "--") for h in headers)
        lines.append(f"    {name} & {cells} \\\\")

    if note:
        lines.append(r"    \midrule")
        lines.append(
            rf"    \multicolumn{{{n_cols}}}{{l}}{{\footnotesize {note}}} \\"
        )

    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )

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


def load_baseline_results(
    results_dir: Path, city: str
) -> dict[str, dict[str, float]] | None:
    """Load classical and deep baselines into one normalized mapping.

    Multi-seed deep-baseline aggregates take precedence over the canonical
    single-run JSON when both are available.
    """
    import csv

    results: dict[str, dict[str, float]] = {}
    csv_path = results_dir / "baselines" / f"{city}_baselines.csv"
    if csv_path.exists():
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("Model") or row.get("")
                if name is None:
                    continue
                results[name] = {}
                for key in ("crps", "mae", "rmse"):
                    value = _finite_number(row.get(key))
                    if value is not None:
                        results[name][key] = value
    else:
        logger.warning(f"Baseline CSV not found: {csv_path}")

    baselines_dir = results_dir / "baselines"
    aggregate_path = baselines_dir / f"{city}_seed_matched.json"
    canonical_path = baselines_dir / f"{city}_deep_baselines.json"
    deep_models: dict[str, Any] = {}
    if aggregate_path.exists():
        aggregate = _load_json(aggregate_path)
        if aggregate is not None:
            deep_models = _nested_mapping(aggregate.get("baselines"))
    elif canonical_path.exists():
        canonical = _load_json(canonical_path)
        if canonical is not None:
            deep_models = canonical

    for name, raw_metrics in deep_models.items():
        if name.startswith("_") or not isinstance(raw_metrics, dict):
            continue
        metrics: dict[str, float] = {}
        for key in ("crps", "mae", "rmse"):
            value = _finite_number(raw_metrics.get(key))
            if value is not None:
                metrics[key] = value
        if metrics:
            results[name] = metrics

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
    chicago_conformal: dict[str, Any] | None = None,
    nyc_conformal: dict[str, Any] | None = None,
) -> str:
    """Generate the benchmark table with CRPSS and DM significance."""
    headers = list(MAIN_METRICS)
    lower_is_better = {
        "CRPS": True,
        "MAE": True,
        "RMSE": True,
        "CRPSS vs HA": False,
        r"DM $p$-value": True,
    }
    city_specs = [
        (
            "Chicago",
            chicago_results,
            chicago_baselines,
            chicago_conformal,
        ),
        ("NYC", nyc_results, nyc_baselines, nyc_conformal),
    ]
    all_rows: list[dict[str, str]] = []
    include_city_separators = (
        sum(
            result is not None or baseline is not None or conformal is not None
            for _, result, baseline, conformal in city_specs
        )
        > 1
    )

    for city, model_res, baseline_res, conformal_res in city_specs:
        if model_res is None and baseline_res is None and conformal_res is None:
            continue

        city_rows: list[dict[str, str]] = []
        conformal_skill = _nested_mapping(
            conformal_res.get("skill_scores") if conformal_res else None
        )
        ha_from_csv = (
            baseline_res.get("HA", {}).get("crps")
            if baseline_res and isinstance(baseline_res.get("HA"), dict)
            else None
        )
        ha_crps = _finite_number(
            ha_from_csv
            if ha_from_csv is not None
            else conformal_skill.get("baseline_crps_ha")
        )
        ha_mae = _first_finite_number(
            conformal_skill,
            "baseline_mae_ha",
            "baseline_ha_mae",
            "ha_mae",
        )
        ha_rmse = _first_finite_number(
            conformal_skill,
            "baseline_rmse_ha",
            "baseline_ha_rmse",
            "ha_rmse",
        )

        has_ha_row = False
        if baseline_res is not None:
            for baseline_name, baseline_metrics in baseline_res.items():
                metrics = (
                    baseline_metrics
                    if isinstance(baseline_metrics, dict)
                    else {}
                )
                is_ha = _is_historical_average(baseline_name)
                has_ha_row = has_ha_row or is_ha
                crps = _finite_number(metrics.get("crps"))
                mae = _finite_number(metrics.get("mae"))
                rmse = _finite_number(metrics.get("rmse"))
                if is_ha:
                    crps = crps if crps is not None else ha_crps
                    mae = mae if mae is not None else ha_mae
                    rmse = rmse if rmse is not None else ha_rmse
                city_rows.append(
                    {
                        "name": _latex_escape(baseline_name),
                        "CRPS": _fmt(crps),
                        "MAE": _fmt(mae),
                        "RMSE": _fmt(rmse),
                        "CRPSS vs HA": _fmt(
                            0.0
                            if is_ha and crps is not None
                            else _crpss_vs_ha(metrics, None, ha_crps),
                            "+.4f",
                        ),
                        r"DM $p$-value": "--",
                        r"Bootstrap $p$-value": "--",
                    }
                )

        if not has_ha_row and ha_crps is not None:
            city_rows.append(
                {
                    "name": "Historical Average (HA)",
                    "CRPS": _fmt(ha_crps),
                    "MAE": _fmt(ha_mae),
                    "RMSE": _fmt(ha_rmse),
                    "CRPSS vs HA": _fmt(0.0, "+.4f"),
                    r"DM $p$-value": "--",
                    r"Bootstrap $p$-value": "--",
                }
            )

        model_metrics = _model_metric_source(model_res, conformal_res)
        if model_metrics:
            significance = _significance_source(conformal_res)
            dm = _nested_mapping(significance.get("dm"))
            bootstrap = _nested_mapping(significance.get("bootstrap"))
            dm_p = _finite_number(dm.get("p_value"))
            bootstrap_p = _finite_number(bootstrap.get("p_value"))
            civic_crpss = _crpss_vs_ha(model_metrics, conformal_res, ha_crps)
            city_rows.append(
                {
                    "name": r"\textsc{Civic-Safe} (Ours)",
                    "CRPS": _fmt(model_metrics.get("crps")),
                    "MAE": _fmt(model_metrics.get("mae")),
                    "RMSE": _fmt(model_metrics.get("rmse")),
                    "CRPSS vs HA": _fmt(civic_crpss, "+.4f"),
                    r"DM $p$-value": _fmt_p_value(dm_p),
                    r"Bootstrap $p$-value": _fmt_p_value(bootstrap_p),
                }
            )

        _bold_best_column(city_rows, headers, lower_is_better)
        if include_city_separators:
            all_rows.append(
                {
                    "name": (
                        rf"\multicolumn{{{len(headers) + 1}}}{{l}}"
                        rf"{{\textit{{{city}}}}}"
                    ),
                    "_separator": "true",
                }
            )
        all_rows.extend(city_rows)

    col_spec = "l" + " r" * len(headers)
    header_cells = " & ".join(
        rf"\textbf{{{header}}}" for header in headers
    )
    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{Main spatiotemporal benchmark on the 2023 test set.}",
        r"  \label{tab:main_results}",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        rf"    \textbf{{Method}} & {header_cells} \\",
        r"    \midrule",
    ]
    rendered_data_row = False
    for row in all_rows:
        if "_separator" in row:
            if rendered_data_row:
                lines.append(r"    \midrule")
            lines.append(f"    {row['name']} \\\\")
            lines.append(r"    \midrule")
        else:
            cells = " & ".join(row.get(header, "--") for header in headers)
            lines.append(f"    {row['name']} & {cells} \\\\")
            rendered_data_row = True
    lines.extend(
        [
            r"    \midrule",
            (
                r"    \multicolumn{7}{l}{\footnotesize "
                r"$^*p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$.} \\"
            ),
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


# Table 4: Component Ablation
# ───────────────────────────────────────────────────────────────────
def generate_ablation_table(results: dict[str, Any] | None = None) -> str:
    """Generate Table 4: Component ablation study.

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
        ("no_transformer", r"$-$ Temporal attention (Transformer)"),
        ("nb_only", r"$-$ Zero-inflation (NB only)"),
        ("no_r_reg", r"$-$ $r$-floor regularization"),
        ("no_sharpness", r"$-$ Sharpness penalty"),
        ("no_emos", r"$-$ EMOS weighting"),
        ("no_recal", r"$-$ Recalibration"),
        ("no_grl", r"$-$ GRL (Demographic blindness)"),
    ]

    rows: list[dict[str, str]] = []
    for key, display_name in ablation_variants:
        if results is not None and key in results:
            m = results[key]
            std = m.get("_std", {}) if isinstance(m, dict) else {}
            rows.append({
                "name": display_name,
                "CRPS": _fmt_pm(m.get("crps"), std.get("crps")),
                "MAE": _fmt_pm(m.get("mae"), std.get("mae")),
                "RMSE": _fmt_pm(m.get("rmse"), std.get("rmse")),
                "Brier": _fmt_pm(m.get("brier_zero"), std.get("brier_zero")),
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
def generate_conformal_table(
    results: dict[str, Any] | None = None,
    *,
    chicago_results: dict[str, Any] | None = None,
    nyc_results: dict[str, Any] | None = None,
) -> str:
    """Generate the conformal calibration and fairness comparison table."""
    (
        coverage_header,
        width_header,
        demographic_header,
        category_header,
        abstention_header,
    ) = CONFORMAL_METRICS
    headers = list(CONFORMAL_METRICS)
    methods = [
        ("split_cp", ("split_cp",), "Split CP"),
        (
            "randomized_split_cp",
            ("randomized_split_cp",),
            "Randomized Split CP",
        ),
        ("weighted_cp", ("weighted_cp",), "Weighted CP"),
        (
            "mondrian",
            ("mondrian_demo_x_category", "mondrian", "mondrian_category"),
            "Mondrian",
        ),
        (
            "equalized_coverage",
            ("equalized_coverage",),
            "Equalized Coverage",
        ),
        (
            "variance_scaled_split_cp",
            ("variance_scaled_split_cp",),
            "Variance-Scaled CP",
        ),
        ("ecrc", ("ecrc",), r"\textsc{ECRC}"),
        (
            "adaptive_ecrc",
            ("adaptive_ecrc_rolling", "adaptive_ecrc"),
            "Adaptive ECRC",
        ),
    ]

    if chicago_results is not None or nyc_results is not None:
        city_specs = [
            ("Chicago", chicago_results),
            ("NYC", nyc_results),
        ]
    elif results is not None and "coverage_results" in results:
        dataset = str(
            _nested_mapping(results.get("metadata")).get("dataset", "Results")
        )
        display = "NYC" if dataset.lower() == "nyc" else dataset.title()
        city_specs = [(display, results)]
    elif results is not None:
        city_specs = [
            ("Chicago", results.get("chicago")),
            ("NYC", results.get("nyc")),
        ]
    else:
        city_specs = [("Results", None)]

    city_specs = [
        (city, city_result)
        for city, city_result in city_specs
        if city_result is not None
    ] or [("Results", None)]
    include_city_separators = len(city_specs) > 1
    all_rows: list[dict[str, str]] = []

    for city, city_result in city_specs:
        coverage_data = _nested_mapping(
            city_result.get("coverage_results") if city_result else None
        )
        city_rows: list[dict[str, str]] = []
        for _, aliases, display in methods:
            method_metrics: dict[str, Any] = {}
            for alias in aliases:
                candidate = coverage_data.get(alias)
                if isinstance(candidate, dict):
                    method_metrics = candidate
                    break
            coverage_value = _finite_number(
                method_metrics.get("marginal_coverage")
            )
            abstention = (
                _finite_number(method_metrics.get("abstention_rate", 0.0))
                if method_metrics
                else None
            )
            city_rows.append(
                {
                    "name": display,
                    coverage_header: _fmt(
                        coverage_value * 100.0
                        if coverage_value is not None
                        else None,
                        FMT_PCT,
                    ),
                    width_header: _fmt(
                        _finite_number(method_metrics.get("mean_width")),
                        FMT_2,
                    ),
                    demographic_header: _fmt(
                        _finite_number(
                            method_metrics.get("coverage_disparity")
                        ),
                        FMT_4,
                    ),
                    category_header: _fmt(
                        _category_disparity(method_metrics), FMT_4
                    ),
                    abstention_header: _fmt(
                        abstention * 100.0 if abstention is not None else None,
                        FMT_PCT,
                    ),
                }
            )

        if include_city_separators:
            all_rows.append(
                {
                    "name": (
                        rf"\multicolumn{{{len(headers) + 1}}}{{l}}"
                        rf"{{\textit{{{city}}}}}"
                    ),
                    "_separator": "true",
                }
            )
        all_rows.extend(city_rows)

    col_spec = "l" + " r" * len(headers)
    header_cells = " & ".join(
        rf"\textbf{{{header}}}" for header in headers
    )
    lines = [
        r"\begin{table*}[htbp]",
        r"  \centering",
        (
            r"  \caption{Conformal prediction efficiency and fairness "
            r"on the 2023 test set.}"
        ),
        r"  \label{tab:conformal_fairness}",
        rf"  \begin{{tabular}}{{{col_spec}}}",
        r"    \toprule",
        rf"    \textbf{{Method}} & {header_cells} \\",
        r"    \midrule",
    ]
    rendered_data_row = False
    for row in all_rows:
        if "_separator" in row:
            if rendered_data_row:
                lines.append(r"    \midrule")
            lines.append(f"    {row['name']} \\\\")
            lines.append(r"    \midrule")
        else:
            cells = " & ".join(row.get(header, "--") for header in headers)
            lines.append(f"    {row['name']} & {cells} \\\\")
            rendered_data_row = True
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def generate_uncertainty_table(
    chicago_results: dict[str, Any] | None = None,
    nyc_results: dict[str, Any] | None = None,
) -> str:
    """Generate Hersbach decomposition and variance-source percentages."""
    headers = [
        r"REL $\downarrow$",
        r"RES $\uparrow$",
        "UNC",
        r"Epistemic Variance (\%)",
        r"Aleatoric Variance (\%)",
        "Recalibration",
    ]
    rows: list[dict[str, str]] = []
    for city, result in [
        ("Chicago", chicago_results),
        ("NYC", nyc_results),
    ]:
        if result is None:
            continue
        decomposition = _nested_mapping(result.get("crps_decomposition"))
        ensemble = _nested_mapping(result.get("ensemble"))
        recalibration = _nested_mapping(result.get("recalibration"))
        epistemic_fraction = _finite_number(
            ensemble.get("epistemic_fraction")
        )
        if epistemic_fraction is None:
            epistemic = _finite_number(
                ensemble.get("epistemic_uncertainty")
            )
            aleatoric = _finite_number(
                ensemble.get("aleatoric_uncertainty")
            )
            if epistemic is not None and aleatoric is not None:
                total = epistemic + aleatoric
                if total > 0.0:
                    epistemic_fraction = epistemic / total
        aleatoric_fraction = (
            1.0 - epistemic_fraction
            if epistemic_fraction is not None
            else None
        )
        rows.append(
            {
                "name": city,
                r"REL $\downarrow$": _fmt(
                    _finite_number(decomposition.get("reliability"))
                ),
                r"RES $\uparrow$": _fmt(
                    _finite_number(decomposition.get("resolution"))
                ),
                "UNC": _fmt(
                    _finite_number(decomposition.get("uncertainty"))
                ),
                r"Epistemic Variance (\%)": _fmt(
                    epistemic_fraction * 100.0
                    if epistemic_fraction is not None
                    else None,
                    FMT_PCT,
                ),
                r"Aleatoric Variance (\%)": _fmt(
                    aleatoric_fraction * 100.0
                    if aleatoric_fraction is not None
                    else None,
                    FMT_PCT,
                ),
                "Recalibration": (
                    "Applied"
                    if bool(recalibration.get("recal_applied", False))
                    else "Identity fallback"
                ),
            }
        )

    if not rows:
        rows = [
            {
                "name": city,
                **{header: "--" for header in headers},
            }
            for city in ("Chicago", "NYC")
        ]

    return _build_booktabs_table(
        caption=(
            r"Hersbach (2000) CRPS decomposition and predictive variance "
            r"attribution for the five-seed ensemble."
        ),
        label="tab:uncertainty",
        headers=headers,
        rows=rows,
        name_header="Dataset",
        note=(
            r"REL: reliability; RES: resolution; UNC: climatological "
            r"uncertainty. Variance shares sum to $100\%$."
        ),
    )


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
            std = m.get("_std", {}) if isinstance(m, dict) else {}
            rows.append({
                "name": display,
                "CRPS": _fmt_pm(m.get("crps"), std.get("crps")),
                "MAE": _fmt_pm(m.get("mae"), std.get("mae")),
                "RMSE": _fmt_pm(m.get("rmse"), std.get("rmse")),
                "Brier": _fmt_pm(m.get("brier_zero"), std.get("brier_zero")),
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

    # Component ablation files.
    # This list must stay in sync with the variants rendered by
    # generate_ablation_table() AND with those produced by run_ablations.py:
    # a name present in only one place either silently drops a result that was
    # computed, or renders a permanent dash for a row nothing will ever fill.
    for variant in [
        "full_model",
        "no_gatv2",
        "no_transformer",
        "nb_only",
        "no_r_reg",
        "no_sharpness",
        "no_emos",
        "no_recal",
        "no_grl",
    ]:
        path = ablation_dir / f"{variant}_results.json"
        data = _load_json(path)
        if data is not None:
            # Normalise: accept either top-level metrics or nested "overall"
            metrics = data.get("overall", data)
            # Carry the seed spread through so the table can print mean +/- std
            if isinstance(data, dict) and "std" in data:
                metrics = {**metrics, "_std": data["std"], "_n_seeds": data.get("n_seeds")}
            out["component"][variant] = metrics

    # Loss function ablation. run_ablations.py writes loss_{nll,sac}_results.json
    # and aliases loss_crps to the full model (which is already CRPS-trained).
    for loss in ["nll", "crps", "sac"]:
        path = ablation_dir / f"loss_{loss}_results.json"
        data = _load_json(path)
        if data is not None:
            metrics = data.get("overall", data)
            if isinstance(data, dict) and "std" in data:
                metrics = {**metrics, "_std": data["std"], "_n_seeds": data.get("n_seeds")}
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

    for city in cities:
        logger.info(f"\n  Loading results for {city}...")
        city_model_results[city] = load_model_results(results_dir, city)
        city_baselines[city] = load_baseline_results(results_dir, city)
        city_conformal[city] = load_conformal_results(results_dir, city)

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
        chicago_conformal=city_conformal.get("chicago"),
        nyc_conformal=city_conformal.get("nyc"),
    )
    _save_table(output_dir / "table1_main_results.tex", table1, "Table 1: Main Results")

    # ── Table 2: Conformal Prediction and Fairness ──
    logger.info("[2/6] Generating Table 2: Conformal Prediction and Fairness...")
    table2 = generate_conformal_table(
        chicago_results=city_conformal.get("chicago"),
        nyc_results=city_conformal.get("nyc"),
    )
    _save_table(
        output_dir / "table2_conformal_fairness.tex",
        table2,
        "Table 2: Conformal Prediction and Fairness",
    )

    # ── Table 3: Uncertainty Decomposition ──
    logger.info("[3/6] Generating Table 3: Uncertainty Decomposition...")
    table3 = generate_uncertainty_table(
        chicago_results=city_conformal.get("chicago"),
        nyc_results=city_conformal.get("nyc"),
    )
    _save_table(
        output_dir / "table3_uncertainty.tex",
        table3,
        "Table 3: Uncertainty",
    )

    # ── Table 4: Component Ablation ──
    logger.info("[4/6] Generating Table 4: Component Ablation...")
    comp_results = (
        ablation_data["component"] if ablation_data["component"] else None
    )
    table4 = generate_ablation_table(comp_results)
    _save_table(
        output_dir / "table4_ablation.tex",
        table4,
        "Table 4: Ablation",
    )

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
