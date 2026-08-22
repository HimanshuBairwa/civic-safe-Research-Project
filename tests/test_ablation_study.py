"""Regression tests for publication-table generation."""

from __future__ import annotations

from typing import Any

from scripts.ablation_study import (
    _significance_stars,
    generate_conformal_table,
    generate_main_results_table,
    generate_policy_table,
    generate_uncertainty_table,
)


def _conformal_result(
    *,
    city: str,
    crps: float,
    mae: float,
    rmse: float,
    crpss: float,
    dm_p: float,
    epistemic_fraction: float,
) -> dict[str, Any]:
    coverage_results: dict[str, Any] = {}
    method_names = [
        "split_cp",
        "randomized_split_cp",
        "weighted_cp",
        "mondrian",
        "equalized_coverage",
        "ecrc",
        "adaptive_ecrc",
    ]
    for index, method in enumerate(method_names):
        coverage_results[method] = {
            "marginal_coverage": 0.90 + index * 0.002,
            "mean_width": 18.0 + index * 0.2,
            "coverage_disparity": 0.020 + index * 0.001,
            "abstention_rate": 0.005,
            "per_category": {
                "violent": {"coverage": 0.91},
                "property": {"coverage": 0.89 + index * 0.001},
                "drug": {"coverage": 0.90},
            },
        }
    return {
        "metadata": {"dataset": city, "alpha": 0.1},
        "point_forecast_metrics": {
            "crps": crps,
            "mae": mae,
            "rmse": rmse,
        },
        "skill_scores": {
            "baseline_crps_ha": crps / (1.0 - crpss),
            "crpss_vs_ha": crpss,
        },
        "statistical_significance": {
            "dm": {"p_value": dm_p},
            "bootstrap": {"p_value": dm_p / 2.0},
        },
        "coverage_results": coverage_results,
        "crps_decomposition": {
            "reliability": 0.0123,
            "resolution": 1.2345,
            "uncertainty": 4.5678,
        },
        "ensemble": {"epistemic_fraction": epistemic_fraction},
    }


def test_significance_stars_use_strict_publication_thresholds() -> None:
    assert _significance_stars(0.049) == r"^{*}"
    assert _significance_stars(0.009) == r"^{**}"
    assert _significance_stars(0.0009) == r"^{***}"
    assert _significance_stars(0.05) == ""


def test_main_table_contains_both_cities_skill_and_dm_stars() -> None:
    chicago = _conformal_result(
        city="chicago",
        crps=2.8182,
        mae=3.8917,
        rmse=7.0883,
        crpss=0.0389,
        dm_p=0.015,
        epistemic_fraction=0.10,
    )
    nyc = _conformal_result(
        city="nyc",
        crps=3.1475,
        mae=4.3773,
        rmse=7.6087,
        crpss=0.0472,
        dm_p=0.009,
        epistemic_fraction=0.20,
    )
    table = generate_main_results_table(
        chicago_baselines={
            "HA": {"crps": 2.9322, "mae": 4.0011, "rmse": 7.3600},
            "XGBoost": {"crps": 2.9157, "mae": 4.10, "rmse": 7.40},
        },
        nyc_baselines={
            "HA": {"crps": 3.3034, "mae": 4.4981, "rmse": 7.8540}
        },
        chicago_conformal=chicago,
        nyc_conformal=nyc,
    )

    assert r"\textbf{CRPSS vs HA}" in table
    assert r"\textbf{DM $p$-value}" in table
    assert "Brier" not in table
    assert r"\textit{Chicago}" in table
    assert r"\textit{NYC}" in table
    assert r"$0.0150^{*}$" in table
    assert r"$0.0090^{**}$" in table
    assert r"$^*p<0.05$" in table
    assert "    \\midrule\n    \\midrule" not in table


def test_main_table_uses_json_ha_baseline_without_auxiliary_csv() -> None:
    chicago = _conformal_result(
        city="chicago",
        crps=2.8182,
        mae=3.8917,
        rmse=7.0883,
        crpss=0.0389,
        dm_p=0.015,
        epistemic_fraction=0.10,
    )

    table = generate_main_results_table(chicago_conformal=chicago)

    assert "Historical Average (HA) &" in table
    assert "& -- & -- & +0.0000 & --" in table
    assert r"\textsc{Civic-Safe} (Ours)" in table


def test_conformal_table_contains_full_fairness_schema_for_both_cities() -> None:
    chicago = _conformal_result(
        city="chicago",
        crps=2.8,
        mae=3.8,
        rmse=7.0,
        crpss=0.04,
        dm_p=0.01,
        epistemic_fraction=0.10,
    )
    nyc = _conformal_result(
        city="nyc",
        crps=3.1,
        mae=4.3,
        rmse=7.6,
        crpss=0.05,
        dm_p=0.01,
        epistemic_fraction=0.20,
    )

    table = generate_conformal_table(
        chicago_results=chicago,
        nyc_results=nyc,
    )

    for method in (
        "Split CP",
        "Randomized Split CP",
        "Weighted CP",
        "Mondrian",
        "Equalized Coverage",
        r"\textsc{ECRC}",
        "Adaptive ECRC",
    ):
        assert table.count(f"\n    {method} &") == 2
    assert "Demographic Disparity" in table
    assert "Category Disparity" in table
    assert "Abstention Rate" in table
    assert "0.0200" in table
    assert "0.50" in table
    assert r"\textit{Chicago}" in table
    assert r"\textit{NYC}" in table
    assert "    \\midrule\n    \\midrule" not in table


def test_conformal_table_prefers_fairness_aware_and_rolling_variants() -> None:
    chicago = _conformal_result(
        city="chicago",
        crps=2.8,
        mae=3.8,
        rmse=7.0,
        crpss=0.04,
        dm_p=0.01,
        epistemic_fraction=0.10,
    )
    chicago["coverage_results"]["mondrian_demo_x_category"] = {
        "marginal_coverage": 0.934,
        "mean_width": 17.77,
        "coverage_disparity": 0.004,
        "abstention_rate": 0.0,
        "per_category": {
            "violent": {"coverage": 0.93},
            "property": {"coverage": 0.91},
        },
    }
    chicago["coverage_results"]["adaptive_ecrc_rolling"] = {
        "marginal_coverage": 0.918,
        "mean_width": 18.03,
        "coverage_disparity": 0.0307,
        "abstention_rate": 0.0,
        "per_category": {
            "violent": {"coverage": 0.92},
            "property": {"coverage": 0.90},
        },
    }

    table = generate_conformal_table(chicago_results=chicago)

    assert "Mondrian & 93.40 & 17.77 & 0.0040 & 0.0200 & 0.00" in table
    assert "Adaptive ECRC & 91.80 & 18.03 & 0.0307 & 0.0200 & 0.00" in table


def test_uncertainty_table_reports_hersbach_and_variance_shares() -> None:
    chicago = _conformal_result(
        city="chicago",
        crps=2.8,
        mae=3.8,
        rmse=7.0,
        crpss=0.04,
        dm_p=0.01,
        epistemic_fraction=0.125,
    )
    nyc = _conformal_result(
        city="nyc",
        crps=3.1,
        mae=4.3,
        rmse=7.6,
        crpss=0.05,
        dm_p=0.01,
        epistemic_fraction=0.25,
    )

    table = generate_uncertainty_table(chicago, nyc)

    assert r"\textbf{Dataset}" in table
    assert r"REL $\downarrow$" in table
    assert r"RES $\uparrow$" in table
    assert "UNC" in table
    assert "0.0123" in table
    assert "1.2345" in table
    assert "4.5678" in table
    assert "12.50" in table
    assert "87.50" in table
    assert "25.00" in table
    assert "75.00" in table


def test_main_table_renders_deep_spread_significance_and_divergence() -> None:
    chicago = _conformal_result(
        city="chicago", crps=2.8, mae=3.8, rmse=7.0,
        crpss=0.04, dm_p=0.01, epistemic_fraction=0.1,
    )
    nyc = _conformal_result(
        city="nyc", crps=3.1, mae=4.3, rmse=7.6,
        crpss=0.05, dm_p=0.01, epistemic_fraction=0.2,
    )
    table = generate_main_results_table(
        chicago_conformal=chicago,
        nyc_conformal=nyc,
        chicago_baselines={
            "HA": {"crps": 2.93},
            "LSTM_NB": {"crps": 3.10, "crps_std": 0.04},
        },
        nyc_baselines={
            "HA": {"crps": 3.30},
            "ZINB": {"crps": 924.10, "mae": 1844.0, "rmse": 4270.0},
        },
        chicago_significance={
            "LSTM_NB": {
                "dm": {"p_value": 0.02},
                "bootstrap": {"p_value": 0.01},
            }
        },
        nyc_significance={},
    )

    assert r"3.1000 $\pm$ 0.0400" in table
    assert r"$0.0200^{*}$" in table
    assert r"$0.0100^{*}$" in table
    assert r"Diverged$^{\dagger}$" in table
    assert "924.1000" not in table


def test_policy_table_has_explicit_city_headers() -> None:
    rows = []
    for city in ("chicago", "nyc"):
        rows.append({
            "city": city,
            "policy": "naive_ha",
            "budget": 20,
            "violent_hit_rate": 0.5,
            "demographic_overallocation_ratio": 1.1,
            "idle_wasted_resource_ratio": 0.01,
            "allocation_disparity": 0.02,
        })

    table = generate_policy_table({"status": "ok", "rows": rows})

    assert r"\multicolumn{6}{l}{\textit{Chicago}}" in table
    assert r"\multicolumn{6}{l}{\textit{NYC}}" in table
    assert all(
        line.endswith(r"\\")
        for line in table.splitlines()
        if line.strip().startswith("Naive Ha &")
    )
