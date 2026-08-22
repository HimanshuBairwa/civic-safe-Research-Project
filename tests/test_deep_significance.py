"""Regression tests for post-training deep-baseline significance artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compute_deep_significance import compute_city, load_deep_series


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _per_week(offset: float) -> dict:
    return {
        "week_index": list(range(260, 272)),
        "crps": [offset + index * 0.01 for index in range(12)],
        "aggregation": "mean over spatial units and categories",
    }


def test_deep_seed_series_are_averaged_by_absolute_week(tmp_path: Path) -> None:
    for seed, offset in ((42, 3.0), (137, 3.2), (256, 3.4)):
        _write(
            tmp_path / "baselines" / f"nyc_deep_baselines_seed{seed}.json",
            {"LSTM_NB": {"per_week": _per_week(offset)}},
        )

    series = load_deep_series("nyc", tmp_path)["LSTM_NB"]

    assert series[0] == list(range(260, 272))
    assert series[1][0] == pytest.approx(3.2)
    assert series[1][-1] == pytest.approx(3.31)


def test_missing_civicsafe_series_fails_closed(tmp_path: Path) -> None:
    _write(
        tmp_path / "baselines" / "nyc_deep_baselines_seed42.json",
        {"LSTM_NB": {"per_week": _per_week(3.0)}},
    )
    # Aggregate CRPS is intentionally insufficient for a paired test.
    _write(
        tmp_path / "conformal_evaluation" / "nyc_conformal_results.json",
        {"point_forecast_metrics": {"crps": 3.14}},
    )

    result = compute_city("nyc", tmp_path)

    assert result["status"] == "unavailable"
    assert "per-week CRPS" in result["reason"]
    assert result["baselines_available"] == ["LSTM_NB"]


def test_compute_city_runs_dm_and_bootstrap_from_weekly_sidecar(tmp_path: Path) -> None:
    _write(
        tmp_path / "conformal_evaluation" / "nyc_per_week_crps.json",
        {"per_week": _per_week(2.8)},
    )
    baseline = _per_week(3.2)
    baseline["crps"] = [value + 0.03 * (index % 3) for index, value in enumerate(baseline["crps"])]
    _write(
        tmp_path / "baselines" / "nyc_deep_baselines_seed42.json",
        {"LSTM_NB": {"per_week": baseline}},
    )

    result = compute_city("nyc", tmp_path)

    assert result["status"] == "ok"
    comparison = result["comparisons"]["LSTM_NB"]
    assert comparison["n_weeks_tested"] == 12
    assert comparison["dm"]["mean_diff"] < -0.39
    assert 0.0 <= comparison["bootstrap"]["p_value"] <= 1.0
