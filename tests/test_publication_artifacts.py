"""Regression tests for the final post-training publication artifacts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch

if TYPE_CHECKING:
    from pathlib import Path


def test_baseline_metrics_sanitize_nonfinite_truth_and_predictions() -> None:
    from scripts.baselines import compute_metrics

    y_true = np.array([[[0.0], [np.nan]], [[np.inf], [-3.0]]])
    y_pred = np.array([[[np.nan], [np.inf]], [[-np.inf], [1e30]]])

    metrics = compute_metrics(y_true, y_pred, week_index=[10, 11])

    for key in ("crps", "mae", "rmse"):
        assert np.isfinite(metrics[key])
    assert len(metrics["per_week"]["crps"]) == 2
    assert np.isfinite(metrics["per_week"]["crps"]).all()


def test_conformal_step7_saves_prediction_panel_and_tail_metrics(
    tmp_path: Path,
) -> None:
    from scripts.run_conformal_evaluation import _save_post_training_artifacts

    y = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0]],
        ]
    )
    pi = torch.full_like(y, 0.2)
    mu = y + 1.0
    r = torch.full_like(y, 3.0)
    rolling_ha = y + 0.5
    upper = y + 4.0

    predictions_path, tail_path = _save_post_training_artifacts(
        "synthetic",
        test_y=y,
        test_pi=pi,
        test_mu=mu,
        test_r=r,
        rolling_ha=rolling_ha,
        conformal_upper=upper,
        demographic_group=torch.tensor([0, 1, 1]),
        selected_method="split_cp",
        output_root=tmp_path,
    )

    assert predictions_path == (
        tmp_path / "conformal_evaluation" / "synthetic_predictions.npz"
    )
    with np.load(predictions_path, allow_pickle=False) as panel:
        assert set(panel.files) == {
            "actual_violent",
            "point_prediction",
            "rolling_ha",
            "conformal_upper",
            "demographic_group",
        }
        assert panel["actual_violent"].shape == (2, 3)
        np.testing.assert_allclose(panel["actual_violent"], y[:, :, 0])
        np.testing.assert_allclose(
            panel["point_prediction"], ((1.0 - pi) * mu)[:, :, 0]
        )
        np.testing.assert_allclose(panel["rolling_ha"], rolling_ha[:, :, 0])
        np.testing.assert_allclose(panel["conformal_upper"], upper[:, :, 0])
        np.testing.assert_array_equal(panel["demographic_group"], [0.0, 1.0, 1.0])

    tail = json.loads(tail_path.read_text(encoding="utf-8"))
    assert tail["dataset"] == "synthetic"
    assert tail["selected_calibrator"] == "split_cp"
    assert tail["tail_quantile"] == pytest.approx(0.9)
    assert tail["tail_n"] > 0
    assert np.isfinite(tail["twcrps"])
    assert np.isfinite(tail["twcrps_tail"])


def test_policy_main_discovers_conformal_artifact_and_writes_publication_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import simulate_policy

    inputs_dir = tmp_path / "inputs" / "conformal_evaluation"
    outputs_dir = tmp_path / "publication"
    inputs_dir.mkdir(parents=True)
    actual = np.array([[3.0, 0.0, 2.0, 1.0], [0.0, 4.0, 1.0, 2.0]])
    np.savez_compressed(
        inputs_dir / "chicago_predictions.npz",
        actual_violent=actual,
        point_prediction=actual + 0.5,
        rolling_ha=np.full_like(actual, 1.0),
        conformal_upper=actual + 2.0,
        demographic_group=np.array([0, 0, 1, 1]),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "simulate_policy.py",
            "--data",
            "chicago",
            "--results-dir",
            str(tmp_path / "inputs"),
            "--output-dir",
            str(outputs_dir),
        ],
    )

    simulate_policy.main()

    result = json.loads(
        (outputs_dir / "policy_simulation_results.json").read_text(encoding="utf-8")
    )
    assert result["status"] == "ok"
    assert len(result["rows"]) == 12
    assert {row["budget"] for row in result["rows"]} == {20, 50, 100}
    table_path = outputs_dir / "tables" / "table7_policy_simulation.tex"
    figure_path = outputs_dir / "figures" / "fig9_policy_tradeoff.pdf"
    assert table_path.is_file()
    assert figure_path.is_file() and figure_path.stat().st_size > 0
    data_rows = [
        line
        for line in table_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("Chicago &")
    ]
    assert data_rows and all(line.endswith(r"\\") for line in data_rows)
    assert all(not line.endswith(r"\\\\") for line in data_rows)
