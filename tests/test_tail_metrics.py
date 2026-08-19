"""Regression tests for tail-focused post-training scoring."""

from __future__ import annotations

import numpy as np
import torch

from civicsafe.calibration.metrics_tail import (
    compare_tail_forecasts,
    threshold_weighted_crps,
    twcrps_zinb,
)


def test_threshold_weighted_crps_ignores_below_threshold_support() -> None:
    y = torch.tensor([0.0, 2.0])
    cdf = torch.tensor([[0.2, 0.5, 0.9], [0.1, 0.4, 0.8]])
    score = threshold_weighted_crps(y, cdf, threshold=2.0, reduction="none")
    assert torch.allclose(score, torch.tensor([0.01, 0.04]))


def test_twcrps_accepts_numpy_arrays_and_is_finite() -> None:
    score = twcrps_zinb(
        np.array([0.0, 3.0, 9.0]),
        np.array([0.1, 0.1, 0.1]),
        np.array([1.0, 3.0, 7.0]),
        np.array([2.0, 2.0, 2.0]),
    )
    assert torch.isfinite(score)
    assert float(score) >= 0.0


def test_compare_tail_forecasts_returns_common_threshold() -> None:
    y = torch.tensor([0.0, 2.0, 4.0, 8.0])
    result = compare_tail_forecasts(
        y,
        {"ha": torch.tensor([1.0, 2.0, 3.0, 4.0])},
    )
    assert result["ha"]["threshold"] == float(torch.quantile(y, 0.9))
    assert result["ha"]["twcrps"] >= 0.0
