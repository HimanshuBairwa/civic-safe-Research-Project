"""Focused regression tests for the post-training CIVIC-SAFE upgrades."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from civicsafe.calibration.conformal import (
    _ecrc_quantile_level,
    compute_variance_scaled_cqr_scores,
    zinb_predictive_scale,
)
from civicsafe.calibration.emos import apply_emos_weights, learn_emos_weights
from civicsafe.calibration.ensemble_evaluator import validate_member_alignment
from civicsafe.calibration.recalibration import recalibrate_and_evaluate


def test_category_conditioned_emos_applies_one_weight_vector_per_category() -> None:
    pi = [torch.zeros(4, 2), torch.zeros(4, 2)]
    mu = [torch.tensor([[1.0, 10.0]]).repeat(4, 1), torch.tensor([[3.0, 30.0]]).repeat(4, 1)]
    r = [torch.ones(4, 2), torch.ones(4, 2)]

    _, combined_mu, _ = apply_emos_weights(
        [[1.0, 0.0], [0.0, 1.0]], pi, mu, r
    )

    assert torch.equal(combined_mu[:, 0], mu[0][:, 0])
    assert torch.equal(combined_mu[:, 1], mu[1][:, 1])


def test_category_conditioned_emos_falls_back_per_category() -> None:
    torch.manual_seed(7)
    n = 30
    y = torch.tensor([[5.0, 20.0]]).repeat(n, 1)
    pi = [torch.full((n, 2), 0.05), torch.full((n, 2), 0.05)]
    mu = [
        torch.tensor([[5.0, 20.0]]).repeat(n, 1),
        torch.tensor([[30.0, 20.0]]).repeat(n, 1),
    ]
    r = [torch.full((n, 2), 20.0), torch.full((n, 2), 20.0)]

    result = learn_emos_weights(
        y,
        pi,
        mu,
        r,
        category_wise=True,
        max_iter=40,
        patience=10,
        min_holdout_improvement=0.0025,
    )

    assert result["fallback_by_category"] == [False, True]
    assert result["category_weights"][1] == pytest.approx([0.5, 0.5], abs=1e-5)
    assert result["category_weights"][0][0] > result["category_weights"][0][1]


def test_entropy_regularization_keeps_nonzero_ensemble_support() -> None:
    n = 30
    y = torch.full((n,), 10.0)
    pi = [torch.zeros(n) for _ in range(3)]
    mu = [torch.full((n,), value) for value in (10.0, 40.0, 50.0)]
    r = [torch.full((n,), 100.0) for _ in range(3)]

    result = learn_emos_weights(
        y,
        pi,
        mu,
        r,
        entropy_lambda=0.5,
        holdout_fraction=0.3,
        min_holdout_improvement=0.0,
        max_iter=80,
        patience=15,
    )

    assert min(result["learned_weights"]) > 0.005
    assert sum(result["weights"]) == pytest.approx(1.0)


def test_exact_binomial_ecrc_bound_is_tighter_and_delta_monotone() -> None:
    scores = torch.linspace(-1.0, 1.0, 1000)
    exact = _ecrc_quantile_level(
        scores, alpha=0.1, delta_group=0.0125, bound="exact_binomial"
    )
    hoeffding = _ecrc_quantile_level(
        scores, alpha=0.1, delta_group=0.0125, bound="hoeffding"
    )
    less_confident = _ecrc_quantile_level(
        scores, alpha=0.1, delta_group=0.025, bound="exact_binomial"
    )

    assert exact[1] < hoeffding[1]
    assert exact[0] >= less_confident[0]


def test_variance_scaled_scores_are_non_degenerate_and_scale_local() -> None:
    pi = torch.zeros(4)
    mu = torch.tensor([1.0, 1.0, 100.0, 100.0])
    r = torch.full((4,), 2.0)
    y = torch.tensor([0.0, 8.0, 0.0, 80.0])

    scores = compute_variance_scaled_cqr_scores(y, pi, mu, r)
    scales = zinb_predictive_scale(pi, mu, r)

    assert torch.isfinite(scores).all()
    assert not torch.all(scores == 0)
    assert scales[3] > scales[0]
    assert scores[1] != scores[3]


class _FakeRecalibrator:
    def __init__(self, *, method: str) -> None:
        del method

    def fit(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        del args, kwargs
        return {
            "initial_crps": 1.0,
            "final_crps": 0.5,
            "improvement_pct": 50.0,
            "iterations": 1,
        }

    def transform(self, pi: torch.Tensor, mu: torch.Tensor, r: torch.Tensor):
        return pi, mu + 100.0, r

    def get_params(self) -> dict[str, float]:
        return {"scale_mu": 1.0, "shift_mu": 100.0}


def test_recalibration_gate_returns_identity_when_holdout_degrades(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import civicsafe.calibration.recalibration as module

    monkeypatch.setattr(module, "ZINBRecalibrator", _FakeRecalibrator)
    y = torch.full((10,), 1.0)
    pi = torch.full((10,), 0.05)
    mu = torch.full((10,), 1.0)
    r = torch.full((10,), 10.0)

    transformed, metrics = recalibrate_and_evaluate(
        y, pi, mu, r, y, pi, mu, r, holdout_fraction=0.3
    )

    assert not metrics["recal_applied"]
    assert metrics["recalibration_gate"] == "identity fallback"
    assert torch.equal(transformed[1], mu)


def test_shared_evaluator_rejects_misaligned_member_weeks() -> None:
    base = {"y": torch.ones(2), "week_idx": torch.tensor([1, 2])}
    shifted = {"y": torch.ones(2), "week_idx": torch.tensor([1, 3])}

    with pytest.raises(RuntimeError, match="misaligned weeks"):
        validate_member_alignment([base, shifted], target_key="y")
