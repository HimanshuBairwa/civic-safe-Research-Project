"""Shared post-training inference and ensemble evaluation primitives."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

import torch
from torch import Tensor

from civicsafe.calibration.emos import apply_emos_weights, learn_emos_weights
from civicsafe.training.metrics import crps_zinb
from civicsafe.training.sac_loss import zinb_variance
from civicsafe.utils.checkpointing import resolve_evaluation_checkpoints

logger = logging.getLogger(__name__)


class ModelProtocol(Protocol):
    """Small inference surface shared by both evaluation scripts."""

    def eval(self) -> Any: ...

    def __call__(
        self, features: Tensor, edge_queen: Tensor, edge_knn: Tensor | None
    ) -> dict[str, Tensor]: ...


def resolve_ensemble_checkpoints(
    checkpoint_path: str | Path | None,
    *,
    data_name: str,
    outputs_dir: Path,
) -> list[Path]:
    """Resolve one checkpoint or every seed checkpoint in an evaluation run."""
    return resolve_evaluation_checkpoints(
        checkpoint_path, data_name=data_name, outputs_dir=outputs_dir
    )


@torch.inference_mode()
def rolling_panel_inference(
    model: ModelProtocol,
    *,
    counts: Tensor,
    features: Tensor,
    edge_queen: Tensor,
    edge_knn: Tensor | None,
    target_weeks: Sequence[int],
    window_size: int,
    device: str | torch.device,
) -> dict[str, Tensor]:
    """Run the canonical one-step-ahead panel inference loop."""
    model.eval()
    edge_q = edge_queen.to(device)
    edge_k = edge_knn.to(device) if edge_knn is not None else None
    all_y: list[Tensor] = []
    all_pi: list[Tensor] = []
    all_mu: list[Tensor] = []
    all_r: list[Tensor] = []
    evaluated_weeks: list[int] = []

    for target_week in target_weeks:
        if target_week < window_size or target_week >= counts.shape[1]:
            continue
        x_features = features[
            :, target_week - window_size : target_week, :
        ].to(device)
        x_counts = torch.log1p(
            counts[:, target_week - window_size : target_week, :]
            .float()
            .to(device)
        )
        output = model(torch.cat([x_features, x_counts], dim=-1), edge_q, edge_k)
        all_y.append(counts[:, target_week, :].cpu().float())
        all_pi.append(output["pi"].cpu().float())
        all_mu.append(output["mu"].cpu().float())
        all_r.append(output["r"].cpu().float())
        evaluated_weeks.append(int(target_week))

    if not all_y:
        raise ValueError("No target weeks had sufficient history for evaluation")
    return {
        "y": torch.stack(all_y),
        "pi": torch.stack(all_pi),
        "mu": torch.stack(all_mu),
        "r": torch.stack(all_r),
        "week_idx": torch.tensor(evaluated_weeks, dtype=torch.long),
    }


def select_checkpoint_weight_sets(
    checkpoints: Sequence[Path],
    *,
    requested: str,
    load_model: Callable[[Path, str], ModelProtocol],
    score_model: Callable[[ModelProtocol], float],
) -> tuple[dict[Path, str], dict[Path, dict[str, float]]]:
    """Choose raw or EMA weights independently per seed on validation data."""
    if requested not in ("auto", "raw", "ema"):
        raise ValueError("requested weights must be 'auto', 'raw', or 'ema'")
    if requested != "auto":
        return (
            {checkpoint: requested for checkpoint in checkpoints},
            {checkpoint: {} for checkpoint in checkpoints},
        )

    selected: dict[Path, str] = {}
    diagnostics: dict[Path, dict[str, float]] = {}
    for checkpoint in checkpoints:
        scores: dict[str, float] = {}
        for candidate in ("raw", "ema"):
            try:
                scores[candidate] = float(
                    score_model(load_model(checkpoint, candidate))
                )
            except KeyError:
                continue
        if not scores:
            raise RuntimeError(f"No usable weight set found in {checkpoint}")
        selected[checkpoint] = min(scores, key=scores.__getitem__)
        diagnostics[checkpoint] = scores
    return selected, diagnostics


def validate_member_alignment(
    outputs: Sequence[dict[str, Tensor]],
    *,
    target_key: str,
) -> None:
    """Fail loudly if ensemble members evaluated different cells or weeks."""
    if not outputs:
        raise ValueError("At least one ensemble output is required")
    reference = outputs[0]
    for output in outputs[1:]:
        if not torch.equal(output[target_key], reference[target_key]):
            raise RuntimeError("Ensemble members produced misaligned targets")
        if "week_idx" in reference:
            week_idx = output.get("week_idx")
            if week_idx is None or not torch.equal(
                week_idx, reference["week_idx"]
            ):
                raise RuntimeError("Ensemble members produced misaligned weeks")


def combine_ensemble_outputs(
    calibration_outputs: Sequence[dict[str, Tensor]],
    test_outputs: Sequence[dict[str, Tensor]],
    *,
    target_key: str,
    category_wise: bool = True,
    entropy_lambda: float = 0.005,
    holdout_fraction: float = 0.30,
    min_holdout_improvement: float = 0.0025,
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict[str, Any]]:
    """Learn gated EMOS weights once and apply them consistently to both splits."""
    validate_member_alignment(calibration_outputs, target_key=target_key)
    validate_member_alignment(test_outputs, target_key=target_key)
    if len(calibration_outputs) != len(test_outputs):
        raise ValueError("Calibration and test ensemble sizes do not match")

    if len(calibration_outputs) == 1:
        return (
            dict(calibration_outputs[0]),
            dict(test_outputs[0]),
            {"num_seeds": 1, "method": "single_model"},
        )

    y_cal = calibration_outputs[0][target_key]
    all_pi_cal = [output["pi"] for output in calibration_outputs]
    all_mu_cal = [output["mu"] for output in calibration_outputs]
    all_r_cal = [output["r"] for output in calibration_outputs]
    all_pi_test = [output["pi"] for output in test_outputs]
    all_mu_test = [output["mu"] for output in test_outputs]
    all_r_test = [output["r"] for output in test_outputs]
    emos = learn_emos_weights(
        y_cal,
        all_pi_cal,
        all_mu_cal,
        all_r_cal,
        category_wise=category_wise,
        entropy_lambda=entropy_lambda,
        holdout_fraction=holdout_fraction,
        min_holdout_improvement=min_holdout_improvement,
    )
    applied_weights = emos.get("category_weights") or emos["weights"]
    cal_pi, cal_mu, cal_r = apply_emos_weights(
        applied_weights, all_pi_cal, all_mu_cal, all_r_cal
    )
    test_pi, test_mu, test_r = apply_emos_weights(
        applied_weights, all_pi_test, all_mu_test, all_r_test
    )
    cal_combined = {
        target_key: y_cal,
        "pi": cal_pi,
        "mu": cal_mu,
        "r": cal_r,
    }
    test_combined = {
        target_key: test_outputs[0][target_key],
        "pi": test_pi,
        "mu": test_mu,
        "r": test_r,
    }
    for key in ("week_idx",):
        if key in calibration_outputs[0]:
            cal_combined[key] = calibration_outputs[0][key]
        if key in test_outputs[0]:
            test_combined[key] = test_outputs[0][key]

    equal_weights = [1.0 / len(test_outputs)] * len(test_outputs)
    equal_pi, equal_mu, equal_r = apply_emos_weights(
        equal_weights, all_pi_test, all_mu_test, all_r_test
    )
    y_test = test_outputs[0][target_key].reshape(-1).float()
    per_seed_crps = [
        crps_zinb(
            output[target_key].reshape(-1).float(),
            output["pi"].reshape(-1).float(),
            output["mu"].reshape(-1).float(),
            output["r"].reshape(-1).float(),
        ).mean().item()
        for output in test_outputs
    ]
    point_predictions = torch.stack(
        [(1.0 - output["pi"]) * output["mu"] for output in test_outputs]
    )
    epistemic = point_predictions.var(dim=0).mean().item()
    aleatoric = float(
        sum(
            zinb_variance(output["pi"], output["mu"], output["r"])
            .mean()
            .item()
            for output in test_outputs
        )
        / len(test_outputs)
    )
    metadata: dict[str, Any] = {
        "num_seeds": len(test_outputs),
        "method": "category_conditioned_entropy_regularized_emos",
        "emos_weights": emos["weights"],
        "emos_category_weights": emos.get("category_weights"),
        "emos_fallback_used": emos.get("fallback_used", False),
        "emos_fallback_by_category": emos.get("fallback_by_category", []),
        "emos_holdout_improvement_pct": emos.get("holdout_improvement_pct"),
        "emos_entropy_lambda": entropy_lambda,
        "emos_calibration_crps_equal_weight": emos["initial_crps"],
        "emos_calibration_crps_learned_weight": emos["final_crps"],
        "per_seed_test_crps": per_seed_crps,
        "equal_weight_test_crps": crps_zinb(
            y_test,
            equal_pi.reshape(-1),
            equal_mu.reshape(-1),
            equal_r.reshape(-1),
        ).mean().item(),
        "learned_weight_test_crps": crps_zinb(
            y_test,
            test_pi.reshape(-1),
            test_mu.reshape(-1),
            test_r.reshape(-1),
        ).mean().item(),
        "epistemic_uncertainty": epistemic,
        "aleatoric_uncertainty": aleatoric,
        "epistemic_fraction": epistemic / max(epistemic + aleatoric, 1e-12),
    }
    return cal_combined, test_combined, metadata
