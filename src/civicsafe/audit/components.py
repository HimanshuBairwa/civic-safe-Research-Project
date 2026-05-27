"""Seven modular audit components for equity and sensitivity evaluation.

Each component follows the Strategy pattern via ``_BaseAuditComponent``:
    1. CoverageEquityAudit      — per-group prediction interval coverage
    2. IntervalWidthEquityAudit — per-group average interval width
    3. PointAccuracyEquityAudit — per-group MAE / RMSE / CRPS
    4. CalibrationEquityAudit   — per-group Brier score for zero-inflation
    5. WinklerEquityAudit       — per-group Winkler interval scores
    6. AbstentionEquityAudit    — equity of model abstention rates
    7. ReportingBiasSensitivityAudit — binomial thinning sensitivity sweep

All components accept an ``AuditBundle`` and return an ``AuditResult``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor


# ===================================================================
# Result Container
# ===================================================================


@dataclass
class AuditResult:
    """Structured output from a single audit component.

    Attributes:
        component_name: Human-readable name of the audit component.
        overall_metrics: Global (population-level) metric values.
        per_group_metrics: ``{group_label: {metric_name: value}}``.
        disparity_metrics: Summary disparity stats (max_ratio, max_diff, cv).
        passes_threshold: Whether disparity is within acceptable limits.
        metadata: Extra information (e.g. which strata feature was used).
    """

    component_name: str
    overall_metrics: dict[str, float]
    per_group_metrics: dict[str, dict[str, float]]
    disparity_metrics: dict[str, float]
    passes_threshold: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flatten to JSON-serialisable dictionary."""
        return {
            "component": self.component_name,
            "overall": self.overall_metrics,
            "by_group": self.per_group_metrics,
            "disparity": self.disparity_metrics,
            "passes": self.passes_threshold,
            "metadata": self.metadata,
        }


# ===================================================================
# Base Component (Abstract)
# ===================================================================


class _BaseAuditComponent(ABC):
    """Abstract base for all audit components (Strategy pattern)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable component name."""
        ...

    @abstractmethod
    def evaluate(
        self,
        y_true: Tensor,
        y_pred: Tensor,
        lower: Tensor,
        upper: Tensor,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        groups: Tensor,
        alpha: float,
    ) -> AuditResult:
        """Run this audit and return structured results."""
        ...


# ===================================================================
# Internal helpers
# ===================================================================


def _per_group_metric(
    groups: Tensor,
    metric_fn: Callable[..., float],
    *tensors: Tensor,
) -> dict[str, float]:
    """Compute ``metric_fn`` for each unique group label.

    Args:
        groups: (N,) integer group labels.
        metric_fn: Callable accepting sliced tensors and returning a float.
        *tensors: Tensors to slice by group membership.

    Returns:
        ``{str(group_id): metric_value}``
    """
    unique_groups = torch.unique(groups)
    results: dict[str, float] = {}
    for g in unique_groups:
        mask = groups == g
        sliced = tuple(t[mask] for t in tensors)
        results[str(g.item())] = metric_fn(*sliced)
    return results


def _disparity_stats(values: dict[str, float]) -> dict[str, float]:
    """Compute disparity summary from per-group metric values.

    Returns:
        max_ratio: worst / best group value ratio (≥ 1.0).
        max_diff: absolute difference between max and min.
        cv: coefficient of variation across groups.
    """
    vals = list(values.values())
    if not vals or len(vals) < 2:
        return {"max_ratio": 1.0, "max_diff": 0.0, "cv": 0.0}

    arr = np.array(vals)
    vmin = float(arr.min())
    vmax = float(arr.max())

    if vmin > 0:
        ratio = vmax / vmin
    elif vmax > 0:
        ratio = float("inf")
    else:
        ratio = 1.0

    mean = float(arr.mean())
    cv = float(arr.std() / mean) if mean > 0 else 0.0

    return {
        "max_ratio": round(ratio, 4),
        "max_diff": round(vmax - vmin, 6),
        "cv": round(cv, 4),
    }


# ===================================================================
# Component 1: Coverage Equity
# ===================================================================


class CoverageEquityAudit(_BaseAuditComponent):
    """Audit per-group prediction interval coverage (PICP).

    Tests whether ``PICP(group_g) ≈ 1 - α`` for all groups.
    """

    def __init__(self, max_coverage_gap: float = 0.05) -> None:
        self.max_coverage_gap = max_coverage_gap

    @property
    def name(self) -> str:
        return "coverage_equity"

    def evaluate(
        self,
        y_true: Tensor,
        y_pred: Tensor,
        lower: Tensor,
        upper: Tensor,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        groups: Tensor,
        alpha: float,
    ) -> AuditResult:
        def _picp(*ts: Tensor) -> float:
            y, lo, hi = ts[0], ts[1], ts[2]
            covered = ((y >= lo) & (y <= hi)).float()
            return float(covered.mean().item())

        overall = _picp(y_true, lower, upper)
        per_group = _per_group_metric(groups, _picp, y_true, lower, upper)
        disp = _disparity_stats(per_group)

        target = 1.0 - alpha
        gap_from_target = {
            k: abs(v - target) for k, v in per_group.items()
        }
        max_gap = max(gap_from_target.values()) if gap_from_target else 0.0
        disp["max_gap_from_target"] = round(max_gap, 6)

        return AuditResult(
            component_name=self.name,
            overall_metrics={"picp": round(overall, 6), "target": target},
            per_group_metrics={k: {"picp": round(v, 6)} for k, v in per_group.items()},
            disparity_metrics=disp,
            passes_threshold=max_gap <= self.max_coverage_gap,
            metadata={"alpha": alpha, "threshold": self.max_coverage_gap},
        )


# ===================================================================
# Component 2: Interval Width Equity
# ===================================================================


class IntervalWidthEquityAudit(_BaseAuditComponent):
    """Audit per-group average prediction interval width.

    Checks whether intervals are systematically wider for disadvantaged groups.
    """

    def __init__(self, max_width_ratio: float = 2.0) -> None:
        self.max_width_ratio = max_width_ratio

    @property
    def name(self) -> str:
        return "interval_width_equity"

    def evaluate(
        self,
        y_true: Tensor,
        y_pred: Tensor,
        lower: Tensor,
        upper: Tensor,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        groups: Tensor,
        alpha: float,
    ) -> AuditResult:
        def _aiw(*ts: Tensor) -> float:
            lo, hi = ts[0], ts[1]
            return float((hi - lo).float().mean().item())

        overall = _aiw(lower, upper)
        per_group = _per_group_metric(groups, _aiw, lower, upper)
        disp = _disparity_stats(per_group)

        return AuditResult(
            component_name=self.name,
            overall_metrics={"aiw": round(overall, 4)},
            per_group_metrics={k: {"aiw": round(v, 4)} for k, v in per_group.items()},
            disparity_metrics=disp,
            passes_threshold=disp["max_ratio"] <= self.max_width_ratio,
            metadata={"threshold": self.max_width_ratio},
        )


# ===================================================================
# Component 3: Point Accuracy Equity
# ===================================================================


class PointAccuracyEquityAudit(_BaseAuditComponent):
    """Audit per-group point-prediction error (MAE, RMSE).

    Ensures error rates are equitable across demographic strata.
    """

    def __init__(self, max_error_ratio: float = 2.0) -> None:
        self.max_error_ratio = max_error_ratio

    @property
    def name(self) -> str:
        return "point_accuracy_equity"

    def evaluate(
        self,
        y_true: Tensor,
        y_pred: Tensor,
        lower: Tensor,
        upper: Tensor,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        groups: Tensor,
        alpha: float,
    ) -> AuditResult:
        def _mae(*ts: Tensor) -> float:
            y, yhat = ts[0], ts[1]
            return float((y.float() - yhat.float()).abs().mean().item())

        def _rmse(*ts: Tensor) -> float:
            y, yhat = ts[0], ts[1]
            return float(((y.float() - yhat.float()) ** 2).mean().sqrt().item())

        overall_mae = _mae(y_true, y_pred)
        overall_rmse = _rmse(y_true, y_pred)
        per_group_mae = _per_group_metric(groups, _mae, y_true, y_pred)
        per_group_rmse = _per_group_metric(groups, _rmse, y_true, y_pred)
        disp = _disparity_stats(per_group_mae)

        per_group_combined = {
            k: {"mae": round(per_group_mae[k], 4), "rmse": round(per_group_rmse[k], 4)}
            for k in per_group_mae
        }

        return AuditResult(
            component_name=self.name,
            overall_metrics={
                "mae": round(overall_mae, 4),
                "rmse": round(overall_rmse, 4),
            },
            per_group_metrics=per_group_combined,
            disparity_metrics=disp,
            passes_threshold=disp["max_ratio"] <= self.max_error_ratio,
            metadata={"threshold": self.max_error_ratio},
        )


# ===================================================================
# Component 4: Calibration Equity (Brier Score)
# ===================================================================


class CalibrationEquityAudit(_BaseAuditComponent):
    """Audit per-group Brier score for zero-inflation calibration.

    Brier = mean((π_pred − I(y=0))²).  Lower is better.
    """

    def __init__(self, max_brier_ratio: float = 2.0) -> None:
        self.max_brier_ratio = max_brier_ratio

    @property
    def name(self) -> str:
        return "calibration_equity"

    def evaluate(
        self,
        y_true: Tensor,
        y_pred: Tensor,
        lower: Tensor,
        upper: Tensor,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        groups: Tensor,
        alpha: float,
    ) -> AuditResult:
        def _brier(*ts: Tensor) -> float:
            y, p = ts[0], ts[1]
            is_zero = (y == 0).float()
            return float(((p.clamp(0.0, 1.0) - is_zero) ** 2).mean().item())

        overall = _brier(y_true, pi)
        per_group = _per_group_metric(groups, _brier, y_true, pi)
        disp = _disparity_stats(per_group)

        return AuditResult(
            component_name=self.name,
            overall_metrics={"brier": round(overall, 6)},
            per_group_metrics={k: {"brier": round(v, 6)} for k, v in per_group.items()},
            disparity_metrics=disp,
            passes_threshold=disp["max_ratio"] <= self.max_brier_ratio,
            metadata={"threshold": self.max_brier_ratio},
        )


# ===================================================================
# Component 5: Winkler Equity
# ===================================================================


class WinklerEquityAudit(_BaseAuditComponent):
    """Audit per-group Winkler interval scores.

    Winkler penalises wide intervals AND adds massive penalty for
    observations falling outside the bounds.
    """

    def __init__(self, max_winkler_ratio: float = 2.0) -> None:
        self.max_winkler_ratio = max_winkler_ratio

    @property
    def name(self) -> str:
        return "winkler_equity"

    def evaluate(
        self,
        y_true: Tensor,
        y_pred: Tensor,
        lower: Tensor,
        upper: Tensor,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        groups: Tensor,
        alpha: float,
    ) -> AuditResult:
        def _winkler(*ts: Tensor) -> float:
            y, lo, hi = ts[0].float(), ts[1].float(), ts[2].float()
            width = hi - lo
            penalty_below = (2.0 / alpha) * torch.clamp(lo - y, min=0.0)
            penalty_above = (2.0 / alpha) * torch.clamp(y - hi, min=0.0)
            score = width + penalty_below + penalty_above
            return float(score.mean().item())

        overall = _winkler(y_true, lower, upper)
        per_group = _per_group_metric(groups, _winkler, y_true, lower, upper)
        disp = _disparity_stats(per_group)

        return AuditResult(
            component_name=self.name,
            overall_metrics={"winkler": round(overall, 4)},
            per_group_metrics={k: {"winkler": round(v, 4)} for k, v in per_group.items()},
            disparity_metrics=disp,
            passes_threshold=disp["max_ratio"] <= self.max_winkler_ratio,
            metadata={"alpha": alpha, "threshold": self.max_winkler_ratio},
        )


# ===================================================================
# Component 6: Abstention Equity
# ===================================================================


class AbstentionEquityAudit(_BaseAuditComponent):
    """Audit equity of model abstention rates across groups.

    The model abstains when prediction interval width exceeds a threshold,
    indicating extreme uncertainty.  This checks whether abstention
    disproportionately affects specific demographic groups.
    """

    def __init__(
        self,
        width_threshold: float | None = None,
        max_abstention_ratio: float = 2.0,
    ) -> None:
        self.width_threshold = width_threshold
        self.max_abstention_ratio = max_abstention_ratio

    @property
    def name(self) -> str:
        return "abstention_equity"

    def evaluate(
        self,
        y_true: Tensor,
        y_pred: Tensor,
        lower: Tensor,
        upper: Tensor,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        groups: Tensor,
        alpha: float,
    ) -> AuditResult:
        widths = (upper - lower).float()
        threshold = self.width_threshold
        if threshold is None:
            # Auto: abstain if width > 3× median width
            threshold = float(widths.median().item()) * 3.0

        abstains = (widths > threshold).float()
        overall_rate = float(abstains.mean().item())

        def _abstention_rate(*ts: Tensor) -> float:
            return float(ts[0].mean().item())

        per_group = _per_group_metric(groups, _abstention_rate, abstains)
        disp = _disparity_stats(per_group)

        return AuditResult(
            component_name=self.name,
            overall_metrics={
                "abstention_rate": round(overall_rate, 6),
                "width_threshold": round(threshold, 4),
            },
            per_group_metrics={
                k: {"abstention_rate": round(v, 6)} for k, v in per_group.items()
            },
            disparity_metrics=disp,
            passes_threshold=disp["max_ratio"] <= self.max_abstention_ratio,
            metadata={"threshold": self.max_abstention_ratio},
        )


# ===================================================================
# Component 7: Reporting Bias Sensitivity
# ===================================================================


class ReportingBiasSensitivityAudit(_BaseAuditComponent):
    """Sensitivity analysis for crime reporting bias via binomial thinning.

    Simulates ``Y_obs ~ Binomial(Y_true, p_report)`` for a sweep of
    reporting rates, and measures how coverage and error metrics degrade
    differentially across demographic groups.

    Based on INAR binomial thinning (Weiß 2008) and BJS/NCVS empirical
    reporting rates (violent: ~47%, property: ~34%, drug: <10%).
    """

    def __init__(
        self,
        reporting_rates: list[float] | None = None,
        seed: int = 42,
    ) -> None:
        self.reporting_rates = reporting_rates or [
            0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        ]
        self.seed = seed

    @property
    def name(self) -> str:
        return "reporting_bias_sensitivity"

    def _apply_thinning(self, y_true: Tensor, p_report: float) -> Tensor:
        """Apply binomial thinning: Y_obs[i] ~ Binomial(y_true[i], p_report).

        Args:
            y_true: (N,) integer ground-truth counts.
            p_report: Reporting probability in (0, 1].

        Returns:
            Thinned counts, same shape as y_true.
        """
        if p_report >= 1.0:
            return y_true.clone()

        gen = torch.Generator(device=y_true.device).manual_seed(self.seed)
        y_long = y_true.long()
        # For each count, draw Binomial(count, p_report)
        # Use: sum of count Bernoulli(p_report) trials
        thinned = torch.zeros_like(y_long)
        max_count = int(y_long.max().item())
        if max_count > 0:
            # Vectorised: create (N, max_count) Bernoulli matrix
            probs = torch.full(
                (y_long.shape[0], max_count),
                p_report,
                device=y_true.device,
            )
            bernoulli_draws = torch.bernoulli(probs, generator=gen)
            # Mask out draws beyond each element's count
            counts_expanded = y_long.unsqueeze(-1)  # (N, 1)
            indices = torch.arange(max_count, device=y_true.device).unsqueeze(0)
            valid_mask = indices < counts_expanded  # (N, max_count)
            thinned = (bernoulli_draws * valid_mask.float()).sum(dim=-1).long()

        return thinned

    def evaluate(
        self,
        y_true: Tensor,
        y_pred: Tensor,
        lower: Tensor,
        upper: Tensor,
        pi: Tensor,
        mu: Tensor,
        r: Tensor,
        groups: Tensor,
        alpha: float,
    ) -> AuditResult:
        sweep_results: dict[str, dict[str, float]] = {}

        for rate in self.reporting_rates:
            y_thinned = self._apply_thinning(y_true, rate)

            # Overall coverage on thinned data
            covered = ((y_thinned >= lower) & (y_thinned <= upper)).float()
            picp_overall = float(covered.mean().item())

            # Overall MAE on thinned data
            mae_overall = float(
                (y_thinned.float() - y_pred.float()).abs().mean().item()
            )

            # Per-group coverage
            unique_g = torch.unique(groups)
            max_gap = 0.0
            for g in unique_g:
                mask = groups == g
                g_covered = covered[mask].mean().item()
                gap = abs(float(g_covered) - (1.0 - alpha))
                max_gap = max(max_gap, gap)

            sweep_results[f"p={rate:.1f}"] = {
                "picp": round(picp_overall, 4),
                "mae": round(mae_overall, 4),
                "max_coverage_gap": round(max_gap, 4),
            }

        # Check if coverage degrades beyond threshold at lowest reporting rate
        lowest_rate_key = f"p={self.reporting_rates[0]:.1f}"
        full_rate_key = f"p={self.reporting_rates[-1]:.1f}"

        if lowest_rate_key in sweep_results and full_rate_key in sweep_results:
            degradation = (
                sweep_results[full_rate_key]["picp"]
                - sweep_results[lowest_rate_key]["picp"]
            )
        else:
            degradation = 0.0

        return AuditResult(
            component_name=self.name,
            overall_metrics={
                "n_sweep_points": float(len(self.reporting_rates)),
                "coverage_degradation": round(degradation, 4),
            },
            per_group_metrics=sweep_results,
            disparity_metrics={"coverage_degradation": round(degradation, 4)},
            passes_threshold=degradation < 0.15,
            metadata={
                "reporting_rates": self.reporting_rates,
                "seed": self.seed,
            },
        )


# ===================================================================
# Default component factory
# ===================================================================


def default_components() -> list[_BaseAuditComponent]:
    """Create the standard set of 7 audit components with default thresholds."""
    return [
        CoverageEquityAudit(),
        IntervalWidthEquityAudit(),
        PointAccuracyEquityAudit(),
        CalibrationEquityAudit(),
        WinklerEquityAudit(),
        AbstentionEquityAudit(),
        ReportingBiasSensitivityAudit(),
    ]
