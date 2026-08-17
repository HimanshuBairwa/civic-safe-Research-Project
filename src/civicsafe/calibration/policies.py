"""Decision policies for calibration and forecast-claim reporting.

Keeping these policies independent of the evaluation script prevents console,
JSON, and paper reports from applying subtly different decision rules.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

DEFAULT_COVERAGE_TOLERANCE = 0.005
DEFAULT_MAX_DISPARITY = 0.030
DEFAULT_MAX_ABSTENTION = 0.01
DEFAULT_SIGNIFICANCE_LEVEL = 0.05


def _finite_float(value: Any) -> float | None:
    """Return *value* as a finite float, or ``None`` when unavailable."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@dataclass(frozen=True)
class CalibratorSelection:
    """Outcome of constrained calibrator selection."""

    selected_method: str | None
    eligible_methods: tuple[str, ...]
    fallback_used: bool
    selection_rule: str
    coverage_floor: float
    max_disparity: float
    max_abstention: float
    rejected_reasons: dict[str, tuple[str, ...]]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "selected_method": self.selected_method,
            "eligible_methods": list(self.eligible_methods),
            "fallback_used": self.fallback_used,
            "selection_rule": self.selection_rule,
            "coverage_floor": self.coverage_floor,
            "max_disparity": self.max_disparity,
            "max_abstention": self.max_abstention,
            "rejected_reasons": {
                method: list(reasons)
                for method, reasons in self.rejected_reasons.items()
            },
        }


def select_best_calibrator(
    coverage_results: Mapping[str, Any],
    *,
    alpha: float,
    coverage_tolerance: float = DEFAULT_COVERAGE_TOLERANCE,
    max_disparity: float = DEFAULT_MAX_DISPARITY,
    max_abstention: float = DEFAULT_MAX_ABSTENTION,
) -> CalibratorSelection:
    """Select the narrowest calibrator satisfying all quality constraints.

    The primary pool requires coverage within the pre-specified lower margin,
    demographic disparity at or below the fairness ceiling, at most one percent
    abstention, finite coverage/width, and no explicit ``INELIGIBLE`` status.
    If that pool is empty, the narrowest fundamentally comparable method is
    returned as an explicitly labelled fallback.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if coverage_tolerance < 0.0:
        raise ValueError("coverage_tolerance must be non-negative")
    if max_disparity < 0.0:
        raise ValueError("max_disparity must be non-negative")
    if not 0.0 <= max_abstention <= 1.0:
        raise ValueError("max_abstention must be in [0, 1]")

    coverage_floor = (1.0 - alpha) - coverage_tolerance
    comparable: dict[str, tuple[float, float]] = {}
    eligible: dict[str, float] = {}
    rejected: dict[str, tuple[str, ...]] = {}
    selected: str | None

    for method, raw_metrics in coverage_results.items():
        if not isinstance(raw_metrics, Mapping):
            continue

        coverage = _finite_float(raw_metrics.get("marginal_coverage"))
        width = _finite_float(raw_metrics.get("mean_width"))
        disparity = _finite_float(raw_metrics.get("coverage_disparity"))
        abstention = _finite_float(raw_metrics.get("abstention_rate", 0.0))
        status = str(raw_metrics.get("status", "")).strip().upper()

        fundamental_reasons: list[str] = []
        if status.startswith("INELIGIBLE"):
            fundamental_reasons.append("status is INELIGIBLE")
        if coverage is None:
            fundamental_reasons.append("coverage is not finite")
        if width is None:
            fundamental_reasons.append("width is not finite")
        if abstention is None:
            fundamental_reasons.append("abstention is not finite")
        elif abstention > max_abstention:
            fundamental_reasons.append(
                f"abstention {abstention:.6f} exceeds {max_abstention:.6f}"
            )

        if fundamental_reasons:
            rejected[method] = tuple(fundamental_reasons)
            continue

        assert coverage is not None
        assert width is not None
        comparable[method] = (coverage, width)

        constraint_reasons: list[str] = []
        if coverage < coverage_floor:
            constraint_reasons.append(
                f"coverage {coverage:.6f} is below {coverage_floor:.6f}"
            )
        if disparity is None:
            constraint_reasons.append("demographic disparity is not finite")
        elif disparity > max_disparity:
            constraint_reasons.append(
                f"demographic disparity {disparity:.6f} exceeds "
                f"{max_disparity:.6f}"
            )

        if constraint_reasons:
            rejected[method] = tuple(constraint_reasons)
        else:
            eligible[method] = width

    if eligible:
        selected = min(eligible, key=lambda method: (eligible[method], method))
        return CalibratorSelection(
            selected_method=selected,
            eligible_methods=tuple(sorted(eligible)),
            fallback_used=False,
            selection_rule=(
                "minimum width subject to coverage, demographic disparity, "
                "abstention, and status constraints"
            ),
            coverage_floor=coverage_floor,
            max_disparity=max_disparity,
            max_abstention=max_abstention,
            rejected_reasons=rejected,
        )

    if comparable:
        selected = min(
            comparable, key=lambda method: (comparable[method][1], method)
        )
        rule = (
            "FALLBACK: minimum width among comparable methods; no method "
            "satisfied both coverage and demographic-disparity constraints"
        )
    else:
        selected = None
        rule = "NO SELECTION: no fundamentally comparable calibrator"

    return CalibratorSelection(
        selected_method=selected,
        eligible_methods=(),
        fallback_used=selected is not None,
        selection_rule=rule,
        coverage_floor=coverage_floor,
        max_disparity=max_disparity,
        max_abstention=max_abstention,
        rejected_reasons=rejected,
    )


@dataclass(frozen=True)
class ForecastingGate:
    """Outcome of the CRPSS-plus-significance forecasting claim gate."""

    passed: bool
    crpss_ha: float | None
    crpss_ha_positive: bool
    statistically_significant: bool
    significance_level: float
    dm_stat: float | None
    dm_p_value: float | None
    dm_ci: tuple[float, float] | None
    bootstrap_p_value: float | None
    bootstrap_ci: tuple[float, float] | None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "passed": self.passed,
            "rule": (
                "CRPSS vs rolling HA > 0 and "
                f"(DM p < {self.significance_level:g} or "
                f"block-bootstrap p < {self.significance_level:g})"
            ),
            "crpss_ha": self.crpss_ha,
            "crpss_ha_positive": self.crpss_ha_positive,
            "statistically_significant": self.statistically_significant,
            "significance_level": self.significance_level,
            "dm_stat": self.dm_stat,
            "dm_p_value": self.dm_p_value,
            "dm_ci": list(self.dm_ci) if self.dm_ci is not None else None,
            "bootstrap_p_value": self.bootstrap_p_value,
            "bootstrap_ci": (
                list(self.bootstrap_ci)
                if self.bootstrap_ci is not None
                else None
            ),
        }


def _confidence_interval(result: Mapping[str, Any]) -> tuple[float, float] | None:
    lower = _finite_float(result.get("ci_lower"))
    upper = _finite_float(result.get("ci_upper"))
    if lower is None or upper is None:
        return None
    return (lower, upper)


def assess_forecasting_gate(
    crpss_ha: Any,
    significance_results: Mapping[str, Any],
    *,
    significance_level: float = DEFAULT_SIGNIFICANCE_LEVEL,
) -> ForecastingGate:
    """Validate a forecasting claim using skill and inferential evidence."""
    if not 0.0 < significance_level < 1.0:
        raise ValueError("significance_level must be in (0, 1)")

    dm_raw = significance_results.get("dm", {})
    bootstrap_raw = significance_results.get("bootstrap", {})
    dm = dm_raw if isinstance(dm_raw, Mapping) else {}
    bootstrap = bootstrap_raw if isinstance(bootstrap_raw, Mapping) else {}

    skill = _finite_float(crpss_ha)
    dm_p = _finite_float(dm.get("p_value"))
    bootstrap_p = _finite_float(bootstrap.get("p_value"))
    positive = skill is not None and skill > 0.0
    significant = (dm_p is not None and dm_p < significance_level) or (
        bootstrap_p is not None and bootstrap_p < significance_level
    )

    return ForecastingGate(
        passed=positive and significant,
        crpss_ha=skill,
        crpss_ha_positive=positive,
        statistically_significant=significant,
        significance_level=significance_level,
        dm_stat=_finite_float(dm.get("dm_stat")),
        dm_p_value=dm_p,
        dm_ci=_confidence_interval(dm),
        bootstrap_p_value=bootstrap_p,
        bootstrap_ci=_confidence_interval(bootstrap),
    )
