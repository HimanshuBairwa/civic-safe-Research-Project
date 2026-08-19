"""Conformal calibration procedures for prediction intervals.

Implements six conformal prediction strategies for ZINB crime-count
forecasting, with rigorous coverage guarantees.

Public API:
    - ``create_calibrator``: Factory function to create a calibrator from config.
    - ``SplitConformalCalibrator``: Standard split conformal (exact marginal coverage).
    - ``WeightedConformalCalibrator``: Temporally-weighted for non-stationary data.
    - ``MondrianConformalCalibrator``: Group-conditional per-group coverage.
    - ``EqualizedCoverageCalibrator``: Regularised equalized coverage.
    - ``ECRCCalibrator``: PAC-style per-group guarantees via Hoeffding bounds.
    - ``AdaptiveTemporalECRCCalibrator``: ACI + ECRC for temporal non-exchangeability.
    - ``compute_all_calibration_metrics``: One-call evaluation of PICP, AIW, Winkler.
"""

from civicsafe.calibration.conformal import (
    AdaptiveTemporalECRCCalibrator,
    ECRCCalibrator,
    EqualizedCoverageCalibrator,
    MondrianConformalCalibrator,
    SplitConformalCalibrator,
    VarianceScaledConformalCalibrator,
    WeightedConformalCalibrator,
    compute_cqr_scores,
    compute_variance_scaled_cqr_scores,
    create_calibrator,
    zinb_predictive_scale,
)
from civicsafe.calibration.ensemble_evaluator import (
    combine_ensemble_outputs,
    resolve_ensemble_checkpoints,
    rolling_panel_inference,
)
from civicsafe.calibration.metrics import (
    average_interval_width,
    compute_all_calibration_metrics,
    conditional_coverage,
    coverage_gap,
    picp,
    winkler_score,
)
from civicsafe.calibration.policies import (
    CalibratorSelection,
    ForecastingGate,
    assess_forecasting_gate,
    select_best_calibrator,
)
from civicsafe.calibration.zinb_distribution import (
    zinb_cdf,
    zinb_cdf_full,
    zinb_ppf,
    zinb_ppf_pair,
)

__all__ = [  # noqa: RUF022 - grouped by public API category
    # Calibrators
    "SplitConformalCalibrator",
    "VarianceScaledConformalCalibrator",
    "WeightedConformalCalibrator",
    "MondrianConformalCalibrator",
    "EqualizedCoverageCalibrator",
    "ECRCCalibrator",
    "AdaptiveTemporalECRCCalibrator",
    "create_calibrator",
    "compute_cqr_scores",
    "compute_variance_scaled_cqr_scores",
    "zinb_predictive_scale",
    # Distribution
    "zinb_cdf",
    "zinb_cdf_full",
    "zinb_ppf",
    "zinb_ppf_pair",
    # Metrics
    "picp",
    "average_interval_width",
    "winkler_score",
    "conditional_coverage",
    "coverage_gap",
    "compute_all_calibration_metrics",
    # Decision policies
    "CalibratorSelection",
    "ForecastingGate",
    "select_best_calibrator",
    "assess_forecasting_gate",
    "combine_ensemble_outputs",
    "resolve_ensemble_checkpoints",
    "rolling_panel_inference",
]
