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
    WeightedConformalCalibrator,
    compute_cqr_scores,
    create_calibrator,
)
from civicsafe.calibration.metrics import (
    average_interval_width,
    compute_all_calibration_metrics,
    conditional_coverage,
    coverage_gap,
    picp,
    winkler_score,
)
from civicsafe.calibration.zinb_distribution import (
    zinb_cdf,
    zinb_cdf_full,
    zinb_ppf,
    zinb_ppf_pair,
)

__all__ = [
    # Calibrators
    "SplitConformalCalibrator",
    "WeightedConformalCalibrator",
    "MondrianConformalCalibrator",
    "EqualizedCoverageCalibrator",
    "ECRCCalibrator",
    "AdaptiveTemporalECRCCalibrator",
    "create_calibrator",
    "compute_cqr_scores",
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
]
