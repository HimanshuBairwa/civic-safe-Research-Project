"""Audit harness: 7 modular audit components for fairness and calibration.

Public API
----------
.. autosummary::

   AuditBundle
   AuditHarness
   AuditReport
   AuditResult
   CoverageEquityAudit
   IntervalWidthEquityAudit
   PointAccuracyEquityAudit
   CalibrationEquityAudit
   WinklerEquityAudit
   AbstentionEquityAudit
   ReportingBiasSensitivityAudit
   StratificationEngine
   StratConfig
   BootstrapTest
   PermutationTest
   MultipleComparisonCorrector
"""

from civicsafe.audit.bundle import AuditBundle
from civicsafe.audit.components import (
    AbstentionEquityAudit,
    AuditResult,
    CalibrationEquityAudit,
    CoverageEquityAudit,
    IntervalWidthEquityAudit,
    PointAccuracyEquityAudit,
    ReportingBiasSensitivityAudit,
    WinklerEquityAudit,
    default_components,
)
from civicsafe.audit.harness import AuditHarness
from civicsafe.audit.report import AuditReport
from civicsafe.audit.statistical import (
    BootstrapTest,
    MultipleComparisonCorrector,
    PermutationTest,
)
from civicsafe.audit.stratification import StratConfig, StratificationEngine

__all__ = [
    "AuditBundle",
    "AuditHarness",
    "AuditReport",
    "AuditResult",
    "CoverageEquityAudit",
    "IntervalWidthEquityAudit",
    "PointAccuracyEquityAudit",
    "CalibrationEquityAudit",
    "WinklerEquityAudit",
    "AbstentionEquityAudit",
    "ReportingBiasSensitivityAudit",
    "StratificationEngine",
    "StratConfig",
    "BootstrapTest",
    "PermutationTest",
    "MultipleComparisonCorrector",
    "default_components",
]
