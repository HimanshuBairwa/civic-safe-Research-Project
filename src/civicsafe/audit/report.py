"""Structured audit report container with serialisation support.

``AuditReport`` collects results from all 7 audit components into a
single object that can be serialised to JSON, converted to summary
tables for a research paper, or exported as LaTeX.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from civicsafe.audit.components import AuditResult


@dataclass
class AuditReport:
    """Complete output from a full equity audit run.

    Attributes:
        results: Mapping of component name → audit result.
        bundle_metadata: Metadata from the ``AuditBundle`` (city, period …).
        timestamp: ISO-format timestamp of when the audit ran.
        statistical_tests: Optional layer of bootstrap / permutation results.
    """

    results: dict[str, AuditResult]
    bundle_metadata: dict[str, Any]
    timestamp: str
    statistical_tests: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Full JSON-serialisable dictionary representation."""
        return {
            "metadata": self.bundle_metadata,
            "timestamp": self.timestamp,
            "audits": {k: v.to_dict() for k, v in self.results.items()},
            "statistical_tests": self.statistical_tests,
        }

    def to_json(self, path: Path | str) -> None:
        """Write the full report to a JSON file.

        Args:
            path: Destination file path.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, default=str)

    # ------------------------------------------------------------------
    # Paper-ready tables
    # ------------------------------------------------------------------

    def summary_table(self) -> dict[str, dict[str, float]]:
        """Summary suitable for *Table 1* in a research paper.

        Returns:
            ``{component_name: overall_metrics}``.
        """
        return {
            name: result.overall_metrics
            for name, result in self.results.items()
        }

    def disparity_table(self) -> dict[str, dict[str, float]]:
        """Disparity ratios suitable for *Table 2* in a research paper.

        Returns:
            ``{component_name: disparity_metrics}``.
        """
        return {
            name: result.disparity_metrics
            for name, result in self.results.items()
        }

    def pass_fail_summary(self) -> dict[str, bool]:
        """Traffic-light summary: which components passed their thresholds.

        Returns:
            ``{component_name: True/False}``.
        """
        return {
            name: result.passes_threshold
            for name, result in self.results.items()
        }
