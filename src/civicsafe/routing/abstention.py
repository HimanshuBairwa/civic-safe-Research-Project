"""Audited abstention monitor for routing decisions.

Implements CIVIC-SAFE's core safety principle: when predictive
uncertainty is too high to guarantee a safe route, the system
**refuses to recommend a path** rather than risk harm.

Two abstention criteria:
1. **Peak uncertainty**: Any single edge on the path has an interval
   width exceeding a calibrated threshold.
2. **Cumulative uncertainty**: The total uncertainty along the path
   (sum of interval widths) exceeds a budget.

This is a critical ethical guardrail: advisory systems MUST know
when they don't know enough to advise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from civicsafe.routing.graph import Edge
from civicsafe.routing.tsinghua import PathResult


class AbstentionError(Exception):
    """Raised when the routing engine refuses to recommend a path.

    This is not a software error — it is an *intentional safety mechanism*.
    The system is saying: "I am not confident enough to advise this route."
    """

    def __init__(
        self,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


@dataclass
class AbstentionVerdict:
    """Result of an abstention check.

    Attributes:
        should_abstain: Whether the system should refuse to recommend.
        reason: Human-readable explanation.
        peak_width: Maximum interval width along the path.
        total_width: Sum of interval widths along the path.
        mean_width: Average interval width along the path.
        peak_threshold: The threshold that was checked against.
        budget_threshold: The cumulative budget threshold.
        flagged_edges: Edges that triggered abstention.
    """

    should_abstain: bool
    reason: str
    peak_width: float
    total_width: float
    mean_width: float
    peak_threshold: float
    budget_threshold: float
    flagged_edges: list[Edge]


class AbstentionMonitor:
    """Monitors routing paths for excessive prediction uncertainty.

    Args:
        peak_threshold: Maximum allowable interval width for any single
            edge.  If any edge exceeds this, the path is rejected.
            Default is 20.0 (i.e., the model predicts a range of 20+
            crime events — too uncertain to advise safely).
        budget_threshold: Maximum allowable cumulative uncertainty
            (sum of interval widths) along the entire path.
            Default is 50.0.
        min_edges_for_budget: Minimum path length before the budget
            rule applies (short paths get a pass on cumulative checks).
    """

    def __init__(
        self,
        peak_threshold: float = 20.0,
        budget_threshold: float = 50.0,
        min_edges_for_budget: int = 2,
    ) -> None:
        self.peak_threshold = peak_threshold
        self.budget_threshold = budget_threshold
        self.min_edges_for_budget = min_edges_for_budget

    def evaluate(self, result: PathResult) -> AbstentionVerdict:
        """Evaluate whether a path should be recommended or rejected.

        Args:
            result: The ``PathResult`` from the routing engine.

        Returns:
            ``AbstentionVerdict`` with the decision and diagnostics.
        """
        if not result.edges:
            return AbstentionVerdict(
                should_abstain=False,
                reason="Empty path (source == target).",
                peak_width=0.0,
                total_width=0.0,
                mean_width=0.0,
                peak_threshold=self.peak_threshold,
                budget_threshold=self.budget_threshold,
                flagged_edges=[],
            )

        widths = [e.interval_width for e in result.edges]
        peak = max(widths)
        total = sum(widths)
        mean = total / len(widths)

        # Check 1: Peak uncertainty
        flagged: list[Edge] = []
        if peak > self.peak_threshold:
            flagged = [e for e in result.edges if e.interval_width > self.peak_threshold]
            return AbstentionVerdict(
                should_abstain=True,
                reason=(
                    f"Peak interval width ({peak:.2f}) exceeds threshold "
                    f"({self.peak_threshold:.2f}). Prediction uncertainty "
                    f"is too high to guarantee safety."
                ),
                peak_width=peak,
                total_width=total,
                mean_width=mean,
                peak_threshold=self.peak_threshold,
                budget_threshold=self.budget_threshold,
                flagged_edges=flagged,
            )

        # Check 2: Cumulative uncertainty (only for longer paths)
        if (
            len(result.edges) >= self.min_edges_for_budget
            and total > self.budget_threshold
        ):
            return AbstentionVerdict(
                should_abstain=True,
                reason=(
                    f"Cumulative uncertainty ({total:.2f}) exceeds budget "
                    f"({self.budget_threshold:.2f}). Route traverses too "
                    f"many uncertain areas."
                ),
                peak_width=peak,
                total_width=total,
                mean_width=mean,
                peak_threshold=self.peak_threshold,
                budget_threshold=self.budget_threshold,
                flagged_edges=[],
            )

        # Path is safe to recommend
        return AbstentionVerdict(
            should_abstain=False,
            reason="Path passes all uncertainty checks.",
            peak_width=peak,
            total_width=total,
            mean_width=mean,
            peak_threshold=self.peak_threshold,
            budget_threshold=self.budget_threshold,
            flagged_edges=[],
        )

    def check_or_raise(self, result: PathResult) -> AbstentionVerdict:
        """Evaluate and raise ``AbstentionError`` if unsafe.

        Args:
            result: The ``PathResult`` to check.

        Returns:
            ``AbstentionVerdict`` (only if the path is safe).

        Raises:
            AbstentionError: If the path fails uncertainty checks.
        """
        verdict = self.evaluate(result)
        if verdict.should_abstain:
            raise AbstentionError(
                reason=verdict.reason,
                details={
                    "peak_width": verdict.peak_width,
                    "total_width": verdict.total_width,
                    "mean_width": verdict.mean_width,
                    "path": result.path,
                },
            )
        return verdict
