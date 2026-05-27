"""Multi-objective Pareto cost functions for safe routing.

Supports both scalar (single-objective) and Pareto (multi-objective)
cost formulations.  The default ``ParetoCost`` combines physical
distance with conformal risk upper bounds via a convex combination:

    Cost(e) = w_dist × distance(e) + w_risk × upper_bound(e)

By using the conformal ``upper_bound`` (not the point prediction),
the routing is inherently conservative — it plans against the
*worst case within the prediction interval*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from civicsafe.routing.graph import Edge


class CostFunction(Protocol):
    """Protocol for edge cost functions."""

    def __call__(self, edge: Edge) -> float:
        """Compute the scalar cost of traversing an edge."""
        ...


@dataclass(frozen=True)
class ParetoCost:
    """Convex-combination Pareto cost: distance + risk.

    The cost for an edge is:
        ``w_dist × distance + w_risk × risk_upper``

    where ``risk_upper`` is the conformal prediction upper bound.

    Attributes:
        w_dist: Weight for physical distance. Default 0.3.
        w_risk: Weight for predicted risk (upper bound). Default 0.7.
    """

    w_dist: float = 0.3
    w_risk: float = 0.7

    def __call__(self, edge: Edge) -> float:
        """Compute the weighted cost of traversing an edge.

        Args:
            edge: The edge to evaluate.

        Returns:
            Scalar cost combining distance and risk.
        """
        return self.w_dist * edge.distance + self.w_risk * edge.risk_upper


@dataclass(frozen=True)
class DistanceOnlyCost:
    """Baseline: pure distance cost (ignores risk)."""

    def __call__(self, edge: Edge) -> float:
        return edge.distance


@dataclass(frozen=True)
class RiskOnlyCost:
    """Pure risk cost: only considers predicted upper bounds."""

    def __call__(self, edge: Edge) -> float:
        return edge.risk_upper


@dataclass(frozen=True)
class UncertaintyPenalisedCost:
    """Cost that penalises edges with high prediction uncertainty.

    Cost = w_dist × distance + w_risk × risk + w_unc × interval_width

    This penalises routes through areas where the model is *uncertain*,
    not just areas where the model predicts high risk.

    Attributes:
        w_dist: Distance weight.
        w_risk: Risk weight.
        w_unc: Uncertainty (interval width) penalty weight.
    """

    w_dist: float = 0.2
    w_risk: float = 0.5
    w_unc: float = 0.3

    def __call__(self, edge: Edge) -> float:
        return (
            self.w_dist * edge.distance
            + self.w_risk * edge.risk_upper
            + self.w_unc * edge.interval_width
        )
