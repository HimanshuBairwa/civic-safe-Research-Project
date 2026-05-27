"""Advisory safe-route reference using the Tsinghua 2025 SSSP algorithm.

Public API
----------
.. autosummary::

   RoutingGraph
   Edge
   TsinghuaRouter
   DijkstraRouter
   PathResult
   ParetoCost
   DistanceOnlyCost
   RiskOnlyCost
   UncertaintyPenalisedCost
   AbstentionMonitor
   AbstentionError
   AbstentionVerdict
   AdvisoryRoutingEngine
   SafeRouteResult
"""

from civicsafe.routing.abstention import (
    AbstentionError,
    AbstentionMonitor,
    AbstentionVerdict,
)
from civicsafe.routing.cost import (
    DistanceOnlyCost,
    ParetoCost,
    RiskOnlyCost,
    UncertaintyPenalisedCost,
)
from civicsafe.routing.engine import AdvisoryRoutingEngine, SafeRouteResult
from civicsafe.routing.graph import Edge, RoutingGraph
from civicsafe.routing.tsinghua import DijkstraRouter, PathResult, TsinghuaRouter

__all__ = [
    "AbstentionError",
    "AbstentionMonitor",
    "AbstentionVerdict",
    "AdvisoryRoutingEngine",
    "DijkstraRouter",
    "DistanceOnlyCost",
    "Edge",
    "ParetoCost",
    "PathResult",
    "RiskOnlyCost",
    "RoutingGraph",
    "SafeRouteResult",
    "TsinghuaRouter",
    "UncertaintyPenalisedCost",
]
