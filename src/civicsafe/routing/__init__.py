"""Advisory safe-route reference using exact Dijkstra shortest paths.

Routing runs on conformal-interval edge costs over a city-scale graph. We use
`DijkstraRouter` (exact, fastest at this scale). `BatchedFrontierRouter` (aliased
as `TsinghuaRouter` for backward compatibility) is a batched-frontier heuristic
that returns identical costs but is not faster and is NOT the Duan et al. (2025)
algorithm -- see routing/tsinghua.py for the honest note.

Public API
----------
.. autosummary::

   RoutingGraph
   Edge
   DijkstraRouter
   BatchedFrontierRouter
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
from civicsafe.routing.feedback_aware import (
    ExposureDisparityAudit,
    ExposureDisparityResult,
    LatentCVaRCost,
    correct_node_intervals,
    correct_node_risk,
)
from civicsafe.routing.graph import Edge, RoutingGraph
from civicsafe.routing.tsinghua import (
    BatchedFrontierRouter,
    DijkstraRouter,
    PathResult,
    TsinghuaRouter,
)

__all__ = [
    "AbstentionError",
    "AbstentionMonitor",
    "AbstentionVerdict",
    "AdvisoryRoutingEngine",
    "BatchedFrontierRouter",
    "DijkstraRouter",
    "DistanceOnlyCost",
    "Edge",
    "ExposureDisparityAudit",
    "ExposureDisparityResult",
    "LatentCVaRCost",
    "ParetoCost",
    "PathResult",
    "RiskOnlyCost",
    "RoutingGraph",
    "SafeRouteResult",
    "TsinghuaRouter",
    "UncertaintyPenalisedCost",
    "correct_node_intervals",
    "correct_node_risk",
]
