"""Advisory routing engine — the public-facing facade.

``AdvisoryRoutingEngine`` is the single entry-point for the routing
subsystem.  It wires together the routing graph, Tsinghua SSSP router,
Pareto cost functions, and the abstention monitor into one cohesive
pipeline.

Usage::

    engine = AdvisoryRoutingEngine.from_adjacency(
        edge_index=adj["queen"],
        num_nodes=77,
        upper_bounds=intervals["upper"],
        lower_bounds=intervals["lower"],
    )
    route = engine.safe_route(source=0, target=42)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor

from civicsafe.routing.abstention import AbstentionMonitor, AbstentionVerdict
from civicsafe.routing.cost import CostFunction, ParetoCost
from civicsafe.routing.graph import RoutingGraph
from civicsafe.routing.tsinghua import DijkstraRouter, PathResult, TsinghuaRouter


@dataclass
class SafeRouteResult:
    """Complete result from a safe-route query.

    Attributes:
        path: Ordered list of node indices from source to target.
        total_cost: Total Pareto cost along the path.
        abstention_verdict: Full abstention evaluation details.
        path_result: The raw ``PathResult`` from the routing algorithm.
        algorithm: Which algorithm was used ('tsinghua' or 'dijkstra').
    """

    path: list[int]
    total_cost: float
    abstention_verdict: AbstentionVerdict
    path_result: PathResult
    algorithm: str


class AdvisoryRoutingEngine:
    """Facade wiring graph + Tsinghua router + cost + abstention.

    Args:
        graph: The routing graph with injected predictions.
        cost_fn: Edge cost function (default: ``ParetoCost``).
        abstention_monitor: Uncertainty monitor (default: standard).
        use_dijkstra_fallback: If True, fall back to classic Dijkstra
            if TsinghuaRouter fails. Default True.
    """

    def __init__(
        self,
        graph: RoutingGraph,
        cost_fn: CostFunction | None = None,
        abstention_monitor: AbstentionMonitor | None = None,
        use_dijkstra_fallback: bool = True,
    ) -> None:
        self.graph = graph
        self.cost_fn: CostFunction = cost_fn or ParetoCost()
        self.monitor = abstention_monitor or AbstentionMonitor()
        self.use_dijkstra_fallback = use_dijkstra_fallback

        self.tsinghua = TsinghuaRouter(graph, self.cost_fn)
        self.dijkstra = DijkstraRouter(graph, self.cost_fn)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def safe_route(
        self,
        source: int,
        target: int,
        raise_on_abstention: bool = True,
    ) -> SafeRouteResult:
        """Find the safest route from source to target.

        Uses the Tsinghua SSSP algorithm for pathfinding, then evaluates
        the result through the abstention monitor.

        Args:
            source: Start node index.
            target: Destination node index.
            raise_on_abstention: If True, raises ``AbstentionError``
                when uncertainty is too high.  If False, returns the
                result with ``abstention_verdict.should_abstain = True``.

        Returns:
            ``SafeRouteResult`` with path, cost, and abstention verdict.
        """
        # Primary: Tsinghua router
        path_result = self.tsinghua.shortest_path(source, target)
        algorithm = "tsinghua"

        # Abstention evaluation
        verdict = self.monitor.evaluate(path_result)

        if raise_on_abstention and verdict.should_abstain:
            from civicsafe.routing.abstention import AbstentionError

            raise AbstentionError(
                reason=verdict.reason,
                details={
                    "path": path_result.path,
                    "total_cost": path_result.total_cost,
                    "peak_width": verdict.peak_width,
                    "total_width": verdict.total_width,
                },
            )

        return SafeRouteResult(
            path=path_result.path,
            total_cost=path_result.total_cost,
            abstention_verdict=verdict,
            path_result=path_result,
            algorithm=algorithm,
        )

    def compare_algorithms(
        self,
        source: int,
        target: int,
    ) -> dict[str, Any]:
        """Run both Tsinghua and Dijkstra and compare results.

        Useful for verification and benchmarking.

        Args:
            source: Start node.
            target: End node.

        Returns:
            Comparison dictionary.
        """
        tsinghua_result = self.tsinghua.shortest_path(source, target)
        dijkstra_result = self.dijkstra.shortest_path(source, target)

        cost_match = abs(tsinghua_result.total_cost - dijkstra_result.total_cost) < 1e-6
        path_match = tsinghua_result.path == dijkstra_result.path

        return {
            "tsinghua_path": tsinghua_result.path,
            "dijkstra_path": dijkstra_result.path,
            "tsinghua_cost": round(tsinghua_result.total_cost, 6),
            "dijkstra_cost": round(dijkstra_result.total_cost, 6),
            "cost_match": cost_match,
            "path_match": path_match,
            "tsinghua_settled": tsinghua_result.settled_count,
            "dijkstra_settled": dijkstra_result.settled_count,
            "tsinghua_frontier_reductions": tsinghua_result.frontier_reductions,
        }

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_adjacency(
        cls,
        edge_index: Tensor,
        num_nodes: int,
        upper_bounds: Tensor,
        lower_bounds: Tensor,
        positions: Tensor | None = None,
        cost_fn: CostFunction | None = None,
        peak_threshold: float = 20.0,
        budget_threshold: float = 50.0,
    ) -> "AdvisoryRoutingEngine":
        """Create engine from adjacency + conformal prediction outputs.

        Args:
            edge_index: (2, E) edge index tensor.
            num_nodes: Number of nodes.
            upper_bounds: (N,) conformal upper bounds.
            lower_bounds: (N,) conformal lower bounds.
            positions: Optional (N, 2) node positions.
            cost_fn: Custom cost function (default: ParetoCost).
            peak_threshold: Abstention peak width threshold.
            budget_threshold: Abstention cumulative budget.

        Returns:
            Configured ``AdvisoryRoutingEngine``.
        """
        graph = RoutingGraph.from_edge_index(edge_index, num_nodes, positions)
        graph.inject_predictions(upper_bounds, lower_bounds)

        monitor = AbstentionMonitor(
            peak_threshold=peak_threshold,
            budget_threshold=budget_threshold,
        )

        return cls(
            graph=graph,
            cost_fn=cost_fn,
            abstention_monitor=monitor,
        )
