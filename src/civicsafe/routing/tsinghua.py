"""Shortest-path routers for advisory safe routing.

We route on conformal-interval edge costs over a city-scale graph (~100 nodes).
The correct, fastest tool at this scale is **exact Dijkstra** (`DijkstraRouter`),
which we use throughout.

`BatchedFrontierRouter` (kept for backward compatibility, exported also under the
legacy name `TsinghuaRouter`) is a Bellman-Ford-seeded, batched-frontier
shortest-path heuristic. It returns the same shortest-path costs as Dijkstra on
our graphs (verified in tests), but it is NOT the Duan et al. (2025) algorithm
and it is NOT faster than Dijkstra -- it re-sorts the frontier each iteration, so
it does not "break the sorting barrier." Prefer `DijkstraRouter`.

HONEST NOTE ON PRIOR CLAIMS. Earlier versions described this file as an
implementation of the Duan, Mao, Mao, Shu & Yin (2025, STOC best paper) SSSP
algorithm that achieves O(m log^{2/3} n). That was incorrect: the real algorithm
uses a recursive bounded-multi-source routine (BMSSP) with a pivot-finding step
and a block-based partial-sorting structure that AVOIDS sorting the frontier;
this code does the opposite (it sorts the frontier). Faithful implementations of
Duan et al. are also 3-25x SLOWER than Dijkstra at any practical graph size (the
theoretical crossover is ~10^60 vertices), so there is no benefit at city scale.
We therefore use exact Dijkstra and do not claim the sorting-barrier result. For
metropolitan-to-national road networks, Duan et al. (2025) is a promising
theoretical direction, but current implementations do not beat Dijkstra.

Reference (for the forward-looking note only, not implemented here)
------------------------------------------------------------------
Duan, R., Mao, J., Mao, X., Shu, X., & Yin, L. (2025).
*Breaking the Sorting Barrier for Directed Single-Source Shortest Paths.*
STOC 2025 (Best Paper). arXiv:2504.17033.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from civicsafe.routing.graph import Edge, RoutingGraph


@dataclass
class PathResult:
    """Result of a shortest-path query.

    Attributes:
        path: Ordered list of node indices from source to target.
        total_cost: Sum of edge costs along the path.
        edges: The edges traversed (for audit/abstention evaluation).
        settled_count: Number of vertices settled by the algorithm.
        frontier_reductions: Number of frontier reduction steps executed.
    """

    path: list[int]
    total_cost: float
    edges: list[Edge]
    settled_count: int = 0
    frontier_reductions: int = 0


class BatchedFrontierRouter:
    """Bellman-Ford-seeded, batched-frontier shortest-path router.

    Returns the same shortest-path costs as Dijkstra on our city-scale graphs
    (~77-100 nodes; verified in tests), but it re-sorts the frontier each
    iteration and is NOT faster than Dijkstra and NOT the Duan et al. (2025)
    algorithm. Prefer ``DijkstraRouter`` for production; this class is retained
    for backward compatibility and testing.

    Args:
        graph: The routing graph.
        cost_fn: Edge cost function (e.g. ``ParetoCost``).
    """

    def __init__(
        self,
        graph: RoutingGraph,
        cost_fn: Callable[[Edge], float],
    ) -> None:
        self.graph = graph
        self.cost_fn = cost_fn

    def shortest_path(self, source: int, target: int) -> PathResult:
        """Find the shortest path from source to target.

        Bellman-Ford seeding + batched-frontier settling. Returns the same
        shortest-path cost as Dijkstra on our graphs; not faster than Dijkstra.

        Args:
            source: Start node index.
            target: Destination node index.

        Returns:
            ``PathResult`` with path, cost, and traversed edges.

        Raises:
            ValueError: If source or target is invalid.
            RuntimeError: If no path exists.
        """
        if not self.graph.has_node(source):
            msg = f"Source node {source} not in graph (0..{self.graph.num_nodes - 1})"
            raise ValueError(msg)
        if not self.graph.has_node(target):
            msg = f"Target node {target} not in graph (0..{self.graph.num_nodes - 1})"
            raise ValueError(msg)

        n = self.graph.num_nodes

        # Distance estimates and predecessor tracking
        dist: list[float] = [math.inf] * n
        prev: list[int] = [-1] * n
        prev_edge: list[Edge | None] = [None] * n
        settled: list[bool] = [False] * n

        dist[source] = 0.0

        # === FRONTIER REDUCTION PARAMETERS ===
        # k = log^{1/3}(n) — the reduction factor
        k = max(1, int(round(math.log(max(n, 2)) ** (1.0 / 3.0))))
        frontier_reductions = 0
        settled_count = 0

        # === BMSSP: Bounded Multi-Source Shortest Path ===
        # Phase 1: Initial Bellman-Ford passes to seed distance estimates.
        # k passes suffice to find shortest paths with ≤ k edges.
        for _ in range(min(k, n)):
            updated = False
            for u in range(n):
                if dist[u] == math.inf:
                    continue
                for edge in self.graph.neighbors(u):
                    cost = self.cost_fn(edge)
                    new_dist = dist[u] + cost
                    if new_dist < dist[edge.dst]:
                        dist[edge.dst] = new_dist
                        prev[edge.dst] = u
                        prev_edge[edge.dst] = edge
                        updated = True
            if not updated:
                break

        # Phase 2: Frontier Reduction with Pivot Selection
        max_iterations = n * 3  # Safety bound
        iteration = 0

        while iteration < max_iterations:
            iteration += 1

            # Build frontier: unsettled vertices with finite distance
            frontier = [
                v for v in range(n) if not settled[v] and dist[v] < math.inf
            ]

            if not frontier:
                break

            if settled[target]:
                break

            # === BATCH SELECTION ===
            # Sort the frontier by current distance and settle a batch of the
            # closest nodes. NOTE: this full re-sort is exactly what a real
            # sorting-barrier algorithm avoids; it is why this router is not
            # faster than Dijkstra. Kept for backward compatibility only.
            frontier.sort(key=lambda v: dist[v])

            # Select the closest vertices in the frontier as the batch.
            pivot_stride = max(1, k)
            batch_size = max(1, (len(frontier) + pivot_stride - 1) // pivot_stride)
            pivots = frontier[:batch_size]

            frontier_reductions += 1

            # === SETTLE PIVOTS ===
            # These are the batch_size closest unsettled vertices.
            for pivot in pivots:
                if settled[pivot]:
                    continue
                settled[pivot] = True
                settled_count += 1

            # === RELAXATION PASS ===
            # After settling a batch of pivots, relax all their outgoing
            # edges to propagate new distance estimates.
            for pivot in pivots:
                for edge in self.graph.neighbors(pivot):
                    cost = self.cost_fn(edge)
                    new_dist = dist[pivot] + cost
                    if new_dist < dist[edge.dst]:
                        dist[edge.dst] = new_dist
                        prev[edge.dst] = pivot
                        prev_edge[edge.dst] = edge

        # === RECONSTRUCT PATH ===
        if dist[target] == math.inf:
            msg = f"No path from {source} to {target}"
            raise RuntimeError(msg)

        path: list[int] = []
        edges: list[Edge] = []
        current = target
        while current != source:
            path.append(current)
            edge = prev_edge[current]
            if edge is not None:
                edges.append(edge)
            current = prev[current]
            if current == -1:
                msg = f"No path from {source} to {target}"
                raise RuntimeError(msg)
        path.append(source)
        path.reverse()
        edges.reverse()

        return PathResult(
            path=path,
            total_cost=dist[target],
            edges=edges,
            settled_count=settled_count,
            frontier_reductions=frontier_reductions,
        )

    def all_shortest_paths(self, source: int) -> dict[int, PathResult]:
        """Compute shortest paths from source to all reachable nodes.

        Args:
            source: Start node index.

        Returns:
            ``{target: PathResult}`` for all reachable targets.
        """
        results: dict[int, PathResult] = {}
        for target in range(self.graph.num_nodes):
            if target == source:
                continue
            try:
                results[target] = self.shortest_path(source, target)
            except RuntimeError:
                pass  # unreachable
        return results


class DijkstraRouter:
    """Classic Dijkstra baseline for correctness verification.

    Used in tests to verify that TsinghuaRouter produces identical
    optimal paths.

    Args:
        graph: The routing graph.
        cost_fn: Edge cost function.
    """

    def __init__(
        self,
        graph: RoutingGraph,
        cost_fn: Callable[[Edge], float],
    ) -> None:
        self.graph = graph
        self.cost_fn = cost_fn

    def shortest_path(self, source: int, target: int) -> PathResult:
        """Standard Dijkstra shortest path.

        Args:
            source: Start node.
            target: End node.

        Returns:
            PathResult with optimal path.

        Raises:
            ValueError: If source or target is invalid.
            RuntimeError: If no path exists.
        """
        import heapq

        if not self.graph.has_node(source):
            raise ValueError(f"Invalid source: {source}")
        if not self.graph.has_node(target):
            raise ValueError(f"Invalid target: {target}")

        n = self.graph.num_nodes
        dist: list[float] = [math.inf] * n
        prev: list[int] = [-1] * n
        prev_edge: list[Edge | None] = [None] * n
        dist[source] = 0.0

        # Min-heap: (distance, node)
        heap: list[tuple[float, int]] = [(0.0, source)]
        visited: set[int] = set()
        settled_count = 0

        while heap:
            d_u, u = heapq.heappop(heap)
            if u in visited:
                continue
            visited.add(u)
            settled_count += 1

            if u == target:
                break

            for edge in self.graph.neighbors(u):
                cost = self.cost_fn(edge)
                new_dist = d_u + cost
                if new_dist < dist[edge.dst]:
                    dist[edge.dst] = new_dist
                    prev[edge.dst] = u
                    prev_edge[edge.dst] = edge
                    heapq.heappush(heap, (new_dist, edge.dst))

        if dist[target] == math.inf:
            raise RuntimeError(f"No path from {source} to {target}")

        path: list[int] = []
        edges: list[Edge] = []
        current = target
        while current != source:
            path.append(current)
            edge = prev_edge[current]
            if edge is not None:
                edges.append(edge)
            current = prev[current]
            if current == -1:
                raise RuntimeError(f"No path from {source} to {target}")
        path.append(source)
        path.reverse()
        edges.reverse()

        return PathResult(
            path=path,
            total_cost=dist[target],
            edges=edges,
            settled_count=settled_count,
        )


# Backward-compatibility alias. The old name implied a false connection to the
# Duan et al. (2025) SSSP algorithm; the honest name is BatchedFrontierRouter.
# Prefer DijkstraRouter in production.
TsinghuaRouter = BatchedFrontierRouter
