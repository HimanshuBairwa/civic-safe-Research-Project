"""Comprehensive tests for the advisory safe-route reference module.

Tests cover:
- RoutingGraph construction and prediction injection
- All cost functions (Pareto, DistanceOnly, RiskOnly, UncertaintyPenalised)
- TsinghuaRouter vs DijkstraRouter correctness (same optimal paths!)
- AbstentionMonitor (peak, budget, safe verdicts)
- AdvisoryRoutingEngine end-to-end
- Edge cases (no path, single node, self-loops)
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

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


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture()
def simple_graph() -> RoutingGraph:
    """A 5-node graph with known shortest paths.

    Graph structure:
        0 --1.0-- 1 --1.0-- 2
        |                    |
       2.0                  1.0
        |                    |
        3 -------1.0------- 4

    Shortest 0→2: 0→1→2 (cost=2.0)
    Shortest 0→4: 0→1→2→4 (cost=3.0) or 0→3→4 (cost=3.0)
    """
    graph = RoutingGraph(num_nodes=5)
    edges = [
        (0, 1, 1.0), (1, 0, 1.0),
        (1, 2, 1.0), (2, 1, 1.0),
        (0, 3, 2.0), (3, 0, 2.0),
        (2, 4, 1.0), (4, 2, 1.0),
        (3, 4, 1.0), (4, 3, 1.0),
    ]
    adj: dict[int, list[Edge]] = {i: [] for i in range(5)}
    for s, d, dist in edges:
        adj[s].append(Edge(src=s, dst=d, distance=dist))
    graph.adjacency_list = adj
    return graph


@pytest.fixture()
def risk_graph() -> RoutingGraph:
    """Graph with risk injected — shorter path has higher risk.

    0 --1.0-- 1 --1.0-- 3  (short but risky: risk=10 on nodes 1,3)
    |                    |
    0 --3.0-- 2 --3.0-- 3  (long but safe: risk=1 on nodes 2,3)
    """
    graph = RoutingGraph(num_nodes=4)
    adj: dict[int, list[Edge]] = {i: [] for i in range(4)}

    # Short risky path: 0→1→3
    adj[0].append(Edge(src=0, dst=1, distance=1.0, risk_upper=10.0, interval_width=5.0))
    adj[1].append(Edge(src=1, dst=3, distance=1.0, risk_upper=10.0, interval_width=5.0))

    # Long safe path: 0→2→3
    adj[0].append(Edge(src=0, dst=2, distance=3.0, risk_upper=1.0, interval_width=1.0))
    adj[2].append(Edge(src=2, dst=3, distance=3.0, risk_upper=1.0, interval_width=1.0))

    graph.adjacency_list = adj
    return graph


# ===================================================================
# Test RoutingGraph
# ===================================================================


class TestRoutingGraph:
    """Tests for RoutingGraph construction."""

    def test_from_edge_index(self) -> None:
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        graph = RoutingGraph.from_edge_index(edge_index, num_nodes=3)
        assert graph.num_nodes == 3
        assert graph.edge_count() == 4

    def test_from_adjacency_matrix(self) -> None:
        matrix = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ])
        graph = RoutingGraph.from_adjacency_matrix(matrix)
        assert graph.num_nodes == 3
        assert graph.edge_count() == 4

    def test_inject_predictions(self, simple_graph: RoutingGraph) -> None:
        upper = torch.tensor([5.0, 3.0, 8.0, 2.0, 6.0])
        lower = torch.tensor([1.0, 0.0, 2.0, 0.0, 1.0])
        simple_graph.inject_predictions(upper, lower)

        # Edge 0→1: dst=1, risk_upper=3.0, width=3.0
        edge_0_to_1 = [e for e in simple_graph.neighbors(0) if e.dst == 1][0]
        assert edge_0_to_1.risk_upper == 3.0
        assert edge_0_to_1.interval_width == 3.0

    def test_neighbors(self, simple_graph: RoutingGraph) -> None:
        n = simple_graph.neighbors(0)
        dsts = {e.dst for e in n}
        assert dsts == {1, 3}

    def test_has_node(self, simple_graph: RoutingGraph) -> None:
        assert simple_graph.has_node(0) is True
        assert simple_graph.has_node(4) is True
        assert simple_graph.has_node(5) is False
        assert simple_graph.has_node(-1) is False

    def test_positions_distance(self) -> None:
        positions = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
        edge_index = torch.tensor([[0], [1]], dtype=torch.long)
        graph = RoutingGraph.from_edge_index(edge_index, 2, positions)
        edge = graph.neighbors(0)[0]
        assert abs(edge.distance - 5.0) < 1e-4  # 3-4-5 triangle


# ===================================================================
# Test Cost Functions
# ===================================================================


class TestCostFunctions:
    """Tests for all cost function implementations."""

    def test_pareto_cost(self) -> None:
        edge = Edge(src=0, dst=1, distance=2.0, risk_upper=10.0)
        cost = ParetoCost(w_dist=0.3, w_risk=0.7)
        assert abs(cost(edge) - (0.3 * 2.0 + 0.7 * 10.0)) < 1e-6

    def test_distance_only(self) -> None:
        edge = Edge(src=0, dst=1, distance=5.0, risk_upper=100.0)
        cost = DistanceOnlyCost()
        assert cost(edge) == 5.0

    def test_risk_only(self) -> None:
        edge = Edge(src=0, dst=1, distance=100.0, risk_upper=3.0)
        cost = RiskOnlyCost()
        assert cost(edge) == 3.0

    def test_uncertainty_penalised(self) -> None:
        edge = Edge(src=0, dst=1, distance=1.0, risk_upper=2.0, interval_width=3.0)
        cost = UncertaintyPenalisedCost(w_dist=0.2, w_risk=0.5, w_unc=0.3)
        expected = 0.2 * 1.0 + 0.5 * 2.0 + 0.3 * 3.0
        assert abs(cost(edge) - expected) < 1e-6


# ===================================================================
# Test TsinghuaRouter — CORE CORRECTNESS TESTS
# ===================================================================


class TestTsinghuaRouter:
    """Tests for the Tsinghua 2025 SSSP router."""

    def test_simple_shortest_path(self, simple_graph: RoutingGraph) -> None:
        router = TsinghuaRouter(simple_graph, DistanceOnlyCost())
        result = router.shortest_path(0, 2)
        assert result.path == [0, 1, 2]
        assert abs(result.total_cost - 2.0) < 1e-6

    def test_matches_dijkstra(self, simple_graph: RoutingGraph) -> None:
        """THE CRITICAL TEST: Tsinghua must produce the same cost as Dijkstra."""
        cost_fn = DistanceOnlyCost()
        tsinghua = TsinghuaRouter(simple_graph, cost_fn)
        dijkstra = DijkstraRouter(simple_graph, cost_fn)

        for src in range(simple_graph.num_nodes):
            for dst in range(simple_graph.num_nodes):
                if src == dst:
                    continue
                t_result = tsinghua.shortest_path(src, dst)
                d_result = dijkstra.shortest_path(src, dst)
                assert abs(t_result.total_cost - d_result.total_cost) < 1e-6, (
                    f"Mismatch {src}→{dst}: "
                    f"Tsinghua={t_result.total_cost:.4f}, "
                    f"Dijkstra={d_result.total_cost:.4f}"
                )

    def test_matches_dijkstra_pareto(self, simple_graph: RoutingGraph) -> None:
        """Tsinghua matches Dijkstra even with Pareto cost."""
        upper = torch.tensor([2.0, 5.0, 1.0, 8.0, 3.0])
        lower = torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0])
        simple_graph.inject_predictions(upper, lower)

        cost_fn = ParetoCost()
        tsinghua = TsinghuaRouter(simple_graph, cost_fn)
        dijkstra = DijkstraRouter(simple_graph, cost_fn)

        for src in range(5):
            for dst in range(5):
                if src == dst:
                    continue
                t = tsinghua.shortest_path(src, dst)
                d = dijkstra.shortest_path(src, dst)
                assert abs(t.total_cost - d.total_cost) < 1e-6

    def test_frontier_reductions_tracked(self, simple_graph: RoutingGraph) -> None:
        router = TsinghuaRouter(simple_graph, DistanceOnlyCost())
        result = router.shortest_path(0, 4)
        assert result.frontier_reductions >= 1

    def test_invalid_source_raises(self, simple_graph: RoutingGraph) -> None:
        router = TsinghuaRouter(simple_graph, DistanceOnlyCost())
        with pytest.raises(ValueError, match="Source"):
            router.shortest_path(99, 0)

    def test_invalid_target_raises(self, simple_graph: RoutingGraph) -> None:
        router = TsinghuaRouter(simple_graph, DistanceOnlyCost())
        with pytest.raises(ValueError, match="Target"):
            router.shortest_path(0, 99)

    def test_no_path_raises(self) -> None:
        """Disconnected graph: no path should raise RuntimeError."""
        graph = RoutingGraph(num_nodes=3)
        graph.adjacency_list = {
            0: [Edge(src=0, dst=1, distance=1.0)],
            1: [],
            2: [],  # Node 2 is disconnected
        }
        router = TsinghuaRouter(graph, DistanceOnlyCost())
        with pytest.raises(RuntimeError, match="No path"):
            router.shortest_path(0, 2)


class TestDijkstraRouter:
    """Tests for the Dijkstra baseline."""

    def test_simple_path(self, simple_graph: RoutingGraph) -> None:
        router = DijkstraRouter(simple_graph, DistanceOnlyCost())
        result = router.shortest_path(0, 2)
        assert result.path == [0, 1, 2]
        assert abs(result.total_cost - 2.0) < 1e-6

    def test_settled_count(self, simple_graph: RoutingGraph) -> None:
        router = DijkstraRouter(simple_graph, DistanceOnlyCost())
        result = router.shortest_path(0, 4)
        assert result.settled_count > 0


# ===================================================================
# Test Risk-Aware Routing
# ===================================================================


class TestRiskAwareRouting:
    """Tests that cost functions correctly shift routes away from risk."""

    def test_distance_prefers_short_risky(self, risk_graph: RoutingGraph) -> None:
        """DistanceOnlyCost takes the short path (ignoring risk)."""
        router = TsinghuaRouter(risk_graph, DistanceOnlyCost())
        result = router.shortest_path(0, 3)
        assert result.path == [0, 1, 3]
        assert abs(result.total_cost - 2.0) < 1e-6

    def test_risk_prefers_safe_long(self, risk_graph: RoutingGraph) -> None:
        """RiskOnlyCost takes the long path (lower risk)."""
        router = TsinghuaRouter(risk_graph, RiskOnlyCost())
        result = router.shortest_path(0, 3)
        assert result.path == [0, 2, 3]
        assert abs(result.total_cost - 2.0) < 1e-6  # risk 1+1=2

    def test_pareto_balances(self, risk_graph: RoutingGraph) -> None:
        """ParetoCost with high risk weight prefers the safe path."""
        cost = ParetoCost(w_dist=0.1, w_risk=0.9)
        router = TsinghuaRouter(risk_graph, cost)
        result = router.shortest_path(0, 3)
        # Safe path: 0.1*3+0.9*1 + 0.1*3+0.9*1 = 0.3+0.9+0.3+0.9 = 2.4
        # Risky:     0.1*1+0.9*10 + 0.1*1+0.9*10 = 0.1+9+0.1+9 = 18.2
        assert result.path == [0, 2, 3]


# ===================================================================
# Test AbstentionMonitor
# ===================================================================


class TestAbstentionMonitor:
    """Tests for the uncertainty-based abstention system."""

    def test_safe_path_passes(self) -> None:
        edges = [
            Edge(src=0, dst=1, distance=1.0, interval_width=2.0),
            Edge(src=1, dst=2, distance=1.0, interval_width=3.0),
        ]
        result = PathResult(path=[0, 1, 2], total_cost=2.0, edges=edges)

        monitor = AbstentionMonitor(peak_threshold=10.0, budget_threshold=50.0)
        verdict = monitor.evaluate(result)
        assert verdict.should_abstain is False

    def test_peak_exceeds_threshold(self) -> None:
        edges = [
            Edge(src=0, dst=1, distance=1.0, interval_width=25.0),  # Too wide!
        ]
        result = PathResult(path=[0, 1], total_cost=1.0, edges=edges)

        monitor = AbstentionMonitor(peak_threshold=20.0)
        verdict = monitor.evaluate(result)
        assert verdict.should_abstain is True
        assert "Peak interval width" in verdict.reason

    def test_budget_exceeds_threshold(self) -> None:
        edges = [
            Edge(src=0, dst=1, distance=1.0, interval_width=15.0),
            Edge(src=1, dst=2, distance=1.0, interval_width=15.0),
            Edge(src=2, dst=3, distance=1.0, interval_width=15.0),
        ]
        result = PathResult(path=[0, 1, 2, 3], total_cost=3.0, edges=edges)

        monitor = AbstentionMonitor(peak_threshold=20.0, budget_threshold=40.0)
        verdict = monitor.evaluate(result)
        assert verdict.should_abstain is True
        assert "Cumulative" in verdict.reason

    def test_check_or_raise(self) -> None:
        edges = [Edge(src=0, dst=1, distance=1.0, interval_width=25.0)]
        result = PathResult(path=[0, 1], total_cost=1.0, edges=edges)

        monitor = AbstentionMonitor(peak_threshold=20.0)
        with pytest.raises(AbstentionError):
            monitor.check_or_raise(result)

    def test_empty_path_passes(self) -> None:
        result = PathResult(path=[0], total_cost=0.0, edges=[])
        monitor = AbstentionMonitor()
        verdict = monitor.evaluate(result)
        assert verdict.should_abstain is False


# ===================================================================
# Test AdvisoryRoutingEngine
# ===================================================================


class TestAdvisoryRoutingEngine:
    """Tests for the full routing engine facade."""

    def test_end_to_end(self, simple_graph: RoutingGraph) -> None:
        upper = torch.tensor([2.0, 3.0, 1.0, 4.0, 2.0])
        lower = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0])
        simple_graph.inject_predictions(upper, lower)

        engine = AdvisoryRoutingEngine(
            graph=simple_graph,
            abstention_monitor=AbstentionMonitor(peak_threshold=100.0),
        )
        result = engine.safe_route(0, 2, raise_on_abstention=False)
        assert isinstance(result, SafeRouteResult)
        assert result.path[0] == 0
        assert result.path[-1] == 2
        assert result.algorithm == "tsinghua"

    def test_from_adjacency_factory(self) -> None:
        edge_index = torch.tensor([
            [0, 1, 1, 2, 2, 3],
            [1, 0, 2, 1, 3, 2],
        ], dtype=torch.long)
        upper = torch.tensor([1.0, 2.0, 3.0, 4.0])
        lower = torch.tensor([0.0, 0.0, 1.0, 1.0])

        engine = AdvisoryRoutingEngine.from_adjacency(
            edge_index=edge_index,
            num_nodes=4,
            upper_bounds=upper,
            lower_bounds=lower,
        )
        result = engine.safe_route(0, 3, raise_on_abstention=False)
        assert result.path[0] == 0
        assert result.path[-1] == 3

    def test_abstention_raises(self, simple_graph: RoutingGraph) -> None:
        """Engine raises AbstentionError when uncertainty is too high."""
        upper = torch.tensor([100.0, 100.0, 100.0, 100.0, 100.0])
        lower = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        simple_graph.inject_predictions(upper, lower)

        engine = AdvisoryRoutingEngine(
            graph=simple_graph,
            abstention_monitor=AbstentionMonitor(
                peak_threshold=50.0,
                budget_threshold=100.0,
            ),
        )
        with pytest.raises(AbstentionError):
            engine.safe_route(0, 4, raise_on_abstention=True)

    def test_abstention_no_raise(self, simple_graph: RoutingGraph) -> None:
        """Engine returns result with should_abstain=True when not raising."""
        upper = torch.tensor([100.0, 100.0, 100.0, 100.0, 100.0])
        lower = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
        simple_graph.inject_predictions(upper, lower)

        engine = AdvisoryRoutingEngine(
            graph=simple_graph,
            abstention_monitor=AbstentionMonitor(
                peak_threshold=50.0,
                budget_threshold=100.0,
            ),
        )
        result = engine.safe_route(0, 4, raise_on_abstention=False)
        assert result.abstention_verdict.should_abstain is True

    def test_compare_algorithms(self, simple_graph: RoutingGraph) -> None:
        engine = AdvisoryRoutingEngine(graph=simple_graph)
        comparison = engine.compare_algorithms(0, 4)
        assert comparison["cost_match"] is True
        assert comparison["tsinghua_frontier_reductions"] >= 1


# ===================================================================
# Test Large Random Graph (Stress Test)
# ===================================================================


class TestLargeGraph:
    """Stress test on a 77-node graph (Chicago-scale)."""

    def test_77_node_correctness(self) -> None:
        """Tsinghua matches Dijkstra on a 77-node random graph."""
        torch.manual_seed(42)
        n = 77
        # Generate random positions and build a connected graph
        positions = torch.rand(n, 2)

        # Connect each node to its 6 nearest neighbors
        from scipy.spatial import cKDTree

        tree = cKDTree(positions.numpy())
        src_list, dst_list = [], []
        for i in range(n):
            _, neighbors = tree.query(positions[i].numpy(), k=7)
            for j in neighbors:
                if j != i:
                    src_list.extend([i, int(j)])
                    dst_list.extend([int(j), i])

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        graph = RoutingGraph.from_edge_index(edge_index, n, positions)

        # Inject random risk
        upper = torch.rand(n) * 10
        lower = torch.zeros(n)
        graph.inject_predictions(upper, lower)

        cost_fn = ParetoCost()
        tsinghua = TsinghuaRouter(graph, cost_fn)
        dijkstra = DijkstraRouter(graph, cost_fn)

        # Test 20 random pairs
        torch.manual_seed(123)
        pairs = [(int(torch.randint(0, n, (1,)).item()),
                   int(torch.randint(0, n, (1,)).item())) for _ in range(20)]

        for src, dst in pairs:
            if src == dst:
                continue
            try:
                t = tsinghua.shortest_path(src, dst)
                d = dijkstra.shortest_path(src, dst)
                assert abs(t.total_cost - d.total_cost) < 1e-4, (
                    f"{src}→{dst}: T={t.total_cost:.4f} D={d.total_cost:.4f}"
                )
            except RuntimeError:
                pass  # disconnected pair
