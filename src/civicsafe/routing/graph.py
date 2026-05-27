"""Spatial routing graph with dynamic risk-weighted edges.

Converts CIVIC-SAFE spatial adjacency graphs into routing-ready
weighted graphs where edge costs reflect prediction uncertainty.
Nodes represent spatial units (community areas / precincts) and
edges represent navigable connections between them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor


@dataclass
class Edge:
    """A single directed edge in the routing graph.

    Attributes:
        src: Source node index.
        dst: Destination node index.
        distance: Physical distance (or proxy) between spatial units.
        risk_upper: Conformal prediction upper bound for this edge's area.
        interval_width: Width of the prediction interval (upper - lower).
    """

    src: int
    dst: int
    distance: float
    risk_upper: float = 0.0
    interval_width: float = 0.0


@dataclass
class RoutingGraph:
    """Weighted directed graph for advisory safe routing.

    Supports dynamic injection of CIVIC-SAFE prediction outputs as
    edge weights, enabling uncertainty-aware pathfinding.

    Attributes:
        num_nodes: Number of spatial units (nodes).
        adjacency_list: ``{node_id: [Edge, ...]}``.
        node_positions: Optional 2D positions for visualization.
        metadata: Free-form metadata (city, timestamp, etc.).
    """

    num_nodes: int
    adjacency_list: dict[int, list[Edge]] = field(default_factory=dict)
    node_positions: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Construction from existing adjacency structures
    # ------------------------------------------------------------------

    @classmethod
    def from_edge_index(
        cls,
        edge_index: Tensor,
        num_nodes: int,
        positions: Tensor | None = None,
    ) -> "RoutingGraph":
        """Create a RoutingGraph from a PyG-style edge_index.

        Args:
            edge_index: (2, E) tensor of [src, dst] pairs.
            num_nodes: Number of nodes in the graph.
            positions: Optional (N, 2) node positions.

        Returns:
            A RoutingGraph with unit-distance edges.
        """
        adj: dict[int, list[Edge]] = {i: [] for i in range(num_nodes)}

        if positions is not None:
            pos = positions.float()
        else:
            pos = None

        for e in range(edge_index.shape[1]):
            src = int(edge_index[0, e].item())
            dst = int(edge_index[1, e].item())

            if pos is not None:
                dist = float(
                    torch.sqrt(((pos[src] - pos[dst]) ** 2).sum()).item()
                )
            else:
                dist = 1.0

            adj[src].append(Edge(src=src, dst=dst, distance=dist))

        return cls(
            num_nodes=num_nodes,
            adjacency_list=adj,
            node_positions=positions,
        )

    @classmethod
    def from_adjacency_matrix(
        cls,
        matrix: Tensor,
        positions: Tensor | None = None,
    ) -> "RoutingGraph":
        """Create a RoutingGraph from a dense adjacency matrix.

        Args:
            matrix: (N, N) binary adjacency matrix.
            positions: Optional (N, 2) node positions.

        Returns:
            A RoutingGraph with distance-weighted edges.
        """
        n = matrix.shape[0]
        indices = matrix.nonzero(as_tuple=False)
        edge_index = indices.t().contiguous()
        return cls.from_edge_index(edge_index, n, positions)

    # ------------------------------------------------------------------
    # Dynamic weight injection
    # ------------------------------------------------------------------

    def inject_predictions(
        self,
        upper_bounds: Tensor,
        lower_bounds: Tensor,
    ) -> None:
        """Inject conformal prediction outputs as edge risk weights.

        For each edge (u → v), the risk is derived from the
        *destination* node's predicted upper bound (worst-case scenario
        for entering that area).

        Args:
            upper_bounds: (N,) conformal upper bounds per spatial unit.
            lower_bounds: (N,) conformal lower bounds per spatial unit.
        """
        widths = (upper_bounds - lower_bounds).float()

        for node_edges in self.adjacency_list.values():
            for edge in node_edges:
                dst = edge.dst
                edge.risk_upper = float(upper_bounds[dst].item())
                edge.interval_width = float(widths[dst].item())

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def neighbors(self, node: int) -> list[Edge]:
        """Return all outgoing edges from a node."""
        return self.adjacency_list.get(node, [])

    def edge_count(self) -> int:
        """Total number of directed edges."""
        return sum(len(edges) for edges in self.adjacency_list.values())

    def has_node(self, node: int) -> bool:
        """Check if a node exists."""
        return 0 <= node < self.num_nodes
