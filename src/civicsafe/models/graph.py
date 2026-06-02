"""Dual adjacency graph builder for spatial encoding.

Constructs two sets of edges for the GATv2 spatial encoder:
  1. Queen contiguity: areas sharing a border or corner
  2. K-NN: k nearest neighbors by centroid distance

Both are converted to PyG edge_index (COO sparse) format.
For unit testing and smoke tests, synthetic graphs are supported.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def build_adjacency_from_synthetic(
    num_nodes: int,
    seed: int = 42,
    knn_k: int = 8,
) -> dict[str, Tensor]:
    """Build dual adjacency from random 2D positions (for testing).

    Args:
        num_nodes: Number of spatial units.
        seed: RNG seed for reproducibility.
        knn_k: Number of nearest neighbors.

    Returns:
        Dictionary with 'queen' and 'knn' edge_index tensors.
    """
    rng = np.random.RandomState(seed)
    positions = rng.uniform(0, 1, size=(num_nodes, 2))

    # Queen contiguity: Delaunay triangulation as proxy
    from scipy.spatial import Delaunay

    tri = Delaunay(positions)
    queen_edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                a, b = simplex[i], simplex[j]
                queen_edges.add((a, b))
                queen_edges.add((b, a))

    queen_src = [e[0] for e in queen_edges]
    queen_dst = [e[1] for e in queen_edges]
    edge_index_queen = torch.tensor([queen_src, queen_dst], dtype=torch.long)

    # K-NN adjacency
    from scipy.spatial import cKDTree

    tree = cKDTree(positions)
    knn_edges_src = []
    knn_edges_dst = []
    for i in range(num_nodes):
        _, neighbors = tree.query(positions[i], k=min(knn_k + 1, num_nodes))
        for j in neighbors:
            if j != i:
                knn_edges_src.append(i)
                knn_edges_dst.append(j)

    edge_index_knn = torch.tensor([knn_edges_src, knn_edges_dst], dtype=torch.long)

    logger.info(
        f"  Graph: {num_nodes} nodes, "
        f"{edge_index_queen.shape[1]} queen edges, "
        f"{edge_index_knn.shape[1]} knn edges"
    )

    return {
        "queen": edge_index_queen,
        "knn": edge_index_knn,
    }


def build_adjacency_from_geodataframe(
    gdf: "geopandas.GeoDataFrame",
    knn_k: int = 8,
    meter_crs: str | None = None,
) -> dict[str, Tensor]:
    """Build dual adjacency from real polygon boundaries.

    Uses geopandas spatial joins for mathematically exact adjacency:
      - Queen contiguity: polygons sharing any boundary point (via touches)
      - K-NN: nearest centroids in a projected meter-based CRS

    Args:
        gdf: GeoDataFrame with polygon geometries. Must be in EPSG:4326.
        knn_k: Number of nearest neighbors for KNN graph.
        meter_crs: CRS for meter-based distance computations.
            Default: auto-detect based on centroid longitude.

    Returns:
        Dictionary with 'queen' and 'knn' edge_index tensors.
    """
    import geopandas as gpd
    from scipy.spatial import cKDTree

    n = len(gdf)
    gdf = gdf.reset_index(drop=True)

    # --- Auto-detect meter CRS based on location ---
    if meter_crs is None:
        centroid = gdf.geometry.unary_union.centroid
        if centroid.x < -80:  # Chicago area (~-87.6)
            meter_crs = "EPSG:26971"  # NAD83 / Illinois East (meters)
        else:  # NYC area (~-74.0)
            meter_crs = "EPSG:32118"  # NAD83 / New York (meters)

    # --- Queen contiguity via spatial join ---
    # Fix floating-point precision issues in boundary comparisons
    gdf_clean = gdf.copy()
    try:
        gdf_clean["geometry"] = gdf.geometry.set_precision(grid_size=0.0001)
    except AttributeError:
        # Older shapely versions may not have set_precision
        pass

    adj = gpd.sjoin(gdf_clean, gdf_clean, how="inner", predicate="touches")
    adj = adj[adj.index != adj["index_right"]]

    queen_src = adj.index.tolist()
    queen_dst = adj["index_right"].tolist()

    # Fallback: if touches finds too few edges, use intersects
    if len(queen_src) < n:
        logger.warning(
            f"  Only {len(queen_src)} queen edges from touches. "
            f"Trying intersects fallback..."
        )
        adj2 = gpd.sjoin(gdf, gdf, how="inner", predicate="intersects")
        adj2 = adj2[adj2.index != adj2["index_right"]]
        if len(adj2) > len(adj):
            queen_src = adj2.index.tolist()
            queen_dst = adj2["index_right"].tolist()

    edge_index_queen = torch.tensor([queen_src, queen_dst], dtype=torch.long)

    # --- K-NN via projected centroids ---
    gdf_proj = gdf.to_crs(meter_crs)
    centroids = gdf_proj.geometry.centroid
    coords = np.column_stack([centroids.x.values, centroids.y.values])

    tree = cKDTree(coords)
    _, indices = tree.query(coords, k=min(knn_k + 1, n))

    knn_src, knn_dst = [], []
    for i in range(n):
        for j_idx in range(1, indices.shape[1]):
            knn_src.append(i)
            knn_dst.append(int(indices[i, j_idx]))

    edge_index_knn = torch.tensor([knn_src, knn_dst], dtype=torch.long)

    logger.info(
        f"  Graph (geospatial): {n} nodes, "
        f"{edge_index_queen.shape[1]} queen edges, "
        f"{edge_index_knn.shape[1]} knn edges"
    )

    return {
        "queen": edge_index_queen,
        "knn": edge_index_knn,
    }


def build_adjacency_from_panel(
    spatial_units: list[int],
    knn_k: int = 8,
) -> dict[str, Tensor]:
    """Build dual adjacency from spatial unit IDs (sequential proxy).

    This is a fallback for when real shapefiles are not available.
    For production use, prefer build_adjacency_from_geodataframe().

    Args:
        spatial_units: Sorted list of spatial unit IDs.
        knn_k: Number of nearest neighbors.

    Returns:
        Dictionary with 'queen' and 'knn' edge_index tensors.
    """
    n = len(spatial_units)

    # Queen contiguity proxy: connect sequential IDs
    queen_src, queen_dst = [], []
    for i in range(n):
        for di in [-1, 1]:
            j = i + di
            if 0 <= j < n:
                queen_src.append(i)
                queen_dst.append(j)

    edge_index_queen = torch.tensor([queen_src, queen_dst], dtype=torch.long)

    # K-NN proxy: connect to k nearest by ID distance
    knn_src, knn_dst = [], []
    for i in range(n):
        distances = [(abs(i - j), j) for j in range(n) if j != i]
        distances.sort()
        for _, j in distances[:knn_k]:
            knn_src.append(i)
            knn_dst.append(j)

    edge_index_knn = torch.tensor([knn_src, knn_dst], dtype=torch.long)

    logger.info(
        f"  Graph (proxy): {n} nodes, "
        f"{edge_index_queen.shape[1]} queen edges, "
        f"{edge_index_knn.shape[1]} knn edges"
    )

    return {
        "queen": edge_index_queen,
        "knn": edge_index_knn,
    }

