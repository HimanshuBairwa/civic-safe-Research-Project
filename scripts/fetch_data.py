#!/usr/bin/env python3
"""CIVIC-SAFE Data Acquisition Script — Run on A100 Jupyter Server.

This script downloads, normalizes, and assembles the complete
spatiotemporal panel for both Chicago and NYC. It is designed to:

  1. Stream data via SoQL (server-side filtering) — minimal bandwidth
  2. Cache every year as a verified Parquet file — re-runnable without re-download
  3. Build the final tensor panel ready for GNN training

Usage (on your A100 Jupyter terminal):
    cd /workspace/civic-safe-Research-Project
    python scripts/fetch_data.py

Expected runtime: ~10–20 minutes (depends on API speed).
Expected disk usage: ~500MB for raw Parquet, ~200MB for tensor panels.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch

# Ensure the project is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from civicsafe.data.acs import load_acs_features
from civicsafe.data.chicago import process_chicago_crimes
from civicsafe.data.crosswalks import get_census_crosswalk
from civicsafe.data.nyc import process_nyc_crimes
from civicsafe.data.panel import build_spatiotemporal_panel

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
START_YEAR = 2018
END_YEAR = 2023
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

ACS_VARIABLES = [
    "median_household_income",
    "poverty_rate",
    "unemployment_rate",
    "pct_black",
    "pct_hispanic",
    "pct_renter_occupied",
    "population_density",
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("civic-safe.fetch")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Execute the complete data acquisition pipeline."""
    t0 = time.perf_counter()
    logger.info("=" * 60)
    logger.info("CIVIC-SAFE Data Acquisition Pipeline")
    logger.info(f"Years: {START_YEAR}–{END_YEAR}")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Download Chicago crime data
    # ------------------------------------------------------------------
    logger.info("\n[1/6] Downloading Chicago crime data...")
    chicago_dir = RAW_DIR / "chicago"
    chicago_df = process_chicago_crimes(START_YEAR, END_YEAR, chicago_dir)
    logger.info(
        f"  ✓ Chicago: {len(chicago_df):,} records, "
        f"{chicago_df['spatial_unit'].nunique()} community areas"
    )

    # ------------------------------------------------------------------
    # Step 2: Download NYC crime data
    # ------------------------------------------------------------------
    logger.info("\n[2/6] Downloading NYC crime data...")
    nyc_dir = RAW_DIR / "nyc"
    nyc_df = process_nyc_crimes(START_YEAR, END_YEAR, nyc_dir)
    logger.info(
        f"  ✓ NYC: {len(nyc_df):,} records, "
        f"{nyc_df['spatial_unit'].nunique()} precincts"
    )

    # ------------------------------------------------------------------
    # Step 3: Load ACS demographics (with resilient fallback)
    # ------------------------------------------------------------------
    logger.info("\n[3/6] Loading ACS demographic features...")
    chicago_acs = load_acs_features("chicago", ACS_VARIABLES)
    nyc_acs = load_acs_features("nyc", ACS_VARIABLES)
    logger.info(
        f"  ✓ Chicago ACS: {len(chicago_acs)} tracts × {len(ACS_VARIABLES)} vars"
    )
    logger.info(f"  ✓ NYC ACS: {len(nyc_acs)} tracts × {len(ACS_VARIABLES)} vars")

    # ------------------------------------------------------------------
    # Step 4: Load crosswalks
    # ------------------------------------------------------------------
    logger.info("\n[4/6] Loading spatial crosswalks...")
    chicago_cw = get_census_crosswalk("chicago")
    nyc_cw = get_census_crosswalk("nyc")
    logger.info(f"  ✓ Chicago: {len(chicago_cw)} tract-to-area mappings")
    logger.info(f"  ✓ NYC: {len(nyc_cw)} tract-to-precinct mappings")

    # ------------------------------------------------------------------
    # Step 5: Build Chicago panel
    # ------------------------------------------------------------------
    logger.info("\n[5/6] Building Chicago spatiotemporal panel...")
    chicago_panel = build_spatiotemporal_panel(
        chicago_df, chicago_acs, chicago_cw, START_YEAR, END_YEAR
    )
    _log_panel_summary("Chicago", chicago_panel)

    # ------------------------------------------------------------------
    # Step 6: Build NYC panel
    # ------------------------------------------------------------------
    logger.info("\n[6/6] Building NYC spatiotemporal panel...")
    nyc_panel = build_spatiotemporal_panel(
        nyc_df, nyc_acs, nyc_cw, START_YEAR, END_YEAR
    )
    _log_panel_summary("NYC", nyc_panel)

    # ------------------------------------------------------------------
    # Step 7: Download boundary shapefiles
    # ------------------------------------------------------------------
    logger.info("\n[7/8] Downloading boundary shapefiles...")
    from civicsafe.data.shapefiles import load_boundaries

    chicago_gdf = load_boundaries("chicago", RAW_DIR / "shapefiles")
    nyc_gdf = load_boundaries("nyc", RAW_DIR / "shapefiles")
    logger.info(f"  ✓ Chicago: {len(chicago_gdf)} community area polygons")
    logger.info(f"  ✓ NYC: {len(nyc_gdf)} precinct polygons")

    # ------------------------------------------------------------------
    # Step 8: Build geospatial adjacency graphs
    # ------------------------------------------------------------------
    logger.info("\n[8/8] Building geospatial adjacency graphs...")
    from civicsafe.models.graph import build_adjacency_from_geodataframe

    chicago_graph = build_adjacency_from_geodataframe(
        chicago_gdf, knn_k=8, meter_crs="EPSG:26971"
    )
    nyc_graph = build_adjacency_from_geodataframe(
        nyc_gdf, knn_k=8, meter_crs="EPSG:32118"
    )

    # ------------------------------------------------------------------
    # Save everything
    # ------------------------------------------------------------------
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    chicago_path = PROCESSED_DIR / "chicago_panel.pt"
    nyc_path = PROCESSED_DIR / "nyc_panel.pt"
    chicago_graph_path = PROCESSED_DIR / "chicago_graph.pt"
    nyc_graph_path = PROCESSED_DIR / "nyc_graph.pt"

    torch.save(chicago_panel, chicago_path)
    torch.save(nyc_panel, nyc_path)
    torch.save(chicago_graph, chicago_graph_path)
    torch.save(nyc_graph, nyc_graph_path)

    logger.info(f"\n  ✓ Saved: {chicago_path}")
    logger.info(f"  ✓ Saved: {nyc_path}")
    logger.info(f"  ✓ Saved: {chicago_graph_path}")
    logger.info(f"  ✓ Saved: {nyc_graph_path}")

    elapsed = time.perf_counter() - t0
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Pipeline complete in {elapsed:.1f}s")
    logger.info(f"{'=' * 60}")


def _log_panel_summary(city: str, panel: dict) -> None:
    """Log shape and basic statistics of a panel."""
    counts = panel["counts"]
    features = panel["features"]
    meta = panel["metadata"]

    logger.info(f"  {city} Panel Summary:")
    logger.info(f"    counts shape:   {tuple(counts.shape)}")
    logger.info(f"    features shape: {tuple(features.shape)}")
    logger.info(f"    total crimes:   {counts.sum().item():,}")
    logger.info(f"    spatial units:  {len(meta['spatial_units'])}")
    logger.info(f"    categories:     {meta['categories']}")

    # Per-category totals
    for i, cat in enumerate(meta["categories"]):
        cat_total = counts[:, :, i].sum().item()
        logger.info(f"    {cat:>10s}: {cat_total:>10,}")


if __name__ == "__main__":
    main()
