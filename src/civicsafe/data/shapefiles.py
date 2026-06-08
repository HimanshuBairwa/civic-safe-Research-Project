"""Shapefile downloader and GeoDataFrame loader for spatial boundaries.

Downloads official GeoJSON boundary files from municipal open data portals:
  - Chicago: 77 Community Areas from data.cityofchicago.org
  - NYC: Police Precincts from data.cityofnewyork.us

Both are served as GeoJSON in EPSG:4326 (WGS 84).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.request
from pathlib import Path
from typing import Any

import geopandas as gpd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CHICAGO_COMMUNITY_AREAS_URL = (
    "https://data.cityofchicago.org/resource/igwz-8jzy.geojson"
)
_NYC_PRECINCTS_URL = (
    "https://data.cityofnewyork.us/api/geospatial/y76i-bdw7"
    "?method=export&format=GeoJSON"
)

_TIMEOUT = 120
_MAX_RETRIES = 5
_BACKOFF_BASE = 2.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_chicago_boundaries(save_dir: Path) -> gpd.GeoDataFrame:
    """Download and cache Chicago Community Area boundaries.

    Args:
        save_dir: Directory for the cached GeoJSON file.

    Returns:
        GeoDataFrame with 77 rows, columns: [area_number, community, geometry].
        CRS: EPSG:4326.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = save_dir / "chicago_community_areas.geojson"

    if geojson_path.exists():
        if geojson_path.stat().st_size > 10000:
            logger.info("  Using cached Chicago community area boundaries")
            gdf = gpd.read_file(geojson_path)
            return gdf
        else:
            logger.warning(f"  Cached file {geojson_path} is too small (corrupted). Redownloading...")
            geojson_path.unlink()

    logger.info("  Downloading Chicago Community Area boundaries...")
    _download_with_retry(_CHICAGO_COMMUNITY_AREAS_URL, geojson_path)

    gdf = gpd.read_file(geojson_path)

    # Normalize column names — the portal uses truncated shapefile field names
    # area_numbe or area_num_1 → area_number
    area_col = None
    for candidate in ["area_numbe", "area_num_1", "area_number"]:
        if candidate in gdf.columns:
            area_col = candidate
            break

    if area_col is None:
        raise ValueError(
            f"Could not find area number column. Available: {list(gdf.columns)}"
        )

    gdf["area_number"] = gdf[area_col].astype(int)

    # Normalize community name
    if "community" in gdf.columns:
        gdf["community"] = gdf["community"].str.strip().str.upper()

    # Sort by area number for deterministic ordering
    gdf = gdf.sort_values("area_number").reset_index(drop=True)

    # Keep only necessary columns
    keep_cols = ["area_number", "community", "geometry"]
    keep_cols = [c for c in keep_cols if c in gdf.columns]
    gdf = gdf[keep_cols]

    logger.info(
        f"  Chicago boundaries: {len(gdf)} community areas, "
        f"CRS={gdf.crs}"
    )
    return gdf


def fetch_nyc_boundaries(save_dir: Path) -> gpd.GeoDataFrame:
    """Download and cache NYC Police Precinct boundaries.

    Args:
        save_dir: Directory for the cached GeoJSON file.

    Returns:
        GeoDataFrame with ~77 rows, columns: [precinct, geometry].
        CRS: EPSG:4326.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    geojson_path = save_dir / "nyc_precincts.geojson"

    if geojson_path.exists():
        if geojson_path.stat().st_size > 10000:
            logger.info("  Using cached NYC precinct boundaries")
            gdf = gpd.read_file(geojson_path)
            return gdf
        else:
            logger.warning(f"  Cached file {geojson_path} is too small (corrupted). Redownloading...")
            geojson_path.unlink()

    logger.info("  Downloading NYC Precinct boundaries...")
    _download_with_retry(_NYC_PRECINCTS_URL, geojson_path)

    gdf = gpd.read_file(geojson_path)

    # Normalize precinct column
    precinct_col = None
    for candidate in ["precinct", "Precinct", "PRECINCT"]:
        if candidate in gdf.columns:
            precinct_col = candidate
            break

    if precinct_col is None:
        raise ValueError(
            f"Could not find precinct column. Available: {list(gdf.columns)}"
        )

    gdf["precinct"] = gdf[precinct_col].astype(int)

    # Sort by precinct number for deterministic ordering
    gdf = gdf.sort_values("precinct").reset_index(drop=True)

    # Keep only necessary columns
    keep_cols = ["precinct", "geometry"]
    keep_cols = [c for c in keep_cols if c in gdf.columns]
    gdf = gdf[keep_cols]

    logger.info(
        f"  NYC boundaries: {len(gdf)} precincts, "
        f"CRS={gdf.crs}"
    )
    return gdf


def load_boundaries(city: str, save_dir: Path) -> gpd.GeoDataFrame:
    """Load spatial boundaries for a city.

    Args:
        city: "chicago" or "nyc" (case-insensitive).
        save_dir: Cache directory.

    Returns:
        GeoDataFrame with boundary polygons.
    """
    if city.lower() == "chicago":
        return fetch_chicago_boundaries(save_dir)
    elif city.lower() == "nyc":
        return fetch_nyc_boundaries(save_dir)
    else:
        raise ValueError(f"Unknown city: {city}. Use 'chicago' or 'nyc'.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _download_with_retry(url: str, save_path: Path) -> None:
    """Download a URL to disk with exponential backoff retry."""
    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "CIVIC-SAFE/1.0 (Research)")
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                data = resp.read()
                save_path.write_bytes(data)
                logger.info(f"  Downloaded {len(data):,} bytes to {save_path.name}")
                return
        except Exception as e:
            wait = _BACKOFF_BASE ** attempt
            logger.warning(
                f"  Download attempt {attempt + 1}/{_MAX_RETRIES} failed: {e}. "
                f"Retrying in {wait:.1f}s..."
            )
            time.sleep(wait)

    raise RuntimeError(f"Failed to download after {_MAX_RETRIES} retries: {url[:100]}...")
