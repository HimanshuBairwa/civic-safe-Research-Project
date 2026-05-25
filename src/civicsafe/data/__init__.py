"""Data loading, harmonization, and crosswalk modules."""
from __future__ import annotations

from civicsafe.data.acs import load_acs_features
from civicsafe.data.chicago import fetch_chicago_year, process_chicago_crimes
from civicsafe.data.crosswalks import get_census_crosswalk
from civicsafe.data.nyc import fetch_nyc_year, process_nyc_crimes
from civicsafe.data.panel import build_spatiotemporal_panel
from civicsafe.data.taxonomy import (
    CHICAGO_MAPPING,
    NYC_MAPPING,
    VIOLENT,
    PROPERTY,
    DRUG,
    get_unified_category,
)

__all__ = [
    "load_acs_features",
    "fetch_chicago_year",
    "process_chicago_crimes",
    "get_census_crosswalk",
    "fetch_nyc_year",
    "process_nyc_crimes",
    "build_spatiotemporal_panel",
    "CHICAGO_MAPPING",
    "NYC_MAPPING",
    "VIOLENT",
    "PROPERTY",
    "DRUG",
    "get_unified_category",
]
