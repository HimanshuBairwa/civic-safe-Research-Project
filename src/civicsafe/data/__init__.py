"""Data loading, harmonization, and crosswalk modules."""

from __future__ import annotations

from civicsafe.data.acs import load_acs_features
from civicsafe.data.chicago import fetch_chicago_year, process_chicago_crimes
from civicsafe.data.crosswalks import get_census_crosswalk
from civicsafe.data.nyc import fetch_nyc_year, process_nyc_crimes
from civicsafe.data.panel import build_spatiotemporal_panel
from civicsafe.data.taxonomy import (
    CHICAGO_MAPPING,
    DRUG,
    NYC_MAPPING,
    PROPERTY,
    VIOLENT,
    get_unified_category,
)

__all__ = [
    "CHICAGO_MAPPING",
    "DRUG",
    "NYC_MAPPING",
    "PROPERTY",
    "VIOLENT",
    "build_spatiotemporal_panel",
    "fetch_chicago_year",
    "fetch_nyc_year",
    "get_census_crosswalk",
    "get_unified_category",
    "load_acs_features",
    "process_chicago_crimes",
    "process_nyc_crimes",
]
