"""Tests for Phase 1 Data Acquisition and Harmonization modules."""
from __future__ import annotations

import pytest

from civicsafe.data.acs import load_acs_features
from civicsafe.data.crosswalks import get_census_crosswalk
from civicsafe.data.taxonomy import get_unified_category


def test_taxonomy_chicago() -> None:
    assert get_unified_category("chicago", "HOMICIDE") == "violent"
    assert get_unified_category("chicago", "theft") == "property"
    assert get_unified_category("chicago", "UNKNOWN") is None


def test_taxonomy_nyc() -> None:
    assert get_unified_category("nyc", 101) == "violent"
    assert get_unified_category("nyc", "101") == "violent"
    assert get_unified_category("nyc", 999) is None


def test_crosswalks() -> None:
    cw = get_census_crosswalk("chicago")
    assert not cw.empty
    # Weights should sum to 1 per spatial unit
    sums = cw.groupby("spatial_unit")["weight"].sum()
    assert (sums - 1.0).abs().max() < 1e-6


def test_resilient_acs() -> None:
    df = load_acs_features("chicago", ["median_income"])
    assert "tract_id" in df.columns
    assert "median_income" in df.columns
    assert not df.empty
