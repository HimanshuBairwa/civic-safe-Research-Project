"""Tests for Phase 1 Data Acquisition and Harmonization modules.

Tests cover:
  - Taxonomy mapping correctness (verified against live API data)
  - Crosswalk weight normalization
  - Resilient ACS fallback
  - Panel builder shape correctness
  - Edge cases (null categories, empty DataFrames)
"""

from __future__ import annotations

import pandas as pd
import pytest

from civicsafe.data.acs import load_acs_features
from civicsafe.data.crosswalks import get_census_crosswalk
from civicsafe.data.panel import build_spatiotemporal_panel
from civicsafe.data.taxonomy import (
    NYC_MAPPING,
    get_unified_category,
)

# ===================================================================
# Taxonomy
# ===================================================================


class TestChicagoTaxonomy:
    """Verify Chicago primary_type mappings against live API data."""

    def test_homicide_is_violent(self) -> None:
        assert get_unified_category("chicago", "HOMICIDE") == "violent"

    def test_battery_is_violent(self) -> None:
        assert get_unified_category("chicago", "BATTERY") == "violent"

    def test_both_sexual_assault_variants(self) -> None:
        """Live API has TWO variants — both must map to violent."""
        assert get_unified_category("chicago", "CRIM SEXUAL ASSAULT") == "violent"
        assert get_unified_category("chicago", "CRIMINAL SEXUAL ASSAULT") == "violent"

    def test_criminal_damage_is_property(self) -> None:
        """972K records — must not be silently dropped."""
        assert get_unified_category("chicago", "CRIMINAL DAMAGE") == "property"

    def test_weapons_violation_is_violent(self) -> None:
        assert get_unified_category("chicago", "WEAPONS VIOLATION") == "violent"

    def test_theft_is_property(self) -> None:
        assert get_unified_category("chicago", "THEFT") == "property"

    def test_case_insensitive(self) -> None:
        """get_unified_category uppercases the input."""
        assert get_unified_category("chicago", "theft") == "property"
        assert get_unified_category("chicago", "Homicide") == "violent"

    def test_unknown_returns_none(self) -> None:
        assert get_unified_category("chicago", "UNKNOWN_TYPE") is None
        assert get_unified_category("chicago", "") is None

    def test_narcotics_is_drug(self) -> None:
        assert get_unified_category("chicago", "NARCOTICS") == "drug"


class TestNYCTaxonomy:
    """Verify NYC KY_CD mappings against live API data."""

    def test_murder_is_violent(self) -> None:
        assert get_unified_category("nyc", 101) == "violent"

    def test_robbery_is_violent(self) -> None:
        assert get_unified_category("nyc", 105) == "violent"

    def test_grand_larceny_is_property(self) -> None:
        """KY_CD 109 is GRAND LARCENY (880K records), NOT kidnapping."""
        assert get_unified_category("nyc", 109) == "property"

    def test_kidnapping_correct_code(self) -> None:
        """KY_CD 124 is the real kidnapping code."""
        assert get_unified_category("nyc", 124) == "violent"

    def test_dangerous_drugs_correct_code(self) -> None:
        """KY_CD 235 is DANGEROUS DRUGS (373K records), not 230."""
        assert get_unified_category("nyc", 235) == "drug"
        assert get_unified_category("nyc", 117) == "drug"

    def test_burglary_is_property(self) -> None:
        assert get_unified_category("nyc", 107) == "property"

    def test_motor_vehicle_theft_is_property(self) -> None:
        assert get_unified_category("nyc", 110) == "property"

    def test_string_code_works(self) -> None:
        """API returns KY_CD as strings — must handle conversion."""
        assert get_unified_category("nyc", "101") == "violent"
        assert get_unified_category("nyc", "341") == "property"

    def test_unknown_code_returns_none(self) -> None:
        assert get_unified_category("nyc", 999) is None
        assert get_unified_category("nyc", "invalid") is None

    def test_old_wrong_codes_not_present(self) -> None:
        """Verify the old incorrect codes are NOT in the mapping."""
        assert 230 not in NYC_MAPPING  # Was wrongly mapped as drug
        assert 231 not in NYC_MAPPING  # Was wrongly mapped as drug


# ===================================================================
# Crosswalks
# ===================================================================


class TestCrosswalks:
    """Structural integrity of spatial crosswalk tables."""

    def test_chicago_crosswalk_weights_sum_to_one(self) -> None:
        cw = get_census_crosswalk("chicago")
        sums = cw.groupby("spatial_unit")["weight"].sum()
        assert (sums - 1.0).abs().max() < 1e-6, "Weights must sum to 1 per unit"

    def test_chicago_crosswalk_has_77_areas(self) -> None:
        cw = get_census_crosswalk("chicago")
        assert cw["spatial_unit"].nunique() == 77

    def test_nyc_crosswalk_weights_sum_to_one(self) -> None:
        cw = get_census_crosswalk("nyc")
        sums = cw.groupby("spatial_unit")["weight"].sum()
        assert (sums - 1.0).abs().max() < 1e-6

    def test_nyc_crosswalk_has_precincts(self) -> None:
        cw = get_census_crosswalk("nyc")
        assert cw["spatial_unit"].nunique() == 77

    def test_unknown_city_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown city"):
            get_census_crosswalk("london")


# ===================================================================
# Resilient ACS
# ===================================================================


class TestACS:
    """Verify the rigorous population-weighted ACS loader.

    ``load_acs_features`` reads the areal-interpolation output
    (``data/processed/{city}_demographics.csv``), keyed by ``spatial_unit`` with
    one row per unit — the current, non-simulated data path.
    """

    def test_returns_dataframe_keyed_by_spatial_unit(self) -> None:
        df = load_acs_features("chicago", ["median_household_income", "poverty_rate"])
        assert isinstance(df, pd.DataFrame)
        assert "spatial_unit" in df.columns
        assert "median_household_income" in df.columns
        assert "poverty_rate" in df.columns
        assert not df.empty
        # One row per spatial unit (Chicago community areas).
        assert df["spatial_unit"].is_unique

    def test_population_values_positive(self) -> None:
        df = load_acs_features("chicago", ["total_population"])
        assert (df["total_population"] >= 0).all()


# ===================================================================
# Panel Builder
# ===================================================================


class TestPanelBuilder:
    """Verify spatiotemporal panel construction."""

    @pytest.fixture
    def mini_inputs(self):
        """Minimal valid inputs for the current panel-builder API.

        ``acs_df`` is keyed by ``spatial_unit`` with one row per unit (the
        areal-interpolation output format); no separate crosswalk is needed at
        panel-build time.
        """
        crime_df = pd.DataFrame(
            {
                "id": ["1", "2", "3"],
                "date": pd.to_datetime(["2020-01-15", "2020-01-20", "2020-06-01"]),
                "spatial_unit": [1, 1, 2],
                "category": ["violent", "property", "drug"],
                "latitude": [41.8, 41.8, 41.9],
                "longitude": [-87.6, -87.6, -87.7],
            }
        )
        acs_df = pd.DataFrame(
            {
                "spatial_unit": [1, 2],
                "income": [50.0, 40.0],  # one ACS feature column
            }
        )
        return crime_df, acs_df

    def test_panel_shapes(self, mini_inputs) -> None:
        crime_df, acs_df = mini_inputs
        panel = build_spatiotemporal_panel(crime_df, acs_df, 2020, 2020)
        S = 2  # spatial units
        C = 3  # categories

        assert panel["counts"].shape[0] == S
        assert panel["counts"].shape[2] == C
        assert panel["features"].shape[0] == S
        assert panel["features"].shape[2] == 1  # one ACS variable

    def test_panel_counts_nonnegative(self, mini_inputs) -> None:
        crime_df, acs_df = mini_inputs
        panel = build_spatiotemporal_panel(crime_df, acs_df, 2020, 2020)
        assert (panel["counts"] >= 0).all()

    def test_panel_total_matches_input(self, mini_inputs) -> None:
        crime_df, acs_df = mini_inputs
        panel = build_spatiotemporal_panel(crime_df, acs_df, 2020, 2020)
        assert panel["counts"].sum().item() == 3  # 3 input records

    def test_panel_does_not_mutate_input(self, mini_inputs) -> None:
        crime_df, acs_df = mini_inputs
        cols_before = list(crime_df.columns)
        build_spatiotemporal_panel(crime_df, acs_df, 2020, 2020)
        assert list(crime_df.columns) == cols_before

    def test_panel_empty_crime_df(self, mini_inputs) -> None:
        _, acs_df = mini_inputs
        empty_df = pd.DataFrame(
            columns=["id", "date", "spatial_unit", "category", "latitude", "longitude"]
        )
        panel = build_spatiotemporal_panel(empty_df, acs_df, 2020, 2020)
        assert panel["counts"].sum().item() == 0

    def test_metadata_correct(self, mini_inputs) -> None:
        crime_df, acs_df = mini_inputs
        panel = build_spatiotemporal_panel(crime_df, acs_df, 2020, 2020)
        assert panel["metadata"]["categories"] == ["violent", "property", "drug"]
        assert panel["metadata"]["time_range"] == [2020, 2020]
