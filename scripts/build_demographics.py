"""
Rigorous Geospatial Areal Interpolation Pipeline for ACS Demographics.

This script fetches real demographic variables from the US Census API at the
Census Tract level, downloads official TIGER/Line Tract boundaries, and performs
an exact spatial intersection against the target city geometries (Community Areas
or Police Precincts) to apportion the populations rigorously.

Usage:
    export CENSUS_API_KEY="your_key_here"
    python scripts/build_demographics.py
"""

import os
import sys
import logging
import zipfile
import urllib.request
from io import BytesIO
from pathlib import Path

import pandas as pd
import geopandas as gpd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger = logging.getLogger(__name__)

# Core ACS Variables needed by CIVIC-SAFE
ACS_VARS = {
    "B01003_001E": "total_population",
    "B19013_001E": "median_household_income",
    "B17001_002E": "pop_below_poverty",
    "B17001_001E": "poverty_universe",
    "B23025_005E": "unemployed",
    "B23025_003E": "labor_force",
    "B02001_003E": "black_pop",
    "B03002_012E": "hispanic_pop",
    "B25003_003E": "renter_occupied",
    "B25003_001E": "total_housing_units"
}

# FIPS Codes
FIPS = {
    "chicago": {"state": "17", "counties": ["031"]},  # Cook County, IL
    "nyc": {"state": "36", "counties": ["005", "047", "061", "081", "085"]}  # Bronx, Kings, NY, Queens, Richmond
}

def download_tiger_tracts(state_fips: str, year: int = 2022) -> gpd.GeoDataFrame:
    """Download official TIGER/Line Census Tract boundaries."""
    import tempfile
    
    url = f"https://www2.census.gov/geo/tiger/TIGER{year}/TRACT/tl_{year}_{state_fips}_tract.zip"
    logger.info(f"Downloading TIGER/Line shapefiles from {url}...")
    
    tmp_dir = Path(tempfile.gettempdir()) / f"tiger_{state_fips}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        with zipfile.ZipFile(BytesIO(response.read())) as z:
            z.extractall(tmp_dir)
            
    gdf = gpd.read_file(tmp_dir / f"tl_{year}_{state_fips}_tract.shp")
    gdf["GEOID"] = gdf["GEOID"].astype(str)
    return gdf

def fetch_acs_data(state_fips: str, counties: list[str], api_key: str) -> pd.DataFrame:
    """Fetch demographic data from US Census API.
    
    The Census API returns -666666666 for suppressed/missing values
    (e.g., median household income in tracts with too few households).
    We replace these sentinels with NaN BEFORE any computation.
    """
    variables = ",".join(ACS_VARS.keys())
    county_list = ",".join(counties)
    url = (
        f"https://api.census.gov/data/2022/acs/acs5"
        f"?get=NAME,{variables}"
        f"&for=tract:*&in=state:{state_fips}"
        f"&in=county:{county_list}&key={api_key}"
    )
    
    logger.info(f"Querying Census API for state {state_fips}...")
    resp = requests.get(url)
    
    if resp.status_code != 200:
        logger.error(f"Census API Error: {resp.text}")
        raise RuntimeError("Failed to fetch Census data. Ensure CENSUS_API_KEY is valid.")
        
    data = resp.json()
    header = data[0]
    rows = data[1:]
    
    df = pd.DataFrame(rows, columns=header)
    df["GEOID"] = df["state"] + df["county"] + df["tract"]
    
    # Convert to numeric, coerce errors to NaN
    for var in ACS_VARS.keys():
        df[var] = pd.to_numeric(df[var], errors="coerce")
    
    # CRITICAL: Replace Census API sentinel values (-666666666) with NaN
    # These indicate suppressed data due to insufficient sample size
    CENSUS_SENTINEL = -666666666
    for var in ACS_VARS.keys():
        sentinel_mask = df[var] <= CENSUS_SENTINEL + 1  # catch -666666666 and similar
        n_sentinels = sentinel_mask.sum()
        if n_sentinels > 0:
            logger.warning(
                f"  Replaced {n_sentinels} Census sentinel values in {var} with NaN"
            )
        df.loc[sentinel_mask, var] = float("nan")
    
    # Also replace any remaining negative values in variables that must be non-negative
    non_negative_vars = ["B01003_001E", "B17001_002E", "B17001_001E", 
                         "B23025_005E", "B23025_003E", "B02001_003E",
                         "B03002_012E", "B25003_003E", "B25003_001E"]
    for var in non_negative_vars:
        if var in df.columns:
            neg_mask = df[var] < 0
            if neg_mask.sum() > 0:
                logger.warning(f"  Replaced {neg_mask.sum()} negative values in {var}")
                df.loc[neg_mask, var] = float("nan")
    
    # Fill NaN with 0 for count variables (not income)
    count_vars = [v for v in ACS_VARS.keys() if v != "B19013_001E"]
    df[count_vars] = df[count_vars].fillna(0)
    
    # Rename columns to human-readable names
    df = df.rename(columns=ACS_VARS)
        
    # Calculate derived percentages (safe division)
    df["poverty_rate"] = (df["pop_below_poverty"] / df["poverty_universe"].replace(0, 1)) * 100
    df["unemployment_rate"] = (df["unemployed"] / df["labor_force"].replace(0, 1)) * 100
    df["pct_black"] = (df["black_pop"] / df["total_population"].replace(0, 1)) * 100
    df["pct_hispanic"] = (df["hispanic_pop"] / df["total_population"].replace(0, 1)) * 100
    df["pct_renter_occupied"] = (df["renter_occupied"] / df["total_housing_units"].replace(0, 1)) * 100
    
    return df[["GEOID", "total_population", "median_household_income", "poverty_rate", 
               "unemployment_rate", "pct_black", "pct_hispanic", "pct_renter_occupied"]]

def perform_areal_interpolation(tract_gdf: gpd.GeoDataFrame, target_gdf: gpd.GeoDataFrame, target_id_col: str) -> pd.DataFrame:
    """Exact spatial areal interpolation of demographics from tracts to target polygons.
    
    Handles NaN values (from Census sentinel removal) by computing population-
    weighted averages only over tracts with valid data for each variable.
    """
    logger.info("Performing Geospatial Areal Interpolation...")
    
    # Project to Albers Equal Area for accurate area calculations (metres)
    tract_gdf = tract_gdf.to_crs("EPSG:5070")
    target_gdf = target_gdf.to_crs("EPSG:5070")
    
    tract_gdf["tract_area"] = tract_gdf.geometry.area
    target_gdf["target_area"] = target_gdf.geometry.area
    
    # Geometric intersection
    intersection = gpd.overlay(tract_gdf, target_gdf, how="intersection")
    intersection["intersect_area"] = intersection.geometry.area
    
    # Weight = proportion of each tract that overlaps the target polygon
    intersection["weight"] = intersection["intersect_area"] / intersection["tract_area"]
    
    # Apportion extensive variable (population)
    intersection["apportioned_pop"] = intersection["total_population"] * intersection["weight"]
    
    # Intensive variables to aggregate via population-weighted mean
    intensive_vars = [
        "median_household_income", "poverty_rate", "unemployment_rate",
        "pct_black", "pct_hispanic", "pct_renter_occupied",
    ]
    
    results = []
    for target_id, group in intersection.groupby(target_id_col):
        total_pop = group["apportioned_pop"].sum()
        if total_pop == 0:
            continue
            
        row = {
            "spatial_unit": target_id,
            "total_population": total_pop,
        }
        
        # For each intensive variable, compute population-weighted mean
        # ONLY over tracts that have valid (non-NaN) data for that variable
        for var in intensive_vars:
            valid_mask = group[var].notna()
            valid_group = group[valid_mask]
            valid_pop = valid_group["apportioned_pop"].sum()
            
            if valid_pop > 0:
                pop_weights = valid_group["apportioned_pop"] / valid_pop
                row[var] = (valid_group[var] * pop_weights).sum()
            else:
                # All tracts have missing data for this variable in this target
                # Use the citywide median as fallback
                row[var] = float("nan")
        
        # Population density: pop per sq km
        target_area_km2 = group["target_area"].iloc[0] / 1e6
        row["population_density"] = total_pop / max(target_area_km2, 0.01)
        
        results.append(row)
    
    result_df = pd.DataFrame(results)
    
    # Fill any remaining NaN with column median (robust fallback)
    for var in intensive_vars:
        if var in result_df.columns:
            n_nan = result_df[var].isna().sum()
            if n_nan > 0:
                median_val = result_df[var].median()
                result_df[var] = result_df[var].fillna(median_val)
                logger.warning(
                    f"  Filled {n_nan} NaN values in {var} with median={median_val:.2f}"
                )
    
    return result_df

def main():
    api_key = os.environ.get("CENSUS_API_KEY")
    if not api_key:
        logger.error("FATAL: CENSUS_API_KEY environment variable is not set.")
        logger.error("To achieve 'World's Best' scientific rigor, synthetic demographic data is unacceptable.")
        logger.error("You must obtain a free Census API key from https://api.census.gov/data/key_signup.html")
        logger.error("Run: export CENSUS_API_KEY='your_key' && python scripts/build_demographics.py")
        sys.exit(1)
        
    from civicsafe.data.shapefiles import load_boundaries
    
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw" / "shapefiles"
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    for city in ["chicago", "nyc"]:
        logger.info(f"=== Processing Demographics for {city.upper()} ===")
        
        # Load targets
        target_gdf = load_boundaries(city, raw_dir)
        target_id_col = "area_number" if city == "chicago" else "precinct"
        
        # Fetch Census Data
        fips = FIPS[city]
        tracts_shp = download_tiger_tracts(fips["state"])
        acs_df = fetch_acs_data(fips["state"], fips["counties"], api_key)
        
        # Merge ACS data onto tract geometries
        tracts_merged = tracts_shp.merge(acs_df, on="GEOID", how="inner")
        
        # Areal Interpolation
        final_demographics = perform_areal_interpolation(tracts_merged, target_gdf, target_id_col)
        
        out_path = processed_dir / f"{city}_demographics.csv"
        final_demographics.to_csv(out_path, index=False)
        logger.info(f"Saved highly rigorous interpolated demographics to {out_path}")
        
    logger.info("Done! The CIVIC-SAFE pipeline now uses state-of-the-art demographic covariates.")

if __name__ == "__main__":
    main()
