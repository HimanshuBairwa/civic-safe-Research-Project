"""Downloader and preprocessor for Chicago Crimes dataset via SODA API."""
from __future__ import annotations

import hashlib
import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from civicsafe.data.taxonomy import CHICAGO_MAPPING, get_unified_category

logger = logging.getLogger(__name__)


def fetch_chicago_year(year: int, save_dir: Path) -> Path:
    """Fetch one year of Chicago crime data using SODA SoQL and cache as Parquet."""
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / f"chicago_crimes_{year}.parquet"
    sha_file = save_dir / f"chicago_crimes_{year}.parquet.sha256"

    # Reproducibility check: return cached version if SHA-256 matches
    if out_file.exists() and sha_file.exists():
        with open(out_file, "rb") as f, open(sha_file, "r") as sf:
            if hashlib.sha256(f.read()).hexdigest() == sf.read().strip():
                logger.info(f"Using verified cached Chicago {year} data.")
                return out_file

    logger.info(f"Downloading Chicago crime data for {year} via SoQL...")
    
    types_list = "','".join(CHICAGO_MAPPING.keys())
    where_clause = (
        f"date >= '{year}-01-01T00:00:00' AND "
        f"date <= '{year}-12-31T23:59:59' AND "
        f"community_area IS NOT NULL AND "
        f"primary_type IN ('{types_list}')"
    )
    select_clause = "id,date,primary_type,community_area,latitude,longitude"
    base_url = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
    
    limit = 50000
    offset = 0
    all_records = []
    
    while True:
        query = {
            "$select": select_clause,
            "$where": where_clause,
            "$limit": limit,
            "$offset": offset,
            "$order": "date ASC"
        }
        url = f"{base_url}?{urllib.parse.urlencode(query)}"
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            if not data:
                break
            all_records.extend(data)
            offset += limit
            logger.info(f"  Fetched {len(all_records)} records for {year}...")

    df = pd.DataFrame(all_records)
    if df.empty:
        logger.warning(f"No records found for Chicago {year}")
        df = pd.DataFrame(
            columns=["id", "date", "primary_type", "community_area", "latitude", "longitude"]
        )
    else:
        df["date"] = pd.to_datetime(df["date"])
        df["spatial_unit"] = df["community_area"].astype(float).astype(int)
        df["latitude"] = df["latitude"].astype(float)
        df["longitude"] = df["longitude"].astype(float)
        df["id"] = df["id"].astype(str)
        df["category"] = df["primary_type"].apply(lambda x: get_unified_category("chicago", x))
        
        df = df.drop_duplicates(subset=["id"])
        # Keep only essential normalized columns
        df = df[["id", "date", "spatial_unit", "category", "latitude", "longitude"]]

    df.to_parquet(out_file, index=False)
    
    with open(out_file, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()
    with open(sha_file, "w") as f:
        f.write(sha256_hash)
        
    return out_file


def process_chicago_crimes(start_year: int, end_year: int, save_dir: Path) -> pd.DataFrame:
    """Download and combine Chicago crimes for a year range."""
    dfs = []
    for year in range(start_year, end_year + 1):
        file_path = fetch_chicago_year(year, save_dir)
        dfs.append(pd.read_parquet(file_path))
    
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)
