"""Downloader and preprocessor for NYC NYPD Complaint dataset via SODA API."""
from __future__ import annotations

import hashlib
import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

from civicsafe.data.taxonomy import NYC_MAPPING, get_unified_category

logger = logging.getLogger(__name__)


def fetch_nyc_year(year: int, save_dir: Path) -> Path:
    """Fetch one year of NYC complaint data using SODA SoQL and cache as Parquet."""
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / f"nyc_crimes_{year}.parquet"
    sha_file = save_dir / f"nyc_crimes_{year}.parquet.sha256"

    # Reproducibility check
    if out_file.exists() and sha_file.exists():
        with open(out_file, "rb") as f, open(sha_file, "r") as sf:
            if hashlib.sha256(f.read()).hexdigest() == sf.read().strip():
                logger.info(f"Using verified cached NYC {year} data.")
                return out_file

    logger.info(f"Downloading NYC crime data for {year} via SoQL...")
    
    ky_cds = ",".join(str(k) for k in NYC_MAPPING.keys())
    where_clause = (
        f"cmplnt_fr_dt >= '{year}-01-01T00:00:00' AND "
        f"cmplnt_fr_dt <= '{year}-12-31T23:59:59' AND "
        f"addr_pct_cd IS NOT NULL AND "
        f"ky_cd IN ({ky_cds})"
    )
    select_clause = "cmplnt_num,cmplnt_fr_dt,ky_cd,addr_pct_cd,latitude,longitude"
    base_url = "https://data.cityofnewyork.us/resource/qgea-i56i.json"
    
    limit = 50000
    offset = 0
    all_records = []
    
    while True:
        query = {
            "$select": select_clause,
            "$where": where_clause,
            "$limit": limit,
            "$offset": offset,
            "$order": "cmplnt_fr_dt ASC"
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
        logger.warning(f"No records found for NYC {year}")
        df = pd.DataFrame(
            columns=["id", "date", "spatial_unit", "category", "latitude", "longitude"]
        )
    else:
        df = df.rename(columns={
            "cmplnt_num": "id",
            "cmplnt_fr_dt": "date",
            "addr_pct_cd": "spatial_unit",
        })
        df["date"] = pd.to_datetime(df["date"])
        df["ky_cd"] = df["ky_cd"].astype(int)
        df["spatial_unit"] = df["spatial_unit"].astype(float).astype(int)
        df["latitude"] = df["latitude"].astype(float)
        df["longitude"] = df["longitude"].astype(float)
        df["id"] = df["id"].astype(str)
        df["category"] = df["ky_cd"].apply(lambda x: get_unified_category("nyc", x))
        
        df = df.drop_duplicates(subset=["id"])
        df = df[["id", "date", "spatial_unit", "category", "latitude", "longitude"]]

    df.to_parquet(out_file, index=False)
    
    with open(out_file, "rb") as f:
        sha256_hash = hashlib.sha256(f.read()).hexdigest()
    with open(sha_file, "w") as f:
        f.write(sha256_hash)
        
    return out_file


def process_nyc_crimes(start_year: int, end_year: int, save_dir: Path) -> pd.DataFrame:
    """Download and combine NYC crimes for a year range."""
    dfs = []
    for year in range(start_year, end_year + 1):
        file_path = fetch_nyc_year(year, save_dir)
        dfs.append(pd.read_parquet(file_path))
    
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)
