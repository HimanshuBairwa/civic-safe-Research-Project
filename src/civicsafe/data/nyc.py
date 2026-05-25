"""Downloader and preprocessor for NYC NYPD Complaint dataset via SODA API.

Endpoint: https://data.cityofnewyork.us/resource/qgea-i56i.json
Dataset:  "NYPD Complaint Data Historic"
Verified: 2025-05-25 — field names are lowercase (SODA is case-sensitive).
          KY_CD codes verified against live groupby query.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.parse
import urllib.request
from typing import TYPE_CHECKING

import pandas as pd

from civicsafe.data.taxonomy import NYC_MAPPING, get_unified_category

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BASE_URL = "https://data.cityofnewyork.us/resource/qgea-i56i.json"
_SELECT = "cmplnt_num,cmplnt_fr_dt,ky_cd,addr_pct_cd,latitude,longitude"
_PAGE_SIZE = 50_000
_TIMEOUT_SECONDS = 120
_MAX_RETRIES = 5
_BACKOFF_BASE = 2.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def fetch_nyc_year(year: int, save_dir: Path) -> Path:
    """Fetch one year of NYC complaint data via SoQL and cache as Parquet.

    Uses server-side filtering to minimize bandwidth. Implements:
    - SHA-256 verified caching
    - Exponential backoff retry on 429/5xx errors
    - Per-request timeout to prevent kernel hangs
    - Null-safe lat/lon handling (NYC masks sex-crime coordinates)

    Args:
        year: Calendar year to fetch.
        save_dir: Directory for cached Parquet + SHA files.

    Returns:
        Path to the written Parquet file.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    out_file = save_dir / f"nyc_crimes_{year}.parquet"
    sha_file = save_dir / f"nyc_crimes_{year}.parquet.sha256"

    # --- Cache check ---
    if out_file.exists() and sha_file.exists():
        if _verify_sha256(out_file, sha_file):
            logger.info(f"Using verified cached NYC {year} data.")
            return out_file

    logger.info(f"Downloading NYC crime data for {year} via SoQL...")

    # --- Build SoQL WHERE clause ---
    ky_cds = ",".join(str(k) for k in NYC_MAPPING)
    where_clause = (
        f"cmplnt_fr_dt >= '{year}-01-01T00:00:00' AND "
        f"cmplnt_fr_dt <= '{year}-12-31T23:59:59' AND "
        f"addr_pct_cd IS NOT NULL AND "
        f"ky_cd IN ({ky_cds})"
    )

    # --- Paginated fetch with retry ---
    all_records: list[dict] = []
    offset = 0

    while True:
        query = {
            "$select": _SELECT,
            "$where": where_clause,
            "$limit": _PAGE_SIZE,
            "$offset": offset,
            "$order": "cmplnt_fr_dt ASC",
        }
        url = f"{_BASE_URL}?{urllib.parse.urlencode(query)}"
        data = _fetch_with_retry(url)

        if not data:
            break

        all_records.extend(data)
        offset += _PAGE_SIZE
        logger.info(f"  NYC {year}: {len(all_records):,} records fetched...")

    # --- Normalize to DataFrame ---
    df = _normalize_nyc_df(all_records)

    logger.info(f"  NYC {year}: {len(df):,} records after dedup and normalization.")

    # --- Write Parquet + SHA-256 ---
    df.to_parquet(out_file, index=False, engine="pyarrow")
    _write_sha256(out_file, sha_file)

    return out_file


def process_nyc_crimes(start_year: int, end_year: int, save_dir: Path) -> pd.DataFrame:
    """Download and combine NYC crimes for a year range.

    Args:
        start_year: First year (inclusive).
        end_year: Last year (inclusive).
        save_dir: Cache directory.

    Returns:
        Combined DataFrame with columns:
        [id, date, spatial_unit, category, latitude, longitude]
    """
    dfs = []
    for year in range(start_year, end_year + 1):
        file_path = fetch_nyc_year(year, save_dir)
        dfs.append(pd.read_parquet(file_path))

    if not dfs:
        return pd.DataFrame(
            columns=["id", "date", "spatial_unit", "category", "latitude", "longitude"]
        )
    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _normalize_nyc_df(records: list[dict]) -> pd.DataFrame:
    """Convert raw JSON records to a clean, typed DataFrame."""
    if not records:
        return pd.DataFrame(
            columns=["id", "date", "spatial_unit", "category", "latitude", "longitude"]
        )

    df = pd.DataFrame(records)

    # Rename API field names to our canonical schema
    df = df.rename(
        columns={
            "cmplnt_num": "id",
            "cmplnt_fr_dt": "date",
            "addr_pct_cd": "spatial_unit",
        }
    )

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ky_cd"] = pd.to_numeric(df["ky_cd"], errors="coerce").fillna(-1).astype(int)
    df["spatial_unit"] = (
        pd.to_numeric(df["spatial_unit"], errors="coerce").fillna(-1).astype(int)
    )
    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    df["id"] = df["id"].astype(str)
    df["category"] = df["ky_cd"].apply(lambda x: get_unified_category("nyc", x))

    # Drop records with invalid spatial unit or unmapped category
    df = df[df["spatial_unit"] > 0]
    df = df[df["category"].notna()]
    df = df.drop_duplicates(subset=["id"])

    return df[["id", "date", "spatial_unit", "category", "latitude", "longitude"]]


def _fetch_with_retry(url: str) -> list[dict]:
    """Fetch a single SODA page with exponential backoff retry."""
    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=_TIMEOUT_SECONDS) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                wait = _BACKOFF_BASE**attempt
                logger.warning(
                    f"  HTTP {e.code} on attempt {attempt+1}/{_MAX_RETRIES}. "
                    f"Retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
            else:
                raise
        except (urllib.error.URLError, TimeoutError) as e:
            wait = _BACKOFF_BASE**attempt
            logger.warning(
                f"  Network error on attempt {attempt+1}/{_MAX_RETRIES}: {e}. "
                f"Retrying in {wait:.1f}s..."
            )
            time.sleep(wait)

    raise RuntimeError(f"Failed to fetch after {_MAX_RETRIES} retries: {url[:120]}...")


def _verify_sha256(data_file: Path, sha_file: Path) -> bool:
    """Verify SHA-256 using chunked reading (no full file in memory)."""
    try:
        expected = sha_file.read_text().strip()
        h = hashlib.sha256()
        with open(data_file, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest() == expected
    except Exception:
        return False


def _write_sha256(data_file: Path, sha_file: Path) -> None:
    """Write SHA-256 hash of data_file to sha_file (chunked)."""
    h = hashlib.sha256()
    with open(data_file, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    sha_file.write_text(h.hexdigest())
