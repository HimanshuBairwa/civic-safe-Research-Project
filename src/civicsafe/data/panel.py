"""Master builder for the spatiotemporal panel and chronologically frozen splits."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def build_spatiotemporal_panel(
    crime_df: pd.DataFrame,
    acs_df: pd.DataFrame,
    crosswalk_df: pd.DataFrame,
    start_year: int,
    end_year: int,
    taxonomy_categories: list[str] = ["violent", "property", "drug"]
) -> dict[str, Tensor]:
    """Integrate crime counts and demographics into tensor panel."""
    logger.info("Building spatiotemporal tensor panel...")
    
    # 1. Temporal binning (weekly)
    if not crime_df.empty:
        crime_df["time_block"] = crime_df["date"].dt.to_period("W").dt.start_time
        weekly_counts = (
            crime_df.groupby(["spatial_unit", "time_block", "category"])
            .size()
            .reset_index(name="count")
        )
    else:
        weekly_counts = pd.DataFrame(
            columns=["spatial_unit", "time_block", "category", "count"]
        )
    
    # 2. Get unique spatial units
    spatial_units = sorted(crosswalk_df["spatial_unit"].unique())
    num_spatial = len(spatial_units)
    
    # Generate full time grid
    date_range = pd.date_range(
        start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq="W"
    )
    num_time = len(date_range)
    num_categories = len(taxonomy_categories)
    
    logger.info(
        f"Panel shape target: {num_spatial} spatial units, {num_time} weeks, {num_categories} categories."
    )
    
    counts_tensor = torch.zeros((num_spatial, num_time, num_categories), dtype=torch.long)
    features_tensor = torch.zeros(
        (num_spatial, num_time, len(acs_df.columns) - 1), dtype=torch.float32
    )
    
    # Mapping dictionaries
    su_to_idx = {su: i for i, su in enumerate(spatial_units)}
    cat_to_idx = {cat: i for i, cat in enumerate(taxonomy_categories)}
    
    # Populate counts
    if not weekly_counts.empty:
        for _, row in weekly_counts.iterrows():
            if row["spatial_unit"] in su_to_idx and row["category"] in cat_to_idx:
                # Find nearest week index
                t_idx = None
                for idx, d in enumerate(date_range):
                    if d >= row["time_block"]:
                        t_idx = idx
                        break
                if t_idx is not None:
                    counts_tensor[
                        su_to_idx[row["spatial_unit"]], 
                        t_idx, 
                        cat_to_idx[row["category"]]
                    ] += int(row["count"])

    # Map ACS features using crosswalk (population-weighted sum)
    acs_cols = [c for c in acs_df.columns if c != "tract_id"]
    for su in spatial_units:
        su_tracts = crosswalk_df[crosswalk_df["spatial_unit"] == su]
        su_idx = su_to_idx[su]
        
        merged = pd.merge(su_tracts, acs_df, on="tract_id")
        if not merged.empty:
            weighted_vars = []
            for col in acs_cols:
                weighted_val = (merged["weight"] * merged[col]).sum()
                weighted_vars.append(weighted_val)
            
            f_vec = torch.tensor(weighted_vars, dtype=torch.float32)
            features_tensor[su_idx, :, :] = f_vec.unsqueeze(0).expand(num_time, -1)

    return {
        "counts": counts_tensor,
        "features": features_tensor,
        "metadata": {
            "spatial_units": spatial_units,
            "time_range": [start_year, end_year],
            "categories": taxonomy_categories,
        }
    }
