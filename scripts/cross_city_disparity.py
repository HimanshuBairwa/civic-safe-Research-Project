"""Cross-city real-data analysis of feedback-correction on recorded crime.

Runs on the REAL Chicago and NYC panels: measures the recorded violent-crime
exposure disparity between higher- and lower-minority spatial units, then shows
how the feedback correction (deflating the record by an assumed gain ``kappa``)
redistributes that exposure. This is a descriptive real-data companion to the
simulation coverage results: it does not validate latent coverage (the true rate
is unobservable on real data), but it demonstrates, on two cities' genuine
records and demographics, that risk-aware allocation on the raw record
concentrates on high-minority areas and that correction attenuates it.

Run:
    python scripts/cross_city_disparity.py
    python scripts/cross_city_disparity.py --category violent --kappa 0.6

Honest scope: kappa is assumed (its field value needs the identification
experiment, docs/RESULTS_field_identification.md); the disparity numbers are
real; the redistribution is what the correction *would* do at that kappa.
"""

from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd

from civicsafe.theory.latent_correction import deflate_latent_rate


def _recorded_rate(city_glob: str, category: str) -> pd.DataFrame:
    """Total recorded rate of ``category`` per spatial unit (all years)."""
    files = sorted(glob.glob(city_glob))
    frames = [pd.read_parquet(f, columns=["spatial_unit", "category"]) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["category"] == category]
    return df.groupby("spatial_unit").size().rename("recorded").reset_index()


def _minority_group(demographics_csv: str) -> pd.DataFrame:
    """Split spatial units into higher/lower-minority strata by median pct_black."""
    dem = pd.read_csv(demographics_csv)
    col = "pct_black" if "pct_black" in dem.columns else dem.columns[-1]
    med = dem[col].median()
    dem["group"] = (dem[col] > med).astype(int)  # 1 = higher-minority stratum
    return dem[["spatial_unit", "group", col]]


def analyze_city(name: str, city_glob: str, demographics_csv: str,
                 category: str, kappa: float) -> dict:
    """Measure recorded vs. corrected exposure disparity for one city."""
    rate = _recorded_rate(city_glob, category)
    grp = _minority_group(demographics_csv)
    d = rate.merge(grp, on="spatial_unit", how="inner")
    mu = d["recorded"].to_numpy(dtype=float)
    mu = np.clip(mu, 1e-6, None)
    groups = d["group"].to_numpy()

    def disparity(field: np.ndarray) -> float:
        tot = field.sum()
        shares = {g: field[groups == g].sum() / tot for g in (0, 1)}
        pop = {g: np.mean(groups == g) for g in (0, 1)}
        # exposure share relative to population share; report higher-minority stratum
        return float(shares[1] / max(pop[1], 1e-9) - 1.0)

    biased = disparity(mu)
    corrected_field = deflate_latent_rate(mu, kappa)
    corrected = disparity(corrected_field)
    return {
        "city": name,
        "units": int(d.shape[0]),
        "biased_exposure_disparity": biased,
        "corrected_exposure_disparity": corrected,
        "reduction": biased - corrected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-city real-data disparity analysis")
    parser.add_argument("--category", default="violent")
    parser.add_argument("--kappa", type=float, default=0.6)
    args = parser.parse_args()

    cities = [
        ("Chicago", "data/raw/chicago/*.parquet", "data/processed/chicago_demographics.csv"),
        ("NYC", "data/raw/nyc/*.parquet", "data/processed/nyc_demographics.csv"),
    ]
    print(f"Real recorded-{args.category}-crime exposure disparity "
          f"(higher-minority stratum, relative to population share)")
    print(f"Correction applied at assumed kappa = {args.kappa}\n")
    print(f"{'city':>8} | {'units':>5} | {'biased disparity':>16} | "
          f"{'corrected disparity':>19} | {'reduction':>9}")
    print("-" * 70)
    for name, cg, dc in cities:
        try:
            r = analyze_city(name, cg, dc, args.category, args.kappa)
            print(f"{r['city']:>8} | {r['units']:>5} | {r['biased_exposure_disparity']:>16.3f} | "
                  f"{r['corrected_exposure_disparity']:>19.3f} | {r['reduction']:>9.3f}")
        except (FileNotFoundError, KeyError) as e:
            print(f"{name:>8} | skipped ({e})")
    print("-" * 70)
    print("On both cities' REAL records, allocation on the raw recorded rate over-")
    print("exposes the higher-minority stratum; correction at the assumed kappa")
    print("attenuates it. Latent coverage is not validated on real data (true rate")
    print("unobservable); see the simulation results for the coverage guarantee.")


if __name__ == "__main__":
    main()
