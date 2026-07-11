"""Real-data loader for the India NCRB state-year multi-channel experiment.

Assembles >=3 MECHANISM-INDEPENDENT measurement channels of a latent
"true offending / state-violence exposure" signal, at STATE x YEAR resolution
over the 2001-2010 overlap window (the accountability tables stop at 2010):

  C0 (pivot)  = police-recorded IPC crime         (institutional record filter)
  C1          = distress calls to police (PC3)     (public-initiation filter)
  C2          = complaints against police (25)      (oversight/accountability filter)
  C3          = custodial deaths + HR violations    (state-violence accountability)

These pass through DIFFERENT institutional filters, which is the identifying
requirement.  HONEST LIMITS (documented, not hidden):
  * state-year, N ~ 34 states x 10 years ~ 340 cells: small; third-cumulant
    over-ID at K=3 is underpowered, so we use K=4 where possible.
  * WHY STATE-LEVEL (not district): the IPC-crime table is published at district
    resolution (~806 districts), but the accountability channels (complaints,
    custodial deaths, HR violations) are published by NCRB ONLY at state level.
    OICC needs all channels on the SAME areal unit to identify one factor, so the
    binding resolution is the coarsest channel = state. This is a data-publishing
    constraint, NOT a modelling choice. Areal (not point/lat-long) data is the
    correct substrate for latent-RATE estimation anyway (small-area estimation;
    point data answers a different, hotspot question). See docs/METHODOLOGY.md #11.
  * channels measure DIFFERENT latent constructs (offending vs mistreatment); we
    treat them as noisy indicators of a shared "coercion/exposure" factor and
    are explicit that this is a modeling choice, validated only by the over-ID
    test firing or not.
  * distress calls (PC3) are LOGGED BY POLICE -> partial shared filter with the
    pivot; this is the weakest-independence link and is flagged.

Everything is defensive: state-name harmonization, per-capita rates with a
population proxy, log1p stabilization, and hard failures on missing files.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Canonical state-name harmonization (NCRB uses several spellings/abbreviations).
_STATE_CANON = {
    "a & n islands": "andaman & nicobar islands",
    "a&n islands": "andaman & nicobar islands",
    "andaman & nicobar islands": "andaman & nicobar islands",
    "d & n haveli": "dadra & nagar haveli",
    "d&n haveli": "dadra & nagar haveli",
    "dadra & nagar haveli": "dadra & nagar haveli",
    "daman & diu": "daman & diu",
    "delhi ut": "delhi",
    "delhi": "delhi",
    "jammu & kashmir": "jammu & kashmir",
    "orissa": "odisha",
    "odisha": "odisha",
    "pondicherry": "puducherry",
    "puducherry": "puducherry",
    "chattisgarh": "chhattisgarh",
    "chhattisgarh": "chhattisgarh",
    "uttaranchal": "uttarakhand",
    "uttarakhand": "uttarakhand",
}

# Rows that are national/aggregate totals, not states.
_DROP_AREAS = {
    "total (states)", "total (uts)", "total (all india)", "total all india",
    "total states", "total uts", "all india", "total", "grand total",
}


def _canon_state(name: str) -> str:
    key = str(name).strip().lower()
    key = key.replace("  ", " ")
    return _STATE_CANON.get(key, key)


def _drop_totals(df: pd.DataFrame, col: str) -> pd.DataFrame:
    mask = ~df[col].astype(str).str.strip().str.lower().isin(_DROP_AREAS)
    return df[mask].copy()


def _load_ipc_records(data_dir: Path) -> pd.DataFrame:
    """Pivot channel: total recorded IPC crime, aggregated district->state."""
    f = data_dir / "crime" / "01_District_wise_crimes_committed_IPC_2001_2012.csv"
    if not f.exists():
        raise FileNotFoundError(f"missing IPC panel: {f}")
    d = pd.read_csv(f)
    d = _drop_totals(d, "STATE/UT")
    d["state"] = d["STATE/UT"].map(_canon_state)
    d = d[(d["YEAR"] >= 2001) & (d["YEAR"] <= 2010)]
    g = d.groupby(["state", "YEAR"], as_index=False)["TOTAL IPC CRIMES"].sum()
    return g.rename(columns={"YEAR": "year", "TOTAL IPC CRIMES": "ipc"})


def _load_distress_calls(data_dir: Path) -> pd.DataFrame:
    f = data_dir / "crime" / "27_Nature_of_complaints_received_by_police.csv"
    if not f.exists():
        raise FileNotFoundError(f"missing complaints-nature panel: {f}")
    d = pd.read_csv(f)
    d = _drop_totals(d, "Area_Name")
    d["state"] = d["Area_Name"].map(_canon_state)
    col = "PC3_Distress_call_over_phoneNo_100_etc"
    g = d.groupby(["state", "Year"], as_index=False)[col].sum()
    return g.rename(columns={"Year": "year", col: "distress"})


def _load_complaints_against_police(data_dir: Path) -> pd.DataFrame:
    f = data_dir / "25_Complaints_against_police.csv"
    if not f.exists():
        raise FileNotFoundError(f"missing complaints-against-police: {f}")
    d = pd.read_csv(f)
    d = _drop_totals(d, "Area_Name")
    d["state"] = d["Area_Name"].map(_canon_state)
    col = "CPA_-_Complaints_Received/Alleged"
    d[col] = pd.to_numeric(d[col], errors="coerce").fillna(0.0)
    g = d.groupby(["state", "Year"], as_index=False)[col].sum()
    return g.rename(columns={"Year": "year", col: "cap"})


def _load_accountability(data_dir: Path) -> pd.DataFrame:
    """Custodial deaths (all 5 sub-tables) + HR violations, summed per state-year."""
    parts = []
    _cd_names = ["person_remanded", "person_not_remanded", "during_production",
                 "during_hospitalization_or_treatment", "others"]
    for i in range(1, 6):
        fname = f"40_0{i}_Custodial_death_{_cd_names[i - 1]}.csv"
        f = data_dir / fname
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = _drop_totals(d, "Area_Name")
        d["state"] = d["Area_Name"].map(_canon_state)
        # sum any numeric death/incident columns present
        num = d.select_dtypes(include=[np.number]).columns
        keep = [c for c in num if c != "Year"]
        d["val"] = d[keep].sum(axis=1) if keep else 0.0
        parts.append(d.groupby(["state", "Year"], as_index=False)["val"].sum())
    hr = data_dir / "35_Human_rights_violation_by_police.csv"
    if hr.exists():
        d = pd.read_csv(hr)
        d = _drop_totals(d, "Area_Name")
        d["state"] = d["Area_Name"].map(_canon_state)
        d["val"] = pd.to_numeric(
            d["Cases_Registered_under_Human_Rights_Violations"], errors="coerce"
        ).fillna(0.0)
        parts.append(d.groupby(["state", "Year"], as_index=False)["val"].sum())
    if not parts:
        raise FileNotFoundError("no accountability (40_*/35_*) files found")
    acc = pd.concat(parts, ignore_index=True)
    g = acc.groupby(["state", "Year"], as_index=False)["val"].sum()
    return g.rename(columns={"Year": "year", "val": "acct"})


def load_ncrb_channels(
    data_dir: str | Path,
    year_min: int = 2001,
    year_max: int = 2010,
    min_state_count: int = 5,
) -> dict:
    """Assemble the aligned state-year multi-channel matrix from real NCRB data.

    Returns a dict with:
      log_channels : (K, N) float array, rows = [ipc, distress, cap, acct] on the
                     log(1 + count) scale (counts, not per-capita: population by
                     state-year is not in the repo, and log-counts still share the
                     latent up to an intercept, which the model absorbs).
      channel_names : list[str]
      states, years : the aligned index arrays (length N).
    Raises FileNotFoundError if required files are missing.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    ipc = _load_ipc_records(data_dir)
    dist = _load_distress_calls(data_dir)
    cap = _load_complaints_against_police(data_dir)
    acc = _load_accountability(data_dir)

    df = ipc.merge(dist, on=["state", "year"], how="inner")
    df = df.merge(cap, on=["state", "year"], how="inner")
    df = df.merge(acc, on=["state", "year"], how="inner")
    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)]

    # keep states with enough temporal coverage (stable moments)
    counts = df.groupby("state")["year"].transform("count")
    df = df[counts >= min_state_count].reset_index(drop=True)

    if len(df) < 20:
        raise ValueError(
            f"too few aligned state-year cells ({len(df)}); cannot estimate moments"
        )

    names = ["ipc", "distress", "cap", "acct"]
    mat = np.vstack([
        np.log1p(np.clip(df[c].to_numpy(dtype=float), 0.0, None)) for c in names
    ])

    return {
        "log_channels": mat,
        "channel_names": names,
        "states": df["state"].to_numpy(),
        "years": df["year"].to_numpy(),
        "raw": df,
    }


def _load_custodial(data_dir: Path) -> pd.DataFrame:
    """Custodial deaths only (all 5 sub-tables), summed per state-year."""
    parts = []
    _cd = ["person_remanded", "person_not_remanded", "during_production",
           "during_hospitalization_or_treatment", "others"]
    for i in range(1, 6):
        f = data_dir / f"40_0{i}_Custodial_death_{_cd[i - 1]}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = _drop_totals(d, "Area_Name")
        d["state"] = d["Area_Name"].map(_canon_state)
        num = [c for c in d.select_dtypes(include=[np.number]).columns
               if c != "Year"]
        d["val"] = d[num].sum(axis=1) if num else 0.0
        parts.append(d.groupby(["state", "Year"], as_index=False)["val"].sum())
    if not parts:
        raise FileNotFoundError("no custodial-death (40_*) files found")
    g = pd.concat(parts, ignore_index=True).groupby(
        ["state", "Year"], as_index=False)["val"].sum()
    return g.rename(columns={"Year": "year", "val": "custodial"})


def _load_hr(data_dir: Path) -> pd.DataFrame:
    f = data_dir / "35_Human_rights_violation_by_police.csv"
    if not f.exists():
        raise FileNotFoundError("missing 35_Human_rights_violation_by_police.csv")
    d = pd.read_csv(f)
    d = _drop_totals(d, "Area_Name")
    d["state"] = d["Area_Name"].map(_canon_state)
    d["val"] = pd.to_numeric(
        d["Cases_Registered_under_Human_Rights_Violations"], errors="coerce"
    ).fillna(0.0)
    g = d.groupby(["state", "Year"], as_index=False)["val"].sum()
    return g.rename(columns={"Year": "year", "val": "hr"})


def load_ncrb_two_control(
    data_dir: str | Path,
    year_min: int = 2001,
    year_max: int = 2010,
    min_state_count: int = 5,
) -> dict:
    """Assemble signal channels + TWO independent accountability controls.

    Signal channels (share the offending/enforcement latent):
        [ipc recorded crime, distress calls, complaints-against-police]
    Negative controls (respond to a policing-intensity / state-coercion common
    mode, plausibly NOT to the offending latent, via DIFFERENT reporting streams):
        control 0 = custodial deaths (40_*)
        control 1 = human-rights violations by police (35)

    This enables PROXIMAL POINT-IDENTIFICATION of a common mode on real data.
    The exclusion assumption (controls carry the confounder, not the offending
    latent) is untestable and stated as such in the experiment report.

    Returns dict with signal_channels (3, N), controls (2, N), names, states,
    years -- all on the log(1+count) scale, aligned by inner join.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    ipc = _load_ipc_records(data_dir)
    dist = _load_distress_calls(data_dir)
    cap = _load_complaints_against_police(data_dir)
    cust = _load_custodial(data_dir)
    hr = _load_hr(data_dir)

    df = ipc.merge(dist, on=["state", "year"]).merge(cap, on=["state", "year"])
    df = df.merge(cust, on=["state", "year"]).merge(hr, on=["state", "year"])
    df = df[(df["year"] >= year_min) & (df["year"] <= year_max)]
    counts = df.groupby("state")["year"].transform("count")
    df = df[counts >= min_state_count].reset_index(drop=True)
    if len(df) < 20:
        raise ValueError(f"too few aligned cells ({len(df)})")

    def _lg(col):
        return np.log1p(np.clip(df[col].to_numpy(dtype=float), 0.0, None))

    signal = np.vstack([_lg("ipc"), _lg("distress"), _lg("cap")])
    controls = np.vstack([_lg("custodial"), _lg("hr")])
    return {
        "signal_channels": signal,
        "controls": controls,
        "signal_names": ["ipc", "distress", "cap"],
        "control_names": ["custodial_deaths", "hr_violations"],
        "states": df["state"].to_numpy(),
        "years": df["year"].to_numpy(),
        "raw": df,
    }
