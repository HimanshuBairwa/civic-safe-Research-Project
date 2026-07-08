"""Field identification of the feedback gain from a natural experiment.

This module turns the identification theorem (:mod:`civicsafe.theory.feedback_law`,
Thm 3) into a *real-data* analysis on the Chicago crime panel, using the
staggered rollout of acoustic gunshot detection (ShotSpotter) across police
districts as an exogenous **detection-sensitivity shock**.

Identification logic
--------------------
ShotSpotter raises the probability that a gun-involved incident is *recorded*
without changing the underlying rate of incidents. Under observation-biased
recording ``y ~ Poisson(lambda * g(a))``, a shock to the detection channel ``g``
in treated areas shifts recorded rates but not latent ones. A
difference-in-differences on **log recorded violent-crime rates** (treated vs.
control, post vs. pre) therefore isolates the recording response:

* The DiD coefficient ``tau`` **point-identifies the detection elasticity**
  ``rho`` (the recording response to a sensitivity shock) up to the known shock
  size — direct field evidence that records are attention-driven, not a
  reflection of true incidence.
* The full feedback gain ``kappa = beta * rho`` additionally requires the policy
  elasticity ``beta`` (how strongly allocation responds to predicted risk),
  which this single experiment does not identify. We therefore report ``kappa``
  as a transparent **sensitivity table over plausible ``beta``**, and never
  claim a single point value from the DiD alone.

What is (and is not) identified is stated explicitly so the empirical claim
survives review: ``tau`` and the event-study (flat pre-trends, a post jump
localized to treated areas) are the point-identified results; ``kappa`` is a
bounded implication under a stated ``beta``.

Design notes
------------
* Treatment assignment is a **documented, overridable input**
  (``ShotSpotterRollout``): the default encodes the South/West-side rollout, and
  the event-study pre-trends self-diagnose a mis-specified treatment set. Verify
  the treated units and dates against official CPD deployment records before
  publication.
* Estimation uses two-way (unit + month) fixed effects with **cluster-robust
  standard errors by spatial unit** (statsmodels), the standard staggered-DiD
  workhorse. Callaway--Sant'Anna group-time estimators are a drop-in upgrade if
  ``linearmodels``/``pyfixest`` is installed.
* This analysis is CPU/pandas work and runs in seconds; it is independent of the
  GPU model training.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Treatment specification
# ---------------------------------------------------------------------------

@dataclass
class ShotSpotterRollout:
    """A staggered detection-sensitivity shock specification.

    Attributes:
        treated_units: Spatial units (Chicago community areas) that received the
            detection shock.
        rollout_period: ``YYYY-MM`` string of the (common) activation month. For
            genuinely staggered timing pass ``unit_rollout`` instead.
        unit_rollout: Optional ``{spatial_unit: 'YYYY-MM'}`` for per-unit timing;
            overrides ``rollout_period`` when present.
    """

    treated_units: list[int]
    rollout_period: str = "2018-06"
    unit_rollout: dict[int, str] = field(default_factory=dict)

    @classmethod
    def chicago_default(cls) -> "ShotSpotterRollout":
        """Documented South/West-side default. VERIFY against official records.

        These South and West side Chicago community areas fall within the police
        districts that received acoustic gunshot detection during the 2017--2018
        expansion. This is a documented starting point, *not* an official
        crosswalk; confirm the exact treated set and activation months against
        CPD deployment records before drawing published conclusions. The
        event-study pre-trends provide a built-in falsification check.
        """
        south_west_side = [
            23, 25, 26, 27, 29, 30,      # West side (Humboldt Park, Austin, Garfield/Lawndale)
            37, 38, 40, 42, 43, 44, 46,  # near-south / Grand Boulevard / Kenwood / Woodlawn
            49, 53, 54, 61, 63, 66, 67, 68, 69, 71, 73, 75,  # far South side
        ]
        return cls(treated_units=south_west_side, rollout_period="2018-06")


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def build_monthly_panel(
    crime_glob: str = "data/raw/chicago/*.parquet",
    category: str = "violent",
    population_csv: str | None = "data/processed/chicago_demographics.csv",
) -> pd.DataFrame:
    """Build a spatial-unit x month panel of recorded rates for one category.

    Args:
        crime_glob: Glob for the crime parquet shards (columns ``date``,
            ``spatial_unit``, ``category``).
        category: Crime category the shock acts on (``"violent"`` for ShotSpotter
            gun detection).
        population_csv: Optional demographics CSV with ``spatial_unit`` and
            ``total_population`` for per-100k rates; falls back to counts.

    Returns:
        Panel with columns ``spatial_unit``, ``month`` (Period[M] as timestamp),
        ``count``, ``rate`` (per 100k if population available), ``log_rate``.
    """
    files = sorted(glob.glob(crime_glob))
    if not files:
        raise FileNotFoundError(f"No crime parquet files matched {crime_glob!r}")
    frames = [pd.read_parquet(f, columns=["date", "spatial_unit", "category"]) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df = df[df["category"] == category].copy()
    df["month"] = df["date"].dt.to_period("M")

    counts = (
        df.groupby(["spatial_unit", "month"]).size().rename("count").reset_index()
    )
    # Complete the panel (unit x month grid) so structural zeros are explicit.
    units = counts["spatial_unit"].unique()
    months = pd.period_range(counts["month"].min(), counts["month"].max(), freq="M")
    grid = pd.MultiIndex.from_product([units, months], names=["spatial_unit", "month"])
    counts = counts.set_index(["spatial_unit", "month"]).reindex(grid, fill_value=0).reset_index()

    if population_csv:
        try:
            pop = pd.read_csv(population_csv)
            pop_col = "total_population" if "total_population" in pop.columns else None
            if pop_col and "spatial_unit" in pop.columns:
                counts = counts.merge(pop[["spatial_unit", pop_col]], on="spatial_unit", how="left")
                counts["rate"] = counts["count"] / counts[pop_col].clip(lower=1) * 1e5
            else:
                counts["rate"] = counts["count"].astype(float)
        except (FileNotFoundError, KeyError):
            counts["rate"] = counts["count"].astype(float)
    else:
        counts["rate"] = counts["count"].astype(float)

    counts["log_rate"] = np.log1p(counts["rate"])
    counts["month"] = counts["month"].dt.to_timestamp()
    return counts.sort_values(["spatial_unit", "month"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Difference-in-differences estimation
# ---------------------------------------------------------------------------

@dataclass
class DiDResult:
    """Result of the staggered-DiD recording-shock estimation.

    Attributes:
        tau: DiD coefficient on ``treated_post`` (log recorded-rate jump).
        se: Cluster-robust standard error (clustered by spatial unit).
        pvalue: Two-sided p-value for ``tau == 0``.
        ci_low: Lower 95% confidence bound.
        ci_high: Upper 95% confidence bound.
        n_obs: Number of unit-month observations.
        n_treated_units: Number of treated units.
        recording_inflation: ``exp(tau) - 1`` — the multiplicative recording jump.
    """

    tau: float
    se: float
    pvalue: float
    ci_low: float
    ci_high: float
    n_obs: int
    n_treated_units: int
    recording_inflation: float


def estimate_did(panel: pd.DataFrame, rollout: ShotSpotterRollout) -> DiDResult:
    """Two-way fixed-effects DiD of log recorded rate on the detection shock.

    Fits ``log_rate ~ treated_post + C(spatial_unit) + C(month)`` with standard
    errors clustered by spatial unit — the workhorse staggered-DiD estimator.

    Args:
        panel: Output of :func:`build_monthly_panel`.
        rollout: The treatment specification.

    Returns:
        A :class:`DiDResult`.
    """
    import statsmodels.formula.api as smf

    d = panel.copy()
    treated_set = set(rollout.treated_units)
    d["is_treated"] = d["spatial_unit"].isin(treated_set).astype(int)

    # Per-unit or common activation month.
    if rollout.unit_rollout:
        roll = {u: pd.Period(p, freq="M").to_timestamp() for u, p in rollout.unit_rollout.items()}
        d["activate"] = d["spatial_unit"].map(roll)
        d["post"] = (d["month"] >= d["activate"]).fillna(False).astype(int)
    else:
        activate = pd.Period(rollout.rollout_period, freq="M").to_timestamp()
        d["post"] = (d["month"] >= activate).astype(int)
    d["treated_post"] = d["is_treated"] * d["post"]

    model = smf.ols("log_rate ~ treated_post + C(spatial_unit) + C(month)", data=d)
    res = model.fit(cov_type="cluster", cov_kwds={"groups": d["spatial_unit"]})
    tau = float(res.params["treated_post"])
    se = float(res.bse["treated_post"])
    ci = res.conf_int().loc["treated_post"]
    return DiDResult(
        tau=tau,
        se=se,
        pvalue=float(res.pvalues["treated_post"]),
        ci_low=float(ci[0]),
        ci_high=float(ci[1]),
        n_obs=int(d.shape[0]),
        n_treated_units=len(treated_set & set(d["spatial_unit"].unique())),
        recording_inflation=float(np.expm1(tau)),
    )


def event_study(
    panel: pd.DataFrame,
    rollout: ShotSpotterRollout,
    max_lead: int = 6,
    max_lag: int = 12,
) -> pd.DataFrame:
    """Event-study coefficients relative to activation (parallel-trends check).

    Estimates dynamic treatment effects at each month offset from rollout,
    omitting the month before activation as the reference. Flat, near-zero
    pre-period coefficients support the parallel-trends assumption and validate
    the treatment specification; a post jump is the recording shock.

    Args:
        panel: Output of :func:`build_monthly_panel`.
        rollout: Treatment specification (uses the common ``rollout_period``).
        max_lead: Number of pre-period months to include.
        max_lag: Number of post-period months to include.

    Returns:
        DataFrame with columns ``rel_month``, ``coef``, ``se`` (reference month
        -1 omitted).
    """
    import statsmodels.formula.api as smf

    d = panel.copy()
    treated_set = set(rollout.treated_units)
    d["is_treated"] = d["spatial_unit"].isin(treated_set).astype(int)
    activate = pd.Period(rollout.rollout_period, freq="M").to_timestamp()
    d["rel_month"] = (
        (d["month"].dt.year - activate.year) * 12 + (d["month"].dt.month - activate.month)
    )
    d = d[(d["rel_month"] >= -max_lead) & (d["rel_month"] <= max_lag)].copy()
    d["rel_bin"] = d["rel_month"].astype(int)

    # Explicit treated x relative-month interaction dummies (reference bin = -1).
    # Control units and the reference period are the omitted baseline, absorbed by
    # the unit and month fixed effects. This avoids the rank-deficiency of coding
    # controls as a categorical level.
    bins = sorted(b for b in set(d["rel_bin"]) if b != -1)
    dummy_cols: list[str] = []
    for b in bins:
        col = f"evt_{'m' if b < 0 else 'p'}{abs(b)}"
        d[col] = ((d["is_treated"] == 1) & (d["rel_bin"] == b)).astype(int)
        if d[col].sum() > 0:  # skip empty bins
            dummy_cols.append(col)

    formula = "log_rate ~ " + " + ".join(dummy_cols) + " + C(spatial_unit) + C(month)"
    res = smf.ols(formula, data=d).fit(
        cov_type="cluster", cov_kwds={"groups": d["spatial_unit"]}
    )

    rows = []
    for b, col in zip([b for b in bins if f"evt_{'m' if b < 0 else 'p'}{abs(b)}" in dummy_cols], dummy_cols):
        if col in res.params.index:
            rows.append({"rel_month": b, "coef": float(res.params[col]), "se": float(res.bse[col])})
    return pd.DataFrame(rows)


def implied_kappa(
    did: DiDResult,
    beta_grid: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0),
    shock_log: float | None = None,
) -> pd.DataFrame:
    """Translate the DiD recording jump into implied feedback gain ``kappa``.

    The DiD ``tau`` estimates the recording response to the detection shock. With
    a known log-shock size ``shock_log`` the detection elasticity is
    ``rho_hat = tau / shock_log``; otherwise ``tau`` is reported as the
    normalized recording response and ``rho_hat = tau`` (shock treated as unit
    scale). The loop gain is ``kappa = beta * rho_hat`` for each assumed policy
    elasticity ``beta`` — an explicit sensitivity table, since ``beta`` is *not*
    identified by this experiment.

    Args:
        did: The estimated :class:`DiDResult`.
        beta_grid: Plausible policy elasticities to tabulate.
        shock_log: Optional known log detection-sensitivity increase.

    Returns:
        DataFrame with columns ``beta``, ``rho_hat``, ``kappa_hat``,
        ``runaway`` (whether ``kappa_hat >= 1``).
    """
    rho_hat = did.tau / shock_log if shock_log else did.tau
    rows = []
    for beta in beta_grid:
        k = float(beta * rho_hat)
        rows.append({
            "beta": beta,
            "rho_hat": float(rho_hat),
            "kappa_hat": k,
            "runaway": bool(k >= 1.0),
        })
    return pd.DataFrame(rows)
