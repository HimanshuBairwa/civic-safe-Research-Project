"""Field identification of the feedback gain from the Chicago ShotSpotter rollout.

Runs the difference-in-differences of Theorem 3 on the REAL Chicago crime panel:
does an exogenous detection-sensitivity shock (staggered acoustic gunshot
detection) inflate *recorded* violent-crime rates in treated areas? A positive,
significant DiD with flat pre-trends is direct field evidence that crime records
are attention-driven --- the empirical anchor for the whole feedback-correction
program.

Run:
    python scripts/field_identification_shotspotter.py
    python scripts/field_identification_shotspotter.py --category violent --rollout 2018-06

Outputs the DiD estimate (log recorded-rate jump, cluster-robust CI), the
event-study pre-trend check, and the implied-kappa sensitivity table over the
(unidentified) policy elasticity beta. Writes a machine-readable summary to
outputs/field_identification.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from civicsafe.theory.field_identification import (
    ShotSpotterRollout,
    build_monthly_panel,
    estimate_did,
    event_study,
    implied_kappa,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="ShotSpotter DiD field identification")
    parser.add_argument("--crime-glob", default="data/raw/chicago/*.parquet")
    parser.add_argument("--category", default="violent")
    parser.add_argument("--rollout", default="2018-06", help="YYYY-MM activation month")
    parser.add_argument("--out", default="outputs/field_identification.json")
    args = parser.parse_args()

    print("Building monthly panel from real Chicago crime records...")
    panel = build_monthly_panel(crime_glob=args.crime_glob, category=args.category)
    print(f"  panel: {panel.shape[0]} unit-months, "
          f"{panel['spatial_unit'].nunique()} units, "
          f"{panel['month'].min():%Y-%m}..{panel['month'].max():%Y-%m}")

    rollout = ShotSpotterRollout.chicago_default()
    rollout.rollout_period = args.rollout

    print("\nEstimating staggered DiD (two-way FE, cluster-robust by unit)...")
    did = estimate_did(panel, rollout)
    print(f"  tau (log recorded-rate jump) = {did.tau:+.4f}  "
          f"(SE {did.se:.4f}, p={did.pvalue:.4g})")
    print(f"  95% CI = [{did.ci_low:+.4f}, {did.ci_high:+.4f}]")
    print(f"  recorded-rate inflation in treated areas = {did.recording_inflation*100:+.1f}%")
    print(f"  treated units = {did.n_treated_units}, N = {did.n_obs}")

    print("\nEvent-study pre-trend check (coef by month rel. to rollout, ref=-1):")
    es = event_study(panel, rollout)
    pre = es[es["rel_month"] < 0]
    post = es[es["rel_month"] >= 0]
    pre_mean = float(pre["coef"].abs().mean()) if not pre.empty else float("nan")
    post_mean = float(post["coef"].mean()) if not post.empty else float("nan")
    print(f"  mean |pre-trend coef|  = {pre_mean:.4f}  (small => parallel trends OK)")
    print(f"  mean post-period coef  = {post_mean:+.4f}  (the recording shock)")

    print("\nImplied feedback gain kappa (sensitivity over policy elasticity beta):")
    kap = implied_kappa(did)
    for _, r in kap.iterrows():
        flag = "  <-- RUNAWAY (kappa>=1)" if r["runaway"] else ""
        print(f"  beta={r['beta']:.1f}: rho_hat={r['rho_hat']:+.3f}, "
              f"kappa_hat={r['kappa_hat']:+.3f}{flag}")

    summary = {
        "category": args.category,
        "rollout_period": args.rollout,
        "did": {
            "tau": did.tau, "se": did.se, "pvalue": did.pvalue,
            "ci": [did.ci_low, did.ci_high],
            "recording_inflation_pct": did.recording_inflation * 100,
            "n_obs": did.n_obs, "n_treated_units": did.n_treated_units,
        },
        "event_study": es.to_dict(orient="records"),
        "pretrend_mean_abs": pre_mean,
        "post_mean": post_mean,
        "implied_kappa": kap.to_dict(orient="records"),
        "caveats": [
            "tau/rho_hat is point-identified; kappa depends on the UNIDENTIFIED policy elasticity beta.",
            "Treated set is a documented default; verify against official CPD ShotSpotter deployment records.",
            "Flat pre-trends validate the design; a pre-jump indicates treatment mis-specification.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote machine-readable summary to {out}")

    # Report the ACTUAL finding, not a hardcoded conclusion.
    print("\n--- Honest interpretation ---")
    sig = did.pvalue < 0.05
    pretrends_ok = pre_mean < 0.15
    if did.tau > 0 and sig and pretrends_ok:
        print("Positive, significant recording jump with flat pre-trends: field evidence")
        print("that records are attention-driven (Theorem 3 confirmed on this specification).")
    elif not pretrends_ok:
        print("Pre-trends are NOT flat: the parallel-trends assumption fails for this")
        print("treatment specification, so the DiD is not validly identified. The treated")
        print("set and/or rollout date are likely mis-specified.")
    else:
        print(f"No significant positive recording jump under this specification "
              f"(tau={did.tau:+.3f}, p={did.pvalue:.3f}).")
        print("This does NOT refute the mechanism; with the placeholder treatment it is")
        print("under-powered / mis-specified. Valid identification requires the inputs below.")
    print("\nInputs needed for a valid real-data estimate:")
    print("  1. Official CPD ShotSpotter deployment records: exact treated police")
    print("     districts (mapped to community areas) AND per-district activation months.")
    print("  2. A clean pre-period: crime data from BEFORE the first activation")
    print("     (Chicago's rollout began ~2017; extend the panel to 2014-2017).")
    print("  3. Gun-specific incidents (shots-fired / weapons), not the aggregate")
    print("     'violent' category, since ShotSpotter detects gunfire specifically.")
    print("  Run: python scripts/validate_did_estimator.py  # confirms the estimator")
    print("  recovers a KNOWN shock on synthetic data (tool validity, independent of")
    print("  the real-data treatment specification).")


if __name__ == "__main__":
    main()
