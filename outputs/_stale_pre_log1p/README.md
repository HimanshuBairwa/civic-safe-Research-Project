# STALE -- DO NOT CITE

These artifacts were produced on **2026-06-16** from checkpoint
`outputs/run_1781590552/seed_42/best.pt`.

That checkpoint was trained **before** commit `072fc14` (2026-07-28),
*"CRITICAL: concatenate log1p crime history to model input (train side)"*.

Before that commit the model received **static ACS features only**. Because
`input_features` carries no time variation, the forecaster had **zero temporal
signal**: it could not see a single crime count. It was structurally incapable
of beating a seasonal-naive baseline, and the numbers reflect exactly that:

| Metric | Value | Note |
|---|---|---|
| CRPS | 15.4813 | vs seasonal-naive 4.4018 |
| MAE | 18.5353 | approximately the mean count -- consistent with a near-constant forecast |
| CRPSS vs seasonal-naive | **-2.5171** | 3.5x worse than lag-52 |
| Conformal mean width | 50.33 | intervals inflated to cover a bad point forecast |

The coverage numbers in these files are *conditionally* valid -- split CP hits
0.9003 -- but conformal coverage is guaranteed by construction regardless of
model quality. Wide intervals around a broken forecaster are not a result.

**Any conclusion drawn from these files is a conclusion about a model that
could not see its own input.** They are retained only as a provenance record of
the pre-fix state. Regenerate from a post-`072fc14` checkpoint before citing
anything.
