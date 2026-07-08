# Field Identification of the Feedback Gain — Results & Honest Status

*The real-data half of the program: estimate the recording response to an
exogenous detection-sensitivity shock (Chicago ShotSpotter rollout) via
difference-in-differences on real crime records. This document reports the
verified estimator, the real-data result, and — plainly — what it does and does
not establish.*

---

## 1. What is established: the estimator is correct (synthetic validation)

`scripts/validate_did_estimator.py` (tests: `tests/test_field_identification.py`,
4/4 pass) injects a **known** recording shock into synthetic unit×month panels
and confirms the two-way fixed-effects DiD recovers it:

| true shock (log) | recovered $\hat\tau$ | p-value | \|pre-trend\| | error |
|-----------------:|---------------------:|--------:|--------------:|------:|
| 0.00 | +0.004 | 0.43 | 0.014 | 0.004 |
| 0.15 | +0.154 | $10^{-201}$ | 0.014 | 0.004 |
| 0.30 | +0.304 | $\approx 0$ | 0.014 | 0.004 |
| 0.50 | +0.504 | $\approx 0$ | 0.014 | 0.004 |

The estimator recovers the shock to **±0.004**, is **insignificant at zero shock**
(no false positive), and shows **flat pre-trends**. The method is sound.

## 2. What the real data shows (honest null under placeholder treatment)

`scripts/field_identification_shotspotter.py` on the real Chicago panel
(2018–2023, 77 community areas, 1.33M incidents, category = violent), with a
**documented-but-unverified** South/West-side treated set and a 2018-06 common
activation date:

- DiD $\hat\tau = -0.040$ (SE 0.022, $p=0.069$) — **no positive recording jump**.
- Event-study pre-trends $\approx 0.098$ (roughly flat after the estimator fix),
  post-period $\approx -0.05$.
- Implied $\kappa$ is therefore negative/null under this specification.

**This is a null result, reported as such.** It does **not** refute the feedback
mechanism, and given the synthetic validation it is **not** an estimator failure.
It is a treatment-specification / data-window problem:

1. **The treated set is a placeholder**, not the official CPD ShotSpotter
   district→community-area mapping.
2. **The pre-period is missing.** Chicago's rollout began ~2017; the panel starts
   2018-01, so treated areas may already be "post" at the sample start — there is
   little clean pre-period to difference against.
3. **The outcome is too coarse.** ShotSpotter detects *gunfire*; the aggregate
   "violent" category dilutes the gun-specific signal.
4. **Unit mismatch.** Treatment is assigned at the police-district level; the
   panel is at the community-area level.

## 3. What is needed for a valid real-data estimate

1. **Official ShotSpotter deployment records** — exact treated police districts
   (mapped to community areas) and per-district activation months (enables true
   *staggered* DiD via `ShotSpotterRollout.unit_rollout`).
2. **A pre-2018 pre-period** — extend the crime panel to ~2014–2017 so there is a
   clean pre-activation baseline.
3. **Gun-specific incidents** — a shots-fired / weapons-involved outcome rather
   than the aggregate violent category.

With those three inputs the same, already-validated estimator yields the
point-identified recording elasticity; the loop gain $\kappa=\beta\rho$ then
follows from the reported sensitivity table over the policy elasticity $\beta$.

## 4. Honest scope statement (for the paper)

- **Point-identified (given valid treatment):** the recording response $\tau$ to
  a detection shock, and hence the detection elasticity $\rho$ — direct evidence
  that records are attention-driven.
- **Not identified by this experiment:** the policy elasticity $\beta$, and hence
  the full loop gain $\kappa$; reported only as a $\beta$-sensitivity table.
- **Current real-data status:** null under a placeholder treatment; a valid
  estimate awaits the official deployment records above. The estimator's
  correctness is established independently on synthetic data.

This is the truthful state. Claiming a positive real-data $\kappa$ now would be
unsupported; the honest contribution is a *validated identification instrument*
plus a precisely specified data requirement to run it for real.

## Reproduce
```
python scripts/validate_did_estimator.py          # estimator recovers known shocks
python scripts/field_identification_shotspotter.py # real-data DiD (null under placeholder)
pytest tests/test_field_identification.py -q
```
