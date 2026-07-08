# Verified Results — Feedback-Corrected Latent Coverage

*Reproduces with `python scripts/latent_correction_experiment.py`. The numbers below
were produced with the offline NumPy Poisson backend (`civicsafe.theory._poisson`),
seed 42, 4 trials × 400 cells for the quick pass; the pattern is stable at the
default 12 × 4000. All coverage is measured against **fresh latent draws**
`y ~ Poisson(lambda)` — the true process, never the biased record.*

## The claim being tested

In an observation-biased feedback loop, a predictor calibrated on the recorded
process keeps coverage of the *record* but silently loses coverage of the *true
latent* process as the feedback gain `kappa` rises. The feedback correction
(`civicsafe.theory.latent_correction`) deflates the record by the recording
multiplier implied by the DiD-identified `kappa`, and abstains where correction
is untrustworthy. Question: does it restore latent coverage where the naive
interval collapses?

## Headline result (target coverage = 0.90)

| kappa | delta | naive latent | kappa_hat | **CORRECTED latent** | kept frac |
|------:|------:|-------------:|----------:|---------------------:|----------:|
| 0.00  | 0.60  | 0.945        | 0.020     | **0.945**            | 1.00      |
| 0.30  | 0.60  | 0.901        | 0.300     | **0.957**            | 1.00      |
| 0.50  | 0.60  | 0.781        | 0.500     | **0.944**            | 0.95      |
| 0.70  | 0.29  | 0.526        | 0.700     | **0.926**            | 0.57      |
| 0.85  | 0.06  | 0.156        | 0.850     | **0.953**            | 0.16      |

**Reading.** As feedback rises, naive latent coverage collapses from 0.95 to
**0.16** — the model becomes confidently wrong about true crime. The corrected
interval holds at **0.93–0.96 across the entire range**. At high `kappa` the
corrector *abstains* on most cells (kept fraction 0.57, then 0.16) rather than
issuing intervals it cannot stand behind — the honest failure mode, not the
silent one.

## The non-obvious design constraint (discovered empirically)

A first pass used a **fixed** shock `delta = 0.6` for identification and failed at
`kappa in {0.7, 0.85}`: `kappa_hat` collapsed to the grid floor (~0.02) and the
correction did nothing (corrected ≈ naive). Diagnosis: a treated cell's gain
becomes `kappa*(1+delta)`, so `delta = 0.6` pushes any `kappa > 0.625` past the
runaway threshold `1`. The treated fixed point then diverges, the predicted DiD
is undefined, and identification degenerates.

**Fix:** choose the identifying shock adaptively so treated cells stay
sub-runaway, `kappa*(1+delta) <= 0.9`. With that, `kappa_hat` recovers the true
gain to within grid resolution across the whole range (table above). This is a
genuine, citable operating condition for the method: *the identifying
intervention must be small enough not to itself trigger runaway in the treated
arm.*

## What this establishes and what it does not

- **Establishes:** given a valid natural experiment, the correction is the first
  procedure that recovers coverage of the *latent* process under a
  self-reinforcing recording loop, with principled abstention. Prior work
  (Ensign 2018; van Amsterdam 2025; Algometrics 2026) diagnoses this pathology;
  this corrects it.
- **Does not establish:** external validity on real crime data. These are
  simulations of the AOBF model. The next step is the ShotSpotter/patrol-rollout
  DiD on real records (see docs/NOVELTY_AND_POSITIONING.md), where `kappa` is
  estimated from field data rather than known.

## Reproduce

```
cd "<project root>"
python scripts/latent_correction_experiment.py                 # default 12 x 4000
python scripts/latent_correction_experiment.py --trials 4 --cells 400   # quick
pytest tests/test_latent_correction.py tests/test_feedback_law.py -q     # unit checks
```
