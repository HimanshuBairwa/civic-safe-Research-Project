# OICC — Over-Identification-Calibrated Conformal Deconvolution

A small, tested, dependency-light (`numpy` + `scipy`) implementation of the
honest multi-channel latent-crime estimation method designed in
`ELEVATED_MASTER_PLAN_v2_OICC.md`. Formal results: `paper/OICC_THEOREMS.md`;
submission-quality write-up: `paper/oicc_paper.tex`.

## What it does

Given `K >= 3` **mechanism-independent** noisy measurement channels of one
latent per-area/period rate `theta` (e.g. police records, distress calls,
victimization survey, accountability/complaint data), OICC:

1. **estimates** the one-factor measurement structure (`moments.py`),
2. **recovers** the latent rate by empirical-Bayes deconvolution
   (`deconvolve.py`),
3. **tests** the conditional-independence assumption *in the identifiable
   (Delta-perp) directions* via a loading-invariant over-identification test
   (`spec_test.py`),
4. **issues** two nested conformal intervals (`conformal_split.py`): an EXACT
   finite-sample distribution-free interval for the observed pivot value, and a
   model-assisted interval for the *never-observed* latent target; latent-error
   shape via moment-Gaussian (default) or nonparametric CF deconvolution
   (`cf_deconv.py`),
5. **escapes the fatal blind spot** with negative-control / proximal inference
   (`proximal.py`): `proximal_deconfound` removes a common-mode confounder the
   over-ID test cannot see, `point_identify` (Q>=2 controls) **point-identifies**
   the true latent variance, and `exclusion_sensitivity` **quantifies** how much
   the estimate could move if the controls' (untestable) exclusion is violated,
6. **monitors deployment** with an anytime-valid e-process (`monitor.py`):
   time-uniform false-alarm control on a stream of over-ID p-values,
7. **quantifies uncertainty** with (block) bootstrap CIs on every estimator
   (`uncertainty.py`).

## The honest boundary — and the escape (this is the science)

The over-identification test has **~100% power** against a *detectable*
(Delta-perp) dependence violation and **provably zero power** against a
*common-mode* (Delta-parallel) violation loaded on all channels along the factor
direction — proved (`OICC_THEOREMS.md` Thm 3) and demonstrated in running code
(`test_overid_is_blind_to_common_mode`). That common mode is exactly the modal
way crime bias operates, so blindness there is not a corner case.

**The escape (Thm 7):** add NEGATIVE-CONTROL channels that carry the confounder
but *no* latent signal (e.g. a placebo / accountability series). Residualizing
the signal channels on the controls removes the common mode. Verified against
ground truth: at common-mode strength 1.0, latent-recovery RMSE drops from
**0.57 (naive) to 0.28 (proximal)**, with **no harm** when the confounder is
absent (0.21 vs 0.21). With `Q >= 2` valid controls the common mode is
point-identified; with `Q = 1`, detected and partially removed. The controls'
exclusion/completeness assumptions are **untestable** and stated as such.

## Verified behavior (all in `pytest tests_oicc -q` — 62 tests, green)

| Property | Result |
|---|---|
| moment recovery of loadings + Var(theta) | within ~5-8% of truth |
| deconvolution vs best single channel | wins ~all trials |
| **EXACT observed-value coverage** (finite-sample, distribution-free) | **0.90-0.91** |
| **latent coverage** (asymptotic, model-assisted), K=3,4,5 | **~0.89** |
| over-ID test size under H0 | ~0.00-0.05 |
| over-ID test power vs detectable confounder | ~1.00 |
| over-ID test vs common-mode confounder | ~0.00 (invisible, by design) |
| CF deconvolution recovers a skewed error law | mean quantile err small |
| proximal residualization fixes common-mode | RMSE 0.57 -> 0.28 |
| **proximal POINT-ID of Var(theta)** (Q>=2) | exact (0.61 vs naive 3.18 at cm=2) |
| point-ID detection gate, no confounder | no false correction |
| **anytime-valid monitor** false-alarm @H0 | 0.013 <= 0.05 (Ville) |
| anytime-valid monitor power vs drift | ~1.00, median delay ~9 windows |
| **bootstrap CIs cover the truth** (moments, point-ID) | verified ~90% |
| `gamma_cm` monotonically widens intervals | yes |
| stress/property (degenerate, collinear, extreme, NaN) | no uncaught errors |

## Real-data experiments

**India NCRB (state-year 2001-2010, N=253)** — `run_ncrb_experiment.py`.
Four institutional channels (recorded IPC crime, distress calls, complaints-
against-police, custodial-death/HR-violation accountability). Over-ID p=0.088
(does not reject one-factor in detectable directions); recovered latent
face-valid (high: Maharashtra/MP/Andhra; low: small NE states); proximal probe +
anytime-valid monitor (no drift alarm over 2001-2010).

**US Chicago/NYC (2018-2023)** — `run_us_experiment.py`. Crime CATEGORIES share
one police filter, so the over-ID test **rejects** (p<0.001) — the correct,
opposite verdict to India's independent channels. A two-directional validation
that the specification test does its job.

**Figures** — `make_figures.py` regenerates all four paper figures
(`paper/figures/`) from the live, tested computations.

There is no ground-truth latent on real data (that is the point), so latent
coverage is validated only on synthetic ground truth.

## Layout

```
src/oicc/
  measurement.py    model + controlled generators (signal, common-mode, controls)
  moments.py        one-factor moment estimation (averaged tetrads)
  deconvolve.py     BLUP / empirical-Bayes latent recovery (+ blup_from_subset)
  spec_test.py      loading-invariant over-identification Wald test
  cf_deconv.py      nonparametric characteristic-function deconvolution
  conformal.py      leave-pivot-out conformal band + sensitivity inflation
  conformal_split.py  two-interval split conformal (exact observed + latent)
  proximal.py       negative-control / proximal common-mode correction
tests_oicc/         32-test pytest suite (synthetic + real-data; real auto-skips)
experiments/oicc_runs/
  ncrb_loader.py           real India NCRB channel assembly
  run_ncrb_experiment.py   end-to-end real-data run
paper/
  OICC_THEOREMS.md   formal theorems + proofs (matches the code)
  oicc_paper.tex     submission-quality LaTeX write-up
```

## Run

```bash
# from the project root, with src/ on the path
PYTHONPATH=src python -m pytest tests_oicc -q       # 32 passed
python experiments/oicc_runs/run_ncrb_experiment.py
```
