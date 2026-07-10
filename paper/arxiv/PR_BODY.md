# OICC: Over-Identification-Calibrated Conformal Deconvolution

Adds a complete, tested, honest method (`oicc`) for estimating a **latent**
per-area rate (e.g. true crime/victimization) from `K>=3` mechanism-independent
biased measurement channels, plus full A100/reproducibility infrastructure and a
submission-ready paper. Numpy/scipy only for the core (torch used only by the
optional US panel loader).

## What's in this PR

**Method (`src/oicc/`, v0.6.0, 34 exports):**
- one-factor tetrad moment identification; loading-invariant over-identification
  specification test (second-moment **and** a new third-cumulant test with real
  power at K=3; moving-block bootstrap for dependent panels);
- two-interval **leave-pivot-out conformal** predictor: an *exact* finite-sample
  distribution-free interval for the observed pivot value + a model-assisted
  latent interval; nonparametric CF deconvolution;
- **proximal / negative-control point-identification** of a common-mode
  confounder the over-ID test is *provably blind* to, with an
  **exclusion-sensitivity** analysis that quantifies the one untestable
  assumption, and **bootstrap** uncertainty on every estimator;
- an **anytime-valid e-process** deployment monitor (Ville false-alarm control);
- a **baseline comparison** (OICC vs single-channel / naive-average /
  reporting-rate; and, under confounding, proximal vs all naive methods).

**A proved impossibility, honestly:** a common-mode confounder along the signal
direction is unidentified by any over-ID test at any moment order (demonstrated
at 2nd and 3rd order). The escape (negative controls) has an explicit, untestable
assumption — surfaced, not hidden.

**Verification (machine-checked):** 86 tests pass / 1 CUDA-skip;
`reproduce_all.py` asserts **17 headline numbers**; clean under
`-W error::DeprecationWarning`.

**Real data:** India NCRB (over-ID does not reject once dependence is respected)
and a US Chicago/NYC cross-national contrast (correctly rejects same-filter
categories). A US records+NCVS+911 loader scaffold is ready for real data.

**Infrastructure:** Dockerfile (CUDA), pinned `requirements-a100.txt`,
`run_all.py` preflight, device-agnostic training smoke test, `tests_oicc/`
conftest, path resolver — the whole repo runs error-free on a fresh A100.

**Paper + arXiv:** `paper/oicc_paper.tex` (5 figures, all citations resolved) and
`paper/arxiv/` (validated tarball builder + metadata).

## Honest positioning
This is a strong **applied-statistics / FAccT / KDD-ADS**-class contribution: a
composition + the leave-pivot-out primitive + a clean impossibility/escape
pairing. It does **not** claim distribution-free finite-sample *latent* coverage
(proved impossible). The one real lever beyond this is data (NCVS + 911);
templates are in `docs/data_access/`.

## Notes for review
- Nothing in `src/civicsafe/` is removed; the legacy forecaster is retained and
  device-hardened. The `oicc` package is additive and self-contained.
- Merge does not require GPU or external data; tests skip real-data cleanly.
