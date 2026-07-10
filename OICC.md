# OICC — Over-Identification-Calibrated Conformal Deconvolution

**The honest way to estimate a latent rate you can never observe, from several
biased measurements of it.** A complete, tested, reproducible research artifact.

> Estimating *true* crime (or any under-reported quantity) is impossible from a
> single biased record. OICC combines three or more mechanism-independent
> channels — police records, calls-for-service, victimization surveys,
> accountability data — into a latent-rate estimate with conformal uncertainty,
> a specification test that says when the model is wrong, a proved impossibility
> result for the one thing no data can fix, and a principled negative-control
> escape from it.

---

## Why this is a real contribution (and where it honestly stops)

| Claim | Status |
|---|---|
| Recover a latent rate better than any single channel | **proved + verified** (RMSE 0.19–0.22 vs 0.35) |
| An exact finite-sample interval for the observed channel value | **proved, distribution-free** (coverage 0.90) |
| A model-assisted interval for the *latent* target | **asymptotic** (coverage ~0.89) |
| A test that detects when the channels are *not* independent | **verified**: 100% power vs detectable dependence |
| …but is **blind** to a common-mode confounder | **proved impossibility** (0% power, at any K) |
| Negative controls **point-identify** that confounder | **proved + verified** (recovers true variance exactly) |
| Anytime-valid deployment drift monitor | **verified**: false-alarm 0.013 ≤ 0.05 |
| Every estimate carries a bootstrap CI | **implemented + verified to cover truth** |

**Honest ceiling:** a strong applied-statistics / FAccT / KDD-ADS / AOAS paper.
It is *not* "beyond NeurIPS," and the reason is a theorem, not a lack of effort:
a common-mode confounder is unidentifiable from the channels alone; only external
negative controls break it, under an assumption that is untestable by
construction. We prove this rather than paper over it — which is exactly what
makes the rest defensible.

---

## One-command reproduction (machine-checked)

```bash
pip install -r requirements-oicc.txt
PYTHONPATH=src python -m pytest tests_oicc -q            # OICC tests, all green
python experiments/oicc_runs/reproduce_all.py            # 13 headline assertions
```

## Running the WHOLE codebase on an A100 (guaranteed clean)

The full repo (the civicsafe GNN forecaster + OICC) is device-agnostic and
runs error-free on a fresh Linux A100. Three ways, easiest first:

```bash
# 0. one-command preflight -- env report + GPU training smoke + tests + reproduce
python run_all.py                 # add --full to also run the 300-test civicsafe suite

# 1. reproducible container (pins CUDA torch + PyG; no compiled PyG extensions)
docker build -t civicsafe-oicc .
docker run --gpus all civicsafe-oicc           # runs the full test suite on the GPU

# 2. bare metal
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1
pip install -r requirements-a100.txt
PYTHONPATH=src python experiments/oicc_runs/device_smoke.py   # exercises the real
                                                              # cuda + bf16 path
PYTHONPATH=src python -m pytest tests/ tests_oicc/ -q         # 360+ tests
```

**Guarantees baked in:** no hardcoded absolute paths (a resolver finds data or
skips), no unguarded `.cuda()` (device is always `cuda if available else cpu`),
bf16 autocast via the modern `torch.amp` API, matplotlib forced to `Agg`
(headless), numpy-2.x safe, optional deps (`xgboost`/`scikit-learn`/`seaborn`)
lazily imported so no script crashes on import, and `WANDB_MODE=disabled` by
default. A `device_smoke` test auto-runs the real GPU forward/backward path.

`reproduce_all.py` re-runs the whole battery and **asserts** every headline
number lands in range; it exits non-zero if anything drifts. "It reproduces" is
a checked fact here.

## Real-data experiments

```bash
python experiments/oicc_runs/run_ncrb_experiment.py   # India NCRB 2001–2010
python experiments/oicc_runs/run_us_experiment.py     # US Chicago/NYC contrast
python experiments/oicc_runs/make_figures.py          # regenerate paper figures
```

- **India NCRB** (state-year, 4 institutional channels): over-ID test does *not*
  reject (p=0.088); two accountability controls enable point-ID (with honest wide
  CIs and a stated exclusion caveat).
- **US Chicago/NYC** (crime categories, one police filter): over-ID test *rejects*
  (p<0.001) — the correct opposite verdict, validating the test in both
  directions.

## What's inside

```
src/oicc/               the method (numpy + scipy only)
  measurement moments deconvolve spec_test cf_deconv
  conformal conformal_split proximal monitor uncertainty
tests_oicc/             60 tests: synthetic, real-data, stress/property
experiments/oicc_runs/  loaders, runners, figures, reproduce_all
paper/                  oicc_paper.tex, OICC_THEOREMS.md, figures/
```

Full method docs: [`src/oicc/README.md`](src/oicc/README.md).
Formal theorems + proofs: [`paper/OICC_THEOREMS.md`](paper/OICC_THEOREMS.md).
Paper: [`paper/oicc_paper.tex`](paper/oicc_paper.tex) (compiles on any TeX host).

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT.
