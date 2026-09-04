<div align="center">

# 🏛️ CIVIC-SAFE

### Honest Uncertainty Under Observation Bias: Online Conformal Prediction for Count Forecasting with Closed-Loop Feedback Evaluation

[![Tests](https://img.shields.io/badge/tests-406%20collected-brightgreen?style=for-the-badge)](tests/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Params](https://img.shields.io/badge/params-688K-blue?style=for-the-badge)]()
[![mypy](https://img.shields.io/badge/mypy-strict-blue?style=for-the-badge)](http://mypy-lang.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000?style=for-the-badge)](https://docs.astral.sh/ruff/)

**Uncertainty-aware spatiotemporal forecasting with conformal prediction intervals, audited equity guarantees, and advisory safe-route routing.**

[IEEE Manuscript](paper/civic_safe_ieee.tex) · [Submission Bundle](paper/submission_bundle/) · [Reviewer Defense](docs/REVIEWER_DEFENSE_DOSSIER.md) · [Equations](MATHEMATICS.md) · [Reproducibility](REPRODUCIBILITY.md)

</div>

---

> ## 📌 What this repository contains (read first)
>
> **Two separable research contributions live here.** Read this before anything
> else, because they are evaluated differently and claimed differently.
>
> ### 1. CIVIC-SAFE — feedback-corrected conformal prediction (`src/civicsafe/`)
>
> The submitted IEEE manuscript. Records of crime are produced by a loop: past
> records direct patrols, patrols produce records. A forecaster fit on such records
> can be well calibrated against the record and badly wrong about reality, and no
> validation on records will reveal it. We deflate the recorded rate by an estimated
> feedback multiplier and issue intervals for the **latent** process, with principled
> abstention where the correction stops being trustworthy.
>
> - **Manuscript:** [`paper/civic_safe_ieee.tex`](paper/civic_safe_ieee.tex) (IEEEtran, two-column)
> - **Ready-to-compile archive:** [`paper/civic_safe_submission_bundle.zip`](paper/civic_safe_submission_bundle.zip) — upload straight to Overleaf
> - **Reviewer defense:** [`docs/REVIEWER_DEFENSE_DOSSIER.md`](docs/REVIEWER_DEFENSE_DOSSIER.md)
> - **Equations:** [`MATHEMATICS.md`](MATHEMATICS.md) — source of truth; the paper's Table I matches its notation
>
> ### 2. OICC — over-identification-calibrated conformal deconvolution (`src/oicc/`)
>
> A separate line: honest latent-rate estimation from ≥3 biased measurement
> channels, with a proved impossibility theorem and an honest negative control. See
> [`RESEARCH_ROADMAP.md`](RESEARCH_ROADMAP.md), [`OICC.md`](OICC.md) and
> [`paper/oicc_paper.tex`](paper/oicc_paper.tex).
>
> ### A correction to earlier versions of this banner
>
> This section previously stated that the CIVIC-SAFE forecaster "does **not** beat a
> seasonal-naive baseline (CRPSS vs seasonal-naive is not positive)" and was
> therefore "**not** claimed as a forecasting contribution". **Both statements are
> now false and are retracted.** They were written before level anchoring landed.
> Current results on the 2023 test set: CRPSS vs seasonal naive **+0.3577**
> (Chicago) and **+0.3362** (NYC); against the *rolling* historical average, which
> the evaluation code marks as the binding baseline, **+0.0360** and **+0.0494**.
>
> The forecaster is presented in the manuscript as competitive, with the caveats
> stated plainly there and summarised below. It builds on STZINB-GNN (Zhuang,
> KDD'22) and STMGNN-ZINB (Wang, 2024) and claims no architectural novelty; the
> contribution is the feedback correction.
>
> ### What the manuscript concedes, up front
>
> - The **rolling historical average beats all four deep spatiotemporal baselines on both cities**, eight comparisons out of eight. CIVIC-SAFE is the only method in the study that improves on a trailing mean.
> - A **single seed loses** to TFT-ZINB; the win requires five-seed EMOS ensembling.
> - **Chicago's PIT fails** uniformity at p = 5.25×10⁻⁴⁷, and 65 of its 77 areas are under-predicted. NYC passes cleanly.
> - The latent-coverage result is **simulation-only** by construction — the true rate is unobservable on real data — and at feedback gain 0.85 the corrector **abstains on 85% of cells**.
> - κ is **not point-identified**; real-data results are sensitivity analyses at an assumed gain, and the real-data DiD is a null.

---

## 🔑 Pipeline Components

*Items 1 and 3-8 are correct implementations of **established** methods and are
not claimed as novel. The contribution is the feedback correction (item 9), which
is what the IEEE manuscript is about.*

| # | Component | Method (prior art) | Role |
|---|-----------|--------|---------------|
| 1 | **ZINB Distributional Forecaster** | GATv2 → Causal Transformer → MFFM → ZINB Head (688K params); cf. STZINB-GNN Zhuang KDD'22 | Full count distributions; CRPS 2.8267 (Chicago) / 3.1401 (NYC) on 2023 |
| 2 | **10 Conformal Calibrators** | Split CP, randomized-PIT split CP, recency-weighted CP (Barber 2023), Mondrian ×{group, category, group×category}, Equalized coverage (Romano 2020), variance-scaled split CP, ECRC, rolling adaptive ECRC (per-group ACI, Gibbs–Candès 2021) | Distribution-free coverage + per-group audit; selected under a pre-registered 0.03 disparity ceiling |
| 3 | **EMOS Learned Ensemble** | CRPS-minimized weights (Gneiting et al., 2005) | Optimal combination vs equal-weight |
| 4 | **Post-Hoc Recalibration** | Affine ZINB correction minimizing CRPS | CRPS improvement with zero retraining |
| 5 | **CRPS Decomposition** | Reliability–Resolution–Uncertainty (Hersbach, 2000) | Where forecast skill comes from |
| 6 | **Statistical Significance** | Diebold-Mariano + temporal block bootstrap | Formal p-values under temporal dependence |
| 7 | **Anomaly Skill Coefficient** | ASC diagnostic per demographic group | A skill diagnostic (not a fairness guarantee) |
| 8 | **Advisory Safe Routing** | Exact Dijkstra over conformal edge-costs + abstention protocol | Refuses to route when uncertainty is too high |
| 9 | **Feedback-Corrected Latent Intervals** | Deflation by the recording multiplier + Rosenbaum-style Γ sensitivity envelope | **The contribution.** Restores coverage of the latent process; simulation-validated, abstains near runaway |

---

## 🏗️ Architecture

```
            ╔══════════════════════════════════════════════════════════════════════╗
            ║                     CIVIC-SAFE   (688,649 params)                    ║
            ╠══════════════════════════════════════════════════════════════════════╣
            ║                                                                      ║
            ║   Input: X ∈ ℝ^{S×T×F}     Graphs: E_queen, E_knn                   ║
            ║       │                                                              ║
            ║       ▼                                                              ║
            ║   ┌──────────────────────────────────────┐                           ║
            ║   │  ① GATv2 Spatial Encoder              │  2 layers, 4 heads       ║
            ║   │     Dual adjacency (queen + 8-NN)     │  LayerNorm + ELU         ║
            ║   │     γ_ij = softmax(z^T·LeakyReLU(W·   │  Per-timestep            ║
            ║   │              [h_i ∥ h_j]))             │                          ║
            ║   └──────────────┬───────────────────────┘                           ║
            ║                  │  Stack over T timesteps                            ║
            ║                  ▼                                                    ║
            ║   ┌──────────────────────────────────────┐                           ║
            ║   │  ② Causal Transformer Encoder         │  2 layers, 4 heads       ║
            ║   │     Pre-LN, sinusoidal PE             │  d_ff = 512              ║
            ║   │     Causal mask: M[t,t'] = −∞ if t'>t │  Zero future leakage     ║
            ║   └──────────────┬───────────────────────┘                           ║
            ║                  ▼                                                    ║
            ║   ┌──────────────────────────────────────┐                           ║
            ║   │  ③ Multi-Factor Feature Mixer (MFFM)  │  3 heads, τ = 1.0       ║
            ║   │     Gated cross-attention              │  JSD diversity penalty   ║
            ║   │     Prevents proxy-variable collapse   │  δ_collapse = 0.1        ║
            ║   └──────────────┬───────────────────────┘                           ║
            ║                  │  Last timestep                                     ║
            ║                  ▼                                                    ║
            ║   ┌──────────────────────────────────────┐                           ║
            ║   │  ④ ZINB Distributional Head            │  3 MLPs → (π, μ, r)     ║
            ║   │     π ∈ [0,1]  (zero-inflation)       │  per S×C                 ║
            ║   │     μ ∈ (0,∞)  (NB mean)              │  r_floor = 0.1           ║
            ║   │     r ∈ [0.1,∞) (NB dispersion)       │  logsumexp NLL           ║
            ║   └──────────────┬───────────────────────┘                           ║
            ║                  │                                                    ║
            ║                  ▼                                                    ║
            ║   ┌────────────────────┐  ┌─────────────────┐  ┌──────────────────┐  ║
            ║   │ Conformal          │  │ 7-Component      │  │ Advisory Safe    │  ║
            ║   │ Calibration        │→ │ Equity Audit     │→ │ Routing          │  ║
            ║   │ (5 methods)        │  │ (BH-FDR)         │  │ (Dijkstra)       │  ║
            ║   └────────────────────┘  └─────────────────┘  └──────────────────┘  ║
            ║                                                                      ║
            ╚══════════════════════════════════════════════════════════════════════╝
```

---

## ⚖️ Ethics Commitments

> **These are hard constraints, not aspirational goals. Every module enforces them.**

| # | Commitment | Enforcement |
|---|-----------|-------------|
| 1 | **Civilian-facing only** — no police dashboards | Routing outputs advisory text, not deployment commands |
| 2 | **No person-level prediction** — aggregates only | Input = `(S, T, C)` panel; no individual features |
| 3 | **No protected attributes as inputs** | Demographics used only for post-hoc audit stratification |
| 4 | **Predicts reported incidents, not committed acts** | Mandatory `ReportingBiasSensitivityAudit` with binomial thinning |
| 5 | **Advisory only** — never autonomous | `AdvisoryRoutingEngine` returns text, not actions |
| 6 | **Abstains under diagnostic failure** | `AbstentionMonitor` raises error when uncertainty is too high |

---

## 📊 Results

### Headline Results (2023 test set, 5-seed EMOS ensemble)

Every figure below is read from `outputs/conformal_evaluation/*_conformal_results.json`.

| Metric | Chicago | NYC |
|--------|--------:|----:|
| **CRPS** (↓) | **2.8267** | **3.1401** |
| MAE (↓) | 3.9017 | 4.3675 |
| RMSE (↓) | 7.0983 | 7.6126 |
| Brier, zero event (↓) | 0.0592 | 0.0493 |
| CRPSS vs rolling HA (↑) | +0.0360 | +0.0494 |
| CRPSS vs seasonal naive (↑) | +0.3577 | +0.3362 |
| Selected calibrator | equalized coverage | variance-scaled split CP |
| Marginal coverage (target 90%) | 90.75% | 90.02% |
| Demographic disparity (ceiling 0.03) | 0.0238 | 0.0286 |

All eight head-to-head Diebold–Mariano comparisons against the four deep
spatiotemporal baselines favour CIVIC-SAFE at p < 2×10⁻⁶ (largest 1.671×10⁻⁶).
Against the rolling historical average the margins are smaller and both tests
agree: DM p = 0.0338 / 0.0036, block bootstrap p = 0.0053 / 0.0003.

> ⚠️ **Read the concessions in the banner above before quoting any of these.** In
> particular, the rolling historical average beats every deep baseline in this
> table's comparison set, and a single CIVIC-SAFE seed loses to TFT-ZINB. An
> earlier version of this section reported CRPS 16.90 for NYC from a single
> pre-anchoring seed, alongside a claim that the forecaster does not beat
> seasonal-naive. Both are superseded and retracted.

### Feedback correction (simulation, target coverage 90%)

| κ | naive latent coverage | corrected | cells retained |
|---:|---:|---:|---:|
| 0.00 | 0.950 | 0.949 | 75% |
| 0.50 | 0.780 | 0.948 | 95% |
| 0.85 | **0.162** | **0.930** | **15%** |

Corrected coverage is measured only on retained cells; naive coverage on all
cells. The two do not share a denominator, which is why the retained fraction is
reported everywhere the corrected number appears.

### Datasets

| Property | Chicago | NYC |
|----------|---------|-----|
| Spatial units | 77 community areas | 78 precincts |
| Total incidents | 1.33M | 1.51M |
| Categories | violent, property, drug | violent, property, drug |
| Temporal range | 2018–2023 (weekly) | 2018–2023 (weekly) |
| Demographics | 7 ACS covariates | 7 ACS covariates |
| **Splits** | Train 2018–2021 / Val 2022H1 / Cal 2022H2 / Test 2023 | Same |

> Full 5-seed results (mean ± std) will be published upon paper submission.

---

## 📦 IEEE Submission Package

Everything needed to compile the manuscript, with no dependence on the rest of the
repository.

```
paper/
├── civic_safe_ieee.tex               # IEEEtran two-column manuscript (in-repo copy)
├── references.bib                    # 21 verified BibTeX entries
├── tables/                           # 7 table floats, generated
├── submission_bundle/                # SELF-CONTAINED: tex + bib + figures/ + tables/
│   └── README.md                     #   how to obtain IEEEtran.cls
└── civic_safe_submission_bundle.zip  # the same bundle, zipped for upload
```

**To compile:** upload `civic_safe_submission_bundle.zip` to Overleaf and build with
pdfLaTeX. Overleaf and the IEEE portals both provide `IEEEtran.cls`, which is
deliberately **not** vendored here — a hand-rolled substitute would silently
typeset the paper in something that is not IEEE format. For a local build, install
it from CTAN first (`tlmgr install ieeetran`), then:

```bash
cd paper/submission_bundle
pdflatex civic_safe_ieee && bibtex civic_safe_ieee && pdflatex civic_safe_ieee && pdflatex civic_safe_ieee
```

**Regenerating the package.** Never edit `paper/submission_bundle/` or
`paper/tables/` by hand — both are build artifacts, and the zip is rebuilt with the
bundle so it cannot go stale:

```bash
python scripts/build_paper_tables.py       # outputs/tables/ -> paper/tables/
python scripts/build_submission_bundle.py  # -> submission_bundle/ + .zip
python scripts/validate_latex.py paper/civic_safe_ieee.tex
python scripts/validate_latex.py paper/submission_bundle/civic_safe_ieee.tex
```

`validate_latex.py` checks that every `\cite` resolves to a bib entry, every
`
ef` resolves to a `\label`, and every `\includegraphics` target exists on
disk. It is static analysis and **does not replace compiling** — it cannot see
overfull boxes or float placement.

### Regenerating figures and evidence

```bash
python scripts/make_paper_figures.py            # figs 10-12 from live experiments
python scripts/misspecification_sensitivity.py  # Γ sensitivity table (Section III-C)
python scripts/measure_calibration_latency.py   # calibration latency (dossier Q7)
python scripts/visualize.py                     # spatial residual + bivariate maps
```

Figure scripts write the values they plotted to `outputs/figure_data/*.json`, so any
figure can be checked against the numbers behind it.

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/HimanshuBairwa/civic-safe-Research-Project.git
cd civic-safe-Research-Project

# Install with dev dependencies
pip install -e ".[dev]"

# Verify installation (387 tests (301 civicsafe + 86 OICC), no GPU required)
pytest -v
```

### Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.2 (CUDA 12.x recommended for GPU training)
- PyTorch Geometric ≥ 2.5

### Train a Model

```bash
# Smoke test (2 epochs, 1 seed — ~2 min on A100)
python scripts/train.py training.epochs=2 training.num_seeds=1

# Full experiment (200 epochs, 5 seeds — ~15 hours on A100)
python scripts/train.py

# Override any config via CLI
python scripts/train.py model.spatial.hidden_dim=256 training.lr=0.0005
```

### Experiment Tracking (W&B)

```bash
# Default: W&B disabled (no login required)
python scripts/train.py

# Online mode (live dashboards):
wandb login
WANDB_MODE=online python scripts/train.py

# Offline mode (sync later):
WANDB_MODE=offline python scripts/train.py
wandb sync --sync-all
```

---

## 📂 Project Structure

```
civic-safe-Research-Project/
├── configs/                          # Hydra YAML configurations
│   ├── audit/default.yaml            #   Equity audit settings
│   ├── calibration/                  #   conformal calibration configs
│   │   ├── split_cp.yaml
│   │   ├── weighted_cp.yaml
│   │   ├── mondrian.yaml
│   │   ├── equalized.yaml
│   │   └── ecrc.yaml
│   ├── data/{chicago,nyc}.yaml       #   Dataset configs with taxonomies
│   ├── model/spatiotemporal_zinb.yaml #   Architecture hyperparameters
│   ├── routing/default.yaml          #   Advisory routing settings
│   └── training/default.yaml         #   Training hyperparameters
├── docs/
│   ├── PAPER_OUTLINE.md              # Full paper scaffold (AAAI/NeurIPS)
│   └── METHODOLOGY.md               # Complete mathematical formulation
├── scripts/
│   ├── train.py                      # Multi-seed training entry point
│   ├── fetch_data.py                 # Chicago/NYC data downloader
│   └── reproduce.py                  # Generate paper tables
├── src/civicsafe/
│   ├── data/                         # Taxonomies, crosswalks, ACS, panels
│   ├── models/
│   │   ├── spatial.py                # GATv2 dual-adjacency encoder
│   │   ├── temporal.py               # Causal Transformer encoder
│   │   ├── feature_mixer.py          # Multi-Factor Feature Mixer (MFFM)
│   │   ├── zinb_head.py              # ZINB (π, μ, r) projection head
│   │   ├── zinb_loss.py              # Numerically stable ZINB NLL
│   │   ├── adversarial_head.py       # Adversarial GRL for invariance
│   │   └── civicsafe_model.py        # Full model composition
│   ├── calibration/
│   │   ├── conformal.py              # 10 conformal calibration variants
│   │   ├── zinb_distribution.py      # ZINB CDF/PPF for conformal scores
│   │   └── metrics.py               # Coverage, AIW, calibration metrics
│   ├── audit/
│   │   ├── components.py             # 7 audit components
│   │   ├── harness.py                # Audit orchestration
│   │   ├── statistical.py            # Bootstrap, permutation, BH-FDR
│   │   ├── stratification.py         # Demographic stratification
│   │   └── feedback_loop.py          # Closed-loop and anomaly metrics
│   ├── routing/                      # Dijkstra + abstention engine
│   ├── training/                     # Trainer, scheduler, early stopping
│   ├── synthetic/                    # ZINB/Poisson data generators
│   └── utils/                        # Seeding, numerics, checkpointing
├── tests/                            # 406 tests (civicsafe + OICC)
├── MATHEMATICS.md                    # Equation source of truth (matches paper Table I)
├── REPRODUCIBILITY.md                # NeurIPS reproducibility checklist
├── pyproject.toml                    # Project metadata + tool configs
└── README.md                         # This file
```

---

## 🧪 Test Suite

406 tests collected (civicsafe + OICC) — **no GPU required**. Two `test_data.py::TestACS` cases require a demographics file absent from a fresh clone and fail on environment, not logic; the rest pass.

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_audit.py` | 43 | All 7 audit components, bootstrap, permutation, BH-FDR |
| `test_calibration.py` | 46 | ZINB CDF/PPF, 5 calibrators, metrics, end-to-end |
| `test_checkpointing.py` | 5 | Save/load roundtrip, SHA-256 verification |
| `test_data.py` | 26 | Taxonomies, crosswalks, ACS, panel builder |
| `test_models.py` | 34 | ZINB loss, spatial/temporal encoders, full model |
| `test_numerics.py` | 13 | safe_log, safe_divide, log_sum_exp, clamp_probs |
| `test_routing.py` | 33 | Batched-frontier vs Dijkstra (identical costs), cost functions, abstention, 77-node stress |
| `test_seeding.py` | 7 | Deterministic torch/numpy/python seeding |
| `test_smoke.py` | 1 | Full pipeline smoke test |
| `test_synthetic.py` | 10 | ZINB/Poisson sampling, panel generation |
| `test_training.py` | 46 | CRPS, point metrics, PIT, early stopping, scheduler, trainer |
| **Total** | **406** | **collected across all core modules** |

```bash
pytest -v                         # Run all 406 tests
pytest tests/test_routing.py -v   # Run routing tests only
pytest tests/test_audit.py -v     # Run audit tests only
mypy src/civicsafe/               # Type checking (strict mode)
```

---

## 🔬 Reproducibility

| Component | Detail |
|-----------|--------|
| **Seeds** | `[42, 137, 256, 512, 1024]` — mean ± std across 5 seeds |
| **Hardware** | NVIDIA A100 40GB recommended (12GB sufficient with gradient checkpointing) |
| **Training** | AdamW, lr = 1e-3, cosine warmup (10 epochs), BFloat16, gradient clipping 1.0 |
| **Duration** | ~3 hours per seed on A100 |
| **Determinism** | Full seeding of Python, NumPy, PyTorch, CUDA |

```bash
# Full reproduction
git clone https://github.com/HimanshuBairwa/civic-safe-Research-Project.git
cd civic-safe-Research-Project
pip install -e ".[dev]"
pytest -v                    # 387 passed
python scripts/train.py      # 5-seed training
python scripts/reproduce.py  # Generate paper tables
```

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full NeurIPS checklist.

---

## 📖 Documentation

| Document | Description |
|----------|-----------|
| [PAPER_OUTLINE.md](docs/PAPER_OUTLINE.md) | Complete paper scaffold for top-tier ML venues |
| [METHODOLOGY.md](docs/METHODOLOGY.md) | Full mathematical formulation (10 sections) |
| [MATHEMATICS.md](MATHEMATICS.md) | Concise math specification |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | NeurIPS reproducibility checklist |

---

## 📝 Citation

```bibtex
@article{bairwa2025civicsafe,
  title     = {Honest Uncertainty Under Observation Bias: Online Conformal Prediction for Count Forecasting with Closed-Loop Feedback Evaluation},
  author    = {Bairwa, Himanshu},
  year      = {2025},
  journal   = {arXiv preprint},
  url       = {https://github.com/HimanshuBairwa/civic-safe-Research-Project},
  note      = {GATv2 + Causal Transformer + ZINB distributional head
               with 5 conformal calibrators, 7-component equity audit,
               and Dijkstra-based advisory routing}
}
```

---

## 🔬 Full Evaluation Pipeline

The evaluation pipeline (`scripts/run_conformal_evaluation.py`) produces a comprehensive JSON report with:

```
📊 Ensemble      → Per-seed CRPS, EMOS weights, learned-weight CRPS
📊 Uncertainty   → Aleatoric (ZINB var) vs Epistemic (seed disagreement)
📊 6 Conformal   → Coverage, width, disparity for each method
📊 Rolling ECRC  → Per-window coverage, alpha convergence trajectory
📊 Baselines     → HA CRPS, Seasonal Naive CRPS
📊 CRPSS         → Skill score vs HA and Seasonal Naive
📊 Recalibration → Before/after CRPS, improvement %, learned params
📊 PIT Histogram → Chi-squared uniformity test
📊 CRPS Decomp   → Reliability, Resolution, Uncertainty (Hersbach 2000)
📊 DM Test       → Diebold-Mariano + block bootstrap p-values
📊 ASC           → Anomaly Skill Coefficient per demographic group
📊 Point Metrics → MAE, RMSE, Brier score
```

## 🔄 Reproducibility

```bash
# 1. Data
python scripts/fetch_data.py --city chicago
python scripts/fetch_data.py --city nyc
python scripts/build_demographics.py --city chicago
python scripts/build_demographics.py --city nyc

# 2. Training (5 seeds × 2 cities)
python scripts/train.py data=chicago
python scripts/train.py data=nyc

# 3. Evaluation (produces full JSON report)
python scripts/run_conformal_evaluation.py --data chicago
python scripts/run_conformal_evaluation.py --data nyc

# 4. Baselines (traditional + deep learning)
python scripts/baselines.py data=chicago
python scripts/deep_baselines.py data=chicago

# 5. Visualization
python scripts/generate_figures.py --data chicago

# 6. Ablation tables
python scripts/ablation_study.py --data chicago --data nyc
```

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for details.

---

<div align="center">

**CIVIC-SAFE** — *Because count forecasting without uncertainty quantification, equity auditing, and principled abstention is not ready for the real world.*

</div>
