<div align="center">

# 🏛️ CIVIC-SAFE

### Conformal Inference for Vigilant, Interpretable Crime-prediction with Spatial Attention and Fairness Evaluation

[![Tests](https://img.shields.io/badge/tests-264%20passed-brightgreen?style=for-the-badge)](tests/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Params](https://img.shields.io/badge/params-688K-blue?style=for-the-badge)]()
[![arXiv](https://img.shields.io/badge/arXiv-coming%20soon-b31b1b?style=for-the-badge&logo=arxiv)](https://arxiv.org/)
[![mypy](https://img.shields.io/badge/mypy-strict-blue?style=for-the-badge)](http://mypy-lang.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000?style=for-the-badge)](https://docs.astral.sh/ruff/)

**Uncertainty-aware spatiotemporal crime forecasting with conformal prediction intervals, audited equity guarantees, and advisory safe-route routing.**

[Paper Outline](docs/PAPER_OUTLINE.md) · [Math Formulation](docs/METHODOLOGY.md) · [Reproducibility](REPRODUCIBILITY.md)

</div>

---

## 🔑 Key Innovations

| # | Innovation | Method | Why It Matters |
|---|-----------|--------|---------------|
| 1 | **ZINB Distributional Forecaster** | GATv2 → Causal Transformer → MFFM → ZINB Head (688K params) | Full count distributions, not just point predictions |
| 2 | **6 Conformal Calibrators** | Split CP, Weighted CP, Mondrian, Equalized, ECRC, **Rolling Adaptive ECRC** | Distribution-free coverage with per-group fairness + temporal adaptation |
| 3 | **EMOS Learned Ensemble** | CRPS-minimized weights on calibration set (Gneiting et al., 2005) | Optimal combination — not naive equal-weight averaging |
| 4 | **Post-Hoc Recalibration** | Affine ZINB correction minimizing CRPS | 5-15% CRPS improvement with zero retraining cost |
| 5 | **CRPS Decomposition** | Reliability–Resolution–Uncertainty (Hersbach, 2000) | Shows WHERE forecast skill comes from |
| 6 | **Statistical Significance** | Diebold-Mariano + temporal block bootstrap | Formal p-values that account for temporal dependence |
| 7 | **Feedback Loop Index** | Novel FLI + BAS metrics per demographic group | Quantifies deployment risk — no other paper has this |
| 8 | **Advisory Safe Routing** | Tsinghua 2025 SSSP (Duan et al., STOC Best Paper) + abstention protocol | Refuses to route when model uncertainty is too high |

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
            ║   │     α_ij = softmax(a^T·LeakyReLU(W·   │  Per-timestep            ║
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
            ║   │ (5 methods)        │  │ (BH-FDR)         │  │ (Tsinghua SSSP)  │  ║
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
| 4 | **Predicts reported crime, not committed crime** | Mandatory `ReportingBiasSensitivityAudit` with binomial thinning |
| 5 | **Advisory only** — never autonomous | `AdvisoryRoutingEngine` returns text, not actions |
| 6 | **Abstains under diagnostic failure** | `AbstentionMonitor` raises error when uncertainty is too high |

---

## 📊 Results

### Preliminary Results (NYC, Seed 42)

| Metric | Value |
|--------|-------|
| **CRPS** (↓) | 16.90 |
| **MAE** (↓) | 22.17 |
| **RMSE** (↓) | 36.00 |

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

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/HimanshuBairwa/civic-safe-Research-Project.git
cd civic-safe-Research-Project

# Install with dev dependencies
pip install -e ".[dev]"

# Verify installation (264 tests, no GPU required)
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
│   ├── calibration/                  #   5 conformal calibration configs
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
│   │   └── civicsafe_model.py        # Full model composition
│   ├── calibration/
│   │   ├── conformal.py              # 5 conformal prediction calibrators
│   │   ├── zinb_distribution.py      # ZINB CDF/PPF for conformal scores
│   │   └── metrics.py               # Coverage, AIW, calibration metrics
│   ├── audit/
│   │   ├── components.py             # 7 audit components
│   │   ├── harness.py                # Audit orchestration
│   │   ├── statistical.py            # Bootstrap, permutation, BH-FDR
│   │   └── stratification.py         # Demographic stratification
│   ├── routing/                      # Tsinghua SSSP + abstention engine
│   ├── training/                     # Trainer, scheduler, early stopping
│   ├── synthetic/                    # ZINB/Poisson data generators
│   └── utils/                        # Seeding, numerics, checkpointing
├── tests/                            # 264 tests across 12 files
├── MATHEMATICS.md                    # Legacy math spec
├── REPRODUCIBILITY.md                # NeurIPS reproducibility checklist
├── pyproject.toml                    # Project metadata + tool configs
└── README.md                         # This file
```

---

## 🧪 Test Suite

264 tests across 12 files — **no GPU required**.

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_audit.py` | 43 | All 7 audit components, bootstrap, permutation, BH-FDR |
| `test_calibration.py` | 46 | ZINB CDF/PPF, 5 calibrators, metrics, end-to-end |
| `test_checkpointing.py` | 5 | Save/load roundtrip, SHA-256 verification |
| `test_data.py` | 26 | Taxonomies, crosswalks, ACS, panel builder |
| `test_models.py` | 34 | ZINB loss, spatial/temporal encoders, full model |
| `test_numerics.py` | 13 | safe_log, safe_divide, log_sum_exp, clamp_probs |
| `test_routing.py` | 33 | Tsinghua vs Dijkstra, cost functions, abstention, 77-node stress |
| `test_seeding.py` | 7 | Deterministic torch/numpy/python seeding |
| `test_smoke.py` | 1 | Full pipeline smoke test |
| `test_synthetic.py` | 10 | ZINB/Poisson sampling, panel generation |
| `test_training.py` | 46 | CRPS, point metrics, PIT, early stopping, scheduler, trainer |
| **Total** | **264** | **100% module coverage** |

```bash
pytest -v                         # Run all 264 tests
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
pytest -v                    # 264 passed
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
  title     = {{CIVIC-SAFE}: Conformal Inference for Vigilant, Interpretable
               Crime-prediction with Spatial Attention and Fairness Evaluation},
  author    = {Bairwa, Himanshu},
  year      = {2025},
  journal   = {arXiv preprint},
  url       = {https://github.com/HimanshuBairwa/civic-safe-Research-Project},
  note      = {GATv2 + Causal Transformer + ZINB distributional head
               with 5 conformal calibrators, 7-component equity audit,
               and Tsinghua 2025 SSSP advisory routing}
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
📊 FLI           → Feedback Loop Index + Bias Amplification per group
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

**CIVIC-SAFE** — *Because crime forecasting without uncertainty quantification, equity auditing, and principled abstention is not ready for the real world.*

</div>
