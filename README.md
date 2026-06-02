# CIVIC-SAFE: Audited Conformal Crime Forecasting

<div align="center">

[![Tests](https://img.shields.io/badge/tests-264%20passed-brightgreen?style=for-the-badge)](tests/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/pytorch-2.2%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![mypy](https://img.shields.io/badge/mypy-strict-blue?style=for-the-badge)](http://mypy-lang.org/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000?style=for-the-badge)](https://docs.astral.sh/ruff/)

**Uncertainty-aware spatiotemporal crime forecasting with conformal prediction intervals, audited equity guarantees, and advisory safe-route routing using the Tsinghua 2025 SSSP algorithm.**

</div>

---

## Table of Contents

- [Overview](#overview)
- [Ethics Commitments](#ethics-commitments)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [W&B Experiment Tracking](#wb-experiment-tracking)
- [Project Structure](#project-structure)
- [Modules](#modules)
- [Test Suite](#test-suite)
- [Reproducibility](#reproducibility)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)

---

## Overview

CIVIC-SAFE (**C**onformal **I**ntervals for **V**alidated **I**nference on **C**rime — **S**tatistically **A**udited **F**air **E**stimates) is a research benchmark that addresses three critical gaps in crime forecasting:

1. **Point predictions are overconfident** → We provide *conformal prediction intervals* with finite-sample coverage guarantees (5 calibration methods).
2. **Predictions amplify historical bias** → We enforce *7-component equity audits* with BH-FDR correction across demographic and geographic strata.
3. **Routing ignores model uncertainty** → Our *advisory safe-route engine* uses the state-of-the-art Tsinghua 2025 SSSP algorithm and *refuses to recommend routes* when prediction uncertainty is too high.

### Key Innovations

| Innovation | Method | Reference |
|---|---|---|
| **ZINB Forecaster** | GATv2 spatial + Transformer temporal + ZINB distributional head | — |
| **5 Conformal Calibrators** | Split CP, Weighted CP, Mondrian, Equalized Coverage, ECRC | Romano et al. (2019), Vovk (2005), Feldman et al. (2021) |
| **7-Component Equity Audit** | Coverage, Width, Point, Calibration, Winkler, Abstention, Reporting Bias | AIF360/Fairlearn patterns + BH-FDR (1995) |
| **Tsinghua SSSP Router** | Frontier-reduction shortest path (breaks 40-year Dijkstra sorting barrier) | Duan et al. (STOC 2025, Best Paper) |
| **Audited Abstention** | Conformal interval-width guardrails on routing decisions | Novel |

---

## Ethics Commitments

> These are **hard constraints**, not aspirational goals. Every module enforces them.

| # | Commitment | Enforcement |
|---|---|---|
| 1 | **Civilian-facing only** — no police dashboards, no patrol allocation | Routing outputs advisory text, not deployment commands |
| 2 | **No person-level prediction** — spatial-temporal aggregates only | Model architecture: input = `(S, T, C)` panel, no individual features |
| 3 | **No protected attributes as inputs** — evaluation strata only | Audit module reads demographics, model never sees them |
| 4 | **Predicts reported crime, not committed crime** | Mandatory `ReportingBiasSensitivityAudit` with binomial thinning |
| 5 | **Advisory only** — never autonomous | `AdvisoryRoutingEngine` returns text, not actions |
| 6 | **Abstains under diagnostic failure** | `AbstentionMonitor` raises `AbstentionError` when uncertainty is too high |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CIVIC-SAFE Pipeline                       │
├─────────────┬──────────────┬───────────────┬────────────────────┤
│   Data      │   Model      │  Calibration  │      Output        │
│             │              │               │                    │
│ Chicago PD  │  GATv2       │  Split CP     │  Prediction        │
│ NYC NYPD    │  Spatial     │  Weighted CP  │  Intervals         │
│ ACS Census  │  Encoder     │  Mondrian CP  │  [lower, upper]    │
│             │      ↓       │  Equalized    │        ↓           │
│ Panel       │  Transformer │  ECRC         │  Equity Audit      │
│ Builder     │  Temporal    │               │  (7 components)    │
│ (S,T,C)     │  Encoder     │  Coverage     │        ↓           │
│             │      ↓       │  Guarantee:   │  Advisory          │
│ Crosswalks  │  ZINB Head   │  P(Y∈C)≥1-α  │  Safe-Route        │
│ tract→area  │  (π, μ, r)   │               │  (Tsinghua SSSP)   │
└─────────────┴──────────────┴───────────────┴────────────────────┘
```

---

## Installation

```bash
# Clone
git clone https://github.com/HimanshuBairwa/civic-safe-Research-Project.git
cd civic-safe-Research-Project

# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# Verify
pytest -v  # Should show 264 passed
```

### Requirements

- Python ≥ 3.11
- PyTorch ≥ 2.2 (CUDA 12.x recommended for GPU training)
- PyTorch Geometric ≥ 2.5

---

## Quick Start

### Run Tests (No GPU Required)

```bash
# Full suite: 264 tests across 12 test files
pytest -v

# Just routing tests
pytest tests/test_routing.py -v

# Just audit tests
pytest tests/test_audit.py -v

# Type checking
mypy src/civicsafe/
```

### Train a Model

```bash
# Smoke test (2 epochs, 1 seed — ~2 min on A100)
python scripts/train.py training.epochs=2 training.num_seeds=1

# Full experiment (100 epochs, 5 seeds — ~15 hours on A100)
python scripts/train.py
```

---

## W&B Experiment Tracking

CIVIC-SAFE integrates with [Weights & Biases](https://wandb.ai/) for experiment tracking. Here's how to use it:

### Option 1: Disabled (Default — No Login Required)

```bash
# W&B is DISABLED by default — the training script runs without prompts
python scripts/train.py training.epochs=2 training.num_seeds=1
```

The script already defaults to `WANDB_MODE=disabled` so it **never blocks** execution with login prompts.

### Option 2: Offline Mode (Logs Locally, Sync Later)

```bash
# Log locally without internet — sync to cloud later
export WANDB_MODE=offline
python scripts/train.py

# Later, when you have internet:
wandb sync --sync-all
```

### Option 3: Online Mode (Live Dashboard)

```bash
# 1. Create a free account at https://wandb.ai/
# 2. Login once:
wandb login

# 3. Enable online tracking:
export WANDB_MODE=online
python scripts/train.py
```

This gives you live dashboards with loss curves, metric comparisons across seeds, and hyperparameter tracking.

### What Gets Logged

| Metric | Description |
|---|---|
| `train/loss` | ZINB NLL + diversity penalty (per epoch) |
| `val/crps` | Continuous Ranked Probability Score |
| `val/mae` | Mean Absolute Error |
| `val/rmse` | Root Mean Squared Error |
| `val/brier_zero` | Brier score for zero-inflation calibration |
| `best_*_mean` | Aggregate mean across seeds (summary) |
| `best_*_std` | Standard deviation across seeds (summary) |

---

## Project Structure

```
civic-safe-Research-Project/
├── configs/
│   ├── audit/           # Equity audit configuration
│   ├── calibration/     # 5 conformal calibration configs
│   ├── data/            # Chicago / NYC data configs
│   ├── experiment/      # Full experiment configs
│   ├── model/           # Spatiotemporal ZINB model config
│   ├── routing/         # Advisory routing config
│   └── training/        # Training hyperparameters
├── scripts/
│   ├── train.py         # Multi-seed training entry point
│   └── fetch_data.py    # Chicago/NYC data downloader
├── src/civicsafe/
│   ├── data/            # Data loading, taxonomies, crosswalks
│   ├── models/          # GATv2 + Transformer + ZINB model
│   ├── calibration/     # 5 conformal prediction calibrators
│   ├── training/        # Trainer, scheduler, early stopping, metrics
│   ├── audit/           # 7-component equity audit harness
│   ├── routing/         # Tsinghua SSSP router + abstention
│   ├── synthetic/       # Synthetic data generators
│   └── utils/           # Seeding, numerics, checkpointing
├── tests/               # 264 tests across 12 files
├── pyproject.toml       # Project metadata + tool configs
└── README.md            # This file
```

---

## Modules

### 📊 Data (`civicsafe.data`)
Crime taxonomies for Chicago PD and NYPD, census crosswalks (tract → community area / precinct), and ACS demographic data.

### 🧠 Models (`civicsafe.models`)
Spatiotemporal ZINB forecaster: dual-adjacency GATv2 spatial encoder → causal Transformer temporal encoder → multi-feature mixture module → ZINB distributional head outputting (π, μ, r) per spatial unit per crime category.

### 📐 Calibration (`civicsafe.calibration`)
Five conformal prediction methods providing prediction intervals with coverage guarantees. The primary method (ECRC) provides per-group PAC-style guarantees using Hoeffding bounds.

### ⚖️ Audit (`civicsafe.audit`)
Seven audit components evaluating equity across demographic and geographic strata. Includes bootstrap CIs, permutation tests, and BH-FDR correction for multiple comparisons. The `ReportingBiasSensitivityAudit` performs INAR binomial thinning to assess robustness to under-reporting.

### 🗺️ Routing (`civicsafe.routing`)
Advisory safe-route engine using the **Tsinghua 2025 SSSP algorithm** (Duan et al., STOC 2025 Best Paper). Features Pareto multi-objective cost functions (distance + risk + uncertainty) and `AbstentionMonitor` that refuses to recommend routes when conformal interval widths exceed calibrated thresholds.

### 🔧 Training (`civicsafe.training`)
Native PyTorch training loop with BFloat16 mixed precision, EMA model averaging, cosine warmup scheduler, and W&B integration.

---

## Test Suite

| Test File | Tests | What It Covers |
|---|---|---|
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

---

## Reproducibility

### Hardware

| Component | Specification |
|---|---|
| GPU | NVIDIA A100 40GB (recommended) |
| CPU | Any modern x86_64 |
| RAM | ≥ 16 GB |
| Storage | ≥ 10 GB |

### Reproducing Results

```bash
# 1. Clone and install
git clone https://github.com/HimanshuBairwa/civic-safe-Research-Project.git
cd civic-safe-Research-Project
pip install -e ".[dev]"

# 2. Verify test suite
pytest -v  # 264 passed

# 3. Run full training (5 seeds)
python scripts/train.py

# 4. Results appear in outputs/run_<timestamp>/
```

### Seeds

The default 5 seeds are: `[42, 137, 256, 512, 1024]`. All results are reported as **mean ± std** across seeds for statistical rigor.

---

## Configuration

CIVIC-SAFE uses YAML configs in `configs/`. Override any parameter via CLI:

```bash
# Change model size
python scripts/train.py model.spatial.hidden_dim=256

# Change learning rate
python scripts/train.py training.lr=0.0005

# Change number of seeds
python scripts/train.py training.num_seeds=3
```

---

## Citation

```bibtex
@software{civicsafe2025,
  title     = {CIVIC-SAFE: Audited Conformal Crime Forecasting Under
               Temporal Drift and Reporting Uncertainty},
  author    = {Bairwa, Himanshu},
  year      = {2025},
  url       = {https://github.com/HimanshuBairwa/civic-safe-Research-Project},
  note      = {Features Tsinghua 2025 SSSP routing (Duan et al., STOC 2025)}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
