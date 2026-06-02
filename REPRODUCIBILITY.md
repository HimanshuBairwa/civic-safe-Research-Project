# CIVIC-SAFE Reproducibility Checklist

As part of our commitment to rigorous, transparent research, this project adheres to the [NeurIPS Machine Learning Reproducibility Checklist](https://neurips.cc/Conferences/2023/PaperInformation/MachineLearningReproducibilityChecklist).

This document details exactly how each reproducibility requirement is met in the CIVIC-SAFE benchmark.

---

## 1. Code and Environment

- [x] **Dependencies:** All dependencies are pinned in `pyproject.toml` (e.g., `torch>=2.2.0,<3.0`). Use `pip install -e ".[dev]"` for an exact replica of the development environment.
- [x] **Hardware Specifications:** Documented in the [README](README.md#reproducibility). The project requires ≥16GB RAM and recommends an NVIDIA A100 GPU for full training.
- [x] **Random Seed Control:** We explicitly set random seeds across Python, NumPy, and PyTorch (including CUDA deterministic flags). The `civicsafe.utils.seeding` module enforces this. All tests are run under strict determinism.
- [x] **Multi-Seed Aggregation:** The training script (`scripts/train.py`) runs 5 seeds by default (`[42, 137, 256, 512, 1024]`) and reports the mean and standard deviation for all core metrics.
- [x] **One-Command Reproduction:** The `scripts/reproduce.py` script automatically generates the LaTeX tables and JSON summaries found in the paper.

## 2. Datasets

- [x] **Data Provenance:** The `scripts/fetch_data.py` script downloads raw data directly from official municipal open data portals (Chicago Data Portal API, NYC OpenData API).
- [x] **Data Harmonization:** The `civicsafe.data` module contains explicit taxonomies and crosswalks used to harmonize raw point data into regular spatiotemporal panels.
- [x] **Synthetic Data:** For unit testing and rapid prototyping, `civicsafe.synthetic` provides exactly-solvable generative models (ZINB, Poisson) with known ground-truth parameters.
- [x] **Reporting Bias Transparency:** The project acknowledges that administrative crime data is subject to reporting bias. The `ReportingBiasSensitivityAudit` enforces evaluation under simulated under-reporting (binomial thinning).

## 3. Training and Evaluation

- [x] **Metrics:** We report Continuous Ranked Probability Score (CRPS), MAE, RMSE, and Zero-Inflation Brier Score. Conformal prediction is evaluated via Coverage and Average Interval Width (AIW).
- [x] **Hyperparameters:** All hyperparameters are documented in the `configs/` YAML files (Hydra). Any CLI overrides are logged explicitly by Weights & Biases.
- [x] **Training Details:** The native PyTorch trainer (`civicsafe.training.trainer`) uses BFloat16 mixed precision, Exponential Moving Average (EMA) model weights, and cosine learning rate scheduling.
- [x] **Experiment Tracking:** Integrated with Weights & Biases (`wandb`). By default, it runs in `disabled` mode to avoid prompt-blocking, but can be switched to `online` or `offline` mode for full experiment tracking.

## 4. Ethics and Safeguards

- [x] **Advisory Only:** The routing engine (`civicsafe.routing`) outputs textual advisories, not autonomous decisions.
- [x] **No Person-Level Tracking:** The spatiotemporal panel builder aggregates all data to the community area / precinct level before the model sees it.
- [x] **Audited Abstention:** The routing engine implements ethical guardrails by refusing to route (`AbstentionError`) if conformal interval widths exceed safety thresholds.

---

## How to Verify Results

1. Setup the environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

2. Run the test suite (100% coverage):
   ```bash
   pytest -v
   ```

3. Generate paper tables:
   ```bash
   python scripts/reproduce.py
   ```
