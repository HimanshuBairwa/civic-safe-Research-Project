# NeurIPS Reproducibility Checklist

This document details how CIVIC-SAFE adheres to the machine learning reproducibility standards required by top-tier conferences (NeurIPS, ICLR, ICML).

## 1. Code and Environment
- [x] **Dependencies:** Full dependencies are pinned in `pyproject.toml` and lock files. Use `pip install -e ".[dev]"` for identical reproduction.
- [x] **Hardware:** Tested extensively on NVIDIA A100 (40GB) and consumer CPUs. Training logs report device types and precision modes (`amp=bf16`).
- [x] **Code structure:** `src/civicsafe/` strictly isolates Data, Models, Calibration, Audit, and Routing.

## 2. Experimental Rigor and Randomness
- [x] **Random Seeds:** All results are aggregated across 5 distinct seeds (`[42, 137, 256, 512, 1024]`).
- [x] **Framework Sync:** `civicsafe.utils.seeding` strictly synchronizes seeds across `torch`, `numpy`, `random`, and CUDNN.
- [x] **Determinism:** PyTorch deterministic algorithms are enabled by default for A100/CuBLAS consistency.
- [x] **Reporting:** Predictive metrics (CRPS, MAE, RMSE) and audit thresholds are reported as `mean ± std` across seeds.

## 3. Data Provenance and Splits
- [x] **Sourcing:** Real data pipelines use the official Socrata API for Chicago Data Portal and NYC OpenData.
- [x] **Splits:** A strict chronological split prevents temporal leakage:
  - Train: 2018–2021
  - Val: 2022
  - Test: 2023
- [x] **Protected Attributes:** Demographics are sourced from the US Census (ACS 5-Year) and are **strictly shielded** from the model during training.

## 4. Hyperparameters and Training
- [x] **Configuration:** Managed via Hydra (`configs/`). Default configuration represents the exact settings used for the paper.
- [x] **Optimization:** We document the use of BFloat16, EMA (decay=0.999), Cosine Warmup, and gradient clipping explicitly in the Trainer.
- [x] **Early Stopping:** Triggered objectively via validation CRPS with a patience of 50 epochs.

## 5. Automated Table Generation
- [x] **LaTeX Exporter:** The `scripts/reproduce.py` script automatically parses W&B logs and outputs publication-ready LaTeX tables (`outputs/results/*.tex`).
- [x] **No Manual Edits:** The pipeline ensures that the numbers in the paper exactly match the raw model outputs.

## 6. Ethics Commitments
- [x] Explicitly documented in `README.md`.
- [x] Enforced programmatically (e.g., `ReportingBiasSensitivityAudit` fails if reporting rate decay drops coverage; `AbstentionMonitor` raises `AbstentionError`).
