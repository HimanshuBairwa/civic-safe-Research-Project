# Reproducibility Checklist & Statement

In accordance with the NeurIPS Paper Checklist guidelines, this document explicitly addresses the steps taken to ensure that all models, calibrations, audits, and routing algorithms in CIVIC-SAFE are transparent, reproducible, and verifiable.

## 1. Environment & Dependencies

- [x] **Exact versions:** We provide a strict `pyproject.toml` with pinned major/minor versions for all dependencies.
- [x] **Frameworks:** PyTorch ≥ 2.2, PyTorch Geometric ≥ 2.5, Hydra Core ≥ 1.3.2.
- [x] **Hardware disclosure:** All timing benchmarks (e.g., 2 minutes/epoch) were performed on a single NVIDIA A100 40GB GPU using BFloat16 mixed precision. CPU fallback is fully supported and tested.

## 2. Code & Random Seeds

- [x] **One-click reproducibility:** The full experimental pipeline can be reproduced using:
  ```bash
  python scripts/reproduce.py
  ```
- [x] **Seed control:** We use the `seed_everything(seed)` utility from `civicsafe.utils.seeding` to freeze Python `random`, NumPy `np.random`, and `torch.manual_seed`.
- [x] **Multiple seeds:** The primary training script (`scripts/train.py`) defaults to evaluating across 5 fixed seeds: `[42, 137, 256, 512, 1024]`.
- [x] **Determinism:** `torch.use_deterministic_algorithms(True)` is supported. Note: On A100 GPUs, `FlashAttention` is inherently non-deterministic. If absolute bit-for-bit parity is required, FlashAttention must be disabled.

## 3. Statistical Reporting

- [x] **Aggregated metrics:** We report the `mean ± std` for all evaluation metrics (CRPS, MAE, RMSE, Brier score) across the 5 independent seeds.
- [x] **Audited significance:** The equity audit (`civicsafe.audit.components`) utilizes permutation tests (B=1000) for disparate impact significance, and applies Benjamini-Hochberg False Discovery Rate (BH-FDR) correction when testing across multiple strata.
- [x] **Coverage guarantees:** Our PAC-style conformal calibrator (ECRC) provides explicit probability bounds for group-conditional coverage, scaling the Hoeffding slack $\epsilon$ by the group size $n_g$.

## 4. Data Provenance

- [x] **Public datasets:** This benchmark uses exclusively public open-data portals:
  - **Chicago PD:** City of Chicago Data Portal (Crimes - 2001 to Present)
  - **NYPD:** NYC OpenData (NYPD Complaint Data Historic)
  - **Demographics:** American Community Survey (ACS) 5-Year Estimates via `cenpy`.
- [x] **Data generators:** We provide `civicsafe.synthetic.distributions` which generates zero-inflated negative binomial (ZINB) data with known ground-truth parameters, enabling exact theoretical validation of the algorithms without external data downloads.

## 5. Ethics Commitments

- [x] **Civilian-facing only:** The routing output is advisory and designed for civilian use, never for patrol allocation or predictive policing.
- [x] **No person-level data:** The model operates strictly on `(Spatial, Temporal, Category)` aggregate tensor inputs.
- [x] **Reporting bias disclosure:** The system formally measures robustness to crime under-reporting using the `ReportingBiasSensitivityAudit`.
- [x] **Abstention:** The routing engine raises an `AbstentionError` when the conformal interval widths indicate that the predictive uncertainty is too high to guarantee a safe route.
