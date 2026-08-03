# CIVIC-SAFE Conformal Prediction Audit Report

**Dataset:** chicago  
**Timestamp:** 2026-08-02T04:48:10.890297  
**Alpha (miscoverage):** 0.1  
**Checkpoint:** `best.pt')]`  
**Panel hash:** `4bb2e1e3322b`  

## Point Forecast Metrics (Test Set — 2023)

| Metric | Value |
|--------|-------|
| CRPS | 3.2291 |
| MAE | 4.5144 |
| RMSE | 8.2103 |
| Brier (zero-inflation) | 0.0677 |

## CRPS Skill Score

| Component | Value |
|-----------|-------|
| Baseline CRPS (Historical Average) | 3.8781 |
| Baseline CRPS (Seasonal Naive) | 4.4008283615112305 |
| Model CRPS | 3.2291 |
| CRPSS vs HA | 0.1673 |
| **CRPSS vs Seasonal Naive** | **0.2662** |
| Threshold (≥0.10 vs SN) | ✓ PASS |

## Coverage Results by Calibration Method

| Method | Marginal Coverage | Target | Mean Width | Disparity |
|--------|:-----------------:|:------:|:----------:|:---------:|
| split_cp | ⚠ 0.9278 | 0.90 | 17.15 | 0.0308 |
| weighted_cp | ⚠ 0.9278 | 0.90 | 17.15 | 0.0308 |
| mondrian | ⚠ 0.9210 | 0.90 | 16.79 | 0.0218 |
| equalized_coverage | ✓ 0.9070 | 0.90 | 15.51 | 0.0384 |
| ecrc | ⚠ 0.9554 | 0.90 | 20.79 | 0.0193 |
| adaptive_ecrc | ⚠ 0.9554 | 0.90 | 20.79 | 0.0193 |
| adaptive_ecrc_rolling | ⚠ 0.9194 | 0.90 | 16.48 | 0.0136 |

### Per-Category Coverage (equalized_coverage)

| Category | Coverage | Width | N |
|----------|:--------:|:-----:|--:|
| violent | 0.9079 | 17.76 | 4081 |
| property | 0.8314 | 23.65 | 4081 |
| drug | 0.9816 | 5.13 | 4081 |

### Per-Demographic-Quartile Coverage (equalized_coverage)

| Group | Coverage | Width | N |
|-------|:--------:|:-----:|--:|
| group_0 | 0.9035 | 19.00 | 3180 |
| group_1 | 0.8878 | 14.09 | 3021 |
| group_2 | 0.9106 | 13.43 | 3021 |
| group_3 | 0.9262 | 15.36 | 3021 |

## Methods Paragraph (Paper-Ready)

We apply Conformalized Quantile Regression (CQR; Romano et al., 2019) 
to the ZINB predictive distribution, computing non-conformity scores 
$s_i = \max(\hat{q}_{\alpha/2}(X_i) - Y_i, Y_i - \hat{q}_{1-\alpha/2}(X_i))$ 
on a held-out calibration set (2022 H2, 26 windows, 
6006 observations). The calibration threshold 
$\hat{q}$ is chosen as the $\lceil (1-\alpha)(1+1/n) \rceil$-th empirical 
quantile of the scores, guaranteeing finite-sample marginal coverage 
$P(Y \in [L, U]) \geq 1-\alpha$ under exchangeability. To correct for 
temporal non-exchangeability, we additionally implement Adaptive Conformal 
Inference (ACI; Gibbs & Candès, 2021) with per-demographic-quartile tracking, 
achieving asymptotic conditional coverage $P(Y \in C(X) | G=g) \to 1-\alpha$ 
for each income quartile $g$. On the 2023 test set (53 windows), 
the best calibrator (equalized_coverage) achieves 90.7% marginal 
coverage with mean prediction interval width 15.51 
counts and a maximum cross-group coverage disparity of 
0.0384.

## Ablation TODO Registry (Table 2)

- [ ] ACI gamma sensitivity: γ ∈ {0.01, 0.05, 0.1, adaptive-PI}
- [ ] Calibration set size: 13 vs 26 vs 52 weeks
- [ ] Group granularity: geographic (S groups) vs demographic (4 quartiles)
- [ ] CQR vs ABS vs RAPS non-conformity score functions
- [ ] ECRC delta sensitivity: δ ∈ {0.01, 0.05, 0.1}
- [ ] Cross-city transfer: calibrate on Chicago, test on NYC