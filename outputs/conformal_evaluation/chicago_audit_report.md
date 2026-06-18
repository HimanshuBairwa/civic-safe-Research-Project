# CIVIC-SAFE Conformal Prediction Audit Report

**Dataset:** chicago  
**Timestamp:** 2026-06-16T11:47:45.193975  
**Alpha (miscoverage):** 0.1  
**Checkpoint:** `best.pt`  
**Panel hash:** `4bb2e1e3322b`  

## Point Forecast Metrics (Test Set — 2023)

| Metric | Value |
|--------|-------|
| CRPS | 15.4813 |
| MAE | 18.5353 |
| RMSE | 30.8644 |
| Brier (zero-inflation) | 0.1633 |

## CRPS Skill Score

| Component | Value |
|-----------|-------|
| Baseline CRPS (Historical Average) | 3.8787 |
| Baseline CRPS (Seasonal Naive) | 4.401771545410156 |
| Model CRPS | 15.4813 |
| CRPSS vs HA | -2.9914 |
| **CRPSS vs Seasonal Naive** | **-2.5171** |
| Threshold (≥0.10 vs SN) | ✗ FAIL |

## Coverage Results by Calibration Method

| Method | Marginal Coverage | Target | Mean Width | Disparity |
|--------|:-----------------:|:------:|:----------:|:---------:|
| split_cp | ✓ 0.9003 | 0.90 | 50.33 | 0.1622 |
| weighted_cp | ⚠ 0.8211 | 0.90 | 37.33 | 0.2526 |
| mondrian | ✓ 0.9040 | 0.90 | 50.74 | 0.0144 |
| equalized_coverage | ✓ 0.9003 | 0.90 | 50.33 | 0.1622 |
| ecrc | ⚠ 0.9476 | 0.90 | 68.52 | 0.0142 |
| adaptive_ecrc | ⚠ 0.9476 | 0.90 | 68.52 | 0.0142 |

### Per-Category Coverage (ecrc)

| Category | Coverage | Width | N |
|----------|:--------:|:-----:|--:|
| violent | 0.9784 | 69.18 | 4081 |
| property | 0.8645 | 69.18 | 4081 |
| drug | 1.0000 | 67.18 | 4081 |

### Per-Demographic-Quartile Coverage (ecrc)

| Group | Coverage | Width | N |
|-------|:--------:|:-----:|--:|
| group_0 | 0.9497 | 79.33 | 3021 |
| group_1 | 0.9563 | 53.33 | 3021 |
| group_2 | 0.9427 | 38.33 | 3021 |
| group_3 | 0.9421 | 101.33 | 3180 |

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
the best calibrator (ecrc) achieves 94.8% marginal 
coverage with mean prediction interval width 68.52 
counts and a maximum cross-group coverage disparity of 
0.0142.

## Ablation TODO Registry (Table 2)

- [ ] ACI gamma sensitivity: γ ∈ {0.01, 0.05, 0.1, adaptive-PI}
- [ ] Calibration set size: 13 vs 26 vs 52 weeks
- [ ] Group granularity: geographic (S groups) vs demographic (4 quartiles)
- [ ] CQR vs ABS vs RAPS non-conformity score functions
- [ ] ECRC delta sensitivity: δ ∈ {0.01, 0.05, 0.1}
- [ ] Cross-city transfer: calibrate on Chicago, test on NYC