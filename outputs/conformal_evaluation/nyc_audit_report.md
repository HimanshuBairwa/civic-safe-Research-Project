# CIVIC-SAFE Conformal Prediction Audit Report

**Dataset:** nyc  
**Timestamp:** 2026-08-02T04:59:31.441258  
**Alpha (miscoverage):** 0.1  
**Checkpoint:** `best.pt')]`  
**Panel hash:** `3ff6707b716d`  

## Point Forecast Metrics (Test Set — 2023)

| Metric | Value |
|--------|-------|
| CRPS | 3.5679 |
| MAE | 4.9627 |
| RMSE | 8.5543 |
| Brier (zero-inflation) | 0.0535 |

## CRPS Skill Score

| Component | Value |
|-----------|-------|
| Baseline CRPS (Historical Average) | 4.5942 |
| Baseline CRPS (Seasonal Naive) | 4.7308526039123535 |
| Model CRPS | 3.5679 |
| CRPSS vs HA | 0.2234 |
| **CRPSS vs Seasonal Naive** | **0.2458** |
| Threshold (≥0.10 vs SN) | ✓ PASS |

## Coverage Results by Calibration Method

| Method | Marginal Coverage | Target | Mean Width | Disparity |
|--------|:-----------------:|:------:|:----------:|:---------:|
| split_cp | ✓ 0.9047 | 0.90 | 17.38 | 0.0361 |
| weighted_cp | ✓ 0.9047 | 0.90 | 17.38 | 0.0361 |
| mondrian | ✓ 0.9047 | 0.90 | 17.38 | 0.0361 |
| equalized_coverage | ✓ 0.9047 | 0.90 | 17.38 | 0.0361 |
| ecrc | ⚠ 0.9415 | 0.90 | 19.89 | 0.0185 |
| adaptive_ecrc | ⚠ 0.9415 | 0.90 | 19.89 | 0.0185 |
| adaptive_ecrc_rolling | ⚠ 0.9132 | 0.90 | 17.97 | 0.0245 |

### Per-Category Coverage (split_cp)

| Category | Coverage | Width | N |
|----------|:--------:|:-----:|--:|
| violent | 0.8950 | 12.07 | 4134 |
| property | 0.8628 | 32.66 | 4134 |
| drug | 0.9562 | 7.42 | 4134 |

### Per-Demographic-Quartile Coverage (split_cp)

| Group | Coverage | Width | N |
|-------|:--------:|:-----:|--:|
| group_0 | 0.8934 | 18.21 | 3180 |
| group_1 | 0.8884 | 19.88 | 3021 |
| group_2 | 0.9126 | 15.90 | 3180 |
| group_3 | 0.9245 | 15.57 | 3021 |

## Methods Paragraph (Paper-Ready)

We apply Conformalized Quantile Regression (CQR; Romano et al., 2019) 
to the ZINB predictive distribution, computing non-conformity scores 
$s_i = \max(\hat{q}_{\alpha/2}(X_i) - Y_i, Y_i - \hat{q}_{1-\alpha/2}(X_i))$ 
on a held-out calibration set (2022 H2, 26 windows, 
6084 observations). The calibration threshold 
$\hat{q}$ is chosen as the $\lceil (1-\alpha)(1+1/n) \rceil$-th empirical 
quantile of the scores, guaranteeing finite-sample marginal coverage 
$P(Y \in [L, U]) \geq 1-\alpha$ under exchangeability. To correct for 
temporal non-exchangeability, we additionally implement Adaptive Conformal 
Inference (ACI; Gibbs & Candès, 2021) with per-demographic-quartile tracking, 
achieving asymptotic conditional coverage $P(Y \in C(X) | G=g) \to 1-\alpha$ 
for each income quartile $g$. On the 2023 test set (53 windows), 
the best calibrator (split_cp) achieves 90.5% marginal 
coverage with mean prediction interval width 17.38 
counts and a maximum cross-group coverage disparity of 
0.0361.

## Ablation TODO Registry (Table 2)

- [ ] ACI gamma sensitivity: γ ∈ {0.01, 0.05, 0.1, adaptive-PI}
- [ ] Calibration set size: 13 vs 26 vs 52 weeks
- [ ] Group granularity: geographic (S groups) vs demographic (4 quartiles)
- [ ] CQR vs ABS vs RAPS non-conformity score functions
- [ ] ECRC delta sensitivity: δ ∈ {0.01, 0.05, 0.1}
- [ ] Cross-city transfer: calibrate on Chicago, test on NYC