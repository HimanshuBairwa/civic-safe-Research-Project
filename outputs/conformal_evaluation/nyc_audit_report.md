# CIVIC-SAFE Conformal Prediction Audit Report

**Dataset:** nyc  
**Timestamp:** 2026-08-22T06:24:55.421703  
**Alpha (miscoverage):** 0.1  
**Checkpoint:** `run_nyc_anchor_1786812069`  
**Panel hash:** `3ff6707b716d`  

## Point Forecast Metrics (Test Set — 2023)

| Metric | Value |
|--------|-------|
| CRPS | 3.1401 |
| MAE | 4.3675 |
| RMSE | 7.6126 |
| Brier (zero-inflation) | 0.0493 |

## CRPS Skill Score

| Component | Value |
|-----------|-------|
| Baseline CRPS (Historical Average, rolling) | 3.3034 |
| Baseline CRPS (Historical Average, frozen) | 4.5942 |
| Baseline CRPS (Seasonal Naive) | 4.7308526039123535 |
| Model CRPS | 3.1401 |
| CRPSS vs HA (rolling) | +0.0494 |
| CRPSS vs Seasonal Naive | +0.3362 |
| **CRPSS (min over naive family)** | **+0.0494** |
| Binding baseline | ha_rolling |
| Forecasting claim gate | **PASS** |

## Forecasting Claim Gate

| Evidence | Value |
|----------|-------|
| Rule | CRPSS vs rolling HA > 0 and (DM p < 0.05 or block-bootstrap p < 0.05) |
| CRPSS vs rolling HA | 0.049422 |
| DM statistic | -2.908824 |
| DM p-value | 0.003628 |
| DM 95% CI | [-0.273262, -0.053255] |
| Block-bootstrap p-value | 0.000300 |
| Block-bootstrap 95% CI | [-0.228882, -0.056582] |
| **Decision** | **PASS** |

## Calibrator Selection

**Rule:** minimum width subject to coverage, demographic disparity, abstention, and status constraints  
**Selected method:** `variance_scaled_split_cp`  
**Fallback used:** False  

## Coverage Results by Calibration Method

| Method | Marginal Coverage | Target | Mean Width | Demographic Disparity | Abstention | Policy Status |
|--------|:-----------------:|:------:|:----------:|:---------------------:|:----------:|---------------|
| split_cp | 0.9326 | 0.90 | 18.57 | 0.0201 | 0.00% | eligible |
| randomized_split_cp | 0.9328 | 0.90 | 18.60 | 0.0201 | 0.00% | eligible |
| weighted_cp | 0.9326 | 0.90 | 18.57 | 0.0201 | 0.00% | eligible |
| mondrian | 0.9326 | 0.90 | 18.57 | 0.0201 | 0.00% | eligible |
| mondrian_category | 0.9241 | 0.90 | 17.91 | 0.0132 | 0.00% | eligible |
| mondrian_demo_x_category | 0.9319 | 0.90 | 18.25 | 0.0154 | 0.00% | eligible |
| equalized_coverage | 0.9326 | 0.90 | 18.57 | 0.0201 | 0.00% | eligible |
| variance_scaled_split_cp | 0.9002 | 0.90 | 16.45 | 0.0286 | 0.00% | selected; eligible |
| ecrc | 0.9186 | 0.90 | 17.41 | 0.0138 | 0.00% | eligible |
| adaptive_ecrc_rolling | 0.8918 | 0.90 | 16.31 | 0.0046 | 0.00% | coverage 0.891792 is below 0.895000 |

### Per-Category Coverage (variance_scaled_split_cp)

| Category | Coverage | Width | N |
|----------|:--------:|:-----:|--:|
| violent | 0.8677 | 10.42 | 4134 |
| property | 0.8950 | 33.25 | 4134 |
| drug | 0.9378 | 5.67 | 4134 |

### Per-Demographic-Quartile Coverage (variance_scaled_split_cp)

| Group | Coverage | Width | N |
|-------|:--------:|:-----:|--:|
| group_0 | 0.8858 | 16.94 | 3180 |
| group_1 | 0.9020 | 19.60 | 3021 |
| group_2 | 0.9145 | 14.94 | 3180 |
| group_3 | 0.8984 | 14.35 | 3021 |

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
for each income quartile $g$. On the 2023 test set (53 windows), The selected calibrator (variance_scaled_split_cp) achieves 90.0% marginal coverage with mean prediction interval width 16.45 counts and a maximum cross-group coverage disparity of 0.0286.

## Ablation TODO Registry (Table 2)

- [ ] ACI gamma sensitivity: γ ∈ {0.01, 0.05, 0.1, adaptive-PI}
- [ ] Calibration set size: 13 vs 26 vs 52 weeks
- [ ] Group granularity: geographic (S groups) vs demographic (4 quartiles)
- [ ] CQR vs ABS vs RAPS non-conformity score functions
- [ ] ECRC delta sensitivity: δ ∈ {0.01, 0.05, 0.1}
- [ ] Cross-city transfer: calibrate on Chicago, test on NYC