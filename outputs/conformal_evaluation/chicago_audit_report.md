# CIVIC-SAFE Conformal Prediction Audit Report

**Dataset:** chicago  
**Timestamp:** 2026-08-20T09:48:56.522054  
**Alpha (miscoverage):** 0.1  
**Checkpoint:** `run_chicago_anchor_1786718493`  
**Panel hash:** `4bb2e1e3322b`  

## Point Forecast Metrics (Test Set — 2023)

| Metric | Value |
|--------|-------|
| CRPS | 2.8267 |
| MAE | 3.9017 |
| RMSE | 7.0983 |
| Brier (zero-inflation) | 0.0592 |

## CRPS Skill Score

| Component | Value |
|-----------|-------|
| Baseline CRPS (Historical Average, rolling) | 2.9322 |
| Baseline CRPS (Historical Average, frozen) | 3.8790 |
| Baseline CRPS (Seasonal Naive) | 4.400869369506836 |
| Model CRPS | 2.8267 |
| CRPSS vs HA (rolling) | +0.0360 |
| CRPSS vs Seasonal Naive | +0.3577 |
| **CRPSS (min over naive family)** | **+0.0360** |
| Binding baseline | ha_rolling |
| Forecasting claim gate | **PASS** |

## Forecasting Claim Gate

| Evidence | Value |
|----------|-------|
| Rule | CRPSS vs rolling HA > 0 and (DM p < 0.05 or block-bootstrap p < 0.05) |
| CRPSS vs rolling HA | 0.035966 |
| DM statistic | -2.122039 |
| DM p-value | 0.033834 |
| DM 95% CI | [-0.202865, -0.008055] |
| Block-bootstrap p-value | 0.005300 |
| Block-bootstrap 95% CI | [-0.173608, -0.026323] |
| **Decision** | **PASS** |

## Calibrator Selection

**Rule:** minimum width subject to coverage, demographic disparity, abstention, and status constraints  
**Selected method:** `equalized_coverage`  
**Fallback used:** False  

## Coverage Results by Calibration Method

| Method | Marginal Coverage | Target | Mean Width | Demographic Disparity | Abstention | Policy Status |
|--------|:-----------------:|:------:|:----------:|:---------------------:|:----------:|---------------|
| split_cp | 0.9405 | 0.90 | 16.25 | 0.0182 | 0.00% | eligible |
| randomized_split_cp | 0.9347 | 0.90 | 16.53 | 0.0115 | 0.00% | eligible |
| weighted_cp | 0.9075 | 0.90 | 14.58 | 0.0238 | 0.00% | eligible |
| mondrian | 0.9169 | 0.90 | 15.02 | 0.0319 | 0.00% | demographic disparity 0.031860 exceeds 0.030000 |
| mondrian_category | 0.9313 | 0.90 | 17.19 | 0.0250 | 0.00% | eligible |
| mondrian_demo_x_category | 0.9305 | 0.90 | 17.35 | 0.0119 | 0.00% | eligible |
| equalized_coverage | 0.9075 | 0.90 | 14.58 | 0.0238 | 0.00% | selected; eligible |
| variance_scaled_split_cp | 0.9079 | 0.90 | 14.65 | 0.0242 | 0.00% | eligible |
| ecrc | 0.9235 | 0.90 | 15.89 | 0.0156 | 0.00% | eligible |
| adaptive_ecrc_rolling | 0.8930 | 0.90 | 13.88 | 0.0013 | 0.00% | coverage 0.893000 is below 0.895000 |

### Per-Category Coverage (equalized_coverage)

| Category | Coverage | Width | N |
|----------|:--------:|:-----:|--:|
| violent | 0.8883 | 15.72 | 4081 |
| property | 0.8829 | 25.30 | 4081 |
| drug | 0.9515 | 2.73 | 4081 |

### Per-Demographic-Quartile Coverage (equalized_coverage)

| Group | Coverage | Width | N |
|-------|:--------:|:-----:|--:|
| group_0 | 0.8994 | 17.65 | 3180 |
| group_1 | 0.9037 | 13.36 | 3021 |
| group_2 | 0.9043 | 12.71 | 3021 |
| group_3 | 0.9232 | 14.43 | 3021 |

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
for each income quartile $g$. On the 2023 test set (53 windows), The selected calibrator (equalized_coverage) achieves 90.8% marginal coverage with mean prediction interval width 14.58 counts and a maximum cross-group coverage disparity of 0.0238.

## Ablation TODO Registry (Table 2)

- [ ] ACI gamma sensitivity: γ ∈ {0.01, 0.05, 0.1, adaptive-PI}
- [ ] Calibration set size: 13 vs 26 vs 52 weeks
- [ ] Group granularity: geographic (S groups) vs demographic (4 quartiles)
- [ ] CQR vs ABS vs RAPS non-conformity score functions
- [ ] ECRC delta sensitivity: δ ∈ {0.01, 0.05, 0.1}
- [ ] Cross-city transfer: calibrate on Chicago, test on NYC