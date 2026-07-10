# CIVIC-SAFE: Conformal Inference for Vigilant, Interpretable Crime-prediction with Spatial Attention and Fairness Evaluation

> **Venue target:** AAAI / NeurIPS / KDD / ICML (10-page main + appendix)
> **Status:** Draft scaffold — fill sections marked `[TODO]`

---

## Abstract

> **Template — fill in concrete numbers after 5-seed evaluation completes.**

Urban crime forecasting models deployed in public-safety contexts demand three properties that current systems lack: *calibrated uncertainty*, *audited equity*, and *actionable abstention*.  We present **CIVIC-SAFE**, a probabilistic spatiotemporal framework that unifies graph neural networks, conformal prediction, and fairness auditing into a single end-to-end pipeline.  Our architecture pairs a **GATv2 spatial encoder** operating over dual adjacency graphs with a **causal Transformer temporal encoder**, feeding a **Zero-Inflated Negative Binomial (ZINB)** distributional head that explicitly models the zero-inflation and overdispersion inherent in administrative crime data.  We calibrate the resulting predictive distributions using **five conformal prediction strategies** — including a novel Equalized Conditional Risk Control (ECRC) method with Hoeffding-based per-group coverage guarantees — and subject every forecast to a **7-component equity audit** with Benjamini–Hochberg FDR correction.  On two large-scale urban datasets — **Chicago** (77 community areas, 1.33M incidents) and **New York City** (78 precincts, 1.51M incidents) across 2018–2023 — CIVIC-SAFE achieves a CRPS of `[X.XX ± X.XX]`, MAE of `[X.XX ± X.XX]`, and RMSE of `[X.XX ± X.XX]` while maintaining `[XX.X]`% marginal conformal coverage at the α = 0.10 level across all demographic strata.  A downstream **advisory safe-routing engine** built on exact Dijkstra shortest paths over conformal-interval edge costs translates forecasts into civilian-facing route advisories and implements an *abstention protocol* that refuses to recommend routes when prediction uncertainty exceeds calibrated thresholds.  To our knowledge, CIVIC-SAFE is the first crime-forecasting system to simultaneously provide distribution-free coverage guarantees, audited cross-group equity, and principled abstention — all enforced as hard constraints rather than post-hoc checks.  Code and data are available at `https://github.com/HimanshuBairwa/civic-safe-Research-Project`.

---

## 1  Introduction

### 1.1  Motivation

- Crime forecasting has moved from hot-spot mapping to deep spatiotemporal models, but three critical gaps persist:
  1. **Overconfident point predictions.** Models output $\hat{y}$ with no uncertainty quantification; downstream users (e.g., urban planners, community organisations) cannot assess reliability.
  2. **Unaudited equity.** Predictions may systematically over-predict in historically over-policed minority neighbourhoods due to reporting bias in administrative data.
  3. **No principled abstention.** Routing and resource-allocation systems blindly trust model outputs even when the model is uncertain, creating false assurances of safety.

- CIVIC-SAFE addresses all three gaps simultaneously, enforcing them as **hard architectural constraints** rather than optional post-hoc analyses.

### 1.2  Contributions

We make the following four contributions:

> **C1 — Distributional Spatiotemporal Architecture.**  
> We propose a GATv2 → Causal Transformer → Multi-Fidelity Feature Mixer → ZINB head architecture (688K parameters) that outputs full count distributions, explicitly modelling both the zero-inflation from structural/reporting zeros and the overdispersion of urban crime counts.

> **C2 — Conformal Calibration Suite with Equity Guarantees.**  
> We implement and compare five conformal prediction calibrators — Split CP, Weighted CP, Mondrian CP, Equalized Coverage, and ECRC — the last providing PAC-style per-group coverage via Hoeffding bounds: $\Pr\bigl(\text{coverage}(g) \geq 1 - \alpha - \varepsilon\bigr) \geq 1 - \delta$ for every group $g$.

> **C3 — 7-Component Equity Audit with Multiple-Testing Correction.**  
> We design a comprehensive equity audit harness evaluating coverage parity, interval-width disparity, point-forecast bias, calibration deviation, Winkler score gap, abstention disparity, and reporting-bias sensitivity — all corrected for multiple comparisons via Benjamini–Hochberg FDR control.

> **C4 — Advisory Safe-Routing with Audited Abstention.**  
> We couple probabilistic forecasts with a Pareto-optimal advisory routing engine using exact Dijkstra shortest paths. The engine implements an abstention protocol that refuses to recommend routes when conformal interval widths exceed calibrated safety thresholds, preventing false assurances.

### 1.3  Paper Organisation

Section 2 surveys related work. Section 3 formalises the problem. Section 4 details the method. Section 5 describes the experimental setup. Section 6 presents results. Section 7 discusses ethical considerations and limitations. Section 8 concludes.

---

## 2  Related Work

### 2.1  Spatiotemporal Crime Forecasting

- Classical: kernel density estimation, self-exciting point processes (Mohler et al., 2011)
- Deep learning: ST-ResNet (Zhang et al., 2017), DeepCrime (Huang et al., 2018)
- Graph-based: STGCN (Yu et al., 2018), DCRNN (Li et al., 2018), ST-SHN (Xia et al., 2022)
- **Gap:** Most methods output point predictions; none provide distribution-free coverage guarantees.

### 2.2  Graph Neural Networks for Urban Analytics

- GCN (Kipf & Welling, 2017), GAT (Veličković et al., 2018), GATv2 (Brody et al., ICLR 2022)
- Dual/multi-graph approaches for heterogeneous spatial relationships
- **Our contribution:** Dual-adjacency GATv2 (queen contiguity + K-NN) with LayerNorm to prevent oversmoothing on small urban graphs (N ≤ 78).

### 2.3  Probabilistic Count Forecasting

- Poisson regression, Negative Binomial, Zero-Inflated models (Lambert, 1992)
- DeepAR (Salinas et al., 2020), probabilistic transformer forecasters
- ZINB in single-cell genomics (Eraslan et al., 2019) and epidemiology
- **Our contribution:** First ZINB distributional head integrated with a GNN+Transformer backbone for crime forecasting, with numerically stable logsumexp NLL.

### 2.4  Conformal Prediction

- Split CP (Vovk et al., 2005; Papadopoulos et al., 2002)
- Conformalized Quantile Regression (Romano et al., 2019)
- Weighted CP under covariate shift (Tibshirani et al., 2019)
- Mondrian CP for group-conditional coverage (Vovk, 2005)
- Risk-controlling prediction sets (Bates et al., 2021; Feldman et al., 2021)
- **Our contribution:** ECRC calibrator providing Hoeffding-based PAC per-group coverage for ZINB predictions; systematic comparison of 5 methods on urban count data.

### 2.5  Fairness and Equity in Predictive Systems

- AIF360 (Bellamy et al., 2019), Fairlearn (Bird et al., 2020)
- Fairness in predictive policing (Lum & Isaac, 2016; Ensign et al., 2018; feedback loops)
- Reporting bias and data-generating process critique (Knox et al., 2020)
- **Our contribution:** 7-component audit going beyond standard disparity metrics to include calibration deviation, Winkler score gap, abstention disparity, and reporting-bias sensitivity analysis with INAR binomial thinning.

---

## 3  Problem Formulation

### 3.1  Notation

| Symbol | Definition |
|--------|-----------|
| $\mathcal{S} = \{s_1, \ldots, s_S\}$ | Set of spatial units (community areas / precincts) |
| $\mathcal{T} = \{1, \ldots, T\}$ | Discrete time steps (weeks) |
| $\mathcal{C} = \{\text{violent}, \text{property}, \text{drug}\}$ | Crime categories |
| $Y_{s,t,c} \in \mathbb{Z}_{\geq 0}$ | Observed crime count for unit $s$, week $t$, category $c$ |
| $\mathbf{X}_{s,t} \in \mathbb{R}^F$ | Feature vector (historical counts + 7 ACS covariates) |
| $\mathcal{G} = (\mathcal{S}, \mathcal{E}_\text{queen} \cup \mathcal{E}_\text{knn})$ | Dual spatial graph |
| $(\pi, \mu, r)_{s,c}$ | Predicted ZINB parameters |

### 3.2  Task Definition

**Given:** A spatiotemporal panel $\{\mathbf{X}_{s,t}\}_{s \in \mathcal{S}, t \in [t-W+1, t]}$ over a lookback window of $W$ weeks, and the dual spatial graph $\mathcal{G}$.

**Predict:** For each spatial unit $s$ and crime category $c$, a full predictive distribution $\hat{F}_{s,c}$ over crime counts $Y_{s,t+1,c}$, parameterised as $\text{ZINB}(\pi_{s,c}, \mu_{s,c}, r_{s,c})$.

**Calibrate:** Construct prediction intervals $[L_{s,c}, U_{s,c}]$ such that:
$$\Pr(Y_{s,t+1,c} \in [L_{s,c}, U_{s,c}]) \geq 1 - \alpha \quad \forall s, c$$

**Audit:** Verify that calibration, accuracy, and interval width do not exhibit statistically significant disparity across demographic strata, after Benjamini–Hochberg correction.

### 3.3  Data Constraints

- **No person-level features.** The model operates on spatially aggregated counts only.
- **No protected attributes as inputs.** Demographic variables (from ACS) are used exclusively for post-hoc equity auditing and are never provided to the model.
- **Chronological splits.** Train / validation / calibration / test sets are strictly temporally ordered to prevent data leakage.

---

## 4  Method

### 4.1  Architecture Overview

```
Input: X ∈ ℝ^{S×T×F}          Dual graph: G = (S, E_queen ∪ E_knn)
    │
    ▼
┌──────────────────┐
│  Input Projection │   Linear(F → d)
└──────────────────┘
    │
    ▼  (per timestep t)
┌──────────────────┐
│  GATv2 Spatial   │   2 layers, 4 heads, dual adjacency
│  Encoder         │   LayerNorm + ELU + Dropout
└──────────────────┘
    │
    ▼  (stack over T)
┌──────────────────┐
│  Causal           │   2 layers, 4 heads, d_ff = 512
│  Transformer      │   Pre-LN, sinusoidal PE, causal mask
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Multi-Factor     │   3 heads, τ = 1.0
│  Feature Mixer    │   JSD diversity regularisation
└──────────────────┘
    │  (last timestep)
    ▼
┌──────────────────┐
│  ZINB Head        │   3 independent MLPs → (π, μ, r) per category
└──────────────────┘
    │
    ▼
Output: (π, μ, r) ∈ ℝ^{S×C×3}
```

*Total parameters: 688,649. See docs/METHODOLOGY.md for full equations.*

### 4.2  Spatial Encoder: Dual-Adjacency GATv2

- GATv2Conv (Brody et al., ICLR 2022) with dynamic attention
- Dual adjacency: Queen contiguity ($\mathcal{E}_\text{queen}$) + 8-NN ($\mathcal{E}_\text{knn}$)
- Outputs summed from both graphs; LayerNorm prevents oversmoothing on N ≤ 78

*Equations: see §1–§2 of docs/METHODOLOGY.md*

### 4.3  Temporal Encoder: Causal Transformer

- Pre-LN TransformerEncoder with `is_causal=True`
- Sinusoidal positional encoding (zero learnable parameters, generalises to unseen lengths)
- Causal mask $M_{ij} = -\infty \cdot \mathbb{1}[j > i]$ ensures zero future leakage

*Equations: see §3 of docs/METHODOLOGY.md*

### 4.4  Multi-Factor Feature Mixer (MFFM)

- 3-head gated cross-attention decomposition
- Temperature-controlled softmax attention
- Jensen-Shannon Divergence diversity penalty prevents collapse to proxy variables

*Equations: see §4 of docs/METHODOLOGY.md*

### 4.5  ZINB Distributional Head

- Three independent 2-layer MLPs: $\pi$ (sigmoid), $\mu$ (softplus), $r$ (softplus + floor)
- Models structural zeros (reporting bias) and overdispersion simultaneously
- Numerically stable NLL via logsumexp (zero case) and lgamma (non-zero case)

*Equations: see §5 of docs/METHODOLOGY.md*

### 4.6  Conformal Calibration

- Five methods: Split CP, Weighted CP, Mondrian CP, Equalized Coverage, ECRC
- CQR non-conformity score: $s_i = \max(q_{\alpha/2}^{(i)} - y_i, \; y_i - q_{1-\alpha/2}^{(i)})$
- ECRC provides PAC per-group coverage via Hoeffding bounds

*Algorithms: see §7 of docs/METHODOLOGY.md*

### 4.7  Equity Audit Harness

Seven audit components:

| # | Component | What It Tests |
|---|-----------|---------------|
| 1 | Coverage Parity | Equal conformal coverage across demographic strata |
| 2 | Width Disparity | Equal interval widths (sharpness parity) |
| 3 | Point Bias | Unbiased point forecasts across groups |
| 4 | Calibration Deviation | Equal PIT uniformity across groups |
| 5 | Winkler Score Gap | Combined calibration + sharpness parity |
| 6 | Abstention Disparity | Equal abstention rates across groups |
| 7 | Reporting Bias Sensitivity | Robustness to simulated under-reporting (INAR thinning) |

All tests use bootstrap CIs and permutation tests, corrected via Benjamini–Hochberg FDR (1995).

### 4.8  Advisory Safe Routing

- Pareto-optimal routing: $\min_{P} \bigl(\sum_{e \in P} d_e, \; \sum_{e \in P} \rho_e\bigr)$
- Risk: $\rho_e = (1 - \pi_e)\mu_e + \lambda_\text{unc}(1-\pi_e)\frac{\mu_e(\mu_e + r_e)}{r_e}$
- Exact Dijkstra shortest paths (Duan et al. 2025 noted as future large-scale direction only)
- **Abstention protocol:** refuses routes when peak conformal interval width exceeds threshold

### 4.9  Training Procedure

- AdamW optimiser (lr = 1e-3, weight decay = 1e-2)
- Cosine warmup LR schedule (10-epoch warmup, min LR = 1e-6)
- 200 epochs with early stopping (patience = 10, monitor = val CRPS)
- BFloat16 mixed precision with gradient checkpointing
- Gradient clipping (max norm = 1.0)
- Total loss: $\mathcal{L} = \mathcal{L}_\text{ZINB} + \lambda_\text{div} \cdot \mathcal{L}_\text{diversity}$

---

## 5  Experimental Setup

### 5.1  Datasets

| Property | Chicago | NYC |
|----------|---------|-----|
| Source | Chicago Data Portal | NYC OpenData |
| Spatial units | 77 community areas | 78 precincts |
| Total incidents | 1.33M | 1.51M |
| Categories | violent, property, drug | violent, property, drug |
| Temporal range | 2018–2023 | 2018–2023 |
| Temporal granularity | Weekly | Weekly |
| Demographics | 7 ACS covariates via areal interpolation | 7 ACS covariates via areal interpolation |

**Chronological splits:** Train (2018–2021) → Val (2022 H1) → Cal (2022 H2) → Test (2023).

### 5.2  Demographic Covariates (ACS)

Seven US Census American Community Survey (ACS) variables obtained via Geospatial Areal Interpolation from census tracts to community areas / precincts:

1. Median household income
2. Population density
3. Unemployment rate
4. Percentage below poverty line
5. Percentage with bachelor's degree or higher
6. Racial diversity index (entropy)
7. Percentage renter-occupied housing

*These variables are used only for equity stratification — they are never seen by the model during training.*

### 5.3  Baselines

| Method | Type | Reference |
|--------|------|-----------|
| Historical Average | Naïve | — |
| ARIMA | Statistical | Box & Jenkins (1970) |
| Prophet | Statistical | Taylor & Letham (2018) |
| DeepAR | Neural (RNN) | Salinas et al. (2020) |
| STGCN | Neural (GNN) | Yu et al. (2018) |
| DCRNN | Neural (GNN) | Li et al. (2018) |
| ST-SHN | Neural (GNN) | Xia et al. (2022) |
| CIVIC-SAFE (ours) | Neural (GNN+Transformer+ZINB) | This paper |

### 5.4  Evaluation Metrics

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| CRPS | $\sum_k [F(k) - \mathbb{1}(y \leq k)]^2$ | Distributional calibration + sharpness |
| MAE | $\frac{1}{n}\sum|y_i - \hat{y}_i|$ | Point accuracy |
| RMSE | $\sqrt{\frac{1}{n}\sum(y_i - \hat{y}_i)^2}$ | Point accuracy (penalises large errors) |
| Coverage | $\frac{1}{n}\sum\mathbb{1}(y_i \in [L_i, U_i])$ | Conformal validity |
| AIW | $\frac{1}{n}\sum(U_i - L_i)$ | Interval sharpness |
| Brier (zero) | $\frac{1}{n}\sum(\pi_i - \mathbb{1}(y_i = 0))^2$ | Zero-inflation calibration |

### 5.5  Implementation Details

- **Framework:** PyTorch 2.2+ with PyTorch Geometric 2.5+
- **Hardware:** NVIDIA A100 40GB (single GPU)
- **Parameters:** 688,649
- **Training time:** ~3 hours per seed on A100
- **Seeds:** 5 seeds [42, 137, 256, 512, 1024]; results reported as mean ± std
- **Experiment tracking:** Weights & Biases

---

## 6  Results

### 6.1  Main Results

> `[TODO: Fill after 5-seed evaluation completes]`

**Table 1: Forecasting performance on Chicago test set (2023). Mean ± std over 5 seeds.**

| Method | CRPS ↓ | MAE ↓ | RMSE ↓ | Coverage (90%) ↑ | AIW ↓ |
|--------|--------|-------|--------|-------------------|-------|
| Historical Avg | — | — | — | — | — |
| ARIMA | — | — | — | — | — |
| DeepAR | — | — | — | — | — |
| STGCN | — | — | — | — | — |
| DCRNN | — | — | — | — | — |
| **CIVIC-SAFE** | **—** | **—** | **—** | **—** | **—** |

**Table 2: Forecasting performance on NYC test set (2023). Mean ± std over 5 seeds.**

| Method | CRPS ↓ | MAE ↓ | RMSE ↓ | Coverage (90%) ↑ | AIW ↓ |
|--------|--------|-------|--------|-------------------|-------|
| Historical Avg | — | — | — | — | — |
| ARIMA | — | — | — | — | — |
| DeepAR | — | — | — | — | — |
| STGCN | — | — | — | — | — |
| DCRNN | — | — | — | — | — |
| **CIVIC-SAFE** | **—** | **—** | **—** | **—** | **—** |

*Preliminary NYC result (seed 42): CRPS = 16.90, MAE = 22.17, RMSE = 36.00.*

### 6.2  Conformal Calibration Comparison

**Table 3: Conformal calibration methods on Chicago test set (α = 0.10).**

| Calibrator | Coverage ↑ | AIW ↓ | Max Group Gap ↓ |
|-----------|------------|-------|-----------------|
| Split CP | — | — | — |
| Weighted CP | — | — | — |
| Mondrian CP | — | — | — |
| Equalized Coverage | — | — | — |
| **ECRC (ours)** | **—** | **—** | **—** |

### 6.3  Equity Audit Results

**Table 4: Equity audit summary (income-quartile stratification).**

| Audit Component | Max Disparity | p-value (BH-corrected) | Pass? |
|----------------|---------------|------------------------|-------|
| Coverage Parity | — | — | — |
| Width Disparity | — | — | — |
| Point Bias | — | — | — |
| Calibration Deviation | — | — | — |
| Winkler Gap | — | — | — |
| Abstention Disparity | — | — | — |
| Reporting Bias Sensitivity | — | — | — |

### 6.4  Ablation Study

**Table 5: Ablation study on Chicago test set.**

| Variant | CRPS ↓ | MAE ↓ | RMSE ↓ |
|---------|--------|-------|--------|
| Full model | — | — | — |
| − GATv2 (replace with MLP) | — | — | — |
| − Transformer (replace with MLP) | — | — | — |
| − Zero-inflation (NB only) | — | — | — |
| − MFFM (direct projection) | — | — | — |
| − Dual adjacency (queen only) | — | — | — |

### 6.5  Per-Category Analysis

**Table 6: CRPS by crime category.**

| Category | Chicago CRPS | NYC CRPS | Zero-rate |
|----------|-------------|----------|-----------|
| Violent | — | — | — |
| Property | — | — | — |
| Drug | — | — | — |

### 6.6  Figures

> `[TODO: Generate after experiments]`

- **Figure 1:** Architecture diagram (vector graphic version of §4.1 ASCII art)
- **Figure 2:** Training curves (loss, CRPS, MAE) across 5 seeds with confidence bands
- **Figure 3:** Conformal coverage as a function of α for each calibrator
- **Figure 4:** Per-group coverage heatmap (spatial units × demographic quartiles)
- **Figure 5:** Example prediction intervals for 3 representative community areas (high/mid/low crime)
- **Figure 6:** Choropleth maps of prediction quality (CRPS) across Chicago / NYC
- **Figure 7:** Routing example with abstention demonstration
- **Figure 8:** Ablation spider chart

---

## 7  Ethical Considerations

### 7.1  Intended Use and Misuse Potential

- CIVIC-SAFE is designed as a **research benchmark**, not a deployment-ready system.
- Intended users: urban planners, public health researchers, community organisations.
- **Explicitly not intended** for: police patrol allocation, surveillance targeting, or any use that could create feedback loops amplifying historical over-policing.

### 7.2  Hard Ethical Constraints

Six ethics commitments are enforced architecturally (see README):
1. Civilian-facing only
2. No person-level prediction
3. No protected attributes as model inputs
4. Explicit acknowledgement of reporting bias
5. Advisory-only outputs
6. Abstention under diagnostic failure

### 7.3  Reporting Bias

- Administrative crime data reflects *reported* crime, not *committed* crime.
- We include a mandatory `ReportingBiasSensitivityAudit` that applies INAR binomial thinning to simulate under-reporting at varying rates and evaluates forecast robustness.

### 7.4  Limitations

1. Relies on municipal open data with inherent selection biases.
2. Areal interpolation of census demographics introduces measurement error at spatial unit boundaries.
3. Weekly temporal granularity may miss sub-weekly patterns.
4. The equity audit tests for disparities but cannot eliminate all forms of bias.
5. The routing engine operates on simplified graph representations, not actual road networks.

---

## 8  Conclusion

We presented CIVIC-SAFE, a probabilistic crime-forecasting framework that integrates GATv2 spatial encoding, causal Transformer temporal modelling, ZINB distributional outputs, conformal calibration with equity guarantees, a 7-component fairness audit, and advisory safe-routing with principled abstention.  Our experiments on Chicago and NYC demonstrate that `[TODO: summarise key findings]`.  We hope CIVIC-SAFE establishes a new standard for responsible, uncertainty-aware crime forecasting and inspires further research at the intersection of probabilistic ML, distribution-free inference, and algorithmic fairness.

---

## Appendix

### A  Full Hyperparameter Table

*See `configs/` directory and Table A1 below.*

| Hyperparameter | Value | Search Range |
|---------------|-------|-------------|
| GATv2 layers | 2 | {1, 2, 3} |
| GATv2 heads | 4 | {2, 4, 8} |
| Hidden dim | 128 | {64, 128, 256} |
| Transformer layers | 2 | {1, 2, 4} |
| Transformer heads | 4 | {2, 4, 8} |
| FFN dim | 512 | {256, 512, 1024} |
| MFFM heads | 3 | {2, 3, 4} |
| MFFM temperature | 1.0 | {0.5, 1.0, 2.0} |
| Batch size | 16 | {8, 16, 32} |
| Learning rate | 1e-3 | {1e-4, 5e-4, 1e-3} |
| Weight decay | 1e-2 | {1e-3, 1e-2, 5e-2} |
| Warmup epochs | 10 | {5, 10, 20} |
| Max epochs | 200 | — |
| Early stopping patience | 10 | — |
| Gradient clip norm | 1.0 | — |
| Conformal α | 0.10 | — |
| ECRC δ | 0.05 | — |

### B  Dataset Statistics

*Detailed per-category, per-year count distributions.*

### C  Additional Equity Audit Results

*Stratified by race, geography, and income quintile.*

### D  Routing Case Studies

*Two detailed routing examples (one successful, one abstention).*

### E  NeurIPS Reproducibility Checklist

*See `REPRODUCIBILITY.md` in the repository.*

---

## References

> `[TODO: Format in venue-specific citation style]`

Key references to include:

1. Brody, S., Alon, U., & Yahav, E. (2022). How Attentive are Graph Attention Networks? ICLR.
2. Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized Quantile Regression. NeurIPS.
3. Vovk, V. (2005). Algorithmic Learning in a Random World. Springer.
4. Tibshirani, R. J., et al. (2019). Conformal Prediction Under Covariate Shift. NeurIPS.
5. Feldman, S., et al. (2021). Improving Conditional Coverage via Orthogonal Quantile Regression.
6. Benjamini, Y. & Hochberg, Y. (1995). Controlling the False Discovery Rate. JRSS-B.
7. Duan, R., et al. (2025). Breaking the Sorting Barrier for Directed SSSP. STOC (Best Paper). [cited as future large-scale direction; NOT used at city scale, where Dijkstra is exact and faster]
8. Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
9. Mohler, G. O., et al. (2011). Self-Exciting Point Process Modeling of Crime. JASA.
10. Lum, K. & Isaac, W. (2016). To Predict and Serve? Significance.
11. Salinas, D., et al. (2020). DeepAR: Probabilistic Forecasting with Autoregressive RNNs. IJF.
12. Yu, B., Yin, H., & Zhu, Z. (2018). Spatio-Temporal Graph Convolutional Networks. IJCAI.
13. Li, Y., et al. (2018). Diffusion Convolutional Recurrent Neural Network. ICLR.
14. Lambert, D. (1992). Zero-Inflated Poisson Regression. Technometrics.
15. Bellamy, R. K. E., et al. (2019). AI Fairness 360. IBM J. Res. Dev.
