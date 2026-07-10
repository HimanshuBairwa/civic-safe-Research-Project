# CIVIC-SAFE: Complete Mathematical Formulation

> This document provides the full mathematical specification referenced by `docs/PAPER_OUTLINE.md`.  
> All equations correspond to implementations in `src/civicsafe/`.

---

## Table of Contents

1. [Notation and Problem Setup](#1-notation-and-problem-setup)
2. [Spatial Encoder: Dual-Adjacency GATv2](#2-spatial-encoder-dual-adjacency-gatv2)
3. [Temporal Encoder: Causal Transformer](#3-temporal-encoder-causal-transformer)
4. [Multi-Factor Feature Mixer (MFFM)](#4-multi-factor-feature-mixer-mffm)
5. [ZINB Distributional Head](#5-zinb-distributional-head)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Conformal Prediction](#7-conformal-prediction)
8. [Equity Audit Statistics](#8-equity-audit-statistics)
9. [Advisory Safe Routing](#9-advisory-safe-routing)
10. [Geospatial Areal Interpolation](#10-geospatial-areal-interpolation)

---

## 1  Notation and Problem Setup

### 1.1  Core Notation

| Symbol | Domain | Definition |
|--------|--------|-----------|
| $S$ | $\mathbb{Z}_{>0}$ | Number of spatial units (77 for Chicago, 78 for NYC) |
| $T$ | $\mathbb{Z}_{>0}$ | Number of time steps in the lookback window (up to 52 weeks) |
| $F$ | $\mathbb{Z}_{>0}$ | Number of input features per node per timestep |
| $C$ | $\{3\}$ | Number of crime categories: {violent, property, drug} |
| $d$ | $\mathbb{Z}_{>0}$ | Hidden embedding dimension (default: 128) |
| $\mathbf{X} \in \mathbb{R}^{S \times T \times F}$ | — | Input spatiotemporal feature tensor |
| $Y_{s,t,c} \in \mathbb{Z}_{\geq 0}$ | — | Observed crime count |
| $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ | — | Spatial graph with $\mathcal{V} = \mathcal{S}$, $\mathcal{E} = \mathcal{E}_\text{queen} \cup \mathcal{E}_\text{knn}$ |
| $\mathcal{N}(i)$ | — | Neighbourhood of node $i$ in graph $\mathcal{G}$ |
| $H$ | $\mathbb{Z}_{>0}$ | Number of attention heads |

### 1.2  Model Pipeline

$$\mathbf{X} \xrightarrow{\text{Linear}} \mathbf{Z}^{(0)} \xrightarrow{\text{GATv2}} \mathbf{Z}^{(\text{spatial})} \xrightarrow{\text{CausalTransformer}} \mathbf{Z}^{(\text{temporal})} \xrightarrow{\text{MFFM}} \mathbf{Z}^{(\text{mixed})} \xrightarrow{\text{ZINBHead}} (\pi, \mu, r)$$

---

## 2  Spatial Encoder: Dual-Adjacency GATv2

**Reference implementation:** `src/civicsafe/models/spatial.py`  
**Reference paper:** Brody, Alon & Yahav (ICLR 2022)

### 2.1  Input Projection

The raw features are projected to the hidden dimension:

$$\mathbf{z}_s^{(0)} = \mathbf{W}_\text{proj} \mathbf{x}_s + \mathbf{b}_\text{proj}, \quad \mathbf{W}_\text{proj} \in \mathbb{R}^{d \times F}, \; \mathbf{b}_\text{proj} \in \mathbb{R}^d$$

### 2.2  GATv2 Dynamic Attention

Unlike GATv1 (Veličković et al., 2018), GATv2 applies the nonlinearity *after* concatenation, producing strictly more expressive dynamic attention:

**Attention coefficient** for edge $(j \to i)$ in head $h$:

$$e_{ij}^{(h)} = \mathbf{a}^{(h)\top} \cdot \text{LeakyReLU}\!\left(\mathbf{W}^{(h)} [\mathbf{h}_i \| \mathbf{h}_j]\right)$$

where:
- $\mathbf{h}_i, \mathbf{h}_j \in \mathbb{R}^{d}$ are node embeddings
- $\mathbf{W}^{(h)} \in \mathbb{R}^{d' \times 2d}$ is a learnable weight matrix
- $\mathbf{a}^{(h)} \in \mathbb{R}^{d'}$ is a learnable attention vector
- $[\cdot \| \cdot]$ denotes concatenation
- $\text{LeakyReLU}$ uses negative slope 0.2

**Normalised attention weight** via softmax over the neighbourhood:

$$\alpha_{ij}^{(h)} = \frac{\exp(e_{ij}^{(h)})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik}^{(h)})}$$

### 2.3  Multi-Head Aggregation

**Hidden layers** ($\ell < L$) — concatenation:

$$\mathbf{h}_i^{(\ell+1)} = \Big\|_{h=1}^{H} \sigma\!\left(\sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(h)} \mathbf{W}^{(h)} \mathbf{h}_j^{(\ell)}\right)$$

**Output layer** ($\ell = L$) — averaging:

$$\mathbf{h}_i^{(L)} = \sigma\!\left(\frac{1}{H} \sum_{h=1}^{H} \sum_{j \in \mathcal{N}(i)} \alpha_{ij}^{(h)} \mathbf{W}^{(h)} \mathbf{h}_j^{(L-1)}\right)$$

where $\sigma$ is the ELU activation.

### 2.4  Dual Adjacency Fusion

The same GATv2 layers process both adjacency structures, and outputs are summed:

$$\mathbf{h}_i^{(\ell)} = \text{GATv2}(\mathbf{h}^{(\ell-1)}, \mathcal{E}_\text{queen}) + \text{GATv2}(\mathbf{h}^{(\ell-1)}, \mathcal{E}_\text{knn})$$

followed by LayerNorm, ELU activation, and dropout:

$$\mathbf{z}_i^{(\text{spatial})} = \text{Dropout}\!\left(\text{ELU}\!\left(\text{LayerNorm}\!\left(\mathbf{h}_i^{(L)}\right)\right)\right)$$

**Rationale:**
- **Queen contiguity** ($\mathcal{E}_\text{queen}$): Nodes sharing a border or corner. Captures geographic crime spillover (Tobler's first law).
- **8-Nearest Neighbours** ($\mathcal{E}_\text{knn}$): 8 nearest centroids by Euclidean distance. Captures similarity between non-contiguous but socioeconomically comparable areas.

---

## 3  Temporal Encoder: Causal Transformer

**Reference implementation:** `src/civicsafe/models/temporal.py`  
**Reference paper:** Vaswani et al. (NeurIPS 2017)

### 3.1  Sinusoidal Positional Encoding

For position $t$ and dimension $i$:

$$\text{PE}(t, 2i) = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \quad \text{PE}(t, 2i+1) = \cos\!\left(\frac{t}{10000^{2i/d}}\right)$$

The input to the transformer is:

$$\tilde{\mathbf{z}}_{s,t} = \mathbf{z}_{s,t}^{(\text{spatial})} + \text{PE}(t)$$

### 3.2  Causal (Masked) Self-Attention

The attention mechanism uses a causal mask to prevent information leakage from future timesteps.

**Scaled dot-product attention** for head $h$:

$$\text{Attention}^{(h)}(Q, K, V) = \text{softmax}\!\left(\frac{Q^{(h)} K^{(h)\top}}{\sqrt{d_k}} + M\right) V^{(h)}$$

where:

$$Q^{(h)} = \tilde{Z} \mathbf{W}_Q^{(h)}, \quad K^{(h)} = \tilde{Z} \mathbf{W}_K^{(h)}, \quad V^{(h)} = \tilde{Z} \mathbf{W}_V^{(h)}$$

$$\mathbf{W}_Q^{(h)}, \mathbf{W}_K^{(h)}, \mathbf{W}_V^{(h)} \in \mathbb{R}^{d \times d_k}, \quad d_k = d / H$$

**Causal mask:**

$$M_{t,t'} = \begin{cases} 0 & \text{if } t' \leq t \\ -\infty & \text{if } t' > t \end{cases}$$

This ensures position $t$ can only attend to positions $\{1, \ldots, t\}$, providing a mathematical guarantee of zero future information leakage.

### 3.3  Multi-Head Attention and Feed-Forward

**Multi-head attention** (concatenation + linear projection):

$$\text{MHA}(\tilde{Z}) = \text{Concat}\!\left(\text{Attention}^{(1)}, \ldots, \text{Attention}^{(H)}\right) \mathbf{W}_O$$

where $\mathbf{W}_O \in \mathbb{R}^{d \times d}$.

### 3.4  Transformer Layer (Pre-LN)

We use the Pre-LayerNorm variant for improved training stability:

$$\mathbf{z}' = \mathbf{z} + \text{MHA}\!\left(\text{LayerNorm}(\mathbf{z})\right)$$

$$\mathbf{z}'' = \mathbf{z}' + \text{FFN}\!\left(\text{LayerNorm}(\mathbf{z}')\right)$$

where the feed-forward network is:

$$\text{FFN}(\mathbf{x}) = \text{ReLU}(\mathbf{x} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2$$

with $\mathbf{W}_1 \in \mathbb{R}^{d \times d_\text{ff}}$, $\mathbf{W}_2 \in \mathbb{R}^{d_\text{ff} \times d}$, $d_\text{ff} = 512$.

The output after $L_\text{temp}$ layers is $\mathbf{Z}^{(\text{temporal})} \in \mathbb{R}^{S \times T \times d}$.

---

## 4  Multi-Factor Feature Mixer (MFFM)

**Reference implementation:** `src/civicsafe/models/feature_mixer.py`

The MFFM prevents the model from collapsing attention onto a single proxy variable (e.g., race-correlated income) by decomposing the fused representation into $K$ interpretable factor heads with diversity regularisation.

### 4.1  Gated Cross-Attention

Each factor head $k \in \{1, \ldots, K\}$ produces an attention distribution over feature dimensions:

$$\mathbf{g}_k = \text{softmax}\!\left(\frac{\mathbf{W}_Q^{(k)} \mathbf{z} \cdot (\mathbf{W}_K^{(k)} \mathbf{z})^\top}{\tau}\right)$$

where:
- $\mathbf{W}_Q^{(k)}, \mathbf{W}_K^{(k)} \in \mathbb{R}^{d \times d}$ are learnable projections
- $\tau > 0$ is the temperature parameter (default: 1.0)
- Lower $\tau$ → sharper (winner-take-all); higher $\tau$ → uniform

The mixed output for head $k$:

$$\mathbf{m}_k = \mathbf{g}_k \cdot (\mathbf{W}_V^{(k)} \mathbf{z})$$

The final mixed representation is the mean across heads:

$$\mathbf{z}^{(\text{mixed})} = \frac{1}{K} \sum_{k=1}^{K} \mathbf{m}_k$$

### 4.2  Jensen-Shannon Diversity Regularisation

To prevent attention collapse (multiple heads attending to the same features), we penalise low pairwise JSD between head attention distributions.

Let $P_k$ be the mean attention distribution of head $k$ (averaged over spatial units and timesteps). The diversity loss is:

$$\mathcal{L}_\text{div} = \lambda_\text{div} \sum_{i=1}^{K} \sum_{j=i+1}^{K} \text{ReLU}\!\left(\delta - \text{JSD}(P_i \| P_j)\right)$$

where:

$$\text{JSD}(P \| Q) = \frac{1}{2} D_\text{KL}(P \| M) + \frac{1}{2} D_\text{KL}(Q \| M), \quad M = \frac{1}{2}(P + Q)$$

$$D_\text{KL}(P \| Q) = \sum_i P_i \log \frac{P_i}{Q_i}$$

- $\delta$ is the collapse threshold (default: 0.1)
- $\lambda_\text{div}$ scales the penalty relative to the ZINB NLL
- The ReLU ensures the penalty is zero when heads are already sufficiently diverse

---

## 5  ZINB Distributional Head

**Reference implementation:** `src/civicsafe/models/zinb_head.py`, `src/civicsafe/models/zinb_loss.py`

### 5.1  Output Parameterisation

The ZINB head consists of three independent 2-layer MLPs, each projecting from the $d$-dimensional mixed embedding to $C$ crime categories:

$$\pi_{s,c} = \sigma\!\left(\text{MLP}_\pi(\mathbf{z}_s^{(\text{mixed})})\right) \in [0, 1]$$

$$\mu_{s,c} = \text{softplus}\!\left(\text{MLP}_\mu(\mathbf{z}_s^{(\text{mixed})})\right) \in (0, \infty)$$

$$r_{s,c} = \text{softplus}\!\left(\text{MLP}_r(\mathbf{z}_s^{(\text{mixed})})\right) + r_\text{floor} \in [r_\text{floor}, \infty)$$

where $\sigma$ is the sigmoid function, $\text{softplus}(x) = \log(1 + e^x)$, and $r_\text{floor} = 0.1$ prevents numerical instability.

### 5.2  Probability Mass Function

The ZINB is a mixture of a point mass at zero and a Negative Binomial:

$$P(Y = y \mid \pi, \mu, r) = \begin{cases}
\pi + (1-\pi) \left(\dfrac{r}{r+\mu}\right)^{\!r} & \text{if } y = 0 \\[10pt]
(1-\pi) \cdot \dfrac{\Gamma(y+r)}{\Gamma(r) \, y!} \left(\dfrac{r}{r+\mu}\right)^{\!r} \left(\dfrac{\mu}{r+\mu}\right)^{\!y} & \text{if } y > 0
\end{cases}$$

**Interpretation:**
- $\pi$ = probability that the observation is a structural/reporting zero (zero-inflation)
- $\mu$ = mean of the underlying Negative Binomial process
- $r$ = dispersion (concentration); as $r \to \infty$, NB → Poisson

**Moments:**

$$\mathbb{E}[Y] = (1 - \pi) \mu$$

$$\text{Var}(Y) = (1 - \pi)\mu\!\left(1 + \frac{\mu}{r} + \pi\mu\right)$$

### 5.3  Negative Log-Likelihood (Training Loss)

The NLL is computed in log-space for numerical stability:

**Case $y = 0$** — uses logsumexp to avoid underflow:

Let $a = \log \pi$ and $b = \log(1-\pi) + r \log\!\left(\frac{r}{r+\mu}\right)$:

$$\mathcal{L}_\text{ZINB}(y=0) = -\text{logsumexp}(a, b) = -\log(e^a + e^b)$$

**Case $y > 0$:**

$$\mathcal{L}_\text{ZINB}(y>0) = -\log(1-\pi) - \log\Gamma(y+r) + \log\Gamma(r) + \log(y!) - r\log\!\left(\frac{r}{r+\mu}\right) - y\log\!\left(\frac{\mu}{r+\mu}\right)$$

**Total training loss:**

$$\mathcal{L} = \frac{1}{|\mathcal{B}|} \sum_{(s,t,c) \in \mathcal{B}} \mathcal{L}_\text{ZINB}(Y_{s,t,c} \mid \pi_{s,c}, \mu_{s,c}, r_{s,c}) + \lambda_\text{div} \cdot \mathcal{L}_\text{div}$$

---

## 6  Evaluation Metrics

### 6.1  Continuous Ranked Probability Score (CRPS)

CRPS jointly evaluates the calibration and sharpness of a full predictive distribution:

$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} \left[F(x) - \mathbb{1}(y \leq x)\right]^2 dx$$

For discrete ZINB distributions, we compute the sum up to a truncation limit $K_\text{max} = \mu + 10\sigma$:

$$\text{CRPS}(F_\text{ZINB}, y) \approx \sum_{k=0}^{K_\text{max}} \left[F_\text{ZINB}(k) - \mathbb{1}(y \leq k)\right]^2$$

where the CDF is computed recursively:

$$F_\text{ZINB}(k) = \sum_{j=0}^{k} P(Y = j \mid \pi, \mu, r)$$

**Properties:**
- CRPS = 0 when the distribution is a point mass at the observation
- CRPS is a strictly proper scoring rule: minimised when $F = F_\text{true}$
- CRPS generalises MAE to distributional predictions

### 6.2  Point Metrics

**Mean Absolute Error:**

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|, \quad \hat{y}_i = (1 - \pi_i) \mu_i$$

**Root Mean Squared Error:**

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

### 6.3  Zero-Inflation Brier Score

Evaluates calibration of the zero-inflation probability:

$$\text{Brier}_\text{zero} = \frac{1}{N} \sum_{i=1}^{N} (\pi_i - \mathbb{1}(y_i = 0))^2$$

### 6.4  Probability Integral Transform (PIT)

For a well-calibrated model, the PIT values should be uniformly distributed on $[0, 1]$:

$$\text{PIT}_i = F_\text{ZINB}(y_i \mid \pi_i, \mu_i, r_i)$$

Calibration is assessed via the Kolmogorov–Smirnov test against $\text{Uniform}(0, 1)$.

---

## 7  Conformal Prediction

**Reference implementation:** `src/civicsafe/calibration/conformal.py`

### 7.1  Non-Conformity Score (CQR)

Following Romano et al. (2019), the Conformalized Quantile Regression score is:

$$s_i = \max\!\left(q_{\alpha/2}^{(i)} - y_i, \; y_i - q_{1-\alpha/2}^{(i)}\right)$$

where $q_{\alpha/2}^{(i)}$ and $q_{1-\alpha/2}^{(i)}$ are the ZINB quantiles (PPF) at levels $\alpha/2$ and $1 - \alpha/2$ respectively.

- $s_i < 0$: observation was inside the heuristic interval
- $s_i > 0$: observation was outside

### 7.2  Split Conformal Prediction

**Algorithm:**

1. Compute scores $\{s_1, \ldots, s_n\}$ on calibration set $\mathcal{D}_\text{cal}$
2. Compute the conformal quantile with finite-sample correction:

$$\hat{q} = \text{Quantile}\!\left(\{s_1, \ldots, s_n\}, \; \left\lceil \frac{(1 - \alpha)(n + 1)}{n} \right\rceil\right)$$

3. At test time, for new prediction with ZINB parameters $(\pi, \mu, r)$:

$$L = \max\!\left(0, \; \left\lfloor q_{\alpha/2} - \hat{q} \right\rfloor\right), \quad U = \left\lceil q_{1-\alpha/2} + \hat{q} \right\rceil$$

**Guarantee:** $\Pr(Y \in [L, U]) \geq 1 - \alpha$ (marginal, finite-sample, exchange\-ability).

### 7.3  Weighted Conformal Prediction

For non-stationary crime data, we assign exponentially decaying weights:

$$w_i = \exp(-\lambda \cdot \Delta t_i), \quad w_i \leftarrow \max(w_i, w_\text{min})$$

where $\Delta t_i$ is the temporal distance of calibration point $i$ from the test time, and $\lambda = 0.05$ is the decay rate.

The threshold is the **weighted quantile**: sort scores, compute cumulative normalised weights, and find the first index where $\sum_{j \leq k} \tilde{w}_j \geq 1 - \alpha$.

### 7.4  Mondrian Conformal Prediction

For group-conditional coverage, we run independent Split CP within each group $g$:

$$\hat{q}_g = \text{Quantile}\!\left(\{s_i : i \in \mathcal{D}_\text{cal}^{(g)}\}, \; \left\lceil \frac{(1 - \alpha)(n_g + 1)}{n_g} \right\rceil\right)$$

**Guarantee:** $\Pr(Y \in [L, U] \mid G = g) \geq 1 - \alpha$ for every group $g$.

Groups with $n_g < n_\text{min}$ (default: 40) fall back to the global threshold.

### 7.5  Equalized Coverage

The threshold $\hat{q}$ is chosen to minimise a regularised objective:

$$\hat{q} = \arg\min_q \; \underbrace{|1 - \alpha - \hat{\text{Cov}}(q)|}_{\text{coverage gap}} + \lambda_\text{eq} \cdot \underbrace{\max_g |\hat{\text{Cov}}_g(q) - (1 - \alpha)|}_{\text{max group deviation}}$$

where $\hat{\text{Cov}}(q)$ is the empirical coverage at threshold $q$ and $\hat{\text{Cov}}_g(q)$ is the per-group coverage.

### 7.6  Equalized Conditional Risk Control (ECRC)

The primary calibrator for CIVIC-SAFE. Provides high-probability per-group coverage guarantees using Hoeffding's inequality.

**Hoeffding slack:**

$$\varepsilon = \sqrt{\frac{\log(2G / \delta)}{2 n_\text{cal} / G}}$$

where $G$ is the number of groups and $\delta$ is the failure probability (default: 0.05).

**Adjusted miscoverage level:**

$$\alpha' = \max(\alpha - \varepsilon, \; 0.01)$$

**Per-group calibration** uses Split CP with $\alpha'$:

$$\hat{q}_g = \text{Quantile}\!\left(\{s_i : i \in \mathcal{D}_\text{cal}^{(g)}\}, \; \left\lceil \frac{(1 - \alpha')(n_g + 1)}{n_g} \right\rceil\right)$$

**Guarantee:**

$$\Pr\!\left(\text{coverage}(g) \geq 1 - \alpha - \varepsilon\right) \geq 1 - \delta, \quad \forall g \in \{1, \ldots, G\}$$

---

## 8  Equity Audit Statistics

**Reference implementation:** `src/civicsafe/audit/`

### 8.1  Bootstrap Confidence Intervals

For each audit metric $\theta$, we compute the BCa bootstrap confidence interval:

1. Draw $B = 10{,}000$ bootstrap samples with replacement
2. Compute $\hat{\theta}^*_1, \ldots, \hat{\theta}^*_B$
3. Construct the 95% BCa interval accounting for bias and skewness

### 8.2  Permutation Test for Group Disparities

To test whether metric disparity is statistically significant:

1. Compute observed disparity $d_\text{obs} = |\theta_{g_1} - \theta_{g_2}|$
2. Permute group labels $B = 10{,}000$ times
3. Compute $p = \frac{1 + \sum_{b=1}^B \mathbb{1}(d_b \geq d_\text{obs})}{1 + B}$

### 8.3  Benjamini–Hochberg FDR Correction

Given $m$ simultaneous hypothesis tests with p-values $p_{(1)} \leq \cdots \leq p_{(m)}$:

$$\text{Reject } H_{(i)} \text{ if } p_{(i)} \leq \frac{i}{m} \cdot q$$

where $q$ is the target FDR level (default: 0.05). This controls:

$$\text{FDR} = \mathbb{E}\!\left[\frac{\text{False Positives}}{\max(\text{Rejections}, 1)}\right] \leq q$$

### 8.4  Reporting Bias Sensitivity (INAR Binomial Thinning)

To assess robustness to under-reporting, we apply binomial thinning to observed counts:

$$\tilde{Y}_{s,t,c} \sim \text{Binomial}(Y_{s,t,c}, 1 - p_\text{thin})$$

where $p_\text{thin} \in \{0.05, 0.10, 0.15, 0.20\}$ represents the hypothesised under-reporting rate. The audit re-evaluates all metrics under thinned data.

---

## 9  Advisory Safe Routing

**Reference implementation:** `src/civicsafe/routing/`

### 9.1  Pareto-Optimal Routing Objective

For a path $P$ composed of edges $e \in \mathcal{E}_\text{road}$:

$$\min_{\text{path } P} \left(\sum_{e \in P} d_e, \;\; \sum_{e \in P} \rho_e\right)$$

where $d_e$ is the physical distance and $\rho_e$ is the predicted risk.

### 9.2  Risk Mapping Function

The risk score for edge $e$ incorporates both the expected crime intensity and the model's uncertainty:

$$\rho_e = \underbrace{(1 - \pi_e) \cdot \mu_e}_{\text{expected count}} + \lambda_\text{unc} \cdot \underbrace{(1 - \pi_e) \cdot \frac{\mu_e(\mu_e + r_e)}{r_e}}_{\text{ZINB variance penalty}}$$

where:
- $(1 - \pi_e) \mu_e = \mathbb{E}[Y_e]$ is the expected crime count
- $(1 - \pi_e) \frac{\mu_e(\mu_e + r_e)}{r_e}$ is proportional to $\text{Var}(Y_e)$
- $\lambda_\text{unc}$ controls the risk-averseness of routing

### 9.3  Shortest-path routing (exact Dijkstra)

The engine finds the optimal path over the conformal-interval edge costs using
**exact Dijkstra** ($O(m + n \log n)$ with a binary/Fibonacci heap), which is the
correct and fastest choice at city scale (~77–100 nodes).

> *Forward-looking note (not used here).* Duan, Mao, Mao, Shu & Yin (2025, STOC
> Best Paper) give a deterministic $O(m \log^{2/3} n)$ SSSP algorithm that breaks
> the sorting barrier asymptotically. Faithful implementations are, however,
> 3–25× **slower** than Dijkstra at practical sizes (the crossover is ~$10^{60}$
> vertices), so it offers no benefit at city scale. We note it only as a
> potential direction for metropolitan-to-national road networks. Our code does
> **not** implement it and makes no sorting-barrier claim.

### 9.4  Abstention Protocol

The engine refuses to recommend routes when prediction uncertainty is too high:

$$\text{AbstainIf:} \quad \max_{e \in P^*} \left(U_e - L_e\right) > \theta_\text{abstain}$$

where $[L_e, U_e]$ is the conformal prediction interval for edge $e$ and $\theta_\text{abstain}$ is a calibrated safety threshold. When triggered, the engine returns an advisory message explaining why a safe route cannot be determined.

---

## 10  Geospatial Areal Interpolation

**Reference implementation:** `src/civicsafe/data/`

Census ACS variables are published at the tract level but CIVIC-SAFE operates at the community-area (Chicago) or precinct (NYC) level. We use area-weighted areal interpolation:

$$x_j = \sum_{i \in \mathcal{T}_j} \frac{\text{Area}(T_i \cap P_j)}{\text{Area}(T_i)} \cdot x_i$$

where:
- $x_j$ = estimated variable value for target zone $j$ (community area / precinct)
- $x_i$ = source variable value for census tract $i$
- $T_i$ = geometry of census tract $i$
- $P_j$ = geometry of target zone $j$
- $\mathcal{T}_j = \{i : T_i \cap P_j \neq \emptyset\}$ = set of tracts intersecting zone $j$

**Assumption:** The underlying variable is uniformly distributed within each census tract. This is the standard assumption for extensive (count-like) variables. For intensive (rate-like) variables, we use dasymetric interpolation where population serves as an auxiliary variable:

$$x_j^{(\text{intensive})} = \frac{\sum_{i \in \mathcal{T}_j} \text{Pop}(T_i \cap P_j) \cdot x_i}{\sum_{i \in \mathcal{T}_j} \text{Pop}(T_i \cap P_j)}$$

---

## Summary of Key Dimensions

| Component | Key Hyperparameters |
|-----------|-------------------|
| GATv2 Spatial | $L = 2$, $H = 4$, $d = 128$, dropout = 0.1 |
| Causal Transformer | $L = 2$, $H = 4$, $d = 128$, $d_\text{ff} = 512$, Pre-LN |
| MFFM | $K = 3$ heads, $\tau = 1.0$, $\delta = 0.1$ |
| ZINB Head | $d_\pi = d_\mu = d_r = 64$, $r_\text{floor} = 0.1$ |
| Training | AdamW, lr = 1e-3, warmup = 10, patience = 10, 5 seeds |
| Conformal | $\alpha = 0.10$, ECRC $\delta = 0.05$ |
| **Total parameters** | **688,649** |
