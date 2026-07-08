# CIVIC-SAFE Mathematical Specification

This document provides the formal mathematical specification for the CIVIC-SAFE architecture, including the distributional loss functions, spatial attention mechanisms, bias-mitigation regularisation, and evaluation metrics.

> **Contribution note.** The distributional/GNN/conformal machinery below (§§1–13) is *applied* — it builds on published methods (ZINB-GNN: Zhuang et al. KDD 2022, STZINB-GNN; Wang et al. 2024, STMGNN-ZINB; conformal: Gibbs & Candès 2021; CQR: Romano et al. 2019; EMOS: Gneiting et al. 2005). **The genuinely novel contribution is §0 below**: the Feedback Amplification Law, the passive/active identification duality, and — the part no prior work provides — a *feedback-corrected conformal predictor* that recovers coverage of the true latent process from a biased record. Positioning against the nearest prior art (Ensign 2018; Glaeser–Sacerdote–Scheinkman 2003; Algometrics 2026; van Amsterdam 2025) is in `docs/NOVELTY_AND_POSITIONING.md`; formal proofs in `docs/PROOFS_feedback_law.md`.

## 0. The Contribution: Feedback Amplification and Latent-Coverage Correction

**Model (Allocation under Observation-Biased Feedback).** Latent incidence $\lambda_s>0$ (unobserved); a policy allocates attention $a_s=\pi(\mu_s)$ from the recorded-rate estimate $\mu_s$; recording is observation-biased, $y_s\sim\mathrm{Poisson}(\lambda_s\, g(a_s))$ with $g$ increasing; a consistent learner reaches the fixed point $\mu_s=\lambda_s\,g(\pi(\mu_s))$. Define the feedback gain $\kappa=\bigl(\tfrac{d\log a}{d\log\mu}\bigr)\bigl(\tfrac{d\log g}{d\log a}\bigr)$.

**§0.1 Amplification elasticity (Thm 1).** $\dfrac{d\log\mu_s}{d\log\lambda_s}=\dfrac{1}{1-\kappa}$, so recorded disparity $=$ true disparity$^{1/(1-\kappa)}$, with a pole at $\kappa^\*=1$. *This closed form is the social multiplier (Glaeser–Sacerdote–Scheinkman 2003) / control-theoretic loop gain; our contribution is the coordinate-free elasticity decomposition and the disparity power-law corollary, a quantitative sharpening of Ensign et al. (2018).* Verified: `tests/test_feedback_law.py`.

**§0.2 Passive/active duality (Thms 2–3).** The "confidently wrong" state — observed coverage maintained while latent coverage collapses — is **not identifiable from passive data** (biased and honest worlds are observationally identical), but $\kappa$ is **point-identified by a difference-in-differences** on log recorded rates after an exogenous detection-sensitivity shock (staggered ShotSpotter/patrol rollout). *The duality principle is known (Mendler-Dünner 2022; Algometrics 2026); our contribution is its DiD instantiation for a recording-loop elasticity.*

**§0.3 Feedback-corrected latent conformal prediction (the novel constructive step).** Given the identified $\kappa$, deflate the record by the recording multiplier $m_s=(\mu_s/M)^\kappa$ to recover $\hat\lambda_s=\mu_s/m_s$ and issue prediction intervals valid for the **latent** process, with **abstention** as $\kappa\to1$. **Prior work (Ensign 2018; van Amsterdam 2025; Algometrics 2026) diagnoses the pathology; this corrects it.** Implementation: `src/civicsafe/theory/latent_correction.py`; experiment: `scripts/latent_correction_experiment.py`.

## 1. Output Distribution: Zero-Inflated Negative Binomial (ZINB)

CIVIC-SAFE forecasts crime counts as a full probability distribution rather than a point estimate. Crime data is typically extremely sparse (many structural and reporting zeros) and overdispersed (variance > mean). The ZINB distribution is the statistically correct choice for this domain.

### 1.1 Probability Mass Function
For a spatial unit $s$, time step $t$, and crime category $c$, the model predicts three parameters: $(\pi, \mu, r)$.
The probability of observing $y$ crimes is:

$$P(Y=y \mid \pi, \mu, r) = \begin{cases} \pi + (1-\pi)\left(\frac{r}{r+\mu}\right)^r & \text{if } y = 0 \\[8pt] (1-\pi) \cdot \frac{\Gamma(y+r)}{\Gamma(r)\,y!} \left(\frac{r}{r+\mu}\right)^r \left(\frac{\mu}{r+\mu}\right)^y & \text{if } y > 0 \end{cases}$$

Where:
- $\pi \in [0, 1]$ is the zero-inflation probability (structural and reporting zeros).
- $\mu \in (0, \infty)$ is the mean of the underlying Negative Binomial process.
- $r \in [r_{\text{floor}}, \infty)$ is the dispersion parameter. As $r \to \infty$, the NB reduces to a Poisson distribution.

### 1.2 Negative Log-Likelihood (Training Loss)
To prevent numerical instability (gradient explosion) when $\pi \to 0$ or $r \to 0$, we implement the negative log-likelihood (NLL) using `logsumexp` for the zero case and `torch.lgamma` for the non-zero case.

**For $y = 0$:**
Let $a = \log(\pi)$ and $b = \log(1-\pi) + r \log\left(\frac{r}{r+\mu}\right)$.
$$\mathcal{L}_{\text{ZINB}}(y=0) = -\log(e^a + e^b) = -\text{logsumexp}(a, b)$$

**For $y > 0$:**
$$\mathcal{L}_{\text{ZINB}}(y>0) = -\log(1-\pi) - \log\Gamma(y+r) + \log\Gamma(r) + \log(y!) - r\log\left(\frac{r}{r+\mu}\right) - y\log\left(\frac{\mu}{r+\mu}\right)$$

*Implementation Note: We enforce numerical constraints via activation functions: $\pi = \sigma(\cdot)$, $\mu = \text{softplus}(\cdot)$, and $r = \text{softplus}(\cdot) + 0.1$.*

## 2. Spatial Encoder: Dual-Graph GATv2

To capture spatial diffusion, we use a Graph Attention Network v2 (Brody et al., 2022) operating over two distinct adjacency structures:
1. **Queen Contiguity** ($\mathcal{E}_{\text{queen}}$): Captures immediate geographic spillover.
2. **K-Nearest Neighbors** ($\mathcal{E}_{\text{knn}}$): Captures global spatial autocorrelation between demographically similar but non-contiguous areas.

### 2.1 Dynamic Attention
For any edge $(j \to i)$ in the combined graph, the attention coefficient is computed dynamically:
$$e_{ij} = \mathbf{a}^\top \cdot \text{LeakyReLU}\left(\mathbf{W} \cdot [\mathbf{h}_i \| \mathbf{h}_j]\right)$$
$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k \in \mathcal{N}(i)} \exp(e_{ik})}$$

The updated node representation is the sum over both the Queen and KNN neighborhoods:
$$\mathbf{h}_i' = \sigma\left(\sum_{j \in \mathcal{N}_{\text{queen}}(i) \cup \mathcal{N}_{\text{knn}}(i)} \alpha_{ij} \mathbf{W}\mathbf{h}_j\right)$$

## 3. Bias Mitigation: Multi-Factor Feature Mixer (MFFM)

To prevent the model from overfitting to demographic covariates (a common source of proxy bias in predictive policing), the Temporal Encoder output is passed through a multi-head gating mechanism.

### 3.1 Feature Gating (Squeeze-and-Excitation Style)
Each head $k$ produces an attention distribution over the feature dimensions via a learned linear projection:
$$\mathbf{g}_k = \text{softmax}\left(\frac{\mathbf{W}^{(k)}\mathbf{x}}{\tau}\right)$$
Where $\tau$ is the temperature parameter controlling sparsity. The gated output is the element-wise product:
$$\mathbf{h}_k = \mathbf{x} \odot \mathbf{g}_k$$

This is analogous to a Squeeze-and-Excitation block (Hu et al., 2018) with softmax normalization instead of sigmoid, applied independently per factor head to encourage diverse feature utilization.

### 3.2 Diversity Regularisation (Jensen-Shannon Divergence)
To force the model to distribute its attention across diverse factors rather than collapsing onto a single proxy variable, we apply a pairwise Jensen-Shannon Divergence (JSD) penalty across all $K$ heads.

Let $P_i$ and $P_j$ be the mean attention distributions for heads $i$ and $j$. The diversity loss is:
$$\mathcal{L}_{\text{div}} = \lambda_{\text{div}} \sum_{i=1}^K \sum_{j=i+1}^K \text{ReLU}\left( \delta - \text{JSD}(P_i \| P_j) \right)$$
Where $\delta$ is the collapse threshold (e.g., 0.1) and $\text{JSD}(P\|Q) = \frac{1}{2} D_{\text{KL}}(P \| M) + \frac{1}{2} D_{\text{KL}}(Q \| M)$ with $M = \frac{1}{2}(P+Q)$.

## 4. Evaluation: Continuous Ranked Probability Score (CRPS)

Because CIVIC-SAFE outputs a distribution, point metrics (MAE, RMSE) are insufficient. We use CRPS to evaluate the calibration and sharpness of the full predictive distribution.

For observation $y$ and cumulative distribution function $F$:
$$\text{CRPS}(F, y) = \int_{-\infty}^{\infty} \left[F(x) - \mathbb{1}(y \leq x)\right]^2 dx$$

Since crime counts are discrete, we compute the discrete sum up to a truncation limit $K_{\max} = \mu + 10\sigma$:
$$\text{CRPS}(F_{\text{ZINB}}, y) \approx \sum_{k=0}^{K_{\max}} \left[F_{\text{ZINB}}(k) - \mathbb{1}(y \leq k)\right]^2$$
Where $F_{\text{ZINB}}(k) = \sum_{j=0}^k P(Y=j \mid \pi, \mu, r)$.

## 5. Downstream Application: Advisory Safe Routing

CIVIC-SAFE connects probabilistic forecasts to urban navigation via a Pareto-optimal routing engine.

For a path $P$ composed of edges $e \in \mathcal{E}$, the objective minimizes both physical distance $d_e$ and expected risk $\rho_e$:
$$\min_{\text{path } P} \left(\sum_{e \in P} d_e,\;\; \sum_{e \in P} \rho_e\right)$$

The risk mapping function converts the ZINB parameters for edge $e$ into a scalar risk penalty, incorporating variance to penalize uncertainty:
$$\rho_e = f(\mu_e, r_e, \pi_e) = (1 - \pi_e) \cdot \mu_e + \lambda_{\text{unc}} \cdot (1 - \pi_e) \cdot \frac{\mu_e(\mu_e + r_e)}{r_e}$$

The Tsinghua SSSP algorithm is used to find the optimal path. If the peak uncertainty along the optimal path exceeds a critical threshold, the engine executes an **Abstention Protocol** and refuses to return a route, preventing false assurances of safety.

## 6. CRPS-Direct Training (Applied method; closed-form CRPS for NB after Wei & Held 2014)

### 6.1 The Train-Eval Mismatch Problem

Standard ZINB crime forecasting models (including STMGNN-ZINB, Wang et al. 2024) train by minimizing the negative log-likelihood:
$$\mathcal{L}_{\text{NLL}} = -\frac{1}{N}\sum_{i=1}^{N} \log P(y_i \mid \pi_i, \mu_i, r_i)$$

But they are *evaluated* using CRPS, which measures distributional calibration:
$$\text{CRPS}(F, y) = \sum_{k=0}^{K_{\max}} \left[F_{\text{ZINB}}(k) - \mathbb{1}(y \leq k)\right]^2$$

NLL and CRPS are both strictly proper scoring rules (Gneiting & Raftery, 2007), but they emphasize different aspects: NLL rewards density sharpness at the observed value, while CRPS rewards overall distributional calibration. In practice, NLL-trained models can achieve low NLL while having poor CRPS (the *r-collapse* failure mode), because NLL incentivizes narrowing the distribution around the mode, potentially at the expense of tail calibration.

### 6.2 Differentiable CRPS Loss

CRPS is fully differentiable with respect to the ZINB parameters $(\pi, \mu, r)$:

$$\frac{\partial \text{CRPS}}{\partial \theta} = \sum_{k=0}^{K_{\max}} 2\left[F_{\text{ZINB}}(k; \theta) - \mathbb{1}(y \leq k)\right] \cdot \frac{\partial F_{\text{ZINB}}(k; \theta)}{\partial \theta}$$

where $\theta \in \{\pi, \mu, r\}$. The indicator function $\mathbb{1}(y \leq k)$ has zero gradient (it's a constant for a given observation), so gradients flow entirely through the CDF $F_{\text{ZINB}}$.

The CDF is computed via cumulative summation of the PMF: $F_{\text{ZINB}}(k) = \pi + (1-\pi)\sum_{j=0}^{k} \text{PMF}_{\text{NB}}(j; \mu, r)$, which involves only differentiable operations (`lgamma`, `exp`, `cumsum`).

### 6.3 Blended Loss

For transitional training or hyperparameter search, we support a blended loss:
$$\mathcal{L}_{\text{blend}} = \alpha \cdot \text{CRPS} + (1-\alpha) \cdot \text{NLL}$$

with $\alpha \in [0, 1]$. Setting $\alpha = 1$ gives pure CRPS training; $\alpha = 0$ gives legacy NLL training.

## 7. Spatiotemporal Graph Transformer (V2 Architecture)

### 7.1 Motivation

The sequential V1 architecture applies spatial encoding (GATv2) and temporal encoding (Transformer) independently. This means the temporal encoder processes each spatial unit as an independent sequence — it cannot capture cross-spatial temporal patterns (e.g., "crime rose across the entire south side this week").

### 7.2 Unified Token Representation

We define a spatiotemporal token for each (node, timestep) pair. For $S$ spatial units and $T$ timesteps, the token sequence has length $L = S \times T$.

The positional encoding combines learnable spatial embeddings with sinusoidal temporal encodings:
$$\text{PE}(s, t) = \text{Emb}_{\text{spatial}}(s) + \text{PE}_{\text{sinusoidal}}(t)$$

### 7.3 Structured Attention Mask

The key innovation is a structured attention mask $M \in \{0, -\infty\}^{L \times L}$ that enforces:

1. **Causal temporal self-attention**: Token $(s, t_1)$ can attend to $(s, t_2)$ iff $t_2 \leq t_1$ (same node, past/current time)
2. **Same-timestep spatial cross-attention**: Token $(s_1, t)$ can attend to $(s_2, t)$ iff $(s_2 \to s_1) \in \mathcal{E}$ (graph neighbors at the same time)
3. **No future leakage**: No token can attend to any future timestep

Formally:
$$M[(s_1, t_1), (s_2, t_2)] = \begin{cases} 0 & \text{if } s_1 = s_2 \text{ and } t_2 \leq t_1 \\ 0 & \text{if } (s_2, s_1) \in \mathcal{E} \text{ and } t_1 = t_2 \\ -\infty & \text{otherwise} \end{cases}$$

**Complexity**: $O(S^2T + ST^2)$ per layer with the structured mask (sparse attention), compared to $O(S^2T^2)$ for dense attention. For $S=77, T=52$, this is $\approx 5.2\text{M}$ attention entries vs. $16.1\text{M}$ for dense — feasible on standard hardware.

## 8. Adaptive Temporal ECRC (Applied: per-group Adaptive Conformal Inference, Gibbs & Candès 2021)

### 8.1 Conformal Prediction Background

Given a calibration set $\{(X_i, Y_i)\}_{i=1}^n$ and non-conformity scores $s_i = \max(q_{\alpha/2}^{(i)} - Y_i, Y_i - q_{1-\alpha/2}^{(i)})$ (CQR scores from the ZINB quantile function), the conformal threshold is:
$$\hat{q} = \text{Quantile}\left(\frac{\lceil(n+1)(1-\alpha)\rceil}{n}, \{s_1, \ldots, s_n\}\right)$$

### 8.2 Equalized Conditional Risk Control (ECRC)

For $G$ demographic groups, ECRC computes per-group thresholds $\hat{q}_g$ with Hoeffding-bounded risk control:
$$P\left[\text{Coverage}_g \geq 1 - \alpha - \epsilon_g\right] \geq 1 - \delta_g$$

where $\epsilon_g = \sqrt{\frac{\log(2/\delta_g)}{2n_g}}$ is the Hoeffding slack for group $g$ with $n_g$ calibration samples.

### 8.3 Adaptive Temporal Extension (Our Contribution)

Crime data is non-stationary: crime patterns shift due to policy changes, seasonal effects, and socioeconomic trends. Standard conformal prediction assumes exchangeability, which fails under drift.

Our Adaptive Temporal ECRC extends the ECRC framework with an online PID controller update rule inspired by Adaptive Conformal Inference (Angelopoulos et al. 2023):

$$\alpha_{t,g} \leftarrow \text{clip}\left[\alpha_{t-1,g} + K_P e_{t-1,g} + K_I \sum_{\tau=1}^{t-1} e_{\tau,g} + K_D (e_{t-1,g} - e_{t-2,g}), \; 0.01, \; 0.99\right]$$

where:
- $\alpha_{t,g}$ is the per-group miscoverage target at time $t$
- $e_{t-1,g} = \alpha - \hat{\text{err}}_{t-1,g}$ is the error term (target coverage minus empirical miscoverage)
- $K_P, K_I, K_D$ are the proportional, integral, and derivative gains

**Theorem (Informal)**. Under bounded variance of per-group coverage errors and appropriate PID gains, the time-averaged per-group miscoverage converges asymptotically:
$$\limsup_{T \to \infty} \left| \frac{1}{T} \sum_{t=1}^T \mathbb{1}[Y_t \notin \hat{C}_t \mid G_t = g] - \alpha \right| = 0$$

Note: This provides a long-run asymptotic average coverage guarantee, not an exact per-step marginal guarantee.

## 9. r-Collapse Diagnosis and Regularization (Optimization regularizer; diagnostic)

### 9.1 The r-Collapse Failure Mode

When training ZINB models with NLL, the dispersion parameter $r$ can collapse toward its floor value. This happens because:

1. As $r \to 0$, the NB variance $\sigma^2 = \mu + \mu^2/r \to \infty$, creating a heavy-tailed distribution
2. A heavy-tailed distribution assigns non-negligible probability to the observed $y$, resulting in acceptable NLL
3. But the distribution is poorly calibrated — CRPS degrades because the CDF is too spread

Empirically, we observe $r$ collapsing to $r_{\text{floor}} = 0.1$ across many cells while MAE *improves* (the mode prediction gets better) but CRPS *degrades* (the distribution gets worse). This "hidden Goodhart" effect is the primary reason NLL-trained ZINB models have poor distributional calibration.

### 9.2 Per-Cell Regularization

We apply a per-cell penalty that individually penalizes cells where $r < r_{\text{reg}}$:
$$\mathcal{L}_{r\text{-reg}} = \lambda_r \cdot \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \text{ReLU}(r_{\text{reg}} - r_i)$$

where $r_{\text{reg}} = 0.5$ (regularization floor, distinct from the hard floor $r_{\text{floor}} = 0.1$ in the architecture) and $\lambda_r = 0.1$.

**Why per-cell, not batch-mean**: A batch-mean penalty $\text{ReLU}(r_{\text{reg}} - \bar{r})$ allows some cells to collapse to near-zero while others compensate by staying high. The per-cell formulation prevents any individual cell from collapsing.

## 10. Sharpness-Aware Calibration Loss (Ablation only — disabled by default; see §10.3)

### 10.1 Motivation

Gneiting, Balabdaoui & Raftery (2007) established the foundational principle of probabilistic forecasting: **"maximize sharpness subject to calibration."** CRPS implicitly captures this tradeoff (it decomposes into reliability + resolution/sharpness), but doesn't explicitly control each component.

SAC makes the sharpness-calibration tradeoff *explicit* in the training objective:

$$\mathcal{L}_{\text{SAC}} = \underbrace{\text{CRPS}(F_{\text{ZINB}}, y)}_{\text{calibration}} + \lambda_s \cdot \underbrace{\log(1 + \text{Var}[Y_{\text{ZINB}}])}_{\text{sharpness}} + \lambda_r \cdot \underbrace{\text{ReLU}(r_{\text{floor}} - r)}_{\text{anti-collapse}}$$

### 10.2 ZINB Variance (Differentiable)

The variance of a ZINB random variable is:
$$\text{Var}[Y_{\text{ZINB}}] = (1-\pi)\mu + (1-\pi)\frac{\mu^2}{r} + \pi(1-\pi)\mu^2$$

Decomposition:
- $(1-\pi)\mu$: Poisson-like baseline variance
- $(1-\pi)\mu^2/r$: NB overdispersion (grows as $r \to 0$)
- $\pi(1-\pi)\mu^2$: Zero-inflation variance

The log-variance penalty $\log(1 + \text{Var})$ is used instead of raw variance because crime count variances span many orders of magnitude, and the log transform provides scale invariance.

### 10.3 The Propriety–Sharpness Tradeoff

CRPS is a strictly proper scoring rule: its unique minimiser is the true data-generating distribution $F^*$ (Gneiting & Raftery, 2007). Adding the sharpness penalty $\lambda_s \cdot \log(1 + \text{Var})$ makes the composite objective $\mathcal{L}_{\text{SAC}}$ **improper** — its minimiser $\hat{F}_{\text{SAC}}$ is a biased estimator of $F^*$.

We deliberately accept this bias because:

1. **The bias is bounded**: For small $\lambda_s$, the KL-divergence between $\hat{F}_{\text{SAC}}$ and $\hat{F}_{\text{CRPS}}$ is $O(\lambda_s)$. In our experiments $\lambda_s = 0.1$, producing negligible distributional shift.

2. **The variance reduction is substantial**: The sharpness penalty reduces $\text{Var}[Y_{\text{ZINB}}]$, producing tighter prediction intervals. For downstream conformal calibration, tighter base intervals translate directly to narrower conformalized intervals at the same coverage level.

3. **Conformal correction restores validity**: Even if $\hat{F}_{\text{SAC}}$ is slightly miscalibrated, the conformal calibration layer (§8) provides distribution-free coverage guarantees that hold regardless of the base model's propriety. The conformal guarantee depends only on exchangeability of non-conformity scores, not on the scoring rule used during training.

Thus, SAC implements a principled **calibration–sharpness tradeoff**: sacrifice a small, bounded amount of calibration (corrected by conformal post-processing) to gain substantially sharper intervals. This is the training-time analog of the inference-time tradeoff that conformal prediction manages.

## 11. EMOS Ensemble (Cross-Domain Import from Meteorology)

### 11.1 Mixture ZINB Distribution

Given $K$ trained models (seeds), each producing ZINB parameters $(\pi_k, \mu_k, r_k)$, the ensemble prediction is the mixture:

$$P_{\text{ens}}(Y = y) = \frac{1}{K}\sum_{k=1}^{K} P_{\text{ZINB}}(Y = y \mid \pi_k, \mu_k, r_k)$$

### 11.2 Mixture CDF and CRPS

The mixture CDF is simply the average of individual CDFs:
$$F_{\text{ens}}(j) = \frac{1}{K}\sum_{k=1}^{K} F_{\text{ZINB}}(j \mid \pi_k, \mu_k, r_k)$$

CRPS of the mixture is then:
$$\text{CRPS}(F_{\text{ens}}, y) = \sum_{j=0}^{K_{\max}} \left[\frac{1}{K}\sum_{k=1}^{K} F_{\text{ZINB}}^{(k)}(j) - \mathbb{1}(y \leq j)\right]^2$$

### 11.3 Expected Improvement

In weather forecasting (Gneiting et al., 2005; Raftery et al., 2005), EMOS typically improves CRPS by 10–30% over the best individual model. This is because model diversity (from different random seeds) captures uncertainty that any single model cannot. The improvement is essentially "free" — no additional training cost.

### 11.4 EMOS Weight Learning (Novel Application to ZINB)

Rather than using equal weights $w_k = 1/K$, EMOS learns optimal weights $\mathbf{w}^* \in \Delta_K$ (the probability simplex) by minimizing CRPS on a held-out calibration set:

$$\mathbf{w}^* = \arg\min_{\mathbf{w} \in \Delta_K} \frac{1}{N_{\text{cal}}} \sum_{i=1}^{N_{\text{cal}}} \text{CRPS}\!\left(F_{\text{ZINB}}(\cdot; \bar{\pi}_\mathbf{w}^{(i)}, \bar{\mu}_\mathbf{w}^{(i)}, \bar{r}_\mathbf{w}^{(i)}), y_i\right)$$

where the weighted parameters are:
$$\bar{\pi}_\mathbf{w} = \sum_{k=1}^K w_k \pi_k, \qquad \bar{\mu}_\mathbf{w} = \sum_{k=1}^K w_k \mu_k, \qquad \bar{r}_\mathbf{w} = \sum_{k=1}^K w_k r_k$$

The simplex constraint is enforced via softmax reparameterization: $w_k = \frac{\exp(\ell_k)}{\sum_j \exp(\ell_j)}$ where $\boldsymbol{\ell} \in \mathbb{R}^K$ are unconstrained logits optimized by Adam.

**Novelty**: While EMOS is standard for Gaussian distributions in meteorology (Gneiting et al., 2005), its application to ZINB distributions for crime forecasting is new. The key challenge is that CRPS for ZINB has no closed-form expression (unlike the Gaussian case), requiring numerical computation via CDF summation (§4.1).

## 12. CRPS Decomposition (Hersbach 2000)

### 12.1 The Decomposition

Following Hersbach (2000), the mean CRPS over $N$ forecast-observation pairs can be decomposed analogously to the Brier Score decomposition:

$$\overline{\text{CRPS}} = \text{REL} - \text{RES} + \text{UNC}$$

where:
- **Reliability (REL)**: Measures calibration error. A perfectly calibrated forecast has $\text{REL} = 0$. Computed via PIT (Probability Integral Transform) histogram deviation from uniformity.
- **Resolution (RES)**: Measures the forecast's ability to discriminate between different outcomes — how much the predictive distribution varies from the climatological distribution. Higher resolution is better.
- **Uncertainty (UNC)**: The inherent unpredictability of the observations. This is a property of the data, not the model. $\text{UNC} = \frac{2}{N^2} \sum_{i=1}^{N} \left(i - \frac{N+1}{2}\right) y_{(i)}$ where $y_{(1)} \leq \cdots \leq y_{(N)}$ are the sorted observations.

### 12.2 CRPS Skill Score

The CRPSS relative to climatology is:
$$\text{CRPSS} = 1 - \frac{\overline{\text{CRPS}}}{\text{UNC}} = \frac{\text{RES} - \text{REL}}{\text{UNC}}$$

A model with perfect calibration ($\text{REL}=0$) and maximum resolution achieves $\text{CRPSS} \to 1$. A model no better than climatology has $\text{CRPSS} = 0$.

### 12.3 Connection to PIT Calibration

For a calibrated model, PIT values $p_i = F_i(y_i)$ are uniformly distributed on $[0,1]$. For discrete distributions (ZINB), we use the randomized PIT:
$$p_i = F_i(y_i - 1) + U_i \cdot [F_i(y_i) - F_i(y_i - 1)], \qquad U_i \sim \text{Uniform}(0,1)$$

The reliability component is directly linked to deviation of the PIT histogram from uniformity:
$$\text{REL} \propto \sum_{k=1}^{B} \left(\hat{o}_k - \frac{1}{B}\right)^2$$
where $\hat{o}_k$ is the observed frequency in PIT bin $k$ and $B$ is the number of bins.

## 13. Rolling Adaptive ECRC Algorithm

### 13.1 Problem Setting

Standard conformal prediction treats all test observations as exchangeable. In temporal crime forecasting, we observe test data sequentially (week by week). The Rolling Adaptive ECRC exploits this structure.

### 13.2 Algorithm

**Input**: Calibrated $\hat{\alpha}_g$ for each group $g \in \{1, \ldots, G\}$; learning rate $\gamma > 0$; calibration scores $\{s_i^{(g)}\}$

**For each test window $w = 1, 2, \ldots, W_{\text{test}}$:**

1. **Predict**: Compute conformal intervals using current $\hat{\alpha}_g^{(w)}$:
   $$\hat{q}_g^{(w)} = \text{Quantile}\!\left(\{s_i^{(g)}\}, \min\left(1.0, (1 - \hat{\alpha}_g^{(w)})(1 + 1/n_g^{(w)})\right)\right)$$
   $$C_g^{(w)}(x) = \{y : s(x, y) \leq \hat{q}_g^{(w)}\}$$

2. **Observe**: Receive true counts $y^{(w)}$ for window $w$.

3. **Evaluate**: Compute empirical miscoverage error:
   $$\text{err}_g^{(w)} = 1 - \frac{1}{n_g^{(w)}} \sum_{i \in \text{group}_g} \mathbb{1}\!\left(y_i^{(w)} \in C_g^{(w)}(x_i)\right)$$

4. **Update**: Adjust the target coverage per group:
   $$\hat{\alpha}_g^{(w+1)} = \hat{\alpha}_g^{(w)} + K_P e_g^{(w)} + K_I I_g^{(w)} + K_D D_g^{(w)}$$
   where $e_g^{(w)} = \alpha - \text{err}_g^{(w)}$. This implements a PID controller for conformal inference (Angelopoulos et al. 2023).

5. **Record**: Store $\hat{\alpha}_g^{(w+1)}$ and coverage for convergence diagnostics.

### 13.3 Convergence Guarantee

Under mild regularity conditions (bounded scores, ergodic stationarity), the rolling average coverage converges:
$$\frac{1}{W} \sum_{w=1}^{W} \text{err}_g^{(w)} \xrightarrow{W \to \infty} \alpha \quad \text{for each group } g$$

The convergence rate depends on $\gamma$: smaller $\gamma$ gives slower adaptation but lower variance in the coverage trajectory.

## 14. Statistical Significance Testing

### 14.1 Diebold-Mariano Test

To establish that CIVIC-SAFE significantly outperforms baselines, we use the Diebold & Mariano (1995) test.

**Setup**: Given per-timestep CRPS values $\{L_{1,t}\}_{t=1}^T$ and $\{L_{2,t}\}_{t=1}^T$ for two competing forecasts, define the loss differential:
$$d_t = L_{1,t} - L_{2,t}$$

**Test statistic** (with HAC standard errors):
$$\text{DM} = \frac{\bar{d}}{\sqrt{\hat{\sigma}_d^2 / T}} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{under } H_0: E[d_t] = 0$$

where $\hat{\sigma}_d^2$ is the Newey-West (1987) HAC estimator of the long-run variance:
$$\hat{\sigma}_d^2 = \hat{\gamma}_0 + 2 \sum_{j=1}^{h} \left(1 - \frac{j}{h+1}\right) \hat{\gamma}_j$$

with autocovariance $\hat{\gamma}_j = \frac{1}{T} \sum_{t=j+1}^{T} (d_t - \bar{d})(d_{t-j} - \bar{d})$ and truncation lag $h = \lfloor T^{1/3} \rfloor$.

### 14.2 Temporal Block Bootstrap

As a complementary non-parametric test, we use the stationary block bootstrap (Politis & Romano, 1994):

1. Choose block length $\ell = \lceil T^{1/3} \rceil$
2. For $b = 1, \ldots, B$ bootstrap replicates:
   - Sample $\lceil T/\ell \rceil$ blocks of length $\ell$ with replacement from $\{d_t\}$
   - Compute $\bar{d}^{*(b)}$ from the bootstrap sample
3. Two-sided p-value: $\hat{p} = \frac{1}{B} \sum_{b=1}^{B} \mathbb{1}(|\bar{d}^{*(b)} - \bar{d}| \geq |\bar{d}|)$

This accounts for temporal dependence in the loss differentials without parametric assumptions on their distribution.

## 15. Anomaly Skill Coefficient (ASC)

### 15.1 Motivation

Predictive models are often evaluated on absolute errors, but in spatiotemporal forecasting, it is crucial to measure whether the model successfully predicts *anomalies* (deviations from historical means).

### 15.2 Definition

For demographic group $g$, the ASC is computed over all spatial units in group $g$ and all test time steps $t \in T_{\text{test}}$:

$$\text{ASC}_g = \text{Corr}_{s \in \text{group}_g, t \in T_{\text{test}}}(\hat{y}_{s,t} - \bar{y}_{s}^{\text{hist}}, \; y_{s,t} - \bar{y}_{s}^{\text{hist}})$$

where:
- $\hat{y}_g$ = model predictions for group $g$
- $y_g$ = observed counts for group $g$
- $\bar{y}_g^{\text{hist}}$ = historical training-period mean for group $g$

**Interpretation**:
- $\text{ASC}_g = 1$: Perfect anomaly prediction skill.
- $\text{ASC}_g = 0$: No skill better than historical mean predicting.
- $\text{ASC}_g < 0$: Actively inverse skill.

### 15.3 Bias Amplification Score

$$\text{BAS}_g = \frac{\text{Var}(\hat{y}_g)}{\text{Var}(y_g)} - 1$$

- $\text{BAS}_g > 0$: Model over-predicts variance for group $g$ (amplifies signal)
- $\text{BAS}_g < 0$: Model under-predicts variance (dampens signal)
- $\text{BAS}_g = 0$: Model preserves the natural variance structure

### 15.4 Aggregate Skill Disparity

The overall skill disparity is measured as the maximum absolute difference in ASC across groups:
$$\Delta_{\text{ASC}} = \max_{g, g'} |\text{ASC}_g - \text{ASC}_{g'}|$$

A system exhibits skill parity if $\Delta_{\text{ASC}}$ is near 0.

## 16. Post-Hoc Recalibration

### 16.1 Affine ZINB Recalibration

After training, the ZINB parameters may be slightly miscalibrated. We learn an affine correction on the calibration set:

$$\tilde{\mu} = a_\mu \cdot \mu + b_\mu, \qquad \tilde{r} = a_r \cdot r + b_r$$

The parameters $(a_\mu, b_\mu, a_r, b_r)$ are learned by minimizing CRPS on the calibration set:
$$\theta^* = \arg\min_\theta \frac{1}{N_{\text{cal}}} \sum_{i=1}^{N_{\text{cal}}} \text{CRPS}(F_{\text{ZINB}}(\cdot; \pi_i, \tilde{\mu}_i, \tilde{r}_i), y_i)$$

The recalibrator is initialized at the identity mapping $(a_\mu = 1, b_\mu = 0, a_r = 1, b_r = 0)$ to ensure that no correction is applied if the model is already well-calibrated.

### 16.2 Why CRPS Not NLL?

CRPS optimization for recalibration is strictly preferred over NLL because:
1. CRPS is a proper scoring rule that penalizes both miscalibration and lack of sharpness
2. NLL can be manipulated by overfitting the variance parameter without improving the forecast
3. The recalibrated model can be directly evaluated on the same metric, ensuring consistency between training and evaluation objectives
