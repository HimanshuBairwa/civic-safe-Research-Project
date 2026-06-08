# CIVIC-SAFE Mathematical Specification

This document provides the formal mathematical specification for the CIVIC-SAFE architecture, including the distributional loss functions, spatial attention mechanisms, bias-mitigation regularisation, and evaluation metrics.

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

### 3.1 Gated Cross-Attention
Each head $k$ produces an attention distribution over the feature dimensions:
$$\mathbf{g}_k = \text{softmax}\left(\frac{\mathbf{W}_Q^{(k)}\mathbf{x} \cdot (\mathbf{W}_K^{(k)}\mathbf{x})^\top}{\tau}\right)$$
Where $\tau$ is the temperature parameter controlling sparsity.

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
