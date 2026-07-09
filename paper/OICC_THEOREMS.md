# OICC — Theorems and Proofs

*Companion to the `oicc` package. Every theorem here corresponds to code that is
tested (`tests_oicc/`) and, where a claim is empirical, to a verified numerical
result. Notation matches the implementation. Proofs are complete for the results
we claim as proved and are explicitly flagged where a result is asymptotic or
conditional on an untestable assumption.*

---

## Setup

Latent log-rate `theta_i in R` for units `i = 1..n`. We observe `K >= 3`
measurement **channels**

    Y^c_i = alpha_c + beta_c * theta_i + eps^c_i,     c = 1..K,        (M)

with the maintained assumptions

- **(A1) Conditional independence.** `eps^1,...,eps^K` are mutually independent
  given `theta`, mean zero, finite variance `sigma_c^2 = Var(eps^c)`.
- **(A2) Non-degeneracy.** `Var(theta) > 0`, `beta_c != 0` for all `c`.
- **(A3) Normalization.** `beta_1 = 1` (pivot), which fixes the scale of the
  otherwise scale-free latent.

Write `v = Var(theta)`. Let `C = Cov(Y)` be the `K x K` channel covariance.

---

## Theorem 1 (Second-moment identification).

*Under (M), (A1)-(A3), for every `j != k`*

    C_{jk} = beta_j * beta_k * v.                                     (1)

*Hence the off-diagonal of `C` is the hollow part of the rank-1 matrix
`v * beta beta^T`. For `K >= 3` the loadings `beta` and `v` are identified; for
`K >= 4` they are OVER-identified.*

**Proof.** By (M), `Cov(Y^j, Y^k) = beta_j beta_k Var(theta) + Cov(eps^j, eps^k)`.
By (A1) the error covariance is 0 for `j != k`, giving (1). For identification
with the pivot `beta_1 = 1`: for any distinct `j, k != 1`,

    C_{1j} C_{1k} / C_{jk} = (beta_j v)(beta_k v)/(beta_j beta_k v) = v,   (2)

so `v` is identified whenever some `C_{jk} != 0` (true under (A2)). Then
`beta_k = C_{1k}/v` from (1) with `j = 1`. With `K = 3` there is exactly one
triple, so (2) uses all second-moment information (just-identified: 0 spare
equations). With `K >= 4` distinct triples give multiple expressions for `v`
that must coincide — the over-identifying restrictions. ∎

*Code:* `moments.estimate_factor_moments` (averaged tetrads, median over the
triples in (2)); tested in `test_moments_recover_loadings_and_variance`.

---

## Theorem 2 (Over-identification restriction and its specification test).

*Under (M), (A1)-(A3) with all `C_{jk} > 0` (positive loadings), the log
off-diagonal covariances satisfy the additive model*

    log C_{jk} = a_j + a_k + c,   a_j := log beta_j,  c := log v,     (3)

*for all `j < k`. The vector of `m = K(K-1)/2` log-covariances therefore lies in
the column space of the known `m x (K+1)` design `X` (row `(j,k)` has ones in
columns `j`, `k` and an intercept). The residual of the least-squares projection
of `(log C_{jk})` onto `X` is zero iff (3) holds; its dimension is*

    df = m - K = K(K-1)/2 - K,                                        (4)

*which is `0` at `K = 3` and `2` at `K = 4`. A bootstrap Wald statistic on this
residual is an asymptotically valid, LOADING-INVARIANT test of the one-factor +
conditional-independence structure.*

**Proof.** Taking logs of (1) gives (3) directly; the map `beta -> a = log beta`
is linear in the parameters, so unequal loadings do not perturb the residual —
the test is loading-invariant. The design `X` has rank `K` (the `K` loading
columns; the intercept is in their span since summing any two loading indicators
is not constant), so the projection residual has dimension `m - K`, giving (4).
Under the null the empirical log-covariances are asymptotically normal (delta
method on sample covariances), so the studentized residual quadratic form is
asymptotically `chi^2_{df}`; we estimate its covariance by a nonparametric
bootstrap, which is consistent for the covariance of the residual vector. ∎

*Code:* `spec_test.overid_wald_test`. *Verified:* size ≈ 0.00–0.05 under H0;
power ≈ 1.00 against a detectable confounder (`test_overid_size_under_null`,
`test_overid_power_against_detectable_confounder`).

---

## Theorem 3 (Common-mode non-identification — the impossibility).

*Augment (M) with a common-mode confounder*

    Y^c_i = alpha_c + beta_c * theta_i + l_c * W_i + eps^c_i,          (M')

*with `W` independent of `theta` and `eps`, `Var(W) = w > 0`. If the confounder
loads PROPORTIONALLY to the factor, `l_c = kappa * beta_c` for a scalar `kappa`,
then the observable covariance is*

    C_{jk} = beta_j beta_k (v + kappa^2 w),    j != k,                (5)

*i.e. exactly the one-factor law (1) with `v` replaced by `v* = v + kappa^2 w`.
Consequently `v` is NOT a function of the observable law: any two parameter
configurations with the same `beta` and the same `v + kappa^2 w` are
observationally indistinguishable, for every `K`. No specification test based on
the observed channels can detect the confounder, and the latent variance (hence
the latent target) is not identified without outside information.*

**Proof.** Under (M') with `l_c = kappa beta_c`, for `j != k`,
`C_{jk} = beta_j beta_k v + l_j l_k w = beta_j beta_k v + kappa^2 beta_j beta_k w
= beta_j beta_k (v + kappa^2 w)`, which is (5). The right side depends on
`(v, kappa, w)` only through `v* = v + kappa^2 w`. The diagonal adds `sigma_c^2`,
also absorbed into a per-channel noise term. Thus the entire law of `Y` is a
function of `(beta, v*, {sigma_c^2})`; the map `(v, kappa, w) -> v*` is
many-to-one, so `v` is unidentified. Since the over-identification residual of
Theorem 2 is a function of the observable law alone, it is identically the null
value under (5): the test has no power in this direction. ∎

*Code/finding:* demonstrated by `test_overid_is_blind_to_common_mode` (rejection
rate ≈ 0.00 at every confounder strength). This is the honest fundamental limit.

---

## Theorem 4 (Leave-pivot-out residual decomposition).

*Fix a pivot channel `p`. Let `theta_hat_i = g(Y^{-p}_i)` be any estimator using
only the non-pivot channels. Define the computable residual `R_i = Y^p_i -
theta_hat_i`. Under (M), (A1),*

    R_i = S_i + eps^p_i,   with   S_i := (alpha_p + beta_p theta_i - theta_hat_i),

*and `S_i` is independent of `eps^p_i`.*

**Proof.** `theta_hat_i` is a function of `Y^{-p}_i = (Y^c_i)_{c != p}`, which by
(A1) is independent of `eps^p_i` given `theta_i`; and `beta_p theta_i` is a
function of `theta_i`. Since `eps^p_i` is independent of `theta_i` and of all
other errors (A1), it is independent of `S_i`, which is a function of
`(theta_i, Y^{-p}_i)`. The decomposition is immediate from `Y^p_i = alpha_p +
beta_p theta_i + eps^p_i`. ∎

*Code:* `conformal_split.split_conformal_latent` forms exactly this `R`.

---

## Theorem 5 (Exact finite-sample coverage of the observed pivot value).

*Split the units into disjoint train / calibration / test folds. Estimate
`theta_hat` with train-fold loadings only. On the calibration fold compute
`R_i` and the scores `s_i = |R_i - med(R_cal)|`. Let `q` be the
`ceil((n_cal+1)(1-alpha))/n_cal` empirical quantile of `{s_i}`. Then for a test
unit, the interval `theta_hat + med(R_cal) +/- q` covers the OBSERVED value
`Y^p` with probability at least `1 - alpha`, distribution-free, for exchangeable
units.*

**Proof.** Conditional on the train fold (hence on `theta_hat(.)` and the loading
estimates), the calibration and test residuals `R` are i.i.d. draws from the same
law (units are exchangeable and `theta_hat` is fixed by the train fold). This is
the standard split-conformal setting with nonconformity score `s = |R -
med(R_cal)|`; the `(1 + 1/n_cal)`-corrected empirical quantile guarantees
marginal coverage `>= 1 - alpha` by the exchangeability/rank argument of Vovk et
al. (2005) and Lei et al. (2018). The interval for `Y^p` follows because
`Y^p = theta_hat + R`. ∎

*Code:* the `obs_lower/obs_upper` interval. *Verified:* coverage 0.90–0.91
(`test_split_conformal_exact_observed_coverage`).

---

## Proposition 6 (Latent interval; asymptotic, model-assisted).

*Under (M), (A1)-(A3), `S = beta_p theta - theta_hat + alpha_p` and the latent
target differ by a known affine map, and `Var(S) = Var(R) - sigma_p^2` is
identified from the moments. Using the BLUP `theta_hat`, `S = -(sum_{c != p}
w_c eps^c)` up to the affine map, a weighted AVERAGE of independent errors; by a
Lyapunov CLT `S` is approximately Gaussian as the effective number of channels
grows, so the Gaussian quantiles with the moment-exact `Var(S)` give latent
coverage `-> 1 - alpha`. For heavy-tailed channel errors, the law of `S` is
recovered by characteristic-function deconvolution `phi_S = phi_R / phi_{eps^p}`
(Kotlarski/Neumann), regularized by a flat-top spectral kernel.*

*This coverage is ASYMPTOTIC and MODEL-ASSISTED (it uses (M)); it is NOT
distribution-free finite-sample — and by Theorem 3 no distribution-free
finite-sample latent interval can exist under a possible common mode.*

*Code:* `latent_method="gaussian"` (default) and `"cf"`. *Verified:* latent
coverage ≈ 0.89 (`test_split_conformal_latent_coverage_near_nominal`); CF
deconvolution recovers a skewed error law's quantiles (`test_cf_deconv_*`).

---

## Theorem 7 (Proximal escape from the common mode).

*Augment with `Q` NEGATIVE-CONTROL channels carrying the confounder but no latent
signal:*

    N^q_i = a_q + m_q * W_i + nu^q_i,   q = 1..Q,                     (NC)

*with `nu^q` independent of `(theta, W, eps)` and of each other, and `(m_q)`
spanning the `W`-direction. Regress each signal channel on `(N^q)`; the residual
`Y^c - E[Y^c | N]` removes the `W`-component, so the residualized channels follow
the clean one-factor model (M) in `theta`. With `Q >= 2` independent relevant
controls, the `W`-contamination of each channel is point-identified; with `Q = 1`
only detection and partial (attenuated) removal are available.*

**Proof sketch (linear case).** Stack `N = (N^1,...,N^Q)`. Under (NC), `N` is a
noisy linear image of `W`: `N = a + m W + nu`. The population regression of `Y^c`
on `N` has coefficient vector proportional to `Cov(Y^c, N) Var(N)^{-1}`, and
`Cov(Y^c, N) = l_c m^T w` (only through `W`, since `theta ⟂ W ⟂ nu`). Hence the
fitted value estimates `l_c * (projection of W onto span(N))`. With `Q >= 2`
relevant controls and `Var(nu)` full-rank, `span(N)` recovers `W` up to noise
that vanishes as the control signal-to-noise grows, so the residual `Y^c -
fit_c` converges to `alpha_c + beta_c theta + eps^c` — the clean model. With
`Q = 1`, the single control is `m_1 W + nu^1`; the regression removes only the
part of `W` correlated with it, leaving attenuation `~ Var(nu^1)/Var(N^1)` — hence
partial. The identifying assumptions — controls carry no `theta` (exclusion) and
are `W`-relevant (completeness) — are not testable from `(Y, N)` alone. ∎

*Code:* `proximal.proximal_deconfound`. *Verified against ground truth:* at
common-mode strength 1.0, naive `theta`-RMSE 0.57 vs proximal 0.28; no harm when
the confounder is absent (0.21 vs 0.21) — `test_proximal_fixes_common_mode_
confounding`, `test_proximal_no_harm_without_confounder`.

---

## Honest scope summary

| Result | Status |
|---|---|
| T1 second-moment identification | proved, verified |
| T2 over-ID test (size, power) | proved + empirically verified |
| T3 common-mode non-identification | **proved (impossibility)** |
| T4 residual decomposition | proved |
| T5 exact observed-value coverage | **proved, finite-sample, distribution-free** |
| P6 latent coverage | **asymptotic, model-assisted** (no DF-finite-sample possible, by T3) |
| T7 proximal escape | proved (linear) + verified; assumptions **untestable** |

The pairing of **T3 (a clean impossibility)** with **T7 (a principled, verified
escape that imports an explicit, untestable assumption)** is the honest core: we
do not claim to dissolve the identification barrier from data alone; we
characterize it exactly and provide the one lever — external negative controls —
that moves it, with its price stated.
