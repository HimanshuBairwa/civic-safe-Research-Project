# Reviewer Defense Dossier — CIVIC-SAFE

Prepared answers to the ten hardest questions a hostile TPAMI, TKDE, KDD or
NeurIPS reviewer can put to this paper. Every number below was read from a file
under `outputs/` or produced by a script under `scripts/` and is cited to its
source. Where the evidence does not exist, the answer says so rather than
improvising — a defense that overclaims is worse than no defense, because it hands
the reviewer a second finding.

Q1–Q7 address the paper as submitted. Q8–Q10 address attack surface that the
paper's own later additions opened: the disparity-attribution corollary invites a
charge of unfalsifiability (Q8), the negative control invites the charge that it is
circular (Q9), and the abstention gate invites a post-selection-inference objection
(Q10). Anticipating the consequences of one's own new results is the part of this
exercise that is easy to skip.

Two of these questions rest on premises that are **false about our own system**.
Q5 describes a mechanism our code does not implement, and Q10's earlier
explanation of the low-gain abstention cost was wrong about the mechanism until it
was instrumented. Those are answered by correcting the premise, because a reviewer
who reads the released code will find the same thing, and it is far better to have
said it first. Both are now fixed in the manuscript.

Verification date: 2026-09-05. Manuscript: `paper/civic_safe_ieee.tex`.

---

## Q1. Why community areas and police precincts rather than census tracts or street segments?

**Short answer.** At 77 community areas the drug category is already 50.9% zeros;
at census-tract resolution it would be roughly 90%, and the conformity score
distribution collapses onto a single lattice point where the calibration step has
nothing left to estimate. We are already operating near the edge of where discrete
conformal prediction says anything, and Section V-C shows the edge.

**The measured sparsity**, computed from `data/raw/chicago/*.parquet`
(1,326,056 incidents, 77 units, 313 weeks):

| category | mean count / cell-week | zero cells | max |
|---|---:|---:|---:|
| drug | 2.15 | **50.89%** | 68 |
| violent | 20.92 | 1.09% | 169 |
| property | 31.95 | 0.20% | 532 |

This also answers a sub-question a careful reviewer asks next — *is
zero-inflation even justified at this resolution?* For violent and property
crime, barely: 1.1% and 0.2% zeros. The ZINB atom earns its place on **drug**
crime, at 50.9%, and the model is fit jointly across all three categories, so the
head has to represent that regime. Chicago's Brier score on the zero event is
0.0592 (`point_forecast_metrics.brier_zero`), confirming the atom is doing
measurable work.

**What finer resolution would cost.** Counts scale roughly inversely with unit
count. Chicago has about 801 census tracts against 77 community areas, a factor of
10.4:

| resolution | implied drug mean / cell-week | implied violent mean |
|---|---:|---:|
| community areas (77) | 2.15 | 20.92 |
| census tracts (~801) | ~0.21 | ~2.10 |
| block groups (~2,200) | ~0.08 | ~0.76 |
| street segments (~28,000) | ~0.006 | ~0.06 |

At tract level the drug category is near-degenerate. This is not a hypothetical
concern for us: **we already observe the pathology at the current, coarser
resolution.** Section V-C reports that four NYC calibration methods — split CP,
weighted CP, Mondrian and equalized coverage — return *numerically identical*
coverage, width and disparity (0.9326 / 18.57 / 0.0201), because CQR scores on
this panel are integers and the methods' differing quantile choices land on the
same lattice point. Push resolution finer and that degeneracy spreads from one
city's subset of methods to all of them. The randomized-PIT variant
(`randomized_split_cp`) exists precisely to recover a non-degenerate scale, and
Section V-C is where we show why it is needed.

**Boundary stability.** Chicago's 77 community areas have been fixed since the
1920s. NYC police precincts are the administrative unit on which patrol is
actually allocated, which matters because the policy simulation of Section V-D
allocates to exactly those units — a finer analysis grid would predict at a
resolution no dispatcher can act on. Census tracts are redrawn each decennial
census, which would fracture a 2018–2023 panel mid-window and make the
chronological split incomparable across its own blocks.

**Demographic audit precision.** The fairness audit stratifies on ACS
composition. ACS margins of error grow sharply at tract and block-group level, so
a finer grid would degrade the very group assignments the 0.03 disparity ceiling
is enforced against — we would be measuring fairness against noisier labels while
claiming tighter spatial resolution.

**Concession to volunteer.** This is an aggregation choice and the modifiable
areal unit problem applies: community-area averages hide within-area
heterogeneity, and a hotspot occupying two blocks of a large area is invisible to
us. We do not claim the resolution is optimal, only that it is the finest at which
the distributional and conformal machinery in this paper remains meaningful. A
principled treatment of resolution — nested calibration across scales — is future
work we have not done.

---

## Q2. How robust is Theorem 2 when the policy response deviates from a pure power law?

**Short answer.** Theorem 2(i)'s exactness genuinely requires the power-law fixed
point, and we do not pretend otherwise. Section III-C converts that assumption
into a measurable sensitivity envelope: under a bounded per-cell deviation, a
Γ-inflated interval retains nominal latent coverage for *every* admissible
recording model, and we report both the degradation without inflation and the
width cost with it.

**What the theorem actually assumes.** Theorem 2 is stated for
`μ_s = λ_s(μ_s/M)^κ`. Under that fixed point the deflation
`λ̂_s = μ_s/(μ_s/M)^κ` is an exact algebraic inverse, so `λ̂_s = λ_s` identically —
part (i). Theorem 2(iii) then bounds the error from a wrong *gain*:
`log λ̂_s − log λ_s = (κ − κ̂)log(μ_s/M)`, bounded on the retained set by
`|κ − κ̂|·(log m̄)/κ̂`. That covers a misestimated κ. It does **not** cover a wrong
functional form, and a reviewer is right to press there.

**The sensitivity model (Section III-C).** Assume the true multiplier lies within
a factor Γ ≥ 1 of the assumed one for every cell,
`m_true(s)/m_s ∈ [Γ⁻¹, Γ]`. This implies `λ_s/λ̂_s ∈ [Γ⁻¹, Γ]` directly, so an
interval taking its lower endpoint at `λ̂_s/Γ` and upper at `λ̂_s·Γ` covers a
`Poisson(λ_s)` draw for every recording model in the band. Γ = 1 recovers the
plain interval. This is Rosenbaum's marginal sensitivity model transplanted to a
recording mechanism; implementation in
`src/civicsafe/theory/correction_robustness.py`, 7/7 tests passing in
`tests/test_correction_robustness.py`.

**Measured**, via `scripts/misspecification_sensitivity.py` at κ = 0.6, target
0.90, 6000 cells × 8 trials, building worlds whose true multiplier deviates from
the assumed one by a bounded per-cell factor:

| deviation factor | plain corrected coverage | Γ-inflated coverage | width ratio |
|---:|---:|---:|---:|
| 1.0 | 0.9420 | 0.9420 | 1.00 |
| 1.3 | 0.9219 | 0.9844 | 1.39 |
| 1.6 | 0.8780 | 0.9918 | 1.73 |
| 2.0 | 0.8098 | 0.9943 | 2.13 |
| 3.0 | 0.6787 | 0.9976 | 3.05 |

The plain interval falls below the 0.90 target once the deviation factor reaches
1.6 and reaches 0.679 at 3.0. The inflated interval never does. So the honest
statement is: misspecification hurts, we can measure exactly how much, and there
is a valid remedy whose price is a known widening.

**On stability under heterogeneity.** A related worry is whether the κ < 1
threshold survives when cells differ. It does, exactly. In log coordinates the
fixed-point iteration `log μ ← log λ + κ(log μ − log M)` has Jacobian
`κ(I − 1wᵀ)` with `w_j = μ_j/(SM)` summing to one. That matrix is κ times a
rank-one perturbation of the identity whose eigenvalues are 1 with multiplicity
S−1 and 0, so its spectral radius is **exactly κ** regardless of how
heterogeneous the cells are, and the iteration contracts precisely when κ < 1.
Verified numerically at κ ∈ {0.3, 0.5, 0.7, 0.85, 0.95}, matching to machine
precision.

**Concession to volunteer.** The Γ model assumes the deviation is *bounded*.
Unbounded misspecification — a recording mechanism that is not merely a distorted
power law but structurally different, say non-monotone in attention — is outside
the envelope and we make no claim there. Theorem 1(ii) is likewise stated only for
constant κ; for general smooth φ, g the gain varies with μ and no single global
exponent exists, which is why the manuscript presents κ as a *local* elasticity
at the fixed point rather than a universal constant.

---

## Q3. Does 85% abstention at κ = 0.85 make the system practically useless?

**Short answer.** In that regime, largely yes, and we say so in the manuscript's
Ethical Considerations rather than defend it. But the comparison the question
implies is the wrong one. The alternative is not 93% coverage on all cells — it is
**16.2% coverage on all cells, delivered with no warning.**

**The numbers** (the latent-coverage table in Section V-E of the manuscript, from `outputs/figure_data/fig10_latent_correction.json`):

| κ | naive latent coverage | corrected | cells retained |
|---:|---:|---:|---:|
| 0.00 | 0.950 | 0.949 | 75% |
| 0.30 | 0.903 | 0.952 | 100% |
| 0.50 | 0.780 | 0.948 | 95% |
| 0.70 | 0.502 | 0.937 | 56% |
| 0.85 | **0.162** | **0.930** | **15%** |

**Why abstention is the mathematically necessary cost, not a design failure.**
Theorem 2(iii) gives the error in the deflated rate as
`(κ − κ̂)log(μ_s/M)`, and `log(μ_s/M)` is unbounded — a cell recorded far from the
city mean amplifies any error in the gain without limit. Abstention is the
truncation that makes the sensitivity bound finite: on the retained set
`m_s ∈ [m̄⁻¹, m̄]` the log-error is at most `|κ − κ̂|(log m̄)/κ̂`. Remove abstention
and there is no bound to state. Theorem 2(iv) then makes the guarantee explicitly
conditional on retention, which is what turns "93% coverage" from a
denominator-shifting claim into a true conditional statement. A system that
answered everywhere at κ = 0.85 would be the naive row: confident, wrong 84% of
the time, and undetectable by any validation on records.

**The operational reading.** At κ = 0.85 a deployment mostly returns "I do not
know." That is the correct output and simultaneously a reason not to ship there —
which is exactly what the Ethical Considerations section says, naming the
abstention rate rather than the coverage number as the operational figure of
merit. The useful regime is κ ≤ 0.5, where retention is 95–100% and coverage is
restored from 0.780 to 0.948 at essentially no abstention cost. Whether real
jurisdictions sit at κ = 0.3 or κ = 0.85 is an empirical question our field DiD
was underpowered to answer.

**Concession to volunteer, unprompted.** The abstention rule is inefficient at the
*other* end. At κ = 0 it abstains on 25% of cells while improving coverage not at
all (0.950 → 0.949). The rule is tuned for the high-gain regime and overpays when
there is no feedback to correct. That is a real defect, it is in the manuscript,
and a gain-adaptive threshold is obvious future work we have not done.

---

## Q4. Why did classical ZINB diverge on NYC while CIVIC-SAFE stayed stable?

**Short answer.** The classical baseline is an unregularized maximum-likelihood
fit with no floor on the dispersion parameter and a 30-iteration optimizer cap.
Our parameterization makes dispersion collapse unreachable by construction. But we
also concede the baseline may simply be under-fit, and the manuscript already
excludes it from ranking rather than claiming credit for the gap.

**The measured divergence** (`outputs/baselines/{city}_baselines.csv`):

| | Chicago | NYC |
|---|---:|---:|
| classical ZINB CRPS | 5.5432 | **924.0974** |
| MAE | 6.8976 | 1844.0397 |
| RMSE | 18.8777 | 4270.6474 |

Note `compute_metrics` in `scripts/baselines.py` clips predictions at 1e4, so the
raw fitted values were larger still before clipping.

**The mechanism.** Negative binomial variance is `μ + μ²/r`, which diverges as
`r → 0`. An unconstrained likelihood can therefore buy tolerable NLL by inflating
the tail — the heavier the tail, the more probability it assigns to any
observation — while the predictive distribution becomes useless and CRPS explodes.
This is the same pathology the manuscript documents in Section IV-C as
*r-collapse*, where MAE improves while CRPS degrades.

**Our two guards**, both in the manuscript. A hard floor in the parameterization,
`r = softplus(·) + 0.1` (Section IV-A), which makes `r → 0` unreachable rather
than merely penalized. And a per-cell penalty (Section IV-C, eq. 6)
`L_r-reg = λ_r·mean_i ReLU(r_reg − r_i)` with `r_reg = 0.5`, `λ_r = 0.1`. Per-cell
is load-bearing: a batch-mean penalty lets some cells collapse while others
compensate, so the average looks acceptable while individual predictive
distributions are worthless.

**The baseline's configuration**, stated plainly because a reviewer will check it.
`scripts/baselines.py` uses `statsmodels`
`ZeroInflatedNegativeBinomialP(y, X, exog_infl=X)` fit with
`method='bfgs', maxiter=30`. That is an unregularized MLE with no dispersion
floor, and 30 BFGS iterations is a low cap for a zero-inflated model on a sparse
overdispersed panel.

**Concession to volunteer, and it matters.** Because of that cap we cannot cleanly
separate "the estimator is structurally fragile" from "this particular fit did not
converge." The observed divergence is *consistent* with dispersion collapse and it
is the failure mode our parameterization is explicitly designed to prevent, but we
did not instrument statsmodels' internals to confirm that `alpha` specifically ran
away. We therefore do not claim our margin over classical ZINB as evidence of
method superiority: the main benchmark table footnotes the NYC entry as diverged and **excludes it
from ranking**. Our substantive comparison is against the rolling historical
average and the four deep baselines, and a reviewer is entitled to ask us to
re-fit the ZINB baseline with a higher cap and a dispersion floor. We would expect
it to land near the Chicago figure — poor but finite — and nothing in our claims
depends on that.

---

## Q5. How does the Jensen-Shannon diversity penalty prevent demographic leakage?

**It does not, and the premise needs correcting.** This is the question we most
want to answer before a reviewer asks it, because the answer required fixing the
manuscript.

**What the penalty actually does.** In `src/civicsafe/models/feature_mixer.py`,
`_diversity_loss` computes the Jensen-Shannon divergence between every *pair of
attention heads* and penalizes pairs whose JSD falls below
`collapse_threshold = 0.1`. Its purpose, per the module docstring, is "to prevent
degenerate solutions where all heads learn identical attention patterns." It is a
**head-collapse regularizer**. Protected attributes appear nowhere in it. It
cannot prevent demographic leakage because it never references demographics.

**Where demographic invariance would come from, and why it is not active.** The
repository does implement the right mechanism: `models/adversarial_head.py`
provides a gradient-reversal layer and adversarial discriminator in the manner of
Ganin et al., which sets up a minimax game where the discriminator predicts the
demographic group from the representation while the encoder learns to defeat it.
It is imported by `models/civicsafe_model.py` and instantiated only when
`num_adv_classes > 0`. **`scripts/train.py` never passes that argument**, so it
defaults to 0, `self.adv_head` is `None`, and the trained predictor contains no
representation-level fairness mechanism at all. The trainer's adversarial loss
term is correspondingly gated on `"adv_logits" in output`, which is never true.

**What we corrected.** The Architecture section previously claimed the feature
mixer "discourages routing predictions through demographic covariates — a partial
guard against proxy discrimination." That was unsupported and is now removed. The
manuscript states what the penalty does, notes that the adversarial head exists
but is disabled in the reported configuration, and says explicitly that the
trained predictor carries no representation-level guard against proxy
discrimination. The Limitations section carries the same statement.

**What fairness we do demonstrate.** All of it is downstream and post hoc, and it
is real. Calibrator selection enforces a demographic coverage-disparity ceiling of
0.03 fixed before we looked at results (Section IV-D), audited in Section V-C. The
constraint is load-bearing rather than decorative: rolling adaptive ECRC has the
best disparity in both cities (0.0013, 0.0046) and the narrowest intervals (13.88,
16.31), and a width-minimizing rule would have selected it — its coverage of
89.30% and 89.18% falls below the floor, so the policy rejected it. Selected
disparities are 0.0238 (Chicago) and 0.0286 (NYC), and we flag that NYC clears the
ceiling by only 0.0014.

**The honest framing for a reviewer.** This paper's fairness contribution is
distributional coverage equity under a pre-registered constraint, plus the
exposure-disparity reduction from the feedback correction. It is not
representation-level demographic invariance. Enabling the adversarial head and
measuring whether it buys anything beyond the coverage constraint is the obvious
next experiment and we have not run it.

---

## Q6. Why is the rolling historical average the binding baseline rather than frozen climatology?

**Short answer.** Because a frozen baseline is a strawman, and because the rolling
average is the hardest baseline in the study — it beats every deep model we ran,
in both cities, eight comparisons out of eight.

**The two skill scores**, from `skill_scores` in the conformal results JSONs,
which record `crpss_binding_baseline: "ha_rolling"` explicitly:

| | Chicago | NYC |
|---|---:|---:|
| CRPSS vs **rolling** HA | 0.0360 | 0.0494 |
| CRPSS vs **frozen** HA | 0.2713 | 0.3165 |

Reporting the frozen figure would inflate our headline skill roughly sevenfold.
Any deployed baseline updates its average as weeks arrive; a frozen climatology
corresponds to a system nobody would field. We lead with 3.6% and 4.9%.

**The fact that justifies the choice.** The rolling HA beats all four deep
spatiotemporal baselines on both cities:

| baseline | Chicago CRPS | NYC CRPS |
|---|---:|---:|
| **rolling HA** | **2.9322** | **3.3034** |
| LSTM-NB | 3.1037 | 3.3426 |
| TFT-ZINB | 2.9456 | 3.4244 |
| GraphWaveNet | 3.1189 | 3.8857 |
| STZINB-GNN | 3.3222 | 3.6921 |

Eight comparisons, eight wins for the trailing mean. This reframes what the
eight-of-eight Diebold-Mariano result at p < 2×10⁻⁶ actually establishes: we beat
a set of models that a moving average also beats. The claim worth defending is the
narrower one — **of everything in this study, CIVIC-SAFE is the only method that
improves on the trailing mean.** That observation is now in Section V-A, because a
reviewer who notices it independently will trust the rest of the paper less, and a
reviewer who is told it up front will trust it more.

**Why it is a hard baseline.** Weekly area-level crime counts are strongly
mean-reverting. Aggregating to 77 units over 7 days averages away most
short-horizon variation, so a trailing mean captures nearly all the predictable
signal and a flexible model's extra capacity mostly buys variance. This is why our
own single-seed model (CRPS 3.3622) loses to TFT-ZINB and why the win requires
five-seed EMOS.

**Concession to volunteer.** A 3.6% CRPSS improvement over a moving average is a
modest effect, and on a 53-week test set the paired Diebold-Mariano p-values
against rolling HA are 0.0338 (Chicago) and 0.0036 (NYC) — significant, and
nowhere near the 10⁻⁶ range of the deep-baseline comparisons. We treat a claim as
supported only when the DM test and the stationary block bootstrap agree, which
they do here (0.0053 and 0.0003). Our configuration also notes that 7–8 seeds
would be needed for paired t-tests at power 0.80 with d = 0.8, and we ran five.

---

## Q7. What is the operational latency of the EMOS ensemble and conformal calibration?

**Short answer.** The calibration layer adds about 20 microseconds per cell-week
online, and the expensive step is a one-time offline fit. The forward pass is
**not measured** and we make no end-to-end dispatch claim.

**Measured**, via `scripts/measure_calibration_latency.py` at the manuscript's
true panel sizes (6006 calibration cells, 4081 test cells, 5 ensemble members),
median of 7 repeats on CPU. Raw output in `outputs/calibration_latency.json`:

| stage | when | median |
|---|---|---:|
| EMOS weight learning | offline, once per campaign | 8.3 s |
| calibrator fit (split CP / randomized / ECRC) | offline, once | 68–156 ms |
| EMOS weight application | online, per batch | 0.46 ms |
| calibrator predict, 4081 intervals | online, per batch | 83–86 ms |
| **online total** | **per test panel** | **~82.7 ms, ~20.3 µs/cell** |

**The operational reading.** A weekly dispatch cycle over Chicago's 77 areas and
three categories is 231 cells, so the calibration layer contributes roughly 5 ms
per cycle against a cadence measured in days. Latency is not a binding constraint
for this application, and the paper does not claim a real-time system. The
architecture is a batch weekly forecaster; the conformal layer is negligible
against the forward pass, and the ensemble's marginal cost at inference is five
forward passes rather than one, which is the honest cost of the 16% CRPS gain.

**What is not measured, stated plainly.** The trained checkpoints live on the GPU
server and are absent from this working copy, so `measure_calibration_latency.py`
times everything *downstream* of the forward pass and nothing else. There is no
GPU forward-pass timing, no end-to-end wall-clock from raw incident feed to issued
interval, and no throughput-under-load testing. A reviewer asking for end-to-end
deployment latency is asking for a measurement we have not made, and the answer is
that we have not made it, not an estimate. The `forward_pass_measured: false` flag
in the JSON records this.

**Concession to volunteer.** These are CPU numbers on a development machine, which
is a conservative stand-in rather than a deployment measurement — an operational
system would run the same code on the hardware serving the forward pass. And the
EMOS refit is 8.3 s, so a system that recalibrated on every new week would pay
that; ours refits per campaign, which is the right cadence for a weekly panel but
is a design assumption rather than a measured requirement.

---

## Q8. Corollary 1 attributes 60% of recorded log-disparity to feedback at kappa = 0.6, yet you concede kappa is not identified. Is that attribution not unfalsifiable?

**Short answer.** The corollary is a mapping, not a measurement, and we report it
as one. What makes it more than a tautology is that the mapping is exact, the
sensitivity of its output to the unknown input is bounded, and the machinery is
provably inert at kappa = 0.

**What the corollary is.** Inverting the power law gives
`Delta_lambda = Delta_y^(1-kappa)`, so the share of recorded *log*-disparity
attributable to the loop is exactly `kappa`. That is an identity conditional on
kappa, verified against the numerical fixed point to 1e-12 relative error across
kappa in {0.3, 0.5, 0.6, 0.85}. It is not an estimate of how much real-world
disparity is artificial; it is the function that converts one into the other once
kappa is supplied. Section III states it that way and the real-data passage says
explicitly that the agreement between the predicted 60% share and the observed
58%/61% reductions is a consistency check on magnitude, not a test of the
corollary -- exposure disparity relative to population share is not the
between-group ratio the corollary is stated for.

**Why it is nevertheless disciplined rather than free.** Three reasons. The
sensitivity envelope of Section III-C bounds the consequence of a wrong kappa:
under a bounded per-cell deviation the Gamma-inflated interval retains nominal
coverage, and we tabulate the degradation without inflation (0.942 down to 0.679
across deviation factors 1.0 to 3.0). The corollary is monotone and
sign-preserving, so no value of kappa in [0,1) can turn a recorded disparity into
no disparity: `Delta_lambda > 1` whenever `Delta_y > 1`. And at kappa = 0 the
deflation is the identity map by algebra, not by fitting -- see Q9 -- so the
attribution cannot be manufactured by the method in the absence of a loop.

**Concession to volunteer.** A referee who wants the attribution *number* rather
than the mapping is asking for identified kappa, which we do not have and say so
in Limitations. The corollary's honest use is as a sensitivity instrument: it tells
a city what its recorded gap would imply at each assumed gain, and the assumed gain
remains the reader's to supply.

---

## Q9. The negative control at kappa = 0 runs on your own simulator. Of course the method is inert -- you wrote the generative model.

**Short answer.** The inertness is not empirical and does not depend on the
simulator. At kappa = 0 the deflation is the identity map by algebra.

**The argument.** The recording multiplier is
`m_s = (mu_s / M)^kappa`. At kappa = 0 this equals 1 for every cell regardless of
the distribution of `mu`, so `lambda_hat_s = mu_s / m_s = mu_s` identically and the
corrected interval is the uncorrected interval. No data-generating assumption
enters. The empirical rows -- latent coverage 0.950 to 0.949 in
the latent-coverage table, routing disparity 0.287 to 0.287 with reduction exactly
0.000 in the routing-disparity figure -- are a *check that the implementation
matches the algebra*, not evidence
for the algebra. A provable negative control is strictly stronger than an observed
one, and the distinction is worth making because it is the difference between "we
looked and found nothing" and "there is nothing to find".

**What the simulator is genuinely load-bearing for.** Not the negative control, but
the positive result: the 16.2% to 93.0% restoration at kappa = 0.85 does depend on
our generative model of the loop, because latent coverage cannot be measured
without a known latent rate. Limitations says this plainly. The real-data results
carry no coverage claim at all.

**Concession to volunteer.** The residual 0.950 to 0.949 movement at kappa = 0 is
not exactly zero, and it should be: it comes from the abstention gate, which drops
a quarter of cells at kappa = 0 for reasons diagnosed in Q10, so the two coverage
numbers are computed on slightly different sets. The algebra predicts exact
equality on a common set, and that is what we observe when the gate is disabled.

---

## Q10. The abstention gate selects on m_s after seeing the model's own mu_hat. That is post-selection inference, and your coverage numbers are conditioned on a data-dependent event.

**Short answer.** Correct, and it is stated as such rather than worked around.
Theorem 2(iv) makes the guarantee explicitly conditional on retention, and the
retention rule is measurable with respect to the predictions alone.

**Why the conditioning is admissible.** The gate depends on `mu_hat` and
`kappa_hat` and on no outcome. Applying the same rule to calibration and test cells
therefore yields coverage conditional on the selection event, which is a different
and weaker statement than marginal coverage -- and it is the statement we make. The
one dependence worth flagging, and Section III flags it, is that `M` is the mean of
the same `mu` vector being deflated, so `lambda_hat_s` is not a function of cell
`s` alone. Exchangeability survives because `M` is a symmetric function of the
cells: permuting cells permutes the deflated rates and the retention indicators
together, which is all conformal validity requires. Estimating `M` on the
calibration split alone removes the dependence entirely at the cost of a noisier
normalizer, and we name that as the cleaner choice where exact finite-sample
validity matters more than efficiency.

**The part that is genuinely open, and which we diagnosed rather than assumed.**
The gate has two branches and they fail differently. Instrumenting them separately
over 24 trials: at kappa = 0 the per-cell branch fires on **0.0000%** of cells and
the entire abstention comes from the global `kappa_hat >= 0.9` tripwire, which
crossed in 12% of trials because the estimator has median 0.020 but a maximum of
0.950. At kappa = 0.85 the picture inverts -- per-cell fires on 85.06% of cells,
global on 4% of trials. So the low-gain cost is estimator variance tripping a
threshold that carries no confidence statement, and the high-gain cost is the
deflation genuinely leaving the safe envelope. Our earlier explanation, that the
rule was "tuned for the high-gain regime and overpays at low gain", was wrong about
the mechanism and the manuscript now reports the measured version.

**Concession to volunteer.** The correct fix for the low-gain branch is an interval
estimate for kappa rather than a point estimate, so the tripwire fires on evidence
rather than on a noisy draw. We have not implemented it. Until then the retained
fraction at low gain is pessimistic by roughly the tripwire's false-positive rate.

---

## Cross-cutting: the three weaknesses we would raise ourselves

A reviewer who reads carefully will find these. They are all in the manuscript.

1. **Chicago is misspecified.** PIT uniformity fails at χ² = 241.84,
   p = 5.25×10⁻⁴⁷, against NYC's clean p = 0.293. Independently corroborated by
   the spatial residuals: 65 of 77 areas under-predicted, mean −1.03, and −0.66
   correlation between an area's crime level and its residual, so the shortfall
   concentrates in the busiest areas. Conformal coverage still holds at 90.75%
   because split conformal's guarantee is marginal and does not require correct
   specification — which is precisely why we calibrate post hoc — but a correctly
   specified model would give tighter intervals and ours is not one.

2. **The coverage-restoration result is simulation-only, by construction.** Latent
   coverage cannot be measured on real data because the true rate is unobservable.
   The real-data results (−58.2% Chicago, −60.8% NYC exposure disparity) are a
   sensitivity analysis at an *assumed* κ = 0.6; our real-data DiD is a null.

3. **OICC trades one fairness metric against another.** It attains the best hit
   rate at B = 100 in both cities (96.36%, 98.97%) and the best over-allocation
   ratio (1.036 → 0.643, 1.030 → 0.865), and simultaneously the **worst**
   allocation disparity of the four policies (0.1448 against 0.0132 on Chicago,
   eleven times larger). At B = 20 it is worst on hit rate too. No policy
   dominates, and a deployment decision requires choosing which fairness notion
   to buy.
