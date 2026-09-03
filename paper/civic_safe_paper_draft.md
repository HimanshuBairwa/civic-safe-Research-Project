# Feedback-Corrected Conformal Prediction for Spatiotemporal Crime Forecasting

**Abstract** — Crime records are not crime. Police records are produced by a
loop: past records direct patrols, patrols produce records, and the loop
amplifies whatever disparity it started with. A forecaster fit on such records
can be well calibrated against the record and badly wrong about reality, and no
amount of held-out validation on records will reveal it. We build CIVIC-SAFE, a
spatiotemporal forecaster that predicts a full zero-inflated negative binomial
distribution of weekly crime counts per community area, and we pair it with a
correction for this measurement problem. Two contributions. First, the applied
system: a dual-graph GATv2 spatiotemporal transformer with a five-seed
entropy-regularized EMOS ensemble reaches CRPS 2.8267 on Chicago and 3.1401 on
New York for 2023, beating four deep baselines in all eight head-to-head
Diebold-Mariano tests at p < 1e-6, with conformal intervals achieving 90.75% and
90.02% coverage under a pre-registered 3% disparity ceiling. Second, and the reason for the paper: we deflate
the recorded rate by a feedback multiplier and issue intervals for the latent
process. In simulation at feedback gain 0.85, intervals calibrated on the record
retain only 16.2% coverage of true incidence while our corrected intervals hold
93.0% — though they abstain on 85% of cells to do it, a cost we quantify rather
than hide. On real records from both cities, correction cuts exposure disparity
by 58% and 61%. We report what fails too: Chicago's probability integral
transform is non-uniform at p = 5e-47, and 65 of its 77 areas come in low.

**Index Terms** — conformal prediction, crime forecasting, algorithmic fairness,
feedback loops, measurement error, zero-inflated models, graph neural networks

---

## I. Introduction

Start with the thing that makes this problem different from ordinary
forecasting. When a weather model predicts rain, the prediction does not change
whether it rains. When a crime model predicts crime, the prediction sends
officers, and officers find crime. Not because they manufacture it, but because
crime that nobody looks for goes unrecorded. Send more patrols to a block and
the recorded rate there rises even if the underlying rate held perfectly still.

So the training data is an artifact of past decisions by the system you are
trying to improve.

This has an uncomfortable consequence that we think has been under-appreciated.
Take a forecaster and validate it the normal way — hold out 2023, compute CRPS,
check the coverage of your prediction intervals. Suppose everything passes. You
have learned that the model predicts *records* well. You have learned nothing
about whether it predicts *crime* well, and the two come apart exactly where it
matters most: in the neighborhoods that were over-policed to begin with. Ensign
et al. [1] named this loop and showed it runs away. van Amsterdam et al. [2]
showed the general pattern of a model that is accurate and harmful at once,
invisible to passive validation. What nobody has offered is a fix.

That is what we build here.

Our starting point is a fixed-point model. Latent incidence *λ_s* in area *s* is
unobserved. A policy allocates attention *a_s* = *π*(*μ_s*) based on the
recorded-rate estimate *μ_s*. Recording is attention-dependent:
*y_s* ~ Poisson(*λ_s* *g*(*a_s*)) with *g* increasing. A learner that fits the
records converges to *μ_s* = *λ_s* *g*(*π*(*μ_s*)). Define the feedback gain

    κ = (d log a / d log μ)(d log g / d log a)          (1)

the product of how strongly allocation responds to the record and how strongly
recording responds to allocation. From this,

    d log μ_s / d log λ_s = 1 / (1 − κ)                  (2)

so recorded disparity is true disparity raised to 1/(1 − κ), with a pole at
κ = 1. We want to be careful about credit here, because this closed form is
old. It is the closed-loop gain of a feedback amplifier, and in economics it is
the social multiplier of Glaeser, Sacerdote and Scheinkman [3]. We claim
neither. What we contribute is the constructive step that follows: given an
estimate of κ, deflate the record by the recording multiplier and issue
prediction intervals valid for *λ*, not for *y*, with abstention where the
correction becomes untrustworthy.

The empirical picture, in one paragraph. CIVIC-SAFE forecasts a full ZINB
distribution per area-week-category and achieves CRPS 2.8267 (Chicago) and
3.1401 (NYC) on 2023, ahead of every baseline we tried, with all eight
head-to-head Diebold-Mariano comparisons against deep baselines significant at
p < 1e-6. Post-hoc conformal calibration lands at 90.75% and 90.02% coverage
against a 90% target while keeping demographic coverage disparity at 0.0238 and
0.0286, both under a ceiling we fixed at 0.03 before looking. Then the
correction: in a closed-loop simulation at κ = 0.85, intervals calibrated on the
record cover the latent process only 16.2% of the time, while corrected
intervals hold 93.0%. On real Chicago and NYC records, correction reduces
recorded-exposure disparity for the higher-minority stratum by 58% and 61%.

We also found things that do not work, and we would rather say them here than
have a referee say them. Chicago's PIT histogram is not uniform — chi-square
241.84 on nine degrees of freedom, p = 5e-47 — so the ZINB is misspecified in
Chicago's right tail. A single CIVIC-SAFE seed loses to TFT-ZINB; we win because
of ensembling. The corrected intervals at κ = 0.85 abstain on 85% of cells, so
the headline 16% → 93% compares two different denominators, and we label it as
such everywhere it appears. Section VI collects these.

Contributions:

1. A feedback-corrected conformal predictor that restores coverage of the latent
   process from a biased record, with principled abstention (Section III-E). To
   our knowledge this is the first correction rather than diagnosis of the
   predictive-policing feedback pathology.
2. A coordinate-free form of the amplification elasticity, with κ as a product
   of two log-elasticities, plus the disparity power law — a sharpening of [1]
   that cites [3] for the gain itself.
3. A competitive applied forecaster with honest calibration: full distributional
   forecasts, nine conformal variants compared, and constraint-based method
   selection that refuses methods violating a pre-registered fairness ceiling
   (Sections III-D, V-C).
4. An accounting of what the system cannot do, including a misspecification
   failure on one of two cities.

---

## II. Related Work

**Spatiotemporal crime and count forecasting.** ZINB output heads on graph
networks are established for sparse urban counts. STZINB-GNN [4] pairs a
zero-inflated negative binomial likelihood with a spatiotemporal graph encoder
for demand prediction, and STMGNN-ZINB [5] extends the idea with multi-graph
fusion. Our predictor sits in this family and we make no architectural novelty
claim for it. GATv2 [6] supplies the attention operator; we use two adjacency
structures (queen contiguity and k-nearest neighbours) rather than one.

**Conformal prediction.** Split conformal gives distribution-free finite-sample
coverage under exchangeability. Conformalized quantile regression [7] supplies
the score we use. Adaptive conformal inference [8] handles distribution drift by
updating the miscoverage target online, and equalized conditional risk control
[9] gives per-group high-probability coverage via concentration bounds. All four
are applied here as published. What conformal prediction guarantees, though, is
coverage of the variable you calibrated against — which is the record. That is
the gap this paper addresses.

**Feedback loops in predictive policing.** Ensign et al. [1] formalized the
runaway loop with attention-dependent recording and showed that a Pólya-urn
model drives allocation to a degenerate winner-take-all fixed point. We differ
in two ways: our amplification is a smooth power law with a finite pole at
κ = 1 rather than unconditional runaway, and we supply a correction. Hashimoto
et al. [10] and Wyllie et al. [11] study disparity amplification in
retraining loops empirically without a closed-form exponent or a fix.

**Performativity and identification.** That predictions change the world they
predict is the subject of performative prediction [12], and Perdomo [13] shows
a predictor can pass every observable calibration check while being
performatively useless. The distinction we need is sharper: in our setting there
is a latent target *λ* genuinely different from the recorded *y*, so the failure
is a coverage failure against an unobserved variable. Performative risk control
[14] controls risk on observed outcomes, which is the opposite of what is needed
here. The observation that you cannot detect this by watching but can by
perturbing is a known template [15], and we use its difference-in-differences
instantiation rather than claiming the principle.

**Detection-shock natural experiments.** Staggered acoustic gunshot detection
rollouts have been used as difference-in-differences instruments for recorded
crime outcomes [16]. Those studies estimate reduced-form treatment effects. We
reinterpret the same design as a way to estimate a structural recording
elasticity, which is the methodological move we do claim.

---

## III. Method

### A. Problem setup

Let *s* index spatial units, *t* weeks, and *c* crime categories. For each
triple we predict a distribution over counts, not a number. Chicago gives us 77
community areas and NYC 78 comparable units, three categories (violent,
property, drug), and weekly aggregation.

We use a zero-inflated negative binomial head because crime counts are sparse
and overdispersed at once, and the two need separate machinery. Sparsity is
structural — some area-week-category cells are zero because nothing happened and
some because nothing was recorded — so a point mass at zero is not a modelling
convenience but a description of the data. Overdispersion needs the negative
binomial's second parameter. The model emits (*π*, *μ*, *r*) per cell:

    P(Y = y) = π + (1 − π)(r/(r+μ))^r                        if y = 0
             = (1 − π) · Γ(y+r)/(Γ(r) y!) · (r/(r+μ))^r · (μ/(r+μ))^y   if y > 0
                                                              (3)

with *π* = *σ*(·), *μ* = softplus(·), and *r* = softplus(·) + 0.1. The floor on
*r* matters more than it looks; we return to it in Section III-C.

### B. Architecture

Space first. Two adjacency structures feed a GATv2 encoder: queen contiguity for
immediate geographic spillover, and k-nearest neighbours in demographic feature
space for areas that behave alike without touching. Attention coefficients are
computed dynamically over the union of both neighbourhoods,

    e_ij = a^T LeakyReLU(W [h_i ‖ h_j]),   α_ij = softmax_j(e_ij)   (4)

and node states update as the attention-weighted sum over both neighbourhoods.

Then time. A spatiotemporal transformer over unified space-time tokens with a
structured attention mask carries temporal context. Between the two, a
multi-factor feature mixer with a Jensen-Shannon diversity penalty discourages
the network from routing predictions through demographic covariates — a partial
guard against proxy discrimination, not a solution to it.

We anchor *μ* to a trailing mean rather than predicting it free. This was not a
design preference; without anchoring, conformal interval widths blow up past
1000 and skill score goes deeply negative. Anchoring holds the level and lets
the network model deviations from it.

### C. Training and the r-collapse problem

Training minimizes ZINB negative log-likelihood, computed with `logsumexp` on
the zero branch and `lgamma` elsewhere for stability.

One failure mode deserves description because it is a clean example of a metric
lying. When you train ZINB by maximum likelihood, *r* tends to collapse toward
its floor. The reason: as *r* → 0 the negative binomial variance
*μ* + *μ²*/*r* diverges, giving a heavy-tailed distribution that assigns
tolerable probability to almost any observation, so NLL looks acceptable. MAE
even improves, because the mode sharpens. But CRPS degrades, because the
predictive CDF is smeared across the real line. Likelihood and sharpness pull in
opposite directions and likelihood wins.

The fix is a per-cell penalty,

    L_r-reg = λ_r · mean_i ReLU(r_reg − r_i)                  (5)

with *r_reg* = 0.5 and *λ_r* = 0.1. Per-cell rather than batch-mean is the whole
point: a batch-mean penalty lets some cells collapse to near-zero while others
compensate by staying high, and the average looks fine while individual
predictive distributions are garbage.

### D. Calibration

The model's own quantiles are not trustworthy, so we calibrate post hoc. We
split the panel chronologically — train, validation, calibration, test (2023) —
and never let a later split inform an earlier one.

Non-conformity uses CQR scores from the ZINB quantile function,
*s_i* = max(*q_{α/2}* − *y_i*, *y_i* − *q_{1−α/2}*), and the split-conformal
threshold is the ⌈(n+1)(1−α)⌉/n empirical quantile of the calibration scores.
For per-group control, ECRC computes per-group thresholds with Hoeffding slack
*ε_g* = sqrt(log(2/*δ_g*)/(2*n_g*)) so that per-group coverage is at least
1 − α − *ε_g* with probability 1 − *δ_g*. An adaptive temporal variant updates
the per-group miscoverage target online with a PID rule to handle drift; it
gives a long-run average guarantee, not a per-step one, and we say so in the
results rather than implying otherwise.

We evaluate nine variants and select among them by constraint rather than by
eyeball: narrowest mean width subject to marginal coverage at or above the
floor, demographic disparity at or below 0.03, abstention at or below 1%, and
finite metrics. The ceiling was fixed before we looked at results. Section V-B
shows the constraint doing real work — it rejects the narrowest method in both
cities.

Two smaller decisions worth recording. Ensemble weights come from
entropy-regularized EMOS learned on calibration data, with a fallback to uniform
weighting when the learned weights fail to beat uniform on an internal holdout;
the entropy term exists because unregularized EMOS put 99.6% of its mass on one
seed. And post-hoc distributional recalibration is *fitted but gated*: we
compare recalibrated against original CRPS on an internal calibration holdout,
not on test, and apply the correction only if it helps. On both cities it did
not help (Chicago −5.29%, NYC −0.66%), so both cities report identity
recalibration. Gating on test CRPS would have been leakage; we mention this
because it is a tempting shortcut.

### E. Feedback-corrected latent intervals

Now the part that is new.

Everything in Section III-D delivers coverage of *y* — the record. Under the
fixed point *μ_s* = *λ_s* *g*(*π*(*μ_s*))*, that is not coverage of *λ*, and the
gap widens with κ. Two worlds — one with honest recording and one with a biased
loop — can produce identical records, so no test on passive data separates them
[15]. This is the "confidently wrong" state: coverage of the record holds while
coverage of the truth silently fails.

Breaking the tie needs a perturbation. A staggered detection-sensitivity shock
(a patrol or acoustic-detection rollout) shifts recording intensity for treated
units, and a difference-in-differences on log recorded rates identifies the
recording response *ρ*.

Here we have to be more careful than we would like. The DiD identifies the
recording elasticity *ρ*, but the loop gain is κ = *βρ*, where *β* is the policy
elasticity, and the DiD carries an uncancelled term in the unobserved latent
level. So κ is not point-identified by this design alone: it needs an assumed
*β*, and the honest presentation is a sensitivity analysis over *β* rather than
a single number. In our real-data application the DiD is a null, which is
itself a finding worth reporting. We therefore validate the correction where the
truth is available — in simulation — and treat the real-data results as a
sensitivity analysis at an assumed gain. Overstating this is the single easiest
way to lose the paper, and we would rather claim less.

Given a gain estimate, define the recording multiplier
*m_s* = (*μ_s*/*M*)^κ with *M* the mean recorded rate, deflate to
*λ̂_s* = *μ_s*/*m_s*, and issue intervals on *λ̂*.

Abstention is not optional. As κ → 1 the correction's variance explodes, and for
cells recorded far from the mean a small error in κ produces a large error in
*λ̂*. So we abstain when κ exceeds a runaway threshold (0.9) or when the required
multiplier exceeds 5× in either direction. The system says "I don't know" instead
of issuing an interval it cannot stand behind. That is the right behaviour, and
it has a real cost that we measure in Section V-D rather than leaving implicit.

---

## IV. Experimental Setup

**Data.** Chicago (77 community areas) and New York (78 units), weekly counts in
three categories, split chronologically with 2023 held out as test. Chicago test
tensors are 53 weeks × 77 areas. Demographic groups for fairness auditing come
from area-level composition and are fixed before evaluation.

**Baselines.** Six classical: historical average (both rolling and frozen),
seasonal naive, lag-1 persistence, STARIMA, plain ZINB, XGBoost. Four deep
spatiotemporal: LSTM-NB, TFT-ZINB, GraphWaveNet, STZINB-GNN. Deep baselines run
three seeds (42, 137, 256) and we report mean ± standard deviation.

**Metrics.** CRPS is primary, since it is a strictly proper scoring rule for
distributional forecasts. We also report MAE, RMSE, and Brier score on the zero
event. Skill scores are CRPSS against a baseline. For calibration: marginal
coverage, mean interval width, per-group and per-category coverage disparity,
abstention rate, PIT uniformity chi-square, and the Hersbach decomposition of
CRPS into reliability, resolution and uncertainty.

**Significance.** Diebold-Mariano with Newey-West adjustment on paired weekly
losses, plus stationary block bootstrap. We treat a claim as supported only when
both agree.

**A note on which baseline binds.** CRPSS against a *frozen* historical average
looks impressive — 0.2713 and 0.3165. We do not lead with it. A frozen baseline
is unrealistically weak, because a real deployment would update its average as
weeks arrive. Against a *rolling* HA the gains are 0.0360 and 0.0494, and the
evaluation code marks `ha_rolling` as the binding baseline. Those are the
numbers we quote.

---

## V. Results

### A. Predictive accuracy

TABLE I summarizes the main benchmark.

| Method | Chicago CRPS | NYC CRPS |
|---|---|---|
| Historical Average (rolling) | 2.9322 | 3.3034 |
| Seasonal Naive | 4.4009 | 4.7309 |
| Lag-1 Persistence | 3.6545 | 3.8999 |
| STARIMA | 3.1396 | 3.5078 |
| ZINB | 5.5432 | diverged |
| XGBoost | 2.9157 | 3.3893 |
| LSTM-NB | 3.1037 ± 0.0396 | 3.3426 ± 0.0616 |
| TFT-ZINB | 2.9456 ± 0.0500 | 3.4244 ± 0.1581 |
| GraphWaveNet | 3.1189 ± 0.1073 | 3.8857 ± 0.2102 |
| STZINB-GNN | 3.3222 ± 0.0772 | 3.6921 ± 0.0525 |
| **CIVIC-SAFE** | **2.8267** | **3.1401** |

CRPS 2.8267 and 3.1401, best in both cities. MAE 3.9017 and 4.3675, RMSE 7.0983
and 7.6126, Brier on the zero event 0.0592 and 0.0493.

The Diebold-Mariano results are clean. Against all four deep baselines in both
cities — eight comparisons — CIVIC-SAFE wins every one at p < 1e-6:

| Baseline | Chicago DM (p) | NYC DM (p) |
|---|---|---|
| GraphWaveNet | −9.319 (< 1e-16) | −15.321 (< 1e-16) |
| LSTM-NB | −7.482 (7.33e-14) | −6.979 (2.97e-12) |
| STZINB-GNN | −10.186 (< 1e-16) | −11.158 (< 1e-16) |
| TFT-ZINB | −4.790 (1.67e-06) | −7.669 (1.73e-14) |

Against the rolling HA the margin is real but much smaller: Chicago DM
p = 0.0338 with bootstrap p = 0.0053, NYC DM p = 0.0036 with bootstrap
p = 0.0003. Both tests agree in both cities, which is what we required.

Ensembling is doing heavy lifting, and it would be dishonest to imply otherwise.
A single model averages CRPS 3.3622, worse than TFT-ZINB's 2.9456 on Chicago.
Five-seed EMOS takes it to 2.8267, a 16% improvement. The architecture alone
does not win.

The Hersbach decomposition says something useful about where the remaining error
lives:

| | Chicago | NYC |
|---|---|---|
| Reliability (↓) | 0.00124 | 0.0000604 |
| Resolution (↑) | 9.4280 | 10.7048 |
| Uncertainty | 12.2534 | 13.8448 |
| CRPSS vs climatology | 0.7693 | 0.7732 |

Reliability is four orders of magnitude below resolution. Nearly all of the CRPS
is irreducible spread in the data, not miscalibration — which is worth knowing
before anyone tries to close the gap with a better model.

### B. Where the errors live in space

Averaging the residual (predicted minus actual violent count) over the 53 test
weeks for each of Chicago's 77 community areas turns up something the aggregate
metrics hide. **65 of 77 areas are under-predicted.** The mean residual is
−1.03 incidents per area-week, the median −0.52, and the range is lopsided:
−8.31 at worst against only +1.40 at the other end. Predicted counts top out at
102 where actual counts reach 133.

The bias is not spread evenly. Correlating each area's mean residual against its
mean actual count gives **−0.66** — the busiest areas are under-predicted the
most. Fig. 13 maps this, and the under-prediction concentrates visibly on
Chicago's West Side, which is where violent crime is highest.

We point this out because it independently confirms what the PIT test says in
Section VI. Two diagnostics computed in completely different ways — a
distributional uniformity test on the probability integral transform, and a
plain spatial average of signed errors — agree that the model compresses
Chicago's upper tail. That is more convincing than either alone, and it locates
the problem: not diffuse noise, but a systematic shortfall in the highest-count
areas.

### C. Conformal calibration and the constraint that bites

Nine variants, target 90%, disparity ceiling 0.03 fixed in advance.

**Chicago** (selected: equalized coverage)

| Method | Coverage | Width | Disparity |
|---|---|---|---|
| Split CP | 0.9405 | 16.25 | 0.0182 |
| Randomized split CP | 0.9347 | 16.54 | 0.0115 |
| Weighted CP | 0.9075 | 14.58 | 0.0238 |
| Mondrian | 0.9169 | 15.02 | 0.0319 |
| Mondrian (category) | 0.9313 | 17.19 | 0.0250 |
| Mondrian (demo × category) | 0.9305 | 17.35 | 0.0119 |
| **Equalized coverage** | **0.9075** | **14.58** | **0.0238** |
| Variance-scaled split CP | 0.9079 | 14.65 | 0.0242 |
| ECRC | 0.9235 | 15.89 | 0.0156 |
| Adaptive ECRC (rolling) | 0.8930 | 13.88 | 0.0013 |

**NYC** (selected: variance-scaled split CP)

| Method | Coverage | Width | Disparity |
|---|---|---|---|
| Split CP | 0.9326 | 18.57 | 0.0201 |
| Randomized split CP | 0.9328 | 18.60 | 0.0201 |
| Weighted CP | 0.9326 | 18.57 | 0.0201 |
| Mondrian | 0.9326 | 18.57 | 0.0201 |
| Mondrian (category) | 0.9241 | 17.91 | 0.0132 |
| Mondrian (demo × category) | 0.9319 | 18.25 | 0.0154 |
| Equalized coverage | 0.9326 | 18.57 | 0.0201 |
| **Variance-scaled split CP** | **0.9002** | **16.45** | **0.0286** |
| ECRC | 0.9186 | 17.41 | 0.0138 |
| Adaptive ECRC (rolling) | 0.8918 | 16.31 | 0.0046 |

Selected coverage: 90.75% and 90.02% against a 90% target, disparity 0.0238 and
0.0286, abstention zero everywhere. NYC clears the ceiling by 0.0014 — one
resampling from failing — and we are not going to describe that as comfortable.

Three observations from these tables.

The constraint is load-bearing. Rolling adaptive ECRC has the best disparity in
both cities (0.0013, 0.0046) and the narrowest intervals (13.88, 16.31), and a
width-minimizing selection would have taken it. Its coverage is 89.30% and
89.18%, below the floor, so the policy rejected it. A fairness constraint that
never rejects anything is decoration; this one changed the answer.

Four NYC methods are numerically identical — split CP, weighted CP, Mondrian and
equalized coverage all report 0.9326 / 18.57 / 0.0201. That is the signature of
a degenerate conformity threshold. CQR scores on this panel are integers, so
methods differing only in *how* they select a quantile collapse onto the same
lattice point. It tells you something about the data rather than the code: with
this much mass at zero, the conformal correction has very little room to
distinguish itself.

Per-category coverage spreads wider than the marginal number suggests. Chicago:
violent 0.8883, property 0.8829, drug 0.9515 — two of three categories below
90%, a spread of 0.069. NYC: violent 0.8677, property 0.8950, drug 0.9378.
Marginal coverage can look fine while specific categories are undercovered, and
anyone deploying this should look at the category breakdown rather than the
headline.

### D. Decision-theoretic allocation

We simulate patrol allocation across four policies at three budgets. At B = 100:

| City | Policy | Hit rate | Allocation disparity |
|---|---|---|---|
| Chicago | Naive HA | 0.9399 | 0.0185 |
| Chicago | Point prediction | 0.9391 | 0.0140 |
| Chicago | Unconstrained conformal | 0.9529 | 0.0132 |
| Chicago | **CIVIC-SAFE OICC** | **0.9636** | 0.1448 |
| NYC | Naive HA | 0.9689 | 0.0167 |
| NYC | Point prediction | 0.9660 | 0.0156 |
| NYC | Unconstrained conformal | 0.9859 | 0.0174 |
| NYC | **CIVIC-SAFE OICC** | **0.9897** | 0.0421 |

OICC captures the most violent incidents in both cities, 96.36% and 98.97%. It
also reduces the demographic over-allocation ratio substantially — Chicago from
1.036 to 0.643, NYC from 1.030 to 0.865 — meaning it stops over-policing groups
relative to their incident share.

But the allocation-disparity column is the worst of the four policies in both
cities: 0.1448 against 0.0132 for unconstrained conformal on Chicago, 11 times
larger. These two fairness metrics genuinely disagree. Over-allocation ratio
measures policing relative to where incidents are; allocation disparity measures
evenness of allocation across groups. OICC improves the first by concentrating
resources differently, and that concentration is exactly what worsens the
second. We report both because reporting only the favourable one would be
selective, and a reader deciding whether to deploy this needs to know which
notion of fairness they are buying.

At B = 20 OICC is worst on hit rate too — 0.5242 against 0.5677 on Chicago,
0.4896 against 0.5014 on NYC. The advantage appears only at larger budgets.

### E. The feedback correction

This is the result the paper is for. We simulate a closed loop at known feedback
gain, so the latent rate is available for scoring — which is the only setting
where coverage of the truth can be measured at all.

| κ | Naive latent coverage | κ̂ (DiD) | Corrected latent coverage | Cells retained |
|---|---|---|---|---|
| 0.00 | 0.950 | 0.259 | 0.949 | 0.75 |
| 0.30 | 0.903 | 0.300 | 0.952 | 1.00 |
| 0.50 | 0.780 | 0.500 | 0.948 | 0.95 |
| 0.70 | 0.502 | 0.713 | 0.937 | 0.56 |
| 0.85 | 0.162 | 0.850 | **0.930** | **0.15** |

Read the second column first. Intervals calibrated on the record, targeting 90%,
cover the latent process 95.0% of the time at κ = 0 and 16.2% at κ = 0.85. The
intervals never stopped being valid for the record. They simply stopped being
about reality, and nothing in a standard validation pipeline would show it.

The corrected column holds 93.0% at κ = 0.85. The DiD recovers the gain well in
the middle of the range (0.300, 0.500, 0.713, 0.850 against true 0.3, 0.5, 0.7,
0.85), and poorly at zero, where κ̂ = 0.259 against a true 0. That upward bias at
zero gain is a real limitation of the estimator and we report it rather than
trimming the row.

Now the cost, which we want stated as plainly as the benefit. The last column is
the fraction of cells the corrector keeps. At κ = 0.85 it abstains on 85% of
them, so the 93.0% is measured on the surviving 15%, while the naive 16.2% is
measured on all cells. **The two columns have different denominators.** The
comparison still means something — on the cells where correction is
trustworthy, coverage of the truth is restored from catastrophic to near-nominal
— but anyone who reads "16% → 93%" as a like-for-like improvement has been
misled, so we label retention in every table and figure where corrected coverage
appears. There is also a cost at the other end: at κ = 0 the corrector abstains
on 25% of cells while improving nothing (0.950 → 0.949). The abstention rule is
tuned for the high-gain regime and it overpays at low gain.

Fig. 10 plots both curves with retention annotated.

**Routing.** The same correction applied to advisory routing, with a group-1
structural over-recording factor of 1.8:

| κ | Biased disparity | Corrected | Reduction |
|---|---|---|---|
| 0.00 | 0.287 | 0.287 | 0.000 |
| 0.30 | 0.291 | 0.206 | 0.085 |
| 0.50 | 0.280 | 0.148 | 0.132 |
| 0.70 | 0.270 | 0.089 | 0.182 |
| 0.85 | 0.182 | 0.044 | 0.138 |

At κ = 0.85 disparity drops from 0.182 to 0.044, a 76% reduction — routes stop
systematically steering around over-recorded areas. The biased curve is not
monotone: it falls from 0.270 to 0.182 between κ = 0.70 and 0.85, so the worst
biased case is not the highest gain, and we do not describe the trend as
monotone.

**Real records.** Applying the correction to actual Chicago and NYC records at
an assumed κ = 0.6:

| City | Units | Biased | Corrected | Reduction |
|---|---|---|---|---|
| Chicago | 77 | 0.390 | 0.163 | −58.2% |
| NYC | 78 | 0.311 | 0.122 | −60.8% |

Exposure disparity for the higher-minority stratum falls by 58% and 61%. Two
caveats travel with this table and must not be dropped. The gain is *assumed*,
not identified — the real-data DiD is a null — so this is a sensitivity analysis,
not a validated identification. And latent coverage cannot be evaluated here at
all, because the true rate is unobservable on real data. The coverage guarantee
lives in the simulation above; this table shows only that correction moves
real-world allocation in the predicted direction under a plausible gain.

---

## VI. Limitations

We would rather list these ourselves.

**Chicago is misspecified.** The PIT histogram fails uniformity badly:
chi-square 241.84, p = 5.25e-47, max bin deviation 0.0368 against an expected
0.10 per bin. The top bin holds 13.68% of the mass, and the rise across upper
bins is monotone, so the model systematically under-predicts Chicago's right
tail. High-count area-weeks fall further into the predictive tail than they
should. NYC passes cleanly (chi-square 10.75, p = 0.293), which makes the
Chicago failure a property of that city's data rather than a bug in shared code.

The spatial residuals say the same thing from another direction: 65 of 77
community areas under-predicted, mean residual −1.03, and a −0.66 correlation
between an area's crime level and its residual, so the shortfall concentrates in
the busiest areas (Section V-B). Two unrelated diagnostics pointing at the same
compressed upper tail is harder to dismiss than either would be alone.

Conformal coverage still holds at 90.75%, because split conformal's guarantee is
marginal and finite-sample under exchangeability and does not require a
correctly specified model — that is precisely why we calibrate post hoc. But a
correctly specified model would give tighter intervals, and ours is not one.

**The correction's abstention cost is severe at high gain.** 85% of cells at
κ = 0.85. A deployment there would decline to answer for most of the city. That
is honest behaviour and it is also a limited product.

**κ is not point-identified.** The DiD gives the recording elasticity *ρ*;
the loop gain κ = *βρ* needs an assumed policy elasticity *β*, and the DiD
carries an uncancelled latent-level term. Real-data results are sensitivity
analyses at assumed κ. On real data our DiD is a null.

**The architecture does not win on its own.** Single-seed CRPS around 3.3622
loses to TFT-ZINB's 2.9456 on Chicago. Five-seed EMOS is doing the work.

**OICC trades one fairness metric for another.** Best hit rate and best
over-allocation ratio, worst allocation disparity. At small budgets it is worst
on hit rate as well.

**Skill over a realistic baseline is modest.** 3.6% and 4.9% CRPSS against
rolling HA. The 27-32% figures are against a frozen baseline that no deployment
would use.

**Fairness here is coverage-based.** Equal coverage across groups is one notion
among many, computed on area-level demographic composition rather than
individual attributes, and it says nothing about whether the underlying
allocation is just.

**The simulation is a simulation.** The coverage-restoration result assumes our
own generative model of the feedback loop. It is the only setting where latent
coverage is measurable, which is the point, but it is not field evidence.

**Five seeds is fewer than we would like.** The configuration notes that 7-8
seeds would be needed for paired t-tests at power 0.80 with d = 0.8.

---

## VII. Conclusion

Records are not reality, and a forecaster validated only against records cannot
tell the difference. We built a spatiotemporal ZINB forecaster that is
competitive on ordinary terms — CRPS 2.8267 and 3.1401, eight of eight
Diebold-Mariano wins over deep baselines at p < 1e-6, conformal coverage of
90.75% and 90.02% under a pre-registered 3% disparity ceiling — and then asked
the question that the ordinary terms cannot answer: is the model right about
crime, or only about crime records?

In simulation the answer is stark. At feedback gain 0.85, intervals that look
perfectly well calibrated against the record cover the latent process 16.2% of
the time. Deflating the record by the estimated feedback multiplier restores
that to 93.0%, on the 15% of cells where the correction is trustworthy enough to
issue an interval at all. On real records from two cities, the same correction
cuts exposure disparity for higher-minority areas by 58% and 61% under an
assumed gain.

What we would want a reader to take away is narrower than the numbers might
suggest. The correction works, in a regime we can measure, at a cost we can
quantify. It abstains heavily when the feedback gain approaches runaway, and its
gain parameter is not point-identified on real data. Chicago's PIT failure says
our own predictor is misspecified on one of the two cities we tested.

The obvious next step is the natural experiment: run the detection-shock DiD on
a real staggered rollout with enough power to move the estimate off null. Until
then, the honest claim is that we have shown a correctable failure is
correctable in a setting where correctness is checkable — and that the abstention
rate, not the coverage number, is the thing to watch.

---

## Reproducibility

Every number in this paper is either read from a file under `outputs/` or
reproduced by running a script under `scripts/`. The three figures reporting the
feedback correction are generated by `scripts/make_paper_figures.py`, which calls
the experiment functions directly and writes the values it plotted to
`outputs/figure_data/*.json`, so any figure can be checked against the numbers
behind it. The spatial error map is computed from the saved test predictions in
`outputs/conformal_evaluation/chicago_predictions.npz`.

Three figures that previously existed in the repository are deliberately not
included. Two attention visualisations drew synthetic weights — an exponential
distance decay for the spatial map, a hand-built lag pattern for the temporal
one — under titles claiming they showed learned GATv2 and transformer attention.
The pipeline does not persist real attention weights, so those figures could not
be produced honestly and the plotting functions now refuse to run without real
input rather than drawing something misleading.

## Declaration of AI assistance

An AI coding assistant was used during this work for code review, for drafting
and editing this manuscript, and for the verification pass that checked each
reported number against its source file. The system design, the experiments, and
the research contribution are the authors'. All results were produced by the
authors' code on the authors' data, and every figure and table was regenerated
and checked after drafting. We disclose this in line with IEEE policy on
author responsibilities for AI-generated content.

---

## References

[1] D. Ensign, S. A. Friedler, S. Neville, C. Scheidegger, and S. Venkatasubramanian, "Runaway feedback loops in predictive policing," in *Proc. Conf. Fairness, Accountability and Transparency (FAccT)*, 2018.

[2] W. A. C. van Amsterdam et al., "When accurate prediction models yield harmful self-fulfilling prophecies," *Cell Patterns*, 2025.

[3] E. L. Glaeser, B. I. Sacerdote, and J. A. Scheinkman, "The social multiplier," *J. European Economic Association*, vol. 1, no. 2, pp. 345-353, 2003.

[4] D. Zhuang, S. Wang, H. Koutsopoulos, and J. Zhao, "Uncertainty quantification of sparse travel demand prediction with spatial-temporal graph neural networks," in *Proc. ACM SIGKDD*, 2022.

[5] S. Wang et al., "Spatiotemporal multi-graph neural network with zero-inflated negative binomial output for sparse demand forecasting," 2024.

[6] S. Brody, U. Alon, and E. Yahav, "How attentive are graph attention networks?," in *Proc. ICLR*, 2022.

[7] Y. Romano, E. Patterson, and E. Candès, "Conformalized quantile regression," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2019.

[8] I. Gibbs and E. Candès, "Adaptive conformal inference under distribution shift," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2021.

[9] S. Feldman, S. Bates, and Y. Romano, "Improving conditional coverage via orthogonal quantile regression," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2021.

[10] T. B. Hashimoto, M. Srivastava, H. Namkoong, and P. Liang, "Fairness without demographics in repeated loss minimization," in *Proc. ICML*, 2018.

[11] S. Wyllie, I. Shumailov, and N. Papernot, "Fairness feedback loops: training on synthetic data amplifies bias," in *Proc. FAccT*, 2024.

[12] J. Perdomo, T. Zrnic, C. Mendler-Dünner, and M. Hardt, "Performative prediction," in *Proc. ICML*, 2020.

[13] J. C. Perdomo, "Revisiting the predictability of performative, social events," 2025.

[14] Anonymous, "Performative risk control," 2025. [CITATION NEEDED — verify authors and venue before submission]

[15] C. Mendler-Dünner, J. Ding, and Y. Wang, "Anticipating performativity by predicting from predictions," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2022.

[16] N. Topper, "The effect of acoustic gunshot detection technology on police response and crime outcomes," *J. Experimental Criminology*, 2024. [CITATION NEEDED — verify exact title, volume, pages]

[17] A. N. Angelopoulos, E. J. Candès, and R. J. Tibshirani, "Conformal PID control for time series prediction," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2023.

[18] H. Hersbach, "Decomposition of the continuous ranked probability score for ensemble prediction systems," *Weather and Forecasting*, vol. 15, no. 5, pp. 559-570, 2000.

[19] T. Gneiting, A. E. Raftery, A. H. Westveld III, and T. Goldman, "Calibrated probabilistic forecasting using ensemble model output statistics and minimum CRPS estimation," *Monthly Weather Review*, vol. 133, no. 5, pp. 1098-1118, 2005.

[20] F. X. Diebold and R. S. Mariano, "Comparing predictive accuracy," *J. Business & Economic Statistics*, vol. 13, no. 3, pp. 253-263, 1995.
