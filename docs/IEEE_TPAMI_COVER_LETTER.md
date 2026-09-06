# Cover Letter — IEEE Transactions on Pattern Analysis and Machine Intelligence

To the Editor-in-Chief:

We submit *Feedback-Corrected Conformal Prediction for Spatiotemporal Crime
Forecasting* for consideration as a Regular Paper.

---

## 1. Manuscript metadata

| | |
|---|---|
| **Title** | Feedback-Corrected Conformal Prediction for Spatiotemporal Crime Forecasting |
| **Author** | Himanshu Bairwa, Department of Computer Science |
| **Corresponding author** | bairwahimanshu29@gmail.com |
| **Submission type** | Regular Paper |
| **Primary subject** | Machine learning for societal systems; distribution-free uncertainty quantification |
| **Secondary subject** | Spatiotemporal analytics; algorithmic fairness |
| **Main manuscript** | `paper/civic_safe_ieee.tex` — IEEEtran, two-column, 22 floats (12 figures, 9 tables, 1 algorithm) |
| **Supplementary** | `paper/civic_safe_supplementary.tex` — separate compilation unit (6 sections, 8 tables, 4 figures, 12 pages) |
| **Submission archive** | `paper/civic_safe_submission_bundle.zip` — self-contained, pure ASCII |
| **Code and data** | Public repository; every table and figure regenerable without retraining |

---

## 2. Why this problem is not the problem the field has been solving

Predictive policing has been studied as a *fairness* problem and as a *feedback*
problem. It has not been studied as an *inverse* problem, and that is the gap the
paper fills.

Lum and Isaac (2016) established that police records are a biased sample of crime
rather than a measurement of it, using drug-arrest data against public-health
prevalence estimates. Ensign et al. (FAT\* 2018) formalized the mechanism: a
model trained on records directs patrols, patrols generate records, and a
Pólya-urn analysis shows the loop can run away to a degenerate allocation. van
Amsterdam et al. (Patterns 2025) generalized the epistemic problem — a model can
be accurate and harmful simultaneously, and the harm is invisible to any
validation performed on the outcome the model was fit to.

Those three results establish that the pathology exists, that it is dynamic, and
that it is undetectable by standard means. **None of them inverts it.** The
literature's response has been to constrain the allocation policy, to audit the
outputs, or to argue against deployment. What has remained open is the narrower
technical question: given records produced by such a loop, can one issue
prediction intervals that are valid for the *latent* incidence rate rather than for
the record?

Our answer is yes, under conditions we state, and the conditions are as much of
the contribution as the construction. We would not want the paper read as a claim
that predictive policing is thereby made safe. It is a claim that a specific
failure mode, previously diagnosable only in principle, becomes measurable and
correctable in a regime we characterize.

---

## 3. Methodological and theoretical contributions

**A closed-loop gain that is a product of two log-elasticities.** We define
`kappa` as the allocation elasticity times the detection elasticity, evaluated at
the recording fixed point. Because it is a product of elasticities, it is
coordinate-free: any smooth increasing policy and recording response with the same
product induce the same local amplification. Theorem 1(i) gives
`d log mu / d log lambda = 1/(1 - kappa)` for general smooth responses; Theorem
1(ii) gives the global power law `Delta_y = Delta_lambda^{1/(1-kappa)}` under
constant elasticity, and we are explicit that the global statement needs the
stronger hypothesis. The amplification form itself is classical — it is the
closed-loop gain of a feedback amplifier, and in economics the social multiplier
of Glaeser, Sacerdote and Scheinkman (2003) — and we claim neither.

**Corollary 1, the attribution law.** Read backwards, Theorem 1(ii) gives
`Delta_lambda = Delta_y^{1-kappa}`, so the share of a recorded *log*-disparity
attributable to feedback rather than to incidence is exactly `kappa`. This is the
form the result takes when a policymaker asks how much of an observed gap is real.
The arithmetic is consequential: at `kappa = 0.6`, a recorded 3:1 gap corresponds
to a true 1.55:1 gap. We state both edges, because only one flatters the method —
recorded disparity overstates real disparity without bound as `kappa` approaches
one, *and* the map is sign-preserving, so `Delta_lambda > 1` whenever
`Delta_y > 1`. The loop inflates a real gap; it does not manufacture one.

**Exactness of the deflation.** Theorem 2(i) shows that deflating the recorded
rate by the fitted recording multiplier `m_s = (mu_s/M)^kappa` returns the latent
rate *identically*, with no unidentified scaling constant. The reason is that `M`
is the same panel mean on both sides of the fixed point, so the constant that a
proportionality argument would leave free cancels by construction. Theorem 1(ii)
and Theorem 2(i) are the same identity read in opposite directions, and the paper
says so.

**A provable negative control.** At `kappa = 0` the multiplier is 1 identically,
so the deflation is the identity map by algebra, not by fitting. The method is
inert on an unbiased record as a matter of arithmetic rather than of empirical
observation. We regard this as important for the credibility of the positive
result: a correction that shrank disparity in the absence of a loop would be
shrinking it for some other reason. The simulation confirms it — latent coverage
moves 0.950 to 0.949, routing disparity 0.287 to 0.287 with reduction exactly
0.000 — but the confirmation is a check on the implementation, not evidence for
the claim.

**Abstention as a mathematical necessity.** The log-error of the deflation under a
mismeasured gain is `(kappa - kappa_hat) log(mu_s/M)`, and the leverage
`log(mu_s/M)` is unbounded. Abstention is the truncation that makes the bound
finite: on the retained set the error is at most
`|kappa - kappa_hat| (log m_bar)/kappa_hat`. Removing abstention would leave no
bound to state. The system declines to answer rather than issuing an interval it
cannot stand behind, and Theorem 2(iv) makes the resulting guarantee explicitly
conditional on retention.

**A priced sensitivity envelope.** Theorem 2(i) assumes a power-law recording map.
Rather than defend that assumption, we bound the consequence of its failure with a
Rosenbaum-style marginal sensitivity model: if the true multiplier lies within a
factor `Gamma` of the assumed one at every cell, a `Gamma`-inflated interval
retains nominal coverage for every admissible recording model. Supplementary
Section S2 tabulates the frontier. The un-inflated interval last meets target at
`Gamma = 1.4` and fails from 1.6, so the plain correction tolerates roughly 40%
multiplicative misspecification — a modest tolerance, which we state as such.

---

## 4. Empirical scope, and what we report against ourselves

**Data.** Chicago (77 community areas) and New York (78 precinct aggregates),
weekly counts in three crime categories over 2018–2023, `T = 313` weeks. The split
is chronological by week index and fixed in advance: train `[0, 208)` = 2018–2021,
validation `[208, 234)` = 2022 H1, calibration `[234, 260)` = 2022 H2, test
`[260, 313)` = the 53 weeks of 2023 on which every reported number is computed.

**Comparisons.** Four deep spatiotemporal baselines (LSTM-NB, TFT-ZINB,
GraphWaveNet, STZINB-GNN), each at three seeds, and seven classical baselines
including a rolling historical average. Ten conformal calibration variants,
selected by a rule fixed before we looked at results: narrowest interval subject to
coverage at or above the floor, demographic coverage disparity at or below 0.03,
and abstention at or below 1%.

**Headline results.** CRPS 2.8267 (Chicago) and 3.1401 (New York), ahead of every
baseline. All eight head-to-head Diebold–Mariano comparisons against the deep
baselines favour the method at `p < 2e-6`. Conformal coverage 90.75% and 90.02%
against a 90% target, with disparity 0.0238 and 0.0286. In simulation at
`kappa = 0.85`, intervals calibrated on the record cover latent incidence 16.2% of
the time; corrected intervals hold 93.0%.

**What we report against ourselves.** We would rather the reviewers find these
stated than discover them.

- **A rolling historical average beats all four deep baselines in both cities**,
  eight comparisons out of eight. The eight Diebold–Mariano wins are therefore wins
  over models a trailing mean also beats. The defensible claim is narrower: of
  everything in the study, ours is the only method that improves on the trailing
  mean, by 3.6% and 4.9%.
- **Chicago's probability integral transform is non-uniform** at
  `chi^2 = 241.84` on nine degrees of freedom, `p = 5.25e-47`. New York passes at
  `p = 0.293`. The spatial residuals concur: 65 of 77 Chicago areas are
  under-predicted, with a correlation of −0.66 between an area's mean count and its
  mean residual. Our own predictor is misspecified on one of the two cities.
- **A single seed loses to TFT-ZINB.** Single-seed CRPS averages 3.3622 against
  2.9456. The margin requires five-seed ensembling, and the baselines are not
  ensembled, so the head-to-head compares an ensembled method against
  un-ensembled ones.
- **At `kappa = 0.85` the corrector abstains on 85% of cells.** A deployment in
  that regime would decline to answer for most of the city. We name the abstention
  rate, not the coverage number, as the operational figure of merit.
- **The feedback gain is not point-identified.** The difference-in-differences
  design identifies the recording elasticity; the loop gain also requires an
  assumed policy elasticity, and on real data our difference-in-differences is a
  null. Real-data results are sensitivity analyses at an assumed gain, and the
  latent-coverage guarantee is validated only in simulation, where the latent rate
  is knowable.
- **Fairness here is post hoc only.** The gradient-reversal adversarial
  discriminator that would enforce representation-level demographic invariance is
  implemented but disabled in the reported configuration. The Jensen–Shannon
  penalty in the feature mixer prevents attention-head collapse and is not a
  fairness mechanism. What we demonstrate is distributional coverage equity under a
  pre-registered constraint.
- **Classical baselines are scored generously.** Their point forecasts are dressed
  in a Poisson predictive distribution at the forecast value before CRPS is
  computed, which is why XGBoost's CRPS (2.9157) falls below its MAE (3.9688). This
  favours the baselines, including the rolling average that beats the deep models,
  and Supplementary Section S3 states it.

`docs/REVIEWER_DEFENSE_DOSSIER.md` in the repository addresses ten anticipated
objections in this register, each traced to a source file, each closing with a
concession we volunteer rather than wait to have extracted.

---

## 5. Suggested reviewers

We have no co-authorship, institutional or funding relationship with any person
named below. Each is suggested for a distinct competence, because the paper sits
across four literatures and no single referee spans them.

**Anastasios N. Angelopoulos** — conformal prediction and distribution-free risk
control. The paper's central move is to change the *target variable* of a conformal
guarantee rather than the exchangeability condition on the calibration set, and
Angelopoulos's work on conformal risk control and on conformal PID control for time
series (cited in the manuscript) is the closest technical vantage from which to
judge whether that move is sound.

**Stephen Bates** — distribution-free uncertainty quantification and
group-conditional risk control. Sections IV-D and V-C compare ten calibrators under
a pre-registered group-disparity constraint and report where the constraint binds;
Bates's work on conditional coverage and risk control is directly on point for
whether our selection rule is defensible.

**Sorelle A. Friedler** — algorithmic fairness and the predictive-policing feedback
loop. As a co-author of the runaway-feedback analysis the paper positions itself
against, Friedler is best placed to assess whether our delta over that work is real
and whether we have credited it correctly. *(We note for the editor that Carlos
Scheidegger, a co-author on the same paper, would be a natural alternative;
inviting both would concentrate the review on one prior work rather than
diversifying it.)*

**Celestine Mendler-Dünner** — performative prediction and identification under
feedback. Our identification strategy instantiates the passive/active duality her
work formalizes, and our honest position is that we estimate a recording elasticity
rather than point-identifying the loop gain. She is well placed to judge whether
that scoping is adequate or still overreaches.

**A spatiotemporal graph-learning referee**, for the applied half. The predictor
builds on STZINB-GNN and STMGNN-ZINB and claims no architectural novelty, so the
relevant judgment is whether the zero-inflated graph forecaster is competently
built and fairly compared. We do not name a specific individual here because the
obvious candidates are authors of the baselines we evaluate against, which we
judged to be a position the editor should decide on rather than one we should
propose.

---

## 6. Declarations

**Originality and prior publication.** The manuscript is original, has not been
published previously, and is not under consideration elsewhere in whole or in part.
No portion has appeared in a conference proceeding.

**Concurrent submission.** No related manuscript by this author is under review at
any venue. The repository contains a second, separable research line (over-identified
conformal deconvolution, `src/oicc/`) which is not part of this submission and is
clearly labelled as such in the repository README.

**Conflicts of interest.** None to declare. No funding source imposed conditions on
publication.

**Ethics and data provenance.** All data are publicly released municipal incident
records and American Community Survey estimates. No individual-level or personally
identifying data were used; the finest spatial unit is an administrative area of
tens of thousands of residents. The manuscript contains an Ethical Considerations
section stating what the work does not claim, naming the abstention rate as the
operational figure of merit, and noting that the routing and allocation results are
simulations of decision rules rather than field trials.

**Format compliance.** Both `.tex` files are pure ASCII, with accented names given
as LaTeX escapes, so they compile under any engine without `inputenc`. The
submission archive is self-contained apart from `IEEEtran.cls`, which we
deliberately do not vendor: Overleaf and the IEEE author services provide it, and a
substitute class would silently typeset the paper in a non-IEEE format. The bundle
README documents CTAN installation for local builds.

**Reproducibility.** Every number in the manuscript is either read from a stored
result file or reproduced by a named script, without retraining. Figure scripts
write the values they plotted to JSON alongside the figures, so any panel can be
checked against the numbers behind it. Static validation of both documents and both
bundle copies reports zero errors and zero warnings.

**Compilation verification.** Both the main manuscript and the supplementary material
have been compiled directly against the official CTAN `IEEEtran.cls` distribution.
The main manuscript compiles cleanly to 16 pages with zero overfull horizontal boxes,
zero missing citations, and clean float resolution across all 22 floats. The
supplementary material compiles independently to 12 pages with zero compilation
warnings or structural errors. Both outputs have been visually proofed.

We believe the paper's central observation is durable independent of our particular
model: any forecaster validated on an outcome that its own deployment helps produce
is exposed to a failure that the validation cannot see, and the exposure is
quantifiable. We would welcome review that presses hardest on the conditions under
which our correction holds, since that is where we have tried to be most careful
and where we expect the most useful criticism.

Respectfully submitted,

**Himanshu Bairwa**
Department of Computer Science
bairwahimanshu29@gmail.com
