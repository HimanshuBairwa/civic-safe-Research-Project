# CIVIC-SAFE — Permanent Ground-Truth Memory

Durable project memory. Written to survive session boundaries: architecture,
mathematics, verified numbers, corrected errors, and the one task that remains.

**Anchor commit: `4f560c4`** on `origin/main`, working tree clean.
Every number below was read from a file under `outputs/` or reproduced by a script
under `scripts/`. Nothing here is recalled from conversation.

**Read this first, then `docs/REVIEWER_DEFENSE_DOSSIER.md` for the ten anticipated
objections and `MATHEMATICS.md` for the full equation spec.**

---

## 0. What the paper claims, in one paragraph

Crime records are produced by a loop: records direct patrols, patrols determine
where crime is looked for, and crime nobody looks for goes unrecorded. A
forecaster fit on such records can be exactly calibrated against the record and
badly wrong about reality, and no validation performed on records can distinguish
the two cases. We formalize the loop as a fixed point, derive the elasticity with
which it amplifies latent disparity, and invert it exactly. Prior work
(Lum & Isaac 2016; Ensign et al. FAT\* 2018; van Amsterdam et al. Patterns 2025)
diagnoses this pathology. **The contribution is the correction, not the
diagnosis.**

**Do NOT claim:** a universal law, a new impossibility theorem, a paradigm shift,
`1/(1-kappa)` as a discovery, or point-identified `kappa`. Each is either scooped
or false. See `docs/NOVELTY_AND_POSITIONING.md` — unscoped superlatives are this
project's main rejection trigger.

---

## 1. Repository map

| Path | Role |
|---|---|
| `paper/civic_safe_ieee.tex` | Main manuscript, IEEEtran two-column, 22 floats |
| `paper/civic_safe_supplementary.tex` | Supplementary, IEEEtran **one-column**, S-prefixed numbering |
| `paper/references.bib` | 21 verified entries, pure ASCII |
| `paper/tables/*.tex` | 7 generated table floats — **build artifacts, never hand-edit** |
| `paper/submission_bundle/` | Self-contained copy — **build artifact** |
| `paper/civic_safe_submission_bundle.zip` | 23 files **at archive root** for Overleaf/ScholarOne |
| `docs/IEEE_TPAMI_COVER_LETTER.md` | Cover letter with referee nominations |
| `docs/REVIEWER_DEFENSE_DOSSIER.md` | Ten anticipated objections, each with a volunteered concession |
| `docs/NOVELTY_AND_POSITIONING.md` | Adversarial novelty audit — read before touching claims |
| `docs/AUDIT_2026-07.md` | Records that the point-identification claim is FALSE |
| `MATHEMATICS.md` | Equation source of truth; notation matches Table I |
| `src/civicsafe/` | CIVIC-SAFE (the submission) |
| `src/oicc/` | Separate research line, **not part of this submission** |

### Regeneration commands (order matters)

```bash
python scripts/ablation_study.py            # outputs/tables/
python scripts/build_paper_tables.py        # -> paper/tables/
python scripts/make_paper_figures.py        # figs 9-11 from live experiments
python scripts/generate_framework_diagrams.py   # figs 1-2 schematics
python scripts/visualize.py                 # spatial map (PNG + PDF)
python scripts/build_submission_bundle.py   # -> submission_bundle/ + zip
python scripts/validate_latex.py            # all 4 documents, no args needed
```

---

## 2. The six pipeline stages

From `scripts/generate_framework_diagrams.py`, which draws Fig. 1.

1. **Inputs and urban graphs.** Weekly counts, ACS covariates, dual adjacency.
2. **Backbone (sequential V1).** GATv2 spatial encoder → causal Transformer,
   staged not joint. A unified V2 exists but **was not used for any reported
   number**; `configs/model/spatiotemporal_zinb.yaml` sets
   `architecture: sequential`.
3. **Ensemble.** 5-seed category-conditioned entropy-regularized EMOS, with
   fallback to uniform when learned weights lose to uniform on an internal
   holdout.
4. **Conformal calibration.** Ten variants, selected by constraint.
5. **Latent correction (the contribution).** Deflation + abstention.
6. **Policy.** Patrol allocation (OICC) and advisory routing.

---

## 3. Mathematics

Notation is **identical** across Table I of the manuscript, `MATHEMATICS.md`, the
supplementary, and `src/civicsafe/theory/feedback_law.py`. Five collisions were
resolved and must not drift back:

| Use | Symbol | NOT | Because |
|---|---|---|---|
| allocation policy | `varphi` | `pi` | `pi` is ZINB zero-inflation |
| GATv2 attention weight | `gamma_ij` | `alpha_ij` | `alpha` is conformal miscoverage |
| non-conformity score | `V_i` | `s_i` | `s` indexes spatial units |
| multiplier bound | `m_bar` (=5) | `B` | `B` is the patrol budget |
| GATv2 attention vector | `z` | `a` | `a_s` is attention *allocated* |
| Poisson quantile | `Q^Pois_p` | `Q_p` | `q_p` is the ZINB predictive quantile |

**Setup.** Latent incidence `lambda_s > 0` unobserved. Policy allocates
`a_s = varphi(mu_s)`. Recording is attention-dependent,
`y_s ~ Poisson(lambda_s * g(a_s))` with `g` increasing. A learner fitting records
converges to the fixed point `mu_s = lambda_s * g(varphi(mu_s))`.
Gain `kappa = (d log a / d log mu)(d log g / d log a)`; under
`varphi(mu)=(mu/M)^beta`, `g(a)=a^rho` this is `kappa = beta*rho`.
`M = S^{-1} sum_s mu_s` is the panel mean.

### Theorem 1 — Feedback amplification
- **(i) Local elasticity**, any smooth increasing `varphi, g`:
  `d log mu_s / d log lambda_s = 1/(1-kappa)`.
  Proof: write `E` for the elasticity, differentiate the log fixed point,
  `E = 1 + kappa*E`, solve.
- **(ii) Global power law**, *requires constant kappa*:
  `Delta_y = Delta_lambda^{1/(1-kappa)}`, pole at `kappa=1`.
  Proved **by algebra, not integration** — the log fixed point rearranges to
  `log mu_s = (log lambda_s - kappa log M)/(1-kappa)`, i.e.
  `mu_s = lambda_s^{1/(1-kappa)} * M^{-kappa/(1-kappa)}`. The factor is **one
  scalar shared by every cell**, so it cancels in a group ratio. Verified to
  1e-13.

`kappa` is a **local** elasticity at the fixed point, not a universal constant.

### Corollary 1 — Disparity attribution
`Delta_lambda = Delta_y^{1-kappa}`. The share of recorded **log**-disparity
attributable to feedback is **exactly kappa**. Verified to 1e-12.

| recorded | kappa=0.5 | kappa=0.6 | kappa=0.85 |
|---|---|---|---|
| 2:1 | 1.41:1 | 1.32:1 | 1.11:1 |
| 3:1 | 1.73:1 | 1.55:1 | 1.18:1 |
| 5:1 | 2.24:1 | 1.90:1 | 1.27:1 |

**Both edges must be stated.** Recorded disparity overstates real disparity
without bound as `kappa -> 1`, **and** the map is sign-preserving:
`Delta_lambda > 1` iff `Delta_y > 1`. The loop inflates the *magnitude* of a real
gap; it does not manufacture one.

### Theorem 2 — Exact latent recovery and conditional coverage
With `m_s = (mu_s/M)^kappa` and `lambda_hat_s = mu_s/m_s = M^kappa mu_s^{1-kappa}`:
- **(i) Exactness.** `lambda_hat_s = lambda_s` identically, no unidentified
  constant — because `M` is the *same* panel mean on both sides. Theorem 1(ii) and
  Theorem 2(i) are one identity read in opposite directions.
- **(ii) Coverage.** Poisson quantile interval at `lambda_hat` covers a
  `Poisson(lambda_s)` draw with prob >= `1-alpha`; inequality is lattice
  discreteness only.
- **(iii) Sensitivity.** `log lambda_hat_s - log lambda_s = (kappa - kappa_hat) log(mu_s/M)`.
  Leverage is **unbounded**; on the retained set the error is
  `<= |kappa - kappa_hat| * log(m_bar)/kappa_hat`.
- **(iv) Conditionality.** Retention `R` depends on `mu` alone, so validity is
  **conditional on retention**. This is the formal reason 16.2% -> 93.0% has two
  denominators.

**`M`-dependence caveat:** `lambda_hat_s` is not a function of cell `s` alone.
Exchangeability survives because `M` is permutation-symmetric. Estimating `M` on
the calibration split alone would remove the dependence at the cost of noise.

### Lemma S1 — Exact spectral radius (Supplementary S1.1)
Log-coordinate Jacobian `J = kappa(I - 1 w^T)`, `w_j = mu_j/(S*M)`.
Spectrum is `{kappa with multiplicity S-1, 0}`, so **rho(J) = kappa exactly** for
every configuration — not `<= kappa`. The zero eigenvalue lies along `1` because
the map is invariant to common rescaling. Verified to machine precision at
`kappa in {0.30, 0.50, 0.70, 0.85, 0.95}`.

### Gamma sensitivity (Supplementary S2)
If `m_true/m_hat in [1/Gamma, Gamma]` per cell, the Gamma-inflated interval holds
nominal coverage for every admissible recording model. `Gamma=1` recovers the
plain interval. Plain interval **last meets target at Gamma=1.4 (0.9057), fails
from 1.6** — so it tolerates roughly 40% multiplicative misspecification.
Inflated intervals over-cover heavily (0.9975 at Gamma=3): a worst-case
construction, not an efficient one.

### Algorithm 1 — critical detail
**No additive conformity margin.** A record-calibrated margin would reimport
exactly the bias the deflation removes; the response to a wrong gain is the
multiplicative Gamma band. `latent_prediction_interval` exposes an additive margin
as an option — **it is 0.0 in every reported number.**

---

## 4. Verified empirical results (2023 test set, 53 weeks)

| Metric | Chicago | NYC |
|---|---|---|
| CRPS | **2.8267** | **3.1401** |
| MAE | 3.9017 | 4.3675 |
| RMSE | 7.0983 | 7.6126 |
| Brier (zero event) | 0.0592 | 0.0493 |
| CRPSS vs **rolling** HA | +0.0360 | +0.0494 |
| Selected calibrator | equalized coverage | variance-scaled split CP |
| Coverage (target 90%) | 90.75% | 90.02% |
| Demographic disparity (ceiling 0.03) | 0.0238 | 0.0286 |
| PIT chi2 (9 dof) | **241.84** | 10.75 |
| PIT p | **5.25e-47** | 0.293 |

**Diebold-Mariano:** all 8 deep-baseline comparisons favour CIVIC-SAFE at
**p < 2e-6**. The largest is Chicago/TFT-ZINB at **1.671454e-06**, which is
**greater than 1e-6** — the tight bound is 2e-6, and `p < 1e-6` is FALSE. This
error was in the paper once; do not reintroduce it.
Against rolling HA: DM p = 0.0338 / 0.0036, bootstrap p = 0.0053 / 0.0003.

**Latent correction (simulation, target 0.90):**

| kappa | naive | kappa_hat | corrected | retained |
|---|---|---|---|---|
| 0.00 | 0.950 | 0.259 | 0.949 | 75% |
| 0.30 | 0.903 | 0.300 | 0.952 | 100% |
| 0.50 | 0.780 | 0.500 | 0.948 | 95% |
| 0.70 | 0.502 | 0.713 | 0.937 | 56% |
| 0.85 | **0.162** | 0.850 | **0.930** | **15%** |

**Routing disparity:** 0.287 -> 0.287 (kappa=0, reduction exactly 0.000), then
0.206, 0.148, 0.089, 0.044. **Relative** reductions 0%, 29.3%, 47.1%, 67.1%,
75.6% are **monotone in kappa** even though the absolute biased curve is not —
which is what Theorem 1 predicts.

**Real records (assumed kappa=0.6):** Chicago 0.390 -> 0.163 (-58.2%),
NYC 0.311 -> 0.122 (-60.8%).

**Recalibration was fitted and then GATED OFF** in both cities on an internal
calibration holdout (Chicago -5.29%, NYC -0.66%). `recal_applied: false`. Gating
on test CRPS would have been leakage.

---

## 5. Empirical findings that took real digging

**NYC precinct queen degree zero.** Under queen contiguity one NYC precinct has
**in-degree 0** — it touches no other polygon and would receive no spatial message
passing. The k-NN graph raises it to 2. This is the concrete justification for the
dual-graph design.

| City | Graph | Nodes | Edges | Mean deg | Min | Max |
|---|---|---|---|---|---|---|
| Chicago | queen | 77 | 394 | 5.12 | 1 | 9 |
| Chicago | k-NN | 77 | 616 | 8.00 | 1 | 13 |
| NYC | queen | 78 | 240 | 3.08 | **0** | 6 |
| NYC | k-NN | 78 | 624 | 8.00 | 2 | 13 |

**The k-NN graph is GEOGRAPHIC, k=8.** `build_adjacency_from_geodataframe` queries
a cKDTree over **projected centroids** (EPSG:26971 Chicago, EPSG:32118 NYC). It
**never touches the covariate vector.** The paper and MATHEMATICS.md both once
described it as "k-nearest neighbours in demographic feature space" — that
misstated construction *and* motivation. What it supplies is proximity beyond
adjacency, not similarity of composition.

**ACS panel is 8 fields, and there is NO education variable:**
`total_population`, `median_household_income`, `poverty_rate`,
`unemployment_rate`, `pct_black`, `pct_hispanic`, `pct_renter_occupied`,
`population_density`. Fairness stratification is a **quartile split on
`pct_black` alone**, not a composite index.

**Classical baselines get a Poisson CRPS convention.** `compute_metrics` dresses
each point forecast in a Poisson (`pi=0, mu=y_pred, r=1000`) before scoring. This
is generous and **not neutral** — XGBoost Chicago has MAE 3.9688 but CRPS 2.9157.
It **favours the rolling HA that beats all four deep baselines**, so it must be
disclosed wherever that comparison appears.

**Per-category sparsity (Chicago raw, 1,326,056 incidents):**
drug 50.89% zero cells (mean 2.15/cell-week), violent 1.09% (20.92),
property 0.20% (31.95). Zero-inflation earns its place on **drug** crime.

**Abstention has two branches with different failure modes.** Instrumented over
24 trials: at kappa=0 the per-cell branch fires on **0.0000%** of cells and the
whole loss is the **global tripwire** (`kappa_hat >= 0.9`) firing in 12% of trials
because the estimator has median 0.020 but **max 0.950**. At kappa=0.85 the
per-cell branch fires on **85.06%** and global on 4%. The earlier explanation
("rule tuned for high gain, overpays at low gain") was **wrong about the
mechanism** — at `kappa_hat -> 0` the multiplier -> 1 so the per-cell branch
*cannot* fire.

**GraphWaveNet receptive field is 16 weeks** (dilations 1,2,4,8, kernel 2) —
**less than the 52-week window**, so it cannot see the annual cycle.

**LSTM-NB trains on NB NLL** while the other three train on CRPS, so it is
evaluated on a criterion it was not trained for.

---

## 6. The honest limitations (all in the paper; never quietly drop one)

1. **Rolling HA beats all four deep baselines, 8/8, both cities.** So the eight DM
   wins are wins over models a moving average also beats. CIVIC-SAFE is the *only*
   method in the study that improves on the trailing mean.
2. **Chicago PIT fails** (chi2 241.84, p 5.25e-47); NYC passes. Corroborated
   independently by spatial residuals: **65 of 77 areas under-predicted**, mean
   -1.03, correlation **-0.66** between an area's crime level and its residual.
3. **A single seed loses to TFT-ZINB** (3.3622 vs 2.9456). Five-seed EMOS does the
   work, and baselines are **not** ensembled (3 seeds, reported as mean ± sd).
4. **85% abstention at kappa=0.85.** The abstention rate, not the coverage number,
   is the operational figure of merit.
5. **kappa is NOT point-identified.** DiD gives `rho`; `kappa = beta*rho` needs
   assumed `beta`, and the DiD carries an uncancelled latent-level term. On real
   data **the DiD is a null.** Real-data results are sensitivity analyses.
6. **Fairness is post hoc only.** The gradient-reversal adversarial discriminator
   exists (`models/adversarial_head.py`) but `train.py` never passes
   `num_adv_classes`, so it defaults to 0 and `adv_head is None`. **The reported
   model has no representation-level fairness mechanism.** The Jensen-Shannon
   penalty prevents *attention-head collapse* — it is not a fairness mechanism and
   never sees protected attributes.
7. **OICC trades one fairness metric for another.** Best hit rate at B=100
   (96.36% / 98.97%) and best over-allocation ratio, but the **worst** allocation
   disparity (0.1448 vs 0.0132 Chicago). At B=20 it is worst on hit rate too.
8. **Latent coverage is simulation-only by construction** — the true rate is
   unobservable on real data.
9. **Group-stratified PIT cannot be computed** from persisted artifacts: the npz
   carries no per-cell ZINB parameters. Fixing it is a one-line change to persist
   `(pi, mu, r)`.

---

## 7. Verification state at `4f560c4`

| Check | Result |
|---|---|
| `pytest tests/test_calibration.py tests/test_feedback_law.py` | **67 passed** |
| `scripts/validate_latex.py` (4 documents, no args) | **0 errors, 0 warnings** |
| Main manuscript | 21/21 citations, 51/51 refs, 12/12 figures |
| Supplementary | 28/28 refs, 6 tables (S1-S6), 0 citations by design |
| Zip vs on-disk bundle | 23 files, **0 SHA-256 mismatches** |
| Zip layout | files **at archive root**, 0 nested under `submission_bundle/` |
| ASCII in `.tex`/`.bib` | **0 non-ASCII, 0 tabs, 0 stray control chars** |
| AI disclosures | **none** (the word "generative" in Limitations is the statistical term) |
| Float inventory | 22 floats: 12 figures (10 single-col, 2 full-width), 9 tables, 1 algorithm; 6 starred |
| Float storage | `\extrafloats{48}` -> ceiling 66, **44 slots headroom** |
| Width overflow risk | **0** — every `\includegraphics` width <= 1.0 of its unit |
| Table shrink guards | **7/7** carry the never-upscale `\ifdim` idiom |
| Bundle figures | **12, all PDF vector** |

**Float numbering** (markdown companions cannot track this — refer by name):
Table I = notation, II = main results, III = ablation, IV = loss ablation,
V = ensemble, VI = uncertainty, VII = conformal fairness, VIII = policy,
IX = latent coverage. Fig. 1 = architecture, 2 = feedback loop,
9 = latent correction, 10 = routing, 11 = cross-city.

**Two pre-existing test failures** in `tests/test_data.py::TestACS` resolve
demographics to `C:\Users\kamle\civic-safe-Research-Project\` — the server's path,
not this working copy. Environmental, unrelated to any of this work.

---

## 8. THE ONE REMAINING TASK: Overleaf visual proofing

**Neither document has ever been compiled.** No TeX engine exists on this machine
(`pdflatex`, `xelatex`, `lualatex`, `latexmk`, `tectonic` all absent). Everything
above is **static verification**, which is structurally blind to layout.

```bash
cd paper/submission_bundle
pdflatex civic_safe_ieee && bibtex civic_safe_ieee \
  && pdflatex civic_safe_ieee && pdflatex civic_safe_ieee
pdflatex civic_safe_supplementary && pdflatex civic_safe_supplementary
```

Or upload `paper/civic_safe_submission_bundle.zip` to Overleaf (set compiler to
pdfLaTeX). `IEEEtran.cls` is **deliberately not vendored** — Overleaf and the IEEE
portals provide it, and a hand-rolled substitute would silently typeset the paper
in a non-IEEE format. For local builds: `tlmgr install ieeetran`.

### Checklist

**Log inspection**
- [ ] `Overfull \hbox` over ~20pt — flags a table or equation past the margin
- [ ] `Too many unprocessed floats` — should not occur with 44 slots headroom
- [ ] `LaTeX Warning: Reference ... undefined` — should be zero; static check agrees
- [ ] `Citation ... undefined` — should be zero
- [ ] Two `pdflatex` passes **after** `bibtex`, or refs stay unresolved

**Main manuscript**
- [ ] Abstract fits its box, 247 words
- [ ] Table I (notation, 24 rows) does not overrun the column
- [ ] Tables II and VIII (widest, `table*`) fit the full width without shrinking illegibly
- [ ] Algorithm 1 does not split awkwardly across a column break
- [ ] Figs 1 and 8 (`figure*`) span both columns at page top
- [ ] No float lands more than a page from its discussion
- [ ] Fig. 9's two-panel retention figure keeps both panels legible
- [ ] Fig. 5 spatial map (portrait, 0.86 linewidth) is not squashed

**Supplementary — the highest-risk document, never built**
- [ ] `\renewcommand{\thesection}{S\arabic{section}}` yields S1-S5, not 1-5
- [ ] Tables render as S1-S6
- [ ] Equations render as S1-S10
- [ ] Lemma S1 and Proposition S1 number correctly
- [ ] `IEEEproof` environments close properly (plain `\newtheorem` + `IEEEproof`,
      deliberately **not** `amsthm`, whose `\proof` collides with IEEEtran)
- [ ] Table S3 (baselines, 3-column with wrapped cells) does not overflow
- [ ] One-column layout leaves no orphaned headings

**If both builds are clean, submit.**

---

## 9. Working conventions that earned their place

- **Never hand-edit** `paper/tables/` or `paper/submission_bundle/`. Both are
  build artifacts. The zip is rebuilt with the bundle so it cannot go stale.
- **Every number in the paper must trace to a file** under `outputs/` or
  reproduce from a script. Figure scripts dump plotted values to
  `outputs/figure_data/*.json` for exactly this reason.
- **Do not hardcode float numbers in markdown.** Refer by name; LaTeX numbering
  shifts when floats move.
- **Beware the `\ref` shell hazard.** Heredocs can eat the backslash in `\ref`,
  leaving a literal CR followed by `ef{`. A legitimate CRLF is CR+LF, so CR+`e`
  is unambiguously the bug. Prefer the Edit tool for LaTeX strings.
- **Report the binding baseline.** Skill is against **rolling** HA (3.6%/4.9%),
  never frozen HA (27%/32%).
- **Every claim about the model must be checked against the code**, not the
  config or the docs. Three architecture misstatements were found this way: the
  V2/unified mixup, the demographic-kNN error, and the JS-penalty fairness claim.
