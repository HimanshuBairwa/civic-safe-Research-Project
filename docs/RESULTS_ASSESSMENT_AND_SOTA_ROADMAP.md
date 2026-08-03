# Results Assessment & SOTA Roadmap

**Written 2026-08-03. Supersedes the 2026-08-02 version, which was wrong.**

Audience: the paper team. This is the honest read of where CIVIC-SAFE stands
against the *real* GPU-server results, and the ordered work that turns it into a
defensible top-tier claim.

---

## 0. One-line status

**The forecast wins, on both cities, significantly.** Chicago CRPSS vs
seasonal-naive **+0.2662**, NYC **+0.2459**, both clearing the pre-registered
0.10 gate, both with Diebold-Mariano p < 1e-6 against Historical Average. The
remaining work is not "make the model win" -- it is **completing the comparison
table** so a reviewer cannot dismiss the win.

### Correction to the previous version of this document

The 2026-08-02 draft stated the model *loses* to seasonal-naive (CRPSS -2.52).
That was read off `chicago_conformal_results.json` **before** the real server
results were pulled -- the file at that path was the quarantined 2026-06-16
pre-`072fc14` artifact, from a checkpoint that never saw a crime count. The claim
was retracted. The genuine stale file now lives at
`outputs/_stale_pre_log1p/chicago_conformal_results_JUNE16.json` (CRPSS -2.5171,
timestamp 2026-06-16) and the real results are at
`outputs/conformal_evaluation/` (CRPSS +0.2662, timestamp 2026-08-02T04:48:10).

**Provenance check before citing any artifact:** read
`metadata.checkpoint`. A `C:\Users\...` path is a local toy run; `/workspace/...`
is the A100. Also confirm `metadata.timestamp` is post-`072fc14` (2026-07-28).

---

## 1. The real results

Both cities, 5-seed EMOS ensemble, 53 test weeks (2023), 3 categories.

| | Chicago (77 CAs) | NYC (78 precincts) |
|---|---|---|
| CRPS | **3.2291** | **3.5679** |
| MAE | 4.5144 | 4.9627 |
| RMSE | 8.2103 | 8.5543 |
| Brier (zero-inflation) | 0.0677 | 0.0535 |
| Historical Average CRPS | 3.8781 | 4.5942 |
| Seasonal-naive CRPS | 4.4008 | 4.7309 |
| **CRPSS vs HA** | **+0.1673** | **+0.2234** |
| **CRPSS vs seasonal-naive** | **+0.2662** | **+0.2459** |
| Passes 0.10 gate | YES | YES |

Note HA is the *harder* baseline on both cities (3.88 < 4.40 Chicago,
4.59 < 4.73 NYC). Quote the CRPSS vs HA as the headline; it is the honest floor.

### Significance (vs Historical Average, T=53 weeks)

Sign convention: `mean_diff = CRPS_ours - CRPS_baseline`, so negative favours us.

| | Chicago | NYC |
|---|---|---|
| DM statistic (Newey-West HAC) | -5.1097 | -8.8273 |
| DM p-value | 3.23e-07 | < 1e-16 |
| mean_diff | -0.6490 | -1.0263 |
| 95% CI | [-0.8979, -0.4000] | [-1.2542, -0.7984] |
| Moving-block bootstrap p | 0.0002 | < 1e-4 |
| Block length (Politis-White) | 10 | 4 |

Both CIs exclude zero by a wide margin. This is a real, significant win.

### Per-category CRPSS (vs HA)

| Category | Chicago | NYC |
|---|---|---|
| violent | +0.1433 | +0.1644 |
| property | +0.1483 | +0.2668 |
| drug | +0.3559 | **+0.0538** |

**NYC drug (+0.0538) is a weak result and must be reported as one.** It is below
the 0.10 gate that the aggregate clears. Chicago drug (+0.3559) is the single
best cell in the table; the same category is the worst in NYC. Report both and
say so -- selective reporting here is the easiest thing for a reviewer to catch.

### Ensemble contribution

| | Chicago | NYC |
|---|---|---|
| Per-seed CRPS | 3.3009, 3.2523, 3.5328, 3.2845, 3.4405 | 3.7704, 4.1658, 3.9905, 3.9821, 3.6073 |
| Mean single seed | 3.3622 | 3.9032 |
| Equal-weight ensemble | 3.2735 | 3.6661 |
| EMOS (learned weights) | 3.2291 | 3.5679 |
| EMOS gain over mean seed | 2.09% | 2.55% |
| Aleatoric / epistemic | 39.47 / 4.61 (10.5% epi) | 44.87 / 12.54 (21.8% epi) |

**The margin is architecture, not ensembling.** EMOS buys 2-2.5%; the win over
HA is 17-22%. Even the *worst single seed* (Chicago 3.5328, NYC 4.1658) beats HA
on Chicago and ties it on NYC. That is the right way to present it, and it is a
stronger claim than the ensemble number alone.

NYC's epistemic fraction (21.8%) is double Chicago's -- NYC seeds disagree more.
With 5 seeds that is worth one sentence, not a section.

---

## 2. Bugs found and fixed this session

### 2.1 CRPS truncation, no analytic tail (FIXED, commit 85810cb)

`crps_zinb` truncated its CDF sum at a `k_max` derived only from the predicted
distribution, never from `y`. A confidently-wrong forecast on a large count lost
up to 94% of its penalty (`truth=800, forecast~5` scored 44.75 instead of
793.75). Because each dropped step costs ~1.0, this **systematically flattered
whichever model predicted the smallest counts**. Fixed with an analytic tail
term; now matches a closed-form reference to <0.03 across random, overdispersed
(counts to 2000), pathological, and edge regimes.

### 2.2 The same bug in the EMOS path (FIXED, commit 44d546d)

`scripts/emos_ensemble.py:crps_mixture_zinb` repeated the identical pattern,
capped at 500, with no tail. **This is the function that produced the headline
CRPS numbers in section 1.** Fixed and verified within 0.02 of a brute-force
reference for y up to 900.

### 2.3 Silent upper-quantile clipping in `zinb_ppf` (FIXED, commit 44d546d)

The more serious of the three, and new this session. `zinb_cdf_full` sized its
grid by pairing the batch-max `mu` with the batch-max `r`. **Larger `r` means
smaller variance**, so that pairing yields the *smallest* plausible spread for
the *largest* mean -- it understates the grid needed by the large-mu/small-r
cell, whose CDF saturates slowest. `torch.searchsorted` then ran off the end and
the result was silently clamped to the grid edge:

```
true 95th pct of NB(mu=40, r=0.5) = 154   ->   old code returned 125
```

A one-sided, invisible failure: the value looks like a legitimate quantile, but
intervals come back too narrow and coverage is understated on exactly the
high-count overdispersed cells the interval exists to cover. It feeds **all 7
calibration methods** through `zinb_ppf_pair`.

The truncation rule now lives once, in `civicsafe.utils.numerics.nb_k_max`,
computed per observation then maxed, and `zinb_ppf` raises rather than clipping.

**Why the existing suite missed it:** every prior `zinb_ppf` test used a
homogeneous batch. The bug requires one cell to set `max_r` and a *different*
cell to set `max_mu`. Three regression tests now cover it; all three fail on the
old code. An ordering-only assertion was not enough -- with `mu=[40,40]`,
`r=[0.5,100]` the clipped width (115) still exceeded the tight cell's (25), so
the exact scipy quantiles are pinned instead.

### 2.4 Do these bugs invalidate the section-1 numbers?

**Measured, not assumed.** I rebuilt the real panel from `data/raw/` and checked
the binding conditions directly.

Observed test-set maxima: **Chicago 277** (property), **NYC 168**. The old cap
was 500.

| | Chicago | NYC |
|---|---|---|
| Test observations with y > old k_max | 1 of 12,243 (0.008%) | 0 |
| Mean CRPS understatement at k_max=500 | 0.0000 | 0.0000 |
| Cells whose true 95th pct exceeded k_max | 0 of 231 | 0 of 234 |

**Conclusion: the truncation bugs did not bind on this data.** The CRPS and
CRPSS values in section 1 stand as reported. Full suite: **310 passed**.

This is a narrow escape, not a clean bill of health. The margin was one
observation, and it is data-dependent: the sweep shows that a model emitting
`max_mu=100` with `max_r=50` gets `k_max=274` and the bug bites immediately. Any
re-run on a higher-count city, a finer spatial unit, or a longer horizon would
have shipped silently corrupted numbers. **Re-run the evaluation on the fixed
code before submission** so the published artifacts carry a provably correct
metric, not a metric that happened not to break.

---

## 3. Per-category coverage: diagnosed, and now fixed

Property undercovers in both cities; drug overcovers. Every calibration method
shows it.

| Category (split_cp) | Chicago | NYC |
|---|---|---|
| violent | 0.9348 / w19.7 | 0.8950 / w12.1 |
| property | **0.8589** / w25.6 | **0.8628** / w32.7 |
| drug | 0.9897 / w6.2 | 0.9562 / w7.4 |

I tested three explanations by construction, on a model whose defect is known
exactly.

**Ruled out -- the marginal-coverage guarantee.** On a *perfectly specified*
model (predictions == true generating parameters, category profiles matched to
the real Chicago split), split-CP gives violent 0.9135, property 0.9094, drug
0.9582. Drug over-coverage **reproduces**: you cannot build a discrete interval
on a near-zero series that covers exactly 90%, so drug's 0.99 is expected
behaviour and not a defect. **Property's 0.909 does not reproduce the observed
0.859.**

**Ruled out -- calibration/test drift.** Split CP assumes exchangeability and the
split here is temporal (cal weeks 234-259, test 260-312). But property *falls*
between the windows (Chicago -4.8%, NYC -8.1%), which would make intervals too
**wide** and *over*cover. The observed direction is the opposite.

**Confirmed -- category-specific overconfidence.** Inflating the predicted `r` on
property only (r x 6, i.e. the head claims less dispersion than the truth has)
reproduces the pattern. A single **additive** conformal threshold cannot repair
heteroskedastic miscalibration: the same count-offset is a large relative
widening for drug and a negligible one for property, which is why every method
inherited the spread.

### The fix (implemented, commit 88db19c)

The evaluation grouped **every** calibrator by demographic quartile and **none**
by category. Two variants now exist, `mondrian_category` and
`mondrian_demo_x_category`, on the synthetic analogue of the real defect:

| method | violent | property* | drug | marginal | **spread** |
|---|---|---|---|---|---|
| split_cp | 0.9777 | **0.7336** | 0.9995 | 0.9036 | 0.2659 |
| mondrian (demographic) | 0.9784 | **0.7371** | 0.9995 | 0.9050 | 0.2624 |
| **mondrian (category)** | 0.9231 | **0.9020** | 0.9527 | 0.9259 | **0.0507** |

\* = the category whose predicted `r` is inflated 6x.

Two things worth noting. **Grouping on the demographic axis does not help at all**
(0.7336 -> 0.7371) -- the axis has to match the axis the defect lives on. And the
repair is a *redistribution*, not a uniform widening: the fitted per-category
thresholds are `violent 0.00, property 19.00, drug 0.00`, and mean width is
essentially unchanged (42.0 -> 42.7) while the spread collapses 5x. Widths move
violent 52->40, property 60->82, drug 14->6.

Reporting deliberately stays on the **demographic** axis for every method, so
`coverage_disparity` remains comparable -- a category-conditioned method
reporting category disparity would flatter itself against the others.

A regression test pins all of this
(`test_category_conditioning_repairs_category_specific_overconfidence`),
including the negative result that the demographic axis fails, so the category
axis cannot be silently dropped again.

**Still to do:** re-run the evaluation on the real data to confirm the synthetic
repair transfers. Expect property to land near 0.90 and the spread to fall from
~0.13 (Chicago) / ~0.09 (NYC) to under 0.05. This happens automatically in
Phase 1 below.

Report the marginal coverage *and* the per-category breakdown. Do not report only
the marginal number -- the spread is the interesting finding, and it is a
fairness-relevant one for a FAccT-style venue.

### One honest note on `equalized_coverage`

Its `predict()` takes no groups and applies a single global threshold; it
equalises coverage *marginally* by grid-searching a threshold that minimises
cross-group deviation. That is faithful to Romano et al. (2020)'s marginal
variant, but it means the method **can legitimately return results bit-identical
to `split_cp`** when the pooled quantile already minimises the objective -- which
is exactly what happened on NYC. That is not a bug, but do not present it as a
group-conditional method. `mondrian`/`ecrc` are the group-conditional ones.

---

## 4. What is missing before this is submittable

This is the real gap, and it is about the comparison table, not the model.

1. **`outputs/baselines/` is empty.** No classical baselines (STARIMA, XGBoost,
   LSTM-NB, ZINB-GLM) and no deep baselines have been run. The HA and
   seasonal-naive numbers in section 1 come from inside the conformal script.
2. **`outputs/significance/` does not exist.** The DM test in section 1 is
   against HA only. The BH-corrected DM-vs-all-baselines table -- the artifact
   built specifically to answer "is this win real?" -- has never been produced.
3. **`outputs/ablation/_per_seed/` is empty.** No ablation has been run, so no
   component's contribution is measured.
4. **The baseline comparison is not seed-matched.** Ours is a 5-seed EMOS
   ensemble; the baselines are single fits. The per-seed table in section 1
   partly covers this (the worst seed still beats HA on Chicago), but the
   seed-matched aggregation exists in code and has not been run.
5. **No competitive ST-GNN.** A reviewer will ask "compared to DCRNN?" and there
   is currently no answer.

Everything in 1-4 is already wired into `scripts/run_full_campaign.py`. It is a
compute run, not new code.

---

## 5. Ordered roadmap

### Phase 1 -- Complete the table (do this first; no new modelling)

```bash
python scripts/run_full_campaign.py --smoke-first
```

Runs, in dependency order: OICC -> GNN training -> evaluation -> conformal ->
classical baselines -> deep baselines (matched epoch budget, 3 seeds) ->
ablations -> seed-matched aggregation -> BH-corrected DM significance, both
cities, into a timestamped `results_campaign_*/`.

This run also regenerates every CRPS number on the **fixed** metric (section
2.4) and produces the significance and ablation artifacts. Read
`outputs/significance/` first.

### Phase 2 -- Nothing left to change before the run

6. ~~Add category to the Mondrian grouping.~~ **Done** (section 3). The Phase 1
   run will report `mondrian_category` and `mondrian_demo_x_category` alongside
   the existing six methods.
7. ~~Investigate the `weights_source: raw` anomaly.~~ **Resolved -- already fixed
   in code, never exercised.** Both cities selected raw weights over EMA on a
   large validation gap (Chicago 2.4187 vs 3.2643; NYC 3.3105 vs 5.5531). The
   cause was a hardcoded `decay=0.999`. This project trains ~196 windows at
   batch 16, i.e. **~12 optimizer steps/epoch**, so a 0.999 EMA has a
   ~1000-step horizon against a run that does ~600 steps by a typical epoch-50
   early stop -- the average still carried **54.9%** weight on the *initial
   random snapshot*. It was averaging in noise, which is exactly why the raw
   weights won.

   Commit `f82fa90` (2026-08-02 09:34 IST) replaced this with a decay scaled to
   run length. **Both checkpoints predate it**: `run_chicago_1785214452` is
   2026-07-28 and `run_nyc_1785346837` is 2026-07-29, i.e. 5 and 3 days before
   the fix. So the anomaly is a property of the old checkpoints, not of current
   code, and the section-1 numbers come from the raw weights the selector
   correctly preferred.

   | early stop at | steps | init weight, d=0.999 | init weight, auto-decay |
   |---|---|---|---|
   | epoch 30 | 360 | 69.8% | 47.2% |
   | epoch 50 | 600 | 54.9% | 28.6% |
   | epoch 100 | 1200 | 30.1% | 8.2% |
   | epoch 200 | 2400 | 9.1% | 0.67% |

   One caveat worth keeping: the auto-rule targets a horizon of 1/5 of the
   *configured* 200 epochs, because it cannot know where early stopping will
   land. At an epoch-50 stop it still leaves 28.6% on the initial snapshot. So
   Phase 1 is the **first run with a properly-scaled EMA**, and it may now
   select `ema`. If it does, that is free accuracy the current numbers do not
   include. If it still selects `raw`, the honest reading is that EMA does not
   help this model, and the auto-rule should be re-tuned against the *observed*
   stop epoch rather than the configured one.

### Phase 3 -- Strengthen the contribution

8. **One competitive ST-GNN baseline** (DCRNN, STGCN, or AGCRN) at matched epoch
   budget.
9. **Seasonal anchor on the mu head.** Currently `mu = softplus(mu_mlp(x))` with
   no residual mechanism -- the model predicts counts from scratch while the
   baselines remember last year. `mu(t) = softplus(log(anchor(t)) + h(t))` lets
   it learn a correction to lag-52 and recover the baseline exactly at h=0.
   **This is now an optimisation, not a rescue** -- the model already wins
   without it. Representationally it can only help; optimisation can still fail
   to find it. Lower priority than it was when the model appeared to be losing.
10. **Cross-city zero-shot transfer**: train Chicago, evaluate NYC with no
    retraining. Hard to fake, and it tests whether the learned mechanism
    generalises rather than the data.

---

## 6. Honest ceiling

The forecast block is now a real result: a significant, two-city win over both
naive baselines, driven by architecture rather than ensembling. What it is *not*
yet is a demonstrated win over the published ST-GNN state of the art, because no
such baseline has been run. Until Phase 1 and item 8 land, the defensible claim
is **"beats classical and naive baselines significantly on two cities,"** not
"beats the state of the art."

The OICC honest-measurement contribution (crime estimation under reporting bias)
remains the paper's strongest and most novel component, and it stands
independently of the forecast. The KDD-ADS / FAccT framing still fits: the
contribution is honest measurement under reporting bias, now with a forecaster
that genuinely works underneath it.

Three things would make this top-tier and defensible:

1. **The complete comparison table** with BH-corrected significance (Phase 1).
2. **A fair fight against one real ST-GNN** (item 8).
3. **The OICC block kept as the spine**, with the per-category coverage spread
   (section 3) reported as a fairness finding rather than hidden behind the
   marginal number -- now with a calibrator that actually repairs it.
