# CIVIC-SAFE — Forensic Codebase and Results Analysis

Working notes written before drafting the paper. Every number below was read out
of a file in `outputs/` or reproduced by running a script in `scripts/`. Where a
claim could not be traced to either, it is marked **UNVERIFIED** and the reason
is given.

Verification date: 2026-09-03. Test-set year: 2023 (out-of-sample, rolling
one-step-ahead). Repository HEAD: `6c19150`.

Note on a stale finding: earlier in this review, at HEAD `e247dd4`,
`tests/test_ablation_study.py` failed collection because it imported
`_significance_stars` and `generate_uncertainty_table` before either existed,
which took the whole suite to zero collected tests. Commits `2d60755` through
`6c19150` implemented both. At current HEAD that file passes 8/8. The finding is
recorded here only so the earlier note is not mistaken for a live defect.

---

## 1. What the system actually is

Five components, in the order data flows through them.

1. **ZINB output head.** Every cell-week-category gets three numbers:
   `pi` (structural-zero probability), `mu` (NB mean), `r` (dispersion).
   `MATHEMATICS.md` §1. The zero-inflation is not decorative — Chicago's Brier
   score on the zero event is 0.0592, so the atom at zero carries real mass.
2. **Dual-graph GATv2 spatial encoder** (§2) with a multi-factor feature mixer
   and a Jensen-Shannon diversity penalty (§3).
3. **Spatiotemporal graph transformer** with a structured attention mask (§7).
4. **Post-hoc calibration stack**: split CP, randomized-PIT split CP, weighted
   CP, three Mondrian variants, equalized coverage, variance-scaled split CP,
   ECRC, and a rolling adaptive-temporal ECRC (§8).
5. **Latent-correction layer** (§0) — the part that is actually new. Records are
   a biased view of true crime because policing responds to records. A
   difference-in-differences design on a detection-sensitivity shock estimates
   the recording elasticity, and intervals are then computed for the *latent*
   rate rather than the recorded one. Note that `MATHEMATICS.md` §0.2 is
   explicit that this does **not** point-identify the loop gain `kappa = beta *
   rho`: the DiD carries an uncancelled latent-level term and `beta` must be
   assumed, so `kappa` belongs in a sensitivity table. On real data the DiD is a
   null. The paper follows the spec here, not the looser "point-identified"
   phrasing that appears in the task brief.

The fifth item is the contribution. Items 1-4 are competent engineering on
known methods; a referee will read them as such.

---

## 2. Headline predictive results — all verified

Source: `outputs/conformal_evaluation/{city}_conformal_results.json`,
key `point_forecast_metrics`.

| | Chicago | NYC |
|---|---|---|
| CRPS | **2.8267** | **3.1401** |
| MAE | 3.9017 | 4.3675 |
| RMSE | 7.0983 | 7.6126 |
| Brier (zero event) | 0.0592 | 0.0493 |

Both match the prompt's ground-truth values to four decimals. Ensemble is 5
seeds (42, 137, 256, 512, 1024) from `run_{city}_anchor_*`, combined by
category-conditioned entropy-regularized EMOS.

### Skill scores — and the baseline that actually binds

Source: same files, key `skill_scores`.

| | Chicago | NYC |
|---|---|---|
| CRPSS vs rolling HA | 0.0360 | 0.0494 |
| CRPSS vs *frozen* HA | 0.2713 | 0.3165 |
| CRPSS vs seasonal naive | 0.3577 | 0.3362 |
| binding baseline | `ha_rolling` | `ha_rolling` |

This is worth being careful about in the paper. The impressive-looking 27-32%
skill is against a *frozen* historical average. Against a *rolling* HA — which
is the honest baseline, since it updates as new weeks arrive — the gain is
**3.6% and 4.9%**. The code itself flags `ha_rolling` as the binding baseline.
Lead with the small honest number; a reader who finds the big number and then
discovers it was against a frozen baseline will stop trusting everything else.

### Diebold-Mariano tests — verified, but the count needs care

Source: `outputs/significance/{city}_deep_significance.json`.

| baseline | Chicago DM | Chicago p | NYC DM | NYC p |
|---|---|---|---|---|
| GraphWaveNet | −9.319 | < 1e−16 | −15.321 | < 1e−16 |
| LSTM_NB | −7.482 | 7.33e−14 | −6.979 | 2.97e−12 |
| STZINB_GNN | −10.186 | < 1e−16 | −11.158 | < 1e−16 |
| TFT_ZINB | −4.790 | 1.67e−06 | −7.669 | 1.73e−14 |

All 8 comparisons favour CIVIC-SAFE (DM negative = our loss is lower). **The
"8/8 wins at p < 1e-6" claim is WRONG, and this note previously repeated it.**
The largest of the eight is Chicago/TFT_ZINB at p = 1.671454e-06, which is
*greater* than 1e-6. The tight true bound is p < 2e-6. Every occurrence in the
manuscript has been corrected; a referee checking Table I against the abstract
would have caught this immediately.

Two cautions. These 8 are against *deep* baselines only — the rolling-HA
comparison lives in a different file and is much weaker (Chicago p = 0.0338,
NYC p = 0.0036). And `paper/results_summary.md` reports different DM statistics
for NYC (−6.45, −3.21, −15.59, −9.78) than the significance JSON does
(−6.979, −7.669, −15.321, −11.158). The JSON is newer and is what the campaign
actually produced. **The paper must use the JSON; `results_summary.md` is stale
on this point.**

---

## 3. Conformal calibration — verified, with one real failure

Source: key `coverage_results`. Target coverage 90%, disparity ceiling 0.03.

### Chicago (selected: `equalized_coverage`)

| method | coverage | width | disparity |
|---|---|---|---|
| split_cp | 0.9405 | 16.25 | 0.0182 |
| randomized_split_cp | 0.9347 | 16.54 | 0.0115 |
| weighted_cp | 0.9075 | 14.58 | 0.0238 |
| mondrian | 0.9169 | 15.02 | 0.0319 |
| mondrian_category | 0.9313 | 17.19 | 0.0250 |
| mondrian_demo_x_category | 0.9305 | 17.35 | 0.0119 |
| **equalized_coverage** | **0.9075** | **14.58** | **0.0238** |
| variance_scaled_split_cp | 0.9079 | 14.65 | 0.0242 |
| ecrc | 0.9235 | 15.89 | 0.0156 |
| adaptive_ecrc_rolling | 0.8930 | 13.88 | 0.0013 |

### NYC (selected: `variance_scaled_split_cp`)

| method | coverage | width | disparity |
|---|---|---|---|
| split_cp | 0.9326 | 18.57 | 0.0201 |
| randomized_split_cp | 0.9328 | 18.60 | 0.0201 |
| weighted_cp | 0.9326 | 18.57 | 0.0201 |
| mondrian | 0.9326 | 18.57 | 0.0201 |
| mondrian_category | 0.9241 | 17.91 | 0.0132 |
| mondrian_demo_x_category | 0.9319 | 18.25 | 0.0154 |
| equalized_coverage | 0.9326 | 18.57 | 0.0201 |
| **variance_scaled_split_cp** | **0.9002** | **16.45** | **0.0286** |
| ecrc | 0.9186 | 17.41 | 0.0138 |
| adaptive_ecrc_rolling | 0.8918 | 16.31 | 0.0046 |

Selected coverage 90.75% / 90.02%, disparity 0.0238 / 0.0286 — **both match the
prompt and both sit under 0.03.** Selection was made by
`select_best_calibrator` in `src/civicsafe/calibration/policies.py` (narrowest
width subject to coverage, disparity, abstention, status), `fallback_used:
false` for both cities. Abstention is 0.000 everywhere.

Three honest observations.

**NYC's disparity of 0.0286 clears 0.03 by 0.0014.** That is one resampling away
from failing. Report it as "under the pre-registered ceiling" and do not
editorialize about how comfortably.

**Four NYC methods are numerically identical** (split_cp, weighted_cp, mondrian,
equalized_coverage all at 0.9326 / 18.57 / 0.0201). That is the degenerate-
threshold signature: when the conformity-score quantile lands on the same
integer, methods that differ only in *how they pick* the quantile collapse to
the same interval. Worth a sentence — it is evidence about the data, not a bug.

**`adaptive_ecrc_rolling` has by far the best disparity** (0.0013 Chicago,
0.0046 NYC) and the narrowest intervals, but coverage of 89.30% / 89.18% falls
below the 90% floor, so the policy correctly refused to select it. This is a
nice demonstration that the constraint is load-bearing rather than decorative.

### The Chicago PIT failure — do not bury this

Source: key `calibration_diagnostics`.

| | Chicago | NYC |
|---|---|---|
| PIT max deviation | 0.0368 | 0.0053 |
| PIT chi-square | 241.84 | 10.75 |
| PIT p-value | **5.25e−47** | 0.293 |
| `pit_is_uniform` | **false** | **true** |

Chicago's PIT histogram is not uniform, and it is not close. The top bin holds
13.68% of mass against an expected 10%, and the bottom bin 9.52%. The
monotone rise across the upper bins says the model is systematically
under-predicting the right tail on Chicago — high-count cell-weeks land further
out in the predictive distribution than they should.

NYC passes cleanly (p = 0.293).

This is the paper's most exposed weakness and the first thing a good referee
will find. It has to appear in Limitations in plain language, with the chi-square
statistic. The defensible framing: conformal coverage is a *finite-sample
marginal* guarantee that holds under exchangeability regardless of whether the
underlying model is correctly specified, which is precisely why we calibrate
post hoc instead of trusting the ZINB's own quantiles. Chicago's 90.75%
coverage is real even though its PIT fails. But we should say the model is
misspecified on Chicago's tail, because it is.

Per-category coverage also spreads more than the marginal number suggests:
Chicago violent 0.8883, property 0.8829, drug 0.9515 — a 0.069 spread, with two
of three categories under 90%. NYC: violent 0.8677, property 0.8950, drug
0.9378.

---

## 4. Hersbach decomposition and recalibration

Source: keys `crps_decomposition`, `recalibration`.

| | Chicago | NYC |
|---|---|---|
| Reliability (lower better) | 0.00124 | 0.0000604 |
| Resolution (higher better) | 9.4280 | 10.7048 |
| Uncertainty | 12.2534 | 13.8448 |
| CRPSS vs climatology | 0.7693 | 0.7732 |

Reliability is a rounding error next to resolution — nearly all the CRPS is
irreducible spread, not miscalibration.

**Recalibration was fitted and then deliberately not applied in both cities.**
`recal_applied: false`, `recalibration_gate: "identity fallback"`. The gate
fired on an internal calibration holdout: Chicago −5.29%, NYC −0.66%. So
recalibration *hurt* on held-out data and the pipeline declined it. Note
`test_improvement_pct: 0.0` for both — the gate decided on the holdout, not on
the test set, which is the leakage-free way to do it. This is a genuinely good
design detail and belongs in Methods.

## 5. EMOS

Chicago marginal weights `[0.114, 0.171, 0.481, 0.121, 0.114]`; NYC
`[0.124, 0.266, 0.226, 0.142, 0.243]`. Chicago leans on seed 256 but no longer
collapses onto it.

The category-conditioned weights tell a sharper story. Chicago category 1 is
`[0.0021, 0.0007, 0.9958, 0.0010, 0.0004]` — 99.6% on a single seed. Category 0
sits at exact uniform `[0.2, 0.2, 0.2, 0.2, 0.2]`, and `emos_fallback_used:
true` with `emos_fallback_by_category` recording per-category decisions. So the
entropy regularizer plus fallback gate is working: where the learned weights did
not beat equal weighting on holdout, the pipeline reverted to uniform. Ensembling
buys about 16% CRPS (single-model 3.3622 → 2.8267).

---

## 6. Policy simulation — verified, and the tradeoff is real

Source: `outputs/policy_simulation_results.json`, 24 rows.

At budget B = 100:

| city | policy | hit rate | alloc. disparity |
|---|---|---|---|
| Chicago | naive_ha | 0.9399 | 0.0185 |
| Chicago | point_prediction | 0.9391 | 0.0140 |
| Chicago | unconstrained_conformal | 0.9529 | 0.0132 |
| Chicago | **civic_safe_oicc** | **0.9636** | **0.1448** |
| NYC | naive_ha | 0.9689 | 0.0167 |
| NYC | point_prediction | 0.9660 | 0.0156 |
| NYC | unconstrained_conformal | 0.9859 | 0.0174 |
| NYC | **civic_safe_oicc** | **0.9897** | **0.0421** |

Hit rates 96.36% / 98.97% match the prompt exactly, and OICC is best on hit rate
in both cities.

**But OICC's allocation disparity is the worst of the four policies in both
cities** — Chicago 0.1448 vs 0.0132 for unconstrained conformal (11x worse), NYC
0.0421 vs 0.0156 (2.7x worse). `results_summary.md` frames OICC as fairness-
improving via the *over-allocation ratio* (1.036 → 0.643 Chicago), which is a
different metric measuring something real: OICC stops over-policing relative to
incident share. Both are true. The paper must report both, because claiming
OICC "improves fairness" while its allocation-disparity column is the worst on
the table is the kind of selective reporting that ends a submission.

At B = 20 OICC is also the *worst* on hit rate in both cities (0.5242 vs 0.5677
Chicago; 0.4896 vs 0.5014 NYC). The gains only appear at larger budgets. Say so.

---

## 7. The novel contribution — reproduced, with a denominator problem

### 7.1 Latent coverage correction

Reproduced by running `python scripts/latent_correction_experiment.py
--trials 12 --cells 4000`. Output:

| kappa | delta | naive latent cov | kappa_hat | corrected latent cov | kept frac |
|---|---|---|---|---|---|
| 0.00 | 0.60 | 0.950 | 0.259 | 0.949 | 0.75 |
| 0.30 | 0.60 | 0.903 | 0.300 | 0.952 | 1.00 |
| 0.50 | 0.60 | 0.780 | 0.500 | 0.948 | 0.95 |
| 0.70 | 0.29 | 0.502 | 0.713 | 0.937 | 0.56 |
| 0.85 | 0.06 | 0.162 | 0.850 | 0.930 | 0.15 |

The naive and corrected columns match the prompt's Figure 10 spec exactly. The
DiD estimator recovers `kappa` well at 0.3-0.85 (0.300, 0.500, 0.713, 0.850).
This is a strong, real result: coverage of the *truth* collapses to 16.2% under
a biased record, and the correction holds it at 93%.

**The `kept_frac` column is the problem.** Reading
`scripts/latent_correction_experiment.py` lines ~100-108: `naive` is
`np.mean(...)` over **all** cells, while `corrected` is `np.mean(...)` over
`[keep]` only. At kappa = 0.85 the corrector abstains on **85% of cells** and
the 93.0% is measured on the surviving 15%.

So "16% → 93%" compares two different denominators. The comparison is not
wrong, but stated without the abstention rate it is misleading, and this is
exactly the defect we already fixed once in the conformal evaluation path
(commit `d28c7d0`, where unmasked abstention was being scored as miscoverage).
The fix here is not code — it is disclosure. Every figure, table, and sentence
reporting corrected coverage must carry the retained fraction next to it.
At kappa = 0.0 the corrector also abstains on 25% of cells while adding nothing
(0.950 → 0.949), which is worth one honest sentence: the abstention rule is
tuned for the high-gain regime and costs coverage-neutral cells at low gain.

Also note `kappa_hat = 0.259` when true `kappa = 0.0`. The estimator has real
upward bias at zero gain. It does not hurt the coverage result but it should be
reported.

### 7.2 Routing disparity

Reproduced by `python scripts/routing_disparity_experiment.py`. Group-1
structural over-recording factor 1.8.

| kappa | biased disparity | corrected | reduction |
|---|---|---|---|
| 0.00 | 0.287 | 0.287 | 0.000 |
| 0.30 | 0.291 | 0.206 | 0.085 |
| 0.50 | 0.280 | 0.148 | 0.132 |
| 0.70 | 0.270 | 0.089 | 0.182 |
| 0.85 | 0.182 | 0.044 | 0.138 |

Matches the prompt's Figure 11 spec exactly. At kappa = 0.85, 0.182 → 0.044 is a
**75.8% reduction** — the prompt's "76%" is right.

One caveat a referee will raise: biased disparity *falls* from 0.270 to 0.182
between kappa 0.70 and 0.85, so the worst biased case is not the highest gain.
Do not describe the curve as monotone.

### 7.3 Cross-city disparity

Reproduced by `python scripts/cross_city_disparity.py`.

| city | units | biased | corrected | reduction |
|---|---|---|---|---|
| Chicago | 77 | 0.390 | 0.163 | 0.227 (−58.2%) |
| NYC | 78 | 0.311 | 0.122 | 0.189 (−60.8%) |

Matches the prompt's Figure 12 spec, and −58% / −61% are right.

**The script prints two caveats that must survive into the paper**: the
correction is applied at an *assumed* `kappa = 0.6`, not an identified one; and
"latent coverage is not validated on real data (true rate unobservable)." So
this is a sensitivity analysis on real records under an assumed gain, not a
validated coverage result. The coverage guarantee is simulation-only (§7.1).
Presenting §7.3 as empirical validation of the correction would misrepresent it.

---

## 8. Numbers in the prompt that did NOT verify

Four items, in descending order of how much they matter.

1. **Figure 10's corrected line as given (`[94.9, 95.2, 94.8, 93.7, 93.0]`) is
   real but incomplete** — it hides `kept_frac` `[0.75, 1.00, 0.95, 0.56, 0.15]`.
   Plotting it without the retained fraction shows a flat 93-95% line that
   silently changes denominator across the x-axis. Fix: annotate retention on
   the figure.
2. **`results_summary.md`'s NYC DM statistics are stale** (−6.45 / −3.21 /
   −15.59 / −9.78 and "p = 0.0013" for TFT_ZINB) and disagree with
   `outputs/significance/nyc_deep_significance.json` (−6.979 / −7.669 /
   −15.321 / −11.158, TFT_ZINB p = 1.73e−14). Use the JSON.
3. **"Disparity < 0.03"** is true for both selected calibrators but NYC's margin
   is 0.0014. Do not call it comfortable.
4. **`results_summary.md` claims "20 verified figures"**; `outputs/figures/`
   holds 12 distinct panels (9 numbered + 3 extra) across PNG/PDF. Minor, but
   the paper should not repeat the inflated count.

Nothing in the prompt was fabricated outright. Every headline number traces to a
file or reproduces from a script. The issues are framing and disclosure, not
invention.

---

## 9. What I would tell a referee, unprompted

The strongest thing here is §7.1: a measurement-error problem stated precisely,
a feedback gain estimated from a detection shock rather than simply assumed, and a correction
that restores latent coverage from 16% to 93%. That is a real contribution and
it is the paper's spine.

The weakest points, in the order they will be found:

- **Chicago PIT fails at p = 5e−47.** Own it in Limitations.
- **The corrected-coverage denominator** (85% abstention at kappa = 0.85).
  Disclose it everywhere the number appears.
- **A single seed loses to TFT_ZINB.** The win requires 5-seed EMOS. The
  architecture alone is not enough, and `results_summary.md` already admits
  single-model CRPS is 3.3622 vs TFT_ZINB's 2.9456 on Chicago.
- **Real-data correction uses an assumed kappa = 0.6**, and latent coverage is
  unvalidated on real data by construction.
- **OICC's allocation disparity is the worst on the policy table**, and at
  B = 20 its hit rate is worst too.
- **Rolling-HA skill is 3.6% / 4.9%**, not the 27-32% that a frozen baseline
  would suggest.

None of these sink the paper. All of them, unaddressed, would sink it in review.
Stating them plainly and early is what makes the rest credible.
