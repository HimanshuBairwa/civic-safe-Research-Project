# Results Assessment & SOTA Roadmap (2026-08-02)

Audience: the paper team. This is the honest read of where CIVIC-SAFE stands
*right now* and the ordered work that gets it to a real, defensible SOTA claim.

---

## 0. One-line status

**The model cannot yet claim a win over seasonal-naive, and no numbers that
would support such a claim currently exist in the repo.** Everything below is
about making that statement false.

---

## 1. What the committed "results" actually are

### 1.1 The Chicago conformal audit is a STALE pre-fix artifact

`outputs/conformal_evaluation/chicago_*.json/.md` were produced **2026-06-16**
from checkpoint `outputs/run_1781590552` (timestamp 2026-06-16 11:45:52), and
they are catastrophic:

| Metric | Value |
|---|---|
| CRPS | 15.481 (vs seasonal-naive 4.402) |
| MAE | 18.535 (HA MAE is ~3.9) |
| CRPSS vs seasonal-naive | **-2.517** (3.5x worse) |
| Conformal mean width | 50.33 |

That checkpoint was trained **before** commit `072fc14` (2026-07-28),
*"CRITICAL: concatenate log1p crime history to model input (train side)"*. In
that era the model received **static ACS features only** -- `input_features`
carries no time variation, so the forecaster never saw a single crime count.
It was structurally incapable of beating lag-52.

**The MAE ≈ the mean count is the signature of a near-constant forecast.**
This is not a modeling failure; it is a *provenance* failure: the artifact
describes a model that could not see its own input.

These files are quarantined to `outputs/_stale_pre_log1p/` with a README
explaining exactly that, and the git history now preserves them only as
provenance.

### 1.2 `outputs/baselines/chicago_seed_matched.json` was my own synthetic fixture

A self-test of `aggregate_baseline_seeds.py` wrote `CRPS 2.97 +/- 0.02` (3
seeds) to the **real** results filename at 18:35 today, and `outputs/` is not
gitignored. Deleted. The `.gitignore` now rejects `*_fixture*`, `*_synthetic*`,
and `outputs/_selftest/` so this class of mistake cannot recur.

### 1.3 The evaluation layer after the fixes

The whole significance apparatus added in this session -- per-week CRPS export
on every forecaster, week-index joining, Diebold-Mariano with Newey-West HAC,
moving-block bootstrap, Benjamini-Hochberg, seed-matched baseline aggregation,
and the campaign runner wiring all of it in dependency order -- has **never
been run on real data**. The local machine has no panel and no checkpoints.
Every new result claim must wait for the GPU-server run in section 4.

---

## 2. What the deeper analysis found

### 2.1 CONFIRMED BUG (now fixed): `crps_zinb` silently understated CRPS

The primary metric truncated its CDF sum at `k_max` derived from the predicted
distribution **only**, never from `y`. A forecast that was confidently wrong
on a large count had up to 94% of its penalty discarded:

```
truth=800, forecast~5:  CRPS was 44.75, true value 793.75
```

Because the penalty per discarded step is exactly ~1, this *systematically
flattered whichever model predicted the smallest counts* -- which biased every
model-vs-baseline comparison in the model's favor. Fixed with an analytic tail
term; now exact vs the closed form (Siebert 2023) across the whole regime, and
the 12 CRPS unit tests still pass.

### 2.2 Latent (fixed): trainer-smoke test fed the model the wrong input width

`TestTrainerSmoke` built the model with `num_features=F` but the trainer now
concatenates `log1p(counts)`, so the model saw width 5 vs expected 8. It
predates this session (caused by `072fc14`); fixed and the suite is green.

### 2.3 Diagnosed (unfixed): the model has no seasonal anchor

The relevant comparison, taken from the stale audit's own reported numbers
(these are *inferences from that file*, not fresh measurements -- there is no
panel data locally to measure against):

```
Seasonal-naive (lag-52) CRPS:                 4.40   <- the number to beat
Historical-average CRPS:                      3.88   <- HA is even harder to beat
Model CRPS (stale, pre-log1p):               15.48
Model MAE (stale, pre-log1p):                18.54
```

Note that **HA (3.88) is stronger than seasonal-naive (4.40) on this test
set**, so HA -- not lag-52 -- is the real floor. That the stale model's MAE
(18.54) is several times HA's CRPS is the signature of a near-constant
forecast, consistent with a model that received no temporal input at all.

The gap to bridge is not "tune the head"; it is **architecture**:

1. **The model is asked to predict the count from scratch**, while the
   baselines simply remember the count from 52 weeks ago.
2. **The mu head is a plain Softplus** (`ZINBHead.mu_mlp`), with no
   seasonal-anchor/residual mechanism. It has no way to express
   "last year's count, plus a learned correction."

The fix: make the model learn the *residual* around lag-52, not the raw count.
Concretely, the head should output a multiplicative factor on the seasonal
anchor:

```
mu(t) = softplus( log(anchor(t)) + h(t) )     # anchor = count(t-52) + small const
```

This is the single highest-leverage change in the entire roadmap. It can only
make the model *better* (it can always learn h=0 and collapse to lag-52), it
sharply reduces the burden on the ZINB head, and it directly converts the
"model must beat seasonal-naive" gate from near-impossible to plausible.

**Before you re-run anything, decide: if the model does not beat HA and
seasonal-naive after a real training run, the honest answer is "the
architecture as shipped is not competitive," not "run more seeds."** Treat
HA's 3.88 as the floor to beat, not a baseline to mention.

*Caveat on the anchor: it is a well-motivated hypothesis, not a measured
result. It has not been implemented or tested. The claim "can only make the
model better" is true in the representational sense (h=0 recovers the anchor),
but optimization can still fail to find it.*

### 2.4 What the (quarantined) conformal numbers do *not* mean

The split-CP coverage of 0.9003 is **conditionally valid** -- conformal
coverage is guaranteed by construction regardless of model quality. Wide
intervals around a broken forecaster are not a result. Do not cite them.

---

## 3. The honest current ceiling

- **Routing / feedback-loop / cross-city**: legitimately novel framing, but
  they are *applications on top of a forecast*. If the forecast is broken,
  everything on top inherits the break.
- **The OICC contribution** (the latent-crime-estimation block) is separate
  and the strongest part of the paper. It stands on its own.
- **Any top-tier claim** in the forecast block currently rests on numbers that
  are either stale, synthetic, or not yet generated.

The KDD-ADS / FAccT ceiling stands: the contribution is the honest measurement
of crime under reporting bias, not a claim that a black-box forecaster beats
everyone.

---

## 4. The ordered GPU-server run (single command)

On the A100:

```bash
python scripts/run_full_campaign.py --smoke-first
```

This now runs, in dependency order: OICC contribution -> GNN training ->
evaluation -> conformal -> classical baselines -> deep baselines (matched
epoch budget, 3 seeds) -> ablations -> seed-matched aggregation ->
Diebold-Mariano significance (BH-corrected), for both cities, into a single
timestamped `results_campaign_*/` directory with a live `campaign.log`.

The significance table and the seed-matched table are the artifacts that tell
you whether the model actually wins. **Read those first.**

---

## 5. The SOTA roadmap (ordered)

### Phase 1 -- Make the forecast real (highest leverage)
1. **Add the seasonal anchor to the mu head** (section 2.3). This is the
   single change most likely to turn "loses to seasonal-naive" into a real win.
2. Re-run the campaign; gate every further decision on whether the model
   beats seasonal-naive.
3. Report per-category CRPSS. The NYC drug-crime +0.054 from earlier is a
   weak signal -- if a category is not a win, say so and show which mechanism
   (zero-inflation vs. spatial) is failing there.

### Phase 2 -- Compete against real ST-GNNs
4. Add at least one competitive spatiotemporal baseline (DCRNN, STGCN, or
   AGCRN) at matched epoch budget. Right now the deep baselines are LSTM-NB /
   XGBoost / HA / STARIMA / ZINB / seasonal-naive. A reviewer will ask
   "compared to DCRNN?", and DCRNN should be beaten by a meaningful margin
   for the paper's claim to hold.

### Phase 3 -- Strengthen the contribution
5. **Cross-city zero-shot transfer**: train on Chicago, evaluate on NYC (or
   vice versa) with no retraining. This is a strong, hard-to-fake signal that
   the learned *mechanism* generalizes, not just the data.
6. Keep the OICC block as the paper's spine (honest measurement under
   reporting bias). It is what survives reviewer scrutiny.

---

## 6. What to report at each stage

- **If the model beats seasonal-naive after Phase 1**: report the margin, the
  per-category breakdown, and the DM significance. That is a real result.
- **If it does not**: the honest statement is "CIVIC-SAFE does not beat
  seasonal-naive on this test set," and the architecture needs another pass.
  Do not run more seeds to find a favorable draw.

---

## 7. Bottom line for the paper team

The three things that would make this paper a top-tier, defensible result:

1. **A forecast that beats seasonal-naive** (Phase 1, the anchor).
2. **A fair comparison against at least one real ST-GNN** (Phase 2).
3. **The OICC honest-measurement contribution kept as the spine** -- that is
   the part that is genuinely novel and genuinely defensible.

Everything in this session's commits (per-week export, significance testing,
seed-matched baselines, the campaign runner) exists so that when you *do* run
the model, the paper's numbers are honest, reproducible, and defensible --
and so that a reviewer cannot ask a question the repo cannot answer.
