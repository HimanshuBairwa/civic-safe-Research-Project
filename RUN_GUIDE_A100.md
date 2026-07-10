# A100 Run Guide — the exact command sequence (verified against the code)

> Copy-paste, in order, on your A100 Jupyter/Docker. Every command here was
> checked against the actual scripts. Read the **"What actually matters"** box
> first — it saves you ~30 GPU-hours of confusion.

---

## ⭐ What actually matters (read this once)

Your project has **two parts**, and they are very different to run:

1. **OICC — the research contribution.** It is a *statistical estimator*, NOT a
   trained neural net. It runs in **seconds on CPU**, needs **no GPU, no training,
   no waiting.** This is the part that gets you published. → Section C.
2. **The CIVIC-SAFE ZINB-GNN forecaster — an applied baseline.** This is what
   "training" means here. It takes **~15-20 GPU-hours per city** (200 epochs × 5
   seeds) and, honestly, it does **not beat a seasonal-naive baseline** — it is
   documented as prior-art baseline, not the contribution. Train it if you want
   the full pipeline/figures, but know it is not the headline. → Section B.

**So:** run Section C first (5 minutes, the real result). Then optionally train
the GNN (Section B) if you want the forecasting figures too.

---

## Section A — Sync + housekeeping (do once)

```bash
# 1. go to your existing clone on the A100
cd /path/to/civic-safe-Research-Project

# 2. YOUR workflow: pull the latest masterpiece from GitHub.
#    (Your DATA lives under data/ which is gitignored -> reset/pull NEVER touch it.)
git reset --hard origin/main
git pull origin main

# 3. install the pinned env (numpy<2.1 for the torch stack; oicc needs only numpy+scipy)
pip install -r requirements-a100.txt
pip install -e .

# 4. ARCHIVE the old outputs (do NOT delete -- zero data loss, zero confusion).
#    New runs write to a fresh timestamped dir so they never collide, but
#    archiving the old ones keeps the folder clean.
mkdir -p archive
mv outputs archive/outputs_OLD_$(date +%Y%m%d) 2>/dev/null || true
mkdir -p outputs
#    (archive/ and results_campaign_*/ are gitignored, so the next
#     `git reset --hard` will NOT delete them either.)

# 5. sanity check the whole box (env + GPU smoke + oicc tests + reproduction)
python run_all.py                     # must print ALL GREEN
```

**On `git reset --hard`:** it is safe here. Your datasets, archived outputs, and
campaign results are all gitignored, so reset only refreshes tracked *code*.

**On the "old outputs" question:** I chose **archive, not delete** — nothing is
lost, and the new `outputs/` is clean. If you are 100% sure you want them gone:
`rm -rf archive/outputs_OLD_*` later. Never needed for correctness.

---

## Section A.5 — ⭐ THE ONE-COMMAND MAX-RIGOR CAMPAIGN (use your unlimited A100)

For the "run it for days, everything top-notch" outcome, this single command runs
the **entire** publication pipeline into one clean timestamped dir:

```bash
export OICC_INDIA_DATA=/path/to/crime-detection-ai/data     # your India data
nohup python scripts/run_full_campaign.py --seeds 15 --epochs 200 \
      > campaign.log 2>&1 &                                 # runs for hours; survives disconnect
tail -f campaign.log                                        # watch progress
```

It produces `results_campaign_<timestamp>/` containing: the rigorous OICC
reproduction (tight CIs), the real India NCRB run, the US contrast, all
publication figures (heatmaps + choropleth), and the 15-seed GNN baseline
training for both cities + conformal evaluation. Everything logged; zero collision
with old outputs.

Variants:
```bash
python scripts/run_full_campaign.py --oicc-only    # just the contribution (~5 min, no GPU)
python scripts/run_full_campaign.py --skip-train   # OICC + figures, no GPU training
python scripts/run_full_campaign.py --seeds 15     # 15-seed publication CIs (default)
```

> **What "max rigor" actually means (honest):** unlimited A100 buys **more seeds**
> (15 → tight mean±std confidence intervals) and **bigger Monte-Carlo/bootstrap**
> counts on the OICC contribution (`reproduce_all.py --rigorous` uses 80 seeds /
> 3000 trials). It does **NOT** buy more epochs — the forecaster early-stops at a
> plateau (~epoch 52), so 500 epochs = 200 epochs = wasted time. Seeds and
> bootstrap precision are the levers reviewers actually reward.

---


## Section B — Train the GNN forecaster (optional, GPU, slow)

Your US panels (`data/processed/chicago_panel.pt`, `nyc_panel.pt`, + `_graph.pt`)
are **already on the docker** from the old version — **no download needed.**

```bash
# FIRST: a 2-minute smoke test to confirm training works on YOUR A100
python scripts/train.py data=chicago training.epochs=2 training.num_seeds=1

# If that finishes clean, run the FULL max-rigor training (15 seeds, 200 epochs):
python scripts/train.py data=chicago training.num_seeds=15   # ~2-3 days on 1 A100
python scripts/train.py data=nyc     training.num_seeds=15

#   Each writes to  outputs/run_<city>_<timestamp>/seed_<seed>/best.pt
#   Run them in the background so a dropped session doesn't kill them:
#   nohup python scripts/train.py data=chicago training.num_seeds=15 > train_chicago.log 2>&1 &
#   nohup python scripts/train.py data=nyc     training.num_seeds=15 > train_nyc.log     2>&1 &
```

> **15 seeds is the max-rigor choice** (the seed pool in
> `configs/training/default.yaml` now holds 15). It gives publication-grade
> mean±std. **Do NOT raise epochs past 200** — the model early-stops at a plateau
> (~epoch 52); extra epochs are wasted. Seeds, not epochs, are the lever.

After training, evaluate + calibrate (point `--checkpoint` at the run dir printed
by training, e.g. `outputs/run_chicago_<timestamp>`):

```bash
python scripts/run_conformal_evaluation.py --data chicago --checkpoint outputs/run_chicago_<TS>
python scripts/run_conformal_evaluation.py --data nyc     --checkpoint outputs/run_nyc_<TS>
python scripts/baselines.py data=chicago        # HA / seasonal-naive / ARIMA / XGBoost
python scripts/baselines.py data=nyc
python scripts/generate_figures.py --data chicago
python scripts/generate_figures.py --data nyc
```

---

## Section C — OICC, the actual contribution (fast, no GPU) ⭐

This is the part that matters. It runs in minutes.

```bash
# 1. reproduce every headline number (17 machine-checked assertions)
python experiments/oicc_runs/reproduce_all.py          # prints 17/17 passed

# 2. REAL Indian data run (you already have crime-detection-ai on the box).
#    Point the resolver at it (adjust the path to where it lives on the docker):
export OICC_INDIA_DATA=/path/to/crime-detection-ai/data
python experiments/oicc_runs/run_ncrb_experiment.py    # India NCRB, 4 channels

# 3. US cross-national contrast (uses the panels already on the box)
python experiments/oicc_runs/run_us_experiment.py

# 4. Publication figures (vector PDF + PNG, colorblind-safe, real Chicago geo)
python experiments/oicc_runs/make_pub_figures.py       # -> paper/figures/pub/
python experiments/oicc_runs/make_figures.py           # -> paper/figures/  (core 5)

# 5. full test suite, if you want the green wall (390 pass)
python -m pytest tests_oicc/ -q                        # OICC only (~1 min)
python -m pytest tests/ tests_oicc/ -q                 # whole repo (~13 min)
```

**Routing / heatmaps / diagrams** are produced by step 4 (`make_pub_figures.py`):
the method schematic, the over-ID power heatmap (visually proves the
impossibility), the channel-correlation heatmap, and the recovered-latent Chicago
choropleth. Routing itself is exercised by the test suite and the civicsafe
`generate_figures.py` (Section B).

---

## Do you need to download any dataset for the new version?

**No — nothing new to download for the core project.**
- **US panels (Chicago/NYC):** already on the docker (old version). Used by both
  training and OICC.
- **India NCRB:** the `crime-detection-ai` folder you already have; just set
  `OICC_INDIA_DATA` to point at it.
- **The only *new* data would be NCVS + 911 — that is the optional (b) lever, see
  below. You do NOT need it to run everything above.**

---

## The (b) question — NCVS + 911, and the Indian alternative

**If you skip (b): does it hurt majorly? No.** Here is the honest math:
- With (b): empirical axis ~8/10, whole project ~8/10.
- Without (b): empirical axis ~6/10, whole project stays **~7.5/10** — still a
  solid, publishable KDD-ADS/FAccT paper.
- (b) is a *nice-to-have upgrade*, not a *requirement*. **You can ship without it.**

**The Indian-context alternative (better for you than chasing US NCVS/911):**
India has genuine independent-channel data you can use *instead*:
- **NCRB channels you already run** — recorded crime + complaints-against-police +
  custodial deaths + HR violations are 4 *mechanism-independent* institutional
  channels. That IS the real-data multi-channel OICC run (`run_ncrb_experiment.py`),
  and it is Indian data. This already substitutes much of what (b) would add.
- **For a survey channel (the NCVS analogue):** the **India Human Development
  Survey (IHDS-II)** and **NFHS** contain victimization / safety questions at
  district level — a free, public, Indian survey channel. If you ever want the
  extra lever, that is the low-headache Indian path (no US FOIA needed).

**My recommendation:** ship with the India NCRB multi-channel run as your
real-data anchor; treat NCVS/911/IHDS as a "future work / v2" upgrade. Do **not**
let data-acquisition block the paper.

---

## The minimum "IF" (Impact Factor) — the honest answer

First, a correction that matters: **the top target venues are conferences, not
journals, so they don't have a classical Impact Factor** — and that's a *good*
thing, they're more prestigious than most IF-bearing journals in CS.

| Venue (realistic) | Type | Metric | Honest read |
|---|---|---|---|
| **KDD Applied Data Science** | conference | h5-index ~120+ | top-tier CS; no IF but very high prestige |
| **FAccT** | conference | flagship fairness venue | top-tier; no classical IF |
| **AOAS** (Annals of Applied Statistics) | journal | **IF ≈ 1.8–2.0** | strong applied-stats home |
| **Journal of Quantitative Criminology** | journal | **IF ≈ 3–4** | excellent domain fit |
| **NeurIPS Datasets & Benchmarks** | conference track | — | strong for the benchmark |

**So the honest minimum:** if you publish the *journal* version (AOAS or JQC),
realistically **IF ≈ 2–4**. If you publish at KDD-ADS / FAccT (the natural home),
IF doesn't apply but the prestige is *higher* than a mid-IF journal. Either way
this is a genuinely good, citable outcome for the honest ~7.5/10 project.

**What I will NOT tell you:** that this reaches IF>10 / Nature / NeurIPS-main.
That ceiling is blocked by a *theorem* (the impossibility), which I proved this
session even higher moments/ICA cannot crack. IF 2–4 (or a top conference) is the
real, defensible target — and it's a strong one.

---


## Should we delete any old code? (my call)

**No — leave the code as-is.** The old feedback-law material is already
*corrected and banner-retracted* in the docs, and its code paths are covered by
passing tests. Deleting them would break tests for no benefit. The honesty fixes
already neutralized every false claim; there is nothing harmful left to remove.
The only cleanup that helps is **archiving old outputs** (Section A step 4), which
we already do.

---

## TL;DR — the minimal "greatest" sequence

```bash
cd /path/to/civic-safe-Research-Project
python scripts/a100_sync.py && pip install -r requirements-a100.txt && pip install -e .
mkdir -p archive && mv outputs archive/outputs_OLD_$(date +%Y%m%d); mkdir -p outputs
python run_all.py                                          # verify: ALL GREEN
python experiments/oicc_runs/reproduce_all.py             # the real result: 17/17
export OICC_INDIA_DATA=/path/to/crime-detection-ai/data
python experiments/oicc_runs/run_ncrb_experiment.py       # real Indian data
python experiments/oicc_runs/run_us_experiment.py         # US contrast
python experiments/oicc_runs/make_pub_figures.py          # heatmaps + choropleth
# optional, slow, GPU baseline forecaster:
python scripts/train.py data=chicago training.num_seeds=3 training.epochs=100
python scripts/train.py data=nyc     training.num_seeds=3 training.epochs=100
```

That gives you the whole project — the real contribution (fast), the real Indian
data run, the publication figures, and the trained baseline — with zero
confusion between old and new outputs.
