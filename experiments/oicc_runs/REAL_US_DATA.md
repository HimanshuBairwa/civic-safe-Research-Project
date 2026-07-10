# Dropping real US data into the multi-channel experiment

*This is the last mile of Option (b): once you have the two external channels
(NCVS + 911), put them here and the isolated experiment runs the first genuine
3-channel latent analysis on real US data. Until then, everything runs on demo
data and the rest of the project is completely unaffected.*

## What the runner expects

Three **aligned** 1-D arrays, saved as `.npy`, under `data/processed/`:

```
data/processed/us_records.npy    # police-recorded crime  (count or rate per cell)
data/processed/us_ncvs.npy       # NCVS victimization      (rate per cell, broadcast)
data/processed/us_cfs.npy        # 911 calls-for-service   (count or rate per cell)
```

**Alignment contract (critical):** all three must be the *same length* and in the
*same order*, one entry per area×period cell (e.g. Chicago community-area × month).
Non-negative values; the loader applies `log1p` internally. Then:

```bash
python experiments/oicc_runs/run_us_multichannel_experiment.py --real
```

## How to build the three arrays (concrete recipe)

You already have **records** (in `data/processed/chicago_panel.pt`). For a
Chicago community-area × month grid over 2018–2023:

1. **Records** — aggregate the existing panel to area×month, flatten to a vector
   `us_records.npy` (order: area 0 months 0..M, area 1 months 0..M, ...). Keep
   this ordering fixed for all three channels.

2. **911 CFS** — from the Chicago open-data portal (Calls for Service / OEMC).
   Aggregate to the *same* community-area × month grid and flatten in the *same*
   order. Save `us_cfs.npy`.

3. **NCVS** — NCVS is coarser than area-month. Broadcast/small-area-estimate the
   NCVS victimization *rate* to each Chicago cell (simplest defensible version:
   assign the national/regional NCVS rate for that crime type × year to every
   community area in that year; better: a small-area model using ACS covariates).
   Flatten in the same order. Save `us_ncvs.npy`.

> Helper: `python experiments/oicc_runs/make_us_npy.py --help` scaffolds step 1
> from the existing panel and shows the exact expected shapes.

## What you get

- The over-identification test becomes a **genuine check** that three
  independently-biased channels are consistent with one latent rate.
- Bootstrap-CI'd loadings and `Var(latent)`, and leave-pivot-out latent
  prediction intervals — on real data.
- The first **partial real-data support** for the latent target OICC estimates.

## The honest limit (state it in any writeup)

Three real channels still cannot see a **common-mode** confounder shared by all
of them (e.g. a city-wide reporting-culture shift) — that is the proved
impossibility. NCVS + 911 make the assumption *partially checkable* and the paper
*empirically stronger*; they do not remove the impossibility. Report the over-ID
result as consistency evidence, not as proof of unbiasedness.
