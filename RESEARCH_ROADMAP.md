# CIVIC-SAFE / OICC — Master Research Roadmap

> **One file to understand the entire project.** What it is, why it matters, what
> its honest ceiling is and why, everything we have built, the current state, and
> exactly what is left. Read top-to-bottom and you know the whole story.
>
> **Maintainer note:** update this file whenever a milestone lands. It is the
> single source of truth. Last updated: after the publication-figure + routing-
> honesty work (v0.6.0, merged to `main`).

---

## 0. TL;DR (read this first)

- **What we built:** **OICC** — *Over-Identification-Calibrated Conformal
  Deconvolution* — a method that estimates a **latent** true-crime/victimization
  rate that is **never directly observed**, from **≥3 mechanism-independent
  biased measurement channels** (police records, 911 calls, victimization survey,
  accountability records…), and — the honest part — **quantifies how much of its
  own answer is defensible.**
- **Status:** complete, tested (**387 tests green**), reproducible (17 machine-
  checked headline numbers), **merged to `main`**, runs error-free on an A100,
  arXiv package built, publication-grade figures generated.
- **Honest rating:** a strong **applied-statistics / FAccT / KDD-ADS**-class
  paper (novelty ~6.5/10, publication ~7.5/10). **NOT** "beyond NeurIPS."
- **Why the ceiling is firm:** a **proved impossibility theorem** (see §3). No
  amount of code changes it. The only lever that raises it is **data** (§8).
- **The one thing that lifts it:** real independent channels (NCVS + 911) →
  turns "latent coverage validated only on synthetic data" into "partially
  checkable on real US data." That is data-acquisition work, not code (§8).

---

## 1. The research problem (why this matters)

Administrative records of social harms — recorded crime, 911 calls, complaints,
custodial-death filings — **are not the harm itself.** Each is a *biased,
mechanism-specific measurement* of a latent rate we never observe. Police record
what they choose to; citizens call about what they choose to; surveys capture
recall. Treating any one channel — or a naive average — as "the crime rate"
conflates the harm with the machinery that records it. That conflation drives
biased resource allocation and corrupts fairness audits of the very institutions
producing the data.

**The target:** a latent per-area-period rate `θ` (e.g. true victimization in a
Chicago community area in a month), measured by `K ≥ 3` channels via a one-factor
model:

```
  Y^c = α_c + β_c · θ + ε^c      (c = 1 … K, errors mutually independent given θ)
```

Two facts make it hard and interesting: (1) `θ` is **never observed**, so you can
never directly validate a point estimate or interval on real data; (2) **honesty
about what cannot be known** is as important as the estimate — a method that
silently reports a tight interval around a *confounded* quantity is worse than
useless in a policy/fairness setting.

---

## 2. The method — OICC, component by component

Everything below is implemented in `src/oicc/` (numpy + scipy only) and tested.

| # | Component | File | What it does |
|---|-----------|------|--------------|
| 1 | **Moment identification** | `moments.py` | Recovers loadings `β` and `Var(θ)` in closed form from averaged covariance **tetrads**. Over-identified for K≥4. |
| 2 | **Latent recovery (BLUP)** | `deconvolve.py` | Empirical-Bayes best linear unbiased predictor of `θ` from all channels. |
| 3 | **Over-ID specification test** | `spec_test.py` | *Loading-invariant* test that the channels really share one latent, in **detectable (Δ⊥)** directions. Two flavors: 2nd-moment tetrad (K≥4) **and a new 3rd-cumulant test with real power at K=3** (where 2nd moments are just-identified). Moving-block bootstrap for dependent panels; bootstrap-null p-value option. |
| 4 | **Two-interval conformal** | `conformal_split.py` | *Leave-pivot-out* primitive → (a) an **EXACT finite-sample distribution-free** interval for an observed channel value, and (b) a model-assisted **asymptotic** interval for the latent `θ`. |
| 5 | **CF deconvolution** | `cf_deconv.py` | Nonparametric characteristic-function recovery of the latent-error law (for heavy tails). |
| 6 | **Proximal / negative-control escape** | `proximal.py` | With Q≥2 negative-control channels, **point-identifies** `Var(θ)` free of a common-mode confounder; `exclusion_sensitivity` quantifies the one untestable assumption. |
| 7 | **Anytime-valid monitor** | `monitor.py` | Vovk–Wang p-to-e calibrated **e-process** over the over-ID p-value stream; **Ville's inequality** ⇒ time-uniform false-alarm control in deployment. |
| 8 | **Uncertainty (bootstrap)** | `uncertainty.py` | i.i.d. and moving-block bootstrap CIs on every estimator. |
| 9 | **Baselines** | `baselines.py` | OICC vs single-channel / naive-average / reporting-rate; and under confounding, proximal vs all naive methods. |

**The genuinely novel piece** is the **leave-pivot-out conformal primitive** (a
computable interval for a *never-observed* target) plus the **impossibility/escape
pairing**. Everything else is a careful, correct *composition* of known tools
(Kotlarski/Hu–Schennach deconvolution, Miao–Tchetgen-Tchetgen proximal inference,
conformal prediction, Cinelli–Hazlett sensitivity, Waudby-Smith–Ramdas anytime-
valid). We claim composition + primitive, never "we invented factor analysis."

---

## 3. THE CEILING — the honest core (understand this)

**The impossibility theorem.** A **common-mode confounder** `W` that loads on
*all* channels *proportionally to the signal* (`γ_c = κ·β_c`) is **absorbed into
the estimated factor and is unidentifiable — by any over-ID test, at any K, at any
moment order.** The observable law is exactly a one-factor model in the composite
`F = θ + κW`. So `Var(θ)` is *not a function of the observed covariance* — it is a
free parameter.

**Why it caps the rating.** OICC's estimand is `E[θ]` and `Var(θ)`. A Gaussian
common-mode confounder saturates *exactly those two moments* — it sits precisely
on top of the target. So:
- you **cannot** validate latent coverage on real data (θ is never seen);
- you **cannot** rule out the confounder from the channels alone.

**We proved higher moments don't save it (v0.6.0).** The 3rd-cumulant over-ID
test is *also* blind to the common mode — demonstrated in running code — so the
impossibility holds at 3rd order, not just 2nd. **Deep literature research
(incl. non-Gaussian ICA) confirmed the ceiling is firm:** `W ∝ β` is exactly the
*collinear-column* case where higher-order/ICA identification **provably fails**
(Reiersøl 1950; Comon 1994; Cramér/Linnik). Non-Gaussianity buys a sharper spec
test and distributional-shape robustness — **not** a crack in the impossibility.

**This is a feature, not a bug, for the paper.** We *prove* the barrier, then
provide the only honest escape (external negative controls, with a stated
untestable assumption *plus* a sensitivity analysis bounding its impact). Reviewers
who know the measurement-error literature (Knox–Lowe–Mummolo, selective labels,
Moreno–Girón) will respect that we obey the impossibility instead of hand-waving
past it. **A version that claimed to "solve" it would be a desk-reject.**

**Honest ratings (unchanged and defended):**

| Axis | Score | Why |
|---|---|---|
| Novelty | **6.5 / 10** | real new primitive + composition; every base tool is prior art |
| Publication | **7.5 / 10** | two correct theorems, real data, pre-registered honesty, survives expert review |
| Impact | **7.0 / 10** | deployable bias-audit tool; transfers to epidemiology/ecology under-reporting |
| Patent | **2 / 10** | a mathematical method; not patentable |

Target venues: **KDD Applied Data Science / FAccT** (main), **AOAS / JQC**
(archival stats), **NeurIPS Datasets & Benchmarks** (the benchmark).

---

## 4. What raises the ceiling vs what does NOT

**Does NOT (settled by research — don't waste effort here):**
- ❌ Higher moments / ICA on the same channels (collinear-column failure — proved).
- ❌ A fancier estimator, more tests, more figures (rigor, not ceiling).
- ❌ A cleverer conformal wrapper (the latent is still never observed).

**DOES (the only real levers):**
- ✅ **Real independent-channel data** (NCVS + 911) → makes latent coverage
  *partially checkable* on real US data. **This is the #1 lever.** (§8)
- ✅ A genuinely new *identification* strategy needing a *new data structure*
  (e.g. prospective logged predictions à la Cheng et al. ICML 2024 via an agency
  partnership) — a multi-month research bet, not a re-analysis.
- ✅ A brand-new conformal/martingale *primitive* valid under endogenous feedback
  without prediction logging — none currently exists; would be its own paper.

---

## 5. What we have DONE (chronological, with evidence)

**Core method (v0.1 → v0.6.0):**
- ✅ One-factor moment ID, BLUP recovery, over-ID tetrad test.
- ✅ Leave-pivot-out two-interval conformal (exact observed 0.90; latent ~0.89).
- ✅ Proximal point-ID: recovers true `Var(θ)=0.61` where naive inflates to 3.18.
- ✅ Exclusion-sensitivity band + robustness value `ε*` (answers the #1 reviewer
  objection that the negative-control exclusion is untestable).
- ✅ Anytime-valid e-process monitor (false-alarm 0.013 ≤ 0.05, Ville).
- ✅ Bootstrap CIs (i.i.d. + moving-block) on every estimator.
- ✅ **3rd-cumulant over-ID test** — real power at K=3; demonstrates all-order
  impossibility.
- ✅ **Block-bootstrap correctness fix** — on real dependent panels, India NCRB
  p=0.088 (i.i.d.) → ~0.45 (block); US rejection survives at p<0.001. Honest.
- ✅ **Baseline comparison** — proximal cuts latent RMSE 0.56→0.26 under
  confounding (a 55% reduction) where every naive method fails.

**Real-data experiments:**
- ✅ **India NCRB** (state-year 2001–2010, N=253, 4 institutional channels):
  one-factor not rejected once dependence respected; recovered latent face-valid
  (high: Maharashtra/MP/Andhra; low: small NE states). Two-control point-ID run.
- ✅ **US Chicago/NYC** (2018–2023): crime **categories** share one police filter
  → over-ID **correctly rejects** (p<0.001), the opposite verdict to India's
  independent channels. A **two-directional** validation of the spec test.
- ✅ **Recovered latent choropleth** on real Chicago community-area geography.

**Rigor / reproducibility / deployment:**
- ✅ **387 tests** pass (301 civicsafe + 86 OICC), 1 CUDA-skip.
- ✅ `reproduce_all.py` — asserts **17 headline numbers**, exits non-zero on drift.
- ✅ **A100-ready:** Dockerfile (CUDA), `requirements-a100.txt`, `run_all.py`
  preflight, device-agnostic training smoke test, lazy optional imports (no
  script crashes on a fresh box), numpy-2.x safe, headless matplotlib.
- ✅ **Merged to `main`** (PR #1). The A100 pull-from-main gets everything.
- ✅ **arXiv package** (`paper/arxiv/`): validated tarball builder + metadata.
- ✅ **Publication figures** (`paper/figures/pub/`, Okabe–Ito CVD-safe palette,
  vector PDF): method schematic, over-ID power heatmap, coverage, point-ID +
  exclusion band, channel-correlation heatmap (real), latent choropleth (real),
  anytime-valid monitor.
- ✅ **Data-access paperwork** (`docs/data_access/`): NCVS restricted-data
  application draft + 911 FOIA template + step-by-step checklist.
- ✅ **Isolated (b) US multichannel experiment** — runs on demo today, switches
  to real NCVS+911 when present; imported by nothing (core is independent of it).

---

## 6. Current state (snapshot)

- **Branch/merge:** `main` = complete (merge commit of PR #1); `oicc-method` also
  exists. Repo: `github.com/HimanshuBairwa/civic-safe-Research-Project`.
- **Version:** `oicc` v0.6.0, 34 exports.
- **Tests:** 387 pass / 1 skip (whole repo). OICC-only: 89 pass / 1 skip.
- **Reproduction:** 17/17 assertions green.
- **A100:** `git pull origin main` → `pip install -r requirements-a100.txt` →
  `python run_all.py` = ALL GREEN. **Data is NOT in git (see §8) — must be
  provisioned separately.**

---

## 7. Routing / the "Tsinghua algorithm" — the honest decision (IMPORTANT)

**Finding (deep research on Duan, Mao, Mao, Shu & Yin 2025, STOC best paper):**
- The real algorithm is a **recursive BMSSP** (bounded multi-source SSSP) with a
  FindPivots subroutine and a block-based partial-sorting structure — it achieves
  `O(m log^{2/3} n)` by **avoiding** any full sort of the frontier.
- The current `src/civicsafe/routing/tsinghua.py` does **Bellman-Ford passes then
  repeatedly `frontier.sort()` and settles a batch** — which **re-sorts the
  frontier (the exact thing the paper avoids)**, has no recursion / no pivots / no
  block structure, and is **NOT** the Duan et al. algorithm.
- **Practical fact:** faithful BMSSP is **3–25× SLOWER than Dijkstra** at any real
  graph size; the theoretical crossover is ~`10^60+` nodes. For a ~100-node crime
  graph, Dijkstra is strictly better.

**Decision (to protect a top-tier submission):** **drop the false "breaks the
sorting barrier / O(m log^{2/3} n) / Duan et al." claim.** Routing is a downstream
utility layer, not the contribution. Use **exact Dijkstra** (already implemented
and correct), describe it honestly, and add a **one-sentence forward citation**:
"at metropolitan-to-national scale, advances such as Duan et al. (2025) could
reduce routing cost, though current implementations do not beat Dijkstra below
~10^60 vertices." A false claim about a marquee STOC result is an instant
desk-reject and would poison the credibility of the genuine contribution.
**Status: DONE — router renamed `BatchedFrontierRouter` (legacy alias kept),
engine defaults to exact Dijkstra, all docs corrected + forward citation.**

**UPGRADE (routing is now a genuine SUPPORTING contribution, not just a utility).**
The *algorithm* stays honest Dijkstra, but the routing *problem* now carries two
real, tested guarantees (see `docs/METHODOLOGY.md` §9.5):
- **(G1) Path-level conformal exposure certificate** — a finite-sample,
  distribution-free upper bound on the *realized* risk-exposure of the route a
  policy returns (split conformal on the exposure functional; empirical-coverage
  test passes across 400 splits). Novel in application: prior conformal-navigation
  work calibrates robot obstacle sets; interval-cost robust routing is NP-hard and
  over-conservative. `src/civicsafe/routing/exposure_conformal.py`.
- **(G2) Debiasing breaks the runaway feedback loop** — allocating on the OICC
  latent field (whose 911/survey channels are patrol-independent) keeps belief
  calibrated (≈0.95 vs record-only ≈0.41) and cuts the over-patrolled group's
  exposure disparity by ≈0.9 in a controlled simulation (robust across seeds).
  `experiments/oicc_runs/run_feedback_routing_experiment.py`.
- **Honesty fix:** the earlier "DiD point-identifies the feedback gain κ" claim is
  retracted; κ is now a *sensitivity* knob only, and the identified debiased field
  is OICC (`oicc_routing_field`). Ties routing to the measurement contribution:
  *honest risk → honest routing.* Impact ~7.0 → ~7.5; does NOT change the venue
  ceiling (OICC remains the headline). Figure: `pub_fig8_routing`.

---

## 8. Does it run on REAL data on the A100? — the precise answer

**The code runs on the A100 with zero errors. The DATA is not in git** (correct —
it is large/licensed; `.gitignore` excludes `data/`). So `git pull` gets the code,
not the datasets. To run on real data on the A100 you must provision it:

| Dataset | In git? | How to get it on the A100 |
|---|---|---|
| US panels (Chicago/NYC `.pt`) | ❌ (gitignored) | copy `data/processed/*.pt` to the A100, **or** re-run `scripts/fetch_data.py` (hits the city open-data APIs). |
| India NCRB | ❌ (separate sibling repo `crime-detection-ai`) | copy that folder to the A100 and set `OICC_INDIA_DATA=/path/to/crime-detection-ai/data`. |
| NCVS + 911 (the real lever) | ❌ (not yet obtained) | see `docs/data_access/` — download NCVS (ICPSR, free) + Chicago 911 (open portal, free), format to `.npy`, drop in `data/processed/`. |

**Bottom line for the A100:** synthetic experiments + all tests + reproduction run
with **nothing extra**. Real-data experiments need the data copied over (a `scp`
or an API re-fetch) — the code then runs them without error. The path resolver +
graceful skips mean *missing data never crashes anything*.

**The genuine ceiling lever (do LAST, if time):** obtain NCVS + 911, format to the
3-channel `.npy` contract (`experiments/oicc_runs/REAL_US_DATA.md`), run
`run_us_multichannel_experiment.py --real`. That gives the first real-data latent
over-identification — the one result that strengthens the empirical section.

---

## 9. What is LEFT (prioritized TODO)

**In progress / next:**
1. ✅ **World-class figures** — DONE (`paper/figures/pub/`, 7 vector PDFs, Okabe–Ito).
2. ✅ **Routing honesty fix (§7)** — DONE. `tsinghua.py` → honest
   `BatchedFrontierRouter` (alias kept), engine defaults to exact Dijkstra, all
   docs (README, MATHEMATICS, METHODOLOGY, PAPER_OUTLINE) corrected + forward
   citation. No false sorting-barrier claim remains.
3. ✅ **Whole-repo false-claim audit** — DONE. Corrected the false DiD
   "point-identifies κ" claim (PROOFS/MATHEMATICS retracted + corrected; legacy
   `docs/paper.tex` and `docs/AUDIT_2026-07.md` banner-superseded), demoted the
   ZINB-GNN forecaster to "applied baseline (does not beat seasonal-naive)" with a
   README banner pointing at OICC, removed "novel" tags from standard methods,
   fixed stale test counts (264→387) and the arXiv-coming-soon badge.
4. ✅ **A100 sync tooling** — DONE (`A100_SYNC.md` + `scripts/a100_sync.py`):
   data-preserving update from `main` (data is gitignored → never overwritten).
5. ☐ **Wire the new pub figures into `paper/oicc_paper.tex`** (add heatmaps +
   choropleth; the 5 older figs are already referenced).
6. ☐ **Finalize the plain-text arXiv abstract** in `paper/arxiv/metadata.txt`.

**Optional / higher-effort:**
5. ☐ Runnable external baselines (MSE/LCMCR, reporting-rate scale-up) as a
   comparison table for maximal empirical defensibility.
6. ☐ **(b) real-data run** — NCVS + 911 ingestion → first real latent validation.
   *(the real lever; gated on data, do last)*
7. ☐ Interactive HTML dashboard (hover/tooltips) if a web artifact is wanted.

**Explicitly NOT doing (would lower the rating):**
- ✗ Claiming to beat the impossibility.
- ✗ Claiming the Duan et al. speedup.
- ✗ Inflating novelty beyond composition + the one primitive.

---

## 10. Repository map (where everything lives)

```
src/oicc/                 the method (numpy+scipy): measurement, moments,
                          deconvolve, spec_test, cf_deconv, conformal,
                          conformal_split, proximal, monitor, uncertainty, baselines
src/civicsafe/            legacy GNN forecaster (applied prior art; device-hardened)
tests_oicc/               89 OICC tests (synthetic + real-data + stress + device)
tests/                    301 civicsafe tests
experiments/oicc_runs/    loaders, runners, reproduce_all, figures, US multichannel
  make_pub_figures.py       publication figure suite (Okabe–Ito, vector PDF)
  oicc_style.py             validated palette + NeurIPS/Nature style
  run_ncrb_experiment.py    India NCRB real run
  run_us_experiment.py      US cross-national contrast
  run_us_multichannel_experiment.py   isolated (b) records+NCVS+911 experiment
  REAL_US_DATA.md           how to drop real data in
paper/
  oicc_paper.tex            submission-quality write-up (5 figs; being upgraded)
  OICC_THEOREMS.md          formal theorems + proofs
  figures/pub/              publication-grade vector figures
  arxiv/                    validated tarball builder + metadata + PR body
theory/OICC_identification_theorems.tex   full identification theory
docs/data_access/         NCVS + 911 request templates + checklist
Dockerfile, requirements-a100.txt, run_all.py   A100 reproducibility
RESEARCH_ROADMAP.md       (this file)
```

---

## 11. The elevator pitch (for a stranger)

> "Police records aren't the truth about crime — they're a biased snapshot of what
> got recorded. OICC estimates the *true* latent rate from several independently-
> biased measurements, and — uniquely — it tells you exactly how much of its answer
> is trustworthy: it tests whether the sources agree, proves which kind of bias is
> mathematically impossible to remove without outside data, and gives calibrated
> uncertainty intervals with a formal coverage guarantee. It runs on real Chicago,
> NYC, and India data. It's honest about its ceiling, which is exactly what makes
> it publishable."

*End of roadmap. Keep it current.*
