# Getting the two missing channels (NCVS + 911) — your step-by-step guide

*This is the "what do I actually do" for **Option C**: acquiring the two real
data channels that turn OICC's latent-coverage claim from "validated only on
synthetic data" into "partially checkable on real US data." That is the single
thing that genuinely lifts the honest rating ceiling (from ~6.5/7.5 toward a
stronger empirical paper). It is **data-acquisition work**, not code.*

---

## Why these two channels (the 30-second version)

OICC needs **≥3 mechanism-independent channels** of the same latent crime rate.
You already have **police records** (channel 1). The two highest-value additions,
each biased by a *different* mechanism, are:

| Channel | Who generates it | Bias mechanism (why it's independent) |
|---|---|---|
| **Police records** (have) | police decide what to record | enforcement / recording discretion |
| **NCVS** (need) | household survey of victims | recall + willingness to tell a surveyor; **captures crime never reported to police** |
| **911 calls-for-service** (need) | citizens who choose to call | who-calls filter; citizen-initiated, not police-initiated |

With all three, "latent coverage" becomes **over-identified and partially
testable on real data** — exactly the untestable assumption that caps the paper.

---

## The two tracks, at a glance

| | NCVS | 911 Calls-for-Service |
|---|---|---|
| Public version | **Yes — free** (ICPSR/NACJD) | **Often yes** (city open-data portals) |
| When you need a formal request | only for **sub-national geography** (MSA/county) | only if the city **doesn't** publish it |
| Effort (public) | 1 afternoon (download) | 1 afternoon per city (download) |
| Effort (restricted) | 2–6 months (BJS/FSRDC application) | 2–8 weeks (FOIA per city) |
| Cost | free | free–small (some FOIA fees; ask for waiver) |

**Recommended path (fastest to a real result):** use **public NCVS** (national/
regional) + **open-data 911 CFS** for Chicago and NYC (you already have their
police records). That needs **no formal application** and can be done this week.
Escalate to the restricted-data application only if a reviewer demands
tract/MSA-level NCVS.

---

## TRACK 1 — NCVS (National Crime Victimization Survey)

### 1A. Public use files (do this first — free, no permission)
1. Go to the National Archive of Criminal Justice Data (NACJD) at ICPSR:
   https://www.icpsr.umich.edu/web/NACJD/series/95  (NCVS series).
2. Create a free ICPSR account (any academic/personal email).
3. Download the **NCVS Collection** years you need (2018–2023 to match your
   Chicago/NYC panel). Grab both the **person** and **incident** files.
4. You now have national + region-level victimization rates by crime type. This
   is your near-unbiased "anchor" channel at coarse geography.

**What you can publish with just this:** a national/regional 3-channel OICC run
(records + NCVS + 911) with an honest note that NCVS geography is coarse.

### 1B. Restricted (sub-national) NCVS — only if you need MSA/county detail
This requires a formal application because sub-national identifiers are
confidential. Two routes:
- **NACJD Restricted Data Use Agreement (RDUA)** — lighter weight; or
- **Census/BJS Federal Statistical Research Data Center (FSRDC)** proposal —
  heavier, for the finest geography.

You will need: an **IRB determination** (your institution's ethics board; often
an "exempt" determination suffices for secondary de-identified data), a
**data-security plan**, and **named researchers**. Timeline ~2–6 months.

**→ A ready-to-send request cover letter is in
[`ncvs_restricted_request.md`](ncvs_restricted_request.md). Fill the [brackets].**

---

## TRACK 2 — 911 Calls-for-Service

### 2A. Check the city open-data portal first (free, instant)
Many major cities publish CFS with no request needed:
- **Chicago:** https://data.cityofchicago.org — search "calls for service" /
  OEMC. (You already use Chicago police records from the same portal.)
- **New York City:** https://opendata.cityofnewyork.us — NYPD Calls for Service
  (there are historic + year-to-date CFS datasets).
- Others with CFS portals: Seattle, New Orleans, Dallas, Detroit, Cincinnati,
  Baltimore.

Download call-level records with: date/time, call type/nature code, and a
**geographic unit** (beat / precinct / community area) — **no street address or
caller PII needed**. Aggregate to your area×period grid, exactly like the
records channel.

### 2B. If a city doesn't publish CFS — file a public-records request
File a state public-records / FOIA request with that city's police department.

**→ A ready-to-send FOIA template is in
[`cfs_foia_request.md`](cfs_foia_request.md). Fill the [brackets].**

---

## YOUR CHECKLIST (do these in order)

- [ ] **1.** Create an ICPSR account and download **public NCVS 2018–2023**
      (person + incident files). *(~1 hour)*
- [ ] **2.** Download **911 CFS for Chicago** from the Chicago open-data portal;
      aggregate to community-area × week to match your existing panel. *(~2 hours)*
- [ ] **3.** Download/**FOIA 911 CFS for NYC** (portal first; FOIA if needed).
- [ ] **4.** Tell me when you have any one city's CFS + NCVS — I will wire them
      into the OICC pipeline (I've left `experiments/oicc_runs/` ready for a
      `us_multichannel_loader.py`) and run the **first real 3-channel latent
      over-identification test** on US data.
- [ ] **5.** *(Optional, for the top-tier version)* Get your institution's **IRB
      determination** and submit the **NCVS restricted-data application** (letter
      provided) for MSA-level geography.

## What changes in the paper once this lands

- The over-ID test runs on **genuinely independent US channels** (not
  same-filter categories) → a real specification test on the flagship dataset.
- You can report **cross-channel agreement / partial latent validation on real
  data** — moving the honest limitation ("latent coverage validated only on
  synthetic data") to "partially validated on real US data."
- Realistic effect on rating: pushes the empirical section from "illustrative"
  to "substantiated," which is the difference between a borderline and a solid
  accept at KDD-ADS / FAccT.

## The honest caveat (so you're not surprised)

Even three real channels do **not** defeat the common-mode impossibility (a
confounder like a city-wide reporting-culture shift that moves *all* channels
together stays invisible). NCVS + 911 make the assumption **partially checkable**
and the paper **empirically stronger** — they do not make it "beyond NeurIPS."
That ceiling is a theorem, not a data problem. This is still the highest-value
real move available.
