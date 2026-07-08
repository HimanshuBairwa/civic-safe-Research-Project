# Novelty, Positioning & Related Work — the honest map

*Built from a 4-agent adversarial novelty audit (each agent tried to scoop the contribution across a different literature). This is the document that turns a desk-reject into a defensible submission: every threatening paper is named, and the exact delta against each is stated. Read this before writing the paper's introduction.*

---

## 0. The honest one-paragraph verdict

The project's headline math — the amplification exponent `1/(1−κ)` — is **not new**: it is the closed-loop feedback gain (geometric series `Σκ^n`), known as the **social multiplier** in economics (Glaeser–Sacerdote–Scheinkman 2003) and as Black's feedback-amplifier formula in control theory. The predictive-policing feedback loop with observation-biased recording is **Ensign et al. (2018)**. The passive/active identification duality ("can't detect by watching, can by perturbing") is a **known template** (Mendler-Dünner et al. 2022; Sharma–Hofman–Watts 2015; Algometrics 2026). The "accurate-yet-harmful, invisible to passive validation" phenomenon is **van Amsterdam et al. (2025)**.

**Therefore the paper must NOT claim any of those as its contribution.** What is genuinely unclaimed — and what the paper must be built around — is the **conjunction plus the constructive correction**:

> A coordinate-free amplification elasticity tying an *observation-biased Poisson recording fixed point* to a group-disparity power law, whose gain κ is *point-identified by a detection-sensitivity difference-in-differences*, and — the part no prior work has — a **deployable conformal correction that recovers valid coverage of the latent target**, with principled abstention. Prior work stops at diagnosis; this delivers a fix with a guarantee.

Positioned that way, this is a defensible paper at a good venue. Positioned as "a new universal law / paradigm shift," it is a one-sentence reject.

---

## 1. The threat matrix (name each; state the delta)

| Prior work | What it owns | Our delta (state this in Related Work) |
|---|---|---|
| **Glaeser–Sacerdote–Scheinkman 2003, "The Social Multiplier"** | The closed form `1/(1−κ)` as the endogenous-feedback multiplier; blow-up at κ→1. | We do **not** claim the multiplier. Our delta: κ is a **product of two log-elasticities** (attention-response × recording-response), giving a *coordinate-free* law over any smooth increasing π, g; plus the **disparity power law** `b^{1/(1−κ)}`. Cite them as the origin of the gain. |
| **Ensign et al. 2018, "Runaway Feedback Loops in Predictive Policing"** (arXiv 1706.09847) | The exact loop; observation-biased recording; a parameter κ; a closed-form Pólya-urn fixed point; the disparity link. | Their amplification is **linear / winner-take-all** (degenerate runaway at any disparity); ours is a **smooth power law with a finite pole** at κ*=1. They have no elasticity decomposition, **no identification**, and **no correction**. Ours adds the DiD identification and the coverage-restoring fix. **Read their two-area model in full before finalizing** (the audit agents could not open the PDF). |
| **Algometrics 2026, "Forecasting Under Algorithmic Feedback"** (arXiv 2605.23978) | The passive/active duality as a **named theorem** (passive non-identification; instrumented actions identify feedback). | Finance domain, **IV/randomization** not DiD; **no amplification law**, no Poisson recording model, no disparity power law, **no correction**. We do not claim the duality principle; we claim its **specific DiD instantiation for a recording-loop elasticity** + the correction it enables. |
| **van Amsterdam et al. 2025** (Cell Patterns; arXiv 2312.01210) | "Accurate model → harmful self-fulfilling prophecy; invisible to passive validation." | Exactly our "confidently wrong" intuition — cite it as the precedent. Our delta: a **feedback-gain formalization** with a **coverage** (not calibration) statement and a **correction**. |
| **Perdomo 2025** (arXiv 2503.11713); **Miller–Perdomo–Zrnic 2021** | A predictor can pass all observable calibration tests while being useless (performatively). | Outcome-performativity with **no latent target distinct from observed y**. Ours has a **latent target μ ≠ recorded y**, a **coverage** impossibility, and a **fix**. |
| **Performative Risk Control 2025** (arXiv 2505.24097) | Rigorous risk control **under performativity** on observed outcomes. | Controls the **observed** outcome; the **opposite thrust**. We prove observed-control is insufficient for the latent target, then correct it. |
| **Hashimoto et al. 2018; Wyllie et al. 2024; Taori–Hashimoto 2023** | Disparity amplification / group erasure in performative retraining loops (mostly empirical/stability). | No closed-form exponent, no observation-bias recording channel, **no correction with coverage**. Cite to show "amplification in feedback loops" is established, then differentiate on the constructive fix. |
| **Poisson-tail surveillance (arXiv 2511.12459, 2025)** | Observation-biased Poisson exposure amplifies group disparity; "critical regime." | Static thresholding, **no feedback fixed point**, no `1/(1−κ)`. Cite to pre-empt "Poisson exposure amplification is known." |
| **ShotSpotter DiD (Topper 2024; J. Exp. Crim. 2024); Minneapolis reporting-rate DiD** | The exact empirical design (staggered detection shock + modern DiD) and reduced-form effects on recorded outcomes. | They estimate **reduced-form treatment effects**, not a **structural feedback elasticity κ**. Our reinterpretation (DiD on log recorded rates → κ) is the novel methodological move. Engage the Minneapolis reporting-rate DiD as the nearest identification precedent. |

---

## 2. The contribution, restated to survive review

**Do claim (the surviving delta):**
1. **[C1] Feedback-corrected latent conformal prediction** *(the headline — genuinely unclaimed).* A deployable procedure that deflates the observation-biased record by the identified κ and issues intervals with restored coverage of the *latent* process, with abstention near the runaway threshold. **No prior work corrects; they only diagnose.** *(Implemented: `theory/latent_correction.py`; experiment: `scripts/latent_correction_experiment.py`.)*
2. **[C2] Coordinate-free amplification elasticity** with κ = product of two log-elasticities, and the disparity power law `b^{1/(1−κ)}` — a *quantitative sharpening* of Ensign, explicitly citing the social multiplier as the gain's origin.
3. **[C3] DiD point-identification of κ** from a detection-sensitivity natural experiment — a novel *application* of a known identification template to a recording-loop elasticity.
4. **[C4] The AOBF benchmark** — an open closed-loop simulator (already runs; exhibits the phase transition through the project's own calibrator).

**Do NOT claim:** a "universal law," a "new impossibility theorem," a "paradigm shift," or `1/(1−κ)` as a discovery. Each is scooped and each is a rejection trigger.

**Framing note (important, from the Baltimore 2026 simulation):** present κ as a **local elasticity at the fixed point** (state-dependent), not a global constant — real amplification is non-constant, and a "universal constant" claim is empirically contradicted.

---

## 3. Honest ratings, post-verification

| Axis | If claimed as "landmark law" | If claimed honestly (this doc) + correction verified |
|---|---|---|
| Novelty | 2 (reject — scooped) | **6–7** — a real constructive delta (the correction) over a well-cited base |
| Publication | 2 | **6–7** at a good venue (FAccT / a solid ML or stats venue), *if* the correction experiment holds and the ShotSpotter DiD is run |
| Patent | 1 | **1** (non-patentable) |

These are honest. There is no 10/10 and no IF>20 here; the ceiling for a single-domain method paper with a well-cited base and one constructive result is ~7. That is a genuinely good outcome — a paper that gets in and gets cited — and it is built on the one thing the literature is missing: **a fix, not another diagnosis.**

---

## 4. Pre-submission checklist (kills the remaining rejection risks)
- [ ] Verify **[C1]** numerically (`scripts/latent_correction_experiment.py`): corrected latent coverage ≈ target while naive collapses. *(pending a clean run)*
- [ ] Read **Ensign 2018** §two-area model in full; confirm it never yields `1/(1−κ)`.
- [ ] Write the Related-Work paragraph from §1 verbatim — name GSS, Ensign, Algometrics, van Amsterdam, Perdomo.
- [ ] Reframe κ as a **local, state-dependent** elasticity (Baltimore 2026 caveat).
- [ ] Add STZINB-GNN (KDD 2022) + STMGNN-ZINB (2024) as the base-predictor prior art.
- [ ] Run the **ShotSpotter DiD** on real data for **[C3]** — the most defensible, least-scooped empirical piece.
