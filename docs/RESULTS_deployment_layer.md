# Verified Results — Deployment Layer (monitor · routing · sensitivity)

*All numbers reproduce from the scripts/tests below. This layer turns the
feedback-correction theorem into a deployable pipeline:
**identify $\kappa$ → correct → route on truth → monitor live → bound the trust**.
Every component is tested (`tests/test_feedback_tripwire.py`,
`tests/test_feedback_aware_routing.py`, `tests/test_sensitivity.py`).*

---

## 1. Feedback-aware routing shrinks exposure disparity

Risk-aware routing over the raw (observation-biased) record systematically
diverts people around over-recorded neighborhoods — navigational redlining.
Routing over the feedback-corrected latent risk reduces the worst-group exposure
disparity. Two groups with identical latent incidence; group 1 structurally
over-recorded by $1.8\times$ (`scripts/routing_disparity_experiment.py`):

| $\kappa$ | biased disparity | corrected disparity | reduction |
|------:|-----------------:|--------------------:|----------:|
| 0.00  | 0.287            | 0.287               | 0.000     |
| 0.30  | 0.288            | 0.206               | 0.082     |
| 0.50  | 0.294            | 0.148               | 0.147     |
| 0.70  | 0.269            | 0.089               | 0.180     |
| 0.85  | 0.237            | 0.045               | **0.193** |

**Reading.** At $\kappa=0$ (no feedback) the disparity is purely structural and
the correction — an identity map at zero gain — cannot and should not touch it.
As the feedback loop strengthens, more of the disparity is *feedback-induced* and
therefore correctable: disparity falls from 0.29 to **0.045** at $\kappa=0.85$.
The stronger the loop, the more redlining the correction removes — the honest,
defensible pattern.

## 2. Anytime-valid feedback tripwire (live monitor)

A test supermartingale over the coverage stream (method-of-mixtures betting;
Waudby-Smith & Ramdas 2024) gives a *time-uniform* false-alarm guarantee
(Ville's inequality: $P(\text{ever fire}\mid\text{null})\le$ `alarm_level`),
valid under continuous monitoring with nothing to tune
(`tests/test_feedback_tripwire.py`, 4/4 pass):

- **Null (calibrated stream, 90% coverage):** false-alarm rate within the budget
  across 200 runs.
- **Feedback regime (miscoverage 0.45 vs.\ nominal 0.10):** fires reliably and
  early.
- **Over-coverage (98%):** never fires (one-sided — only under-coverage alarms).
- **Monotone power:** larger miscoverage drift is detected sooner.

This is the online counterpart to the offline impossibility: passive metrics
cannot certify latent coverage, but a betting monitor on periodic anchor
observations gives an anytime-valid *tripwire* that fires as the loop drifts
toward runaway.

## 3. Robustness envelope of the correction

The correction uses an *estimated* $\kappa$; `theory/sensitivity.py` reports how
much misspecification the latent-coverage guarantee tolerates
(`tests/test_sensitivity.py`, 3/3 pass):

- Latent coverage is (near-)maximal when the used gain matches the truth.
- The **robustness value** — the largest gain error keeping coverage above a
  floor — is finite and non-negative, and **shrinks as $\kappa\to1$**: near the
  runaway threshold the correction is more fragile, so the identifying natural
  experiment must pin $\kappa$ more precisely. This directly sets the precision
  target for the real-data ShotSpotter DiD.

---

## What this establishes / does not

- **Establishes (in simulation of the AOBF model):** a complete deployable
  pipeline — identify, correct, route, monitor, and bound trust — each component
  tested. No prior work provides the correction, let alone the surrounding
  safety layer.
- **Does not establish:** external validity on real crime data. The open item
  remains the ShotSpotter/patrol-rollout DiD to estimate $\kappa$ in the field
  (see `docs/NOVELTY_AND_POSITIONING.md`).

## Reproduce

```
python scripts/routing_disparity_experiment.py
pytest tests/test_feedback_tripwire.py tests/test_feedback_aware_routing.py tests/test_sensitivity.py -q
```
