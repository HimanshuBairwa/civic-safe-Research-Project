# Added Results — Misspecification Robustness & Cross-City Real Data

*Two additions that strengthen the paper without new external data. Both are
verified: the robustness guarantee in `tests/test_correction_robustness.py`
(7/7 pass), the cross-city numbers on the real Chicago + NYC records.*

---

## 1. Robustness to a misspecified recording model (hardens the core)

**The objection this answers:** the correction deflates by an *assumed* power-law
recording multiplier; a referee will ask what happens if the true recording
mechanism differs.

**The guarantee (Rosenbaum-style sensitivity model).** If the true recording
multiplier lies within a factor $\Gamma$ of the assumed one for every cell, then
a $\Gamma$-inflated corrected interval (lower quantile at $\hat\lambda/\Gamma$,
upper at $\hat\lambda\,\Gamma$) covers the latent process at the nominal rate,
for *any* admissible recording model. Verified against Poisson latent draws from
a misspecified world (target coverage $0.90$, true $\kappa=0.6$):

| $\Gamma$ | naive-corrected coverage | $\Gamma$-inflated coverage | width ratio |
|-----:|-------------------------:|---------------------------:|------------:|
| 1.0 | 0.949 | 0.949 | 1.00 |
| 1.3 | 0.935 | 0.984 | 1.29 |
| 1.6 | 0.907 | 0.992 | 1.54 |
| 2.0 | 0.857 | **0.995** | 1.84 |

**Reading.** Without inflation, coverage of the latent target degrades as the
recording model is misspecified (0.949 → 0.857, below target at $\Gamma=2$). The
$\Gamma$-inflated interval stays valid throughout, at a bounded, smoothly-growing
width cost (up to $1.84\times$). This converts the correction's main assumption
into a stated robustness envelope: *we tolerate any recording model within a
factor $\Gamma$; the price is a known widening.* The `robustness_gamma` helper
reports the largest $\Gamma$ meeting a target on given data — an interpretable
robustness value. (Implementation: `theory/correction_robustness.py`.)

## 2. Cross-city real-data exposure disparity (external validity)

On the **real** Chicago (77 community areas) and NYC (78 precincts) records,
2018–2023, we measure the recorded violent-crime exposure of the higher-minority
stratum (units above the median `pct_black`) relative to its population share,
and how the correction (assumed $\kappa=0.6$) redistributes it:

| City | biased exposure disparity | corrected | reduction |
|------|--------------------------:|----------:|----------:|
| Chicago | +0.390 | +0.163 | 0.227 |
| NYC | +0.311 | +0.122 | 0.189 |

**Reading.** On both cities' genuine records, risk-aware allocation on the raw
recorded rate over-exposes the higher-minority stratum by 31–39% relative to
population; correction attenuates it to 12–16%. The pattern replicates across two
independent cities with real demographics.

**Honest scope.** $\kappa$ is *assumed* here (its field value needs the
identification experiment, `docs/RESULTS_field_identification.md`); the disparity
magnitudes are real; latent coverage is *not* validated on real data (the true
rate is unobservable) — the coverage guarantee is the simulation result. This is
a descriptive real-data companion, clearly labeled as such.
(Script: `scripts/cross_city_disparity.py`.)

## Reproduce
```
pytest tests/test_correction_robustness.py -q
python scripts/cross_city_disparity.py --category violent --kappa 0.6
```
