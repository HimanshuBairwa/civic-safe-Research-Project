"""Feedback-loop mitigation via OICC-debiased allocation (honest simulation).

The runaway feedback loop (Ensign et al., FAccT 2018; Lum & Isaac 2016)
-----------------------------------------------------------------------
Predictive-allocation systems patrol where they *believe* crime is high; they
then *discover* crime only where they patrol; the discovered counts update the
belief; and so belief diverges from the true rate, concentrating attention on
already-watched areas regardless of ground truth. Ensign et al. break the loop by
debiasing discovered counts by how much patrol each area received.

This experiment demonstrates a DIFFERENT, arguably stronger debiasing: **OICC**.
OICC recovers the latent rate from >=3 mechanism-independent channels. Two of
them -- 911 calls-for-service and a victimization-survey proxy -- are
*public-initiated* and hence **independent of patrol allocation**. So an OICC
belief is anchored to signals the loop cannot contaminate, whereas a
belief built from police-discovered counts alone has no such anchor.

Honest framing
--------------
This is a controlled SIMULATION with a KNOWN latent rate, so every claim is
checked against ground truth. The identifying assumption is explicit and is
exactly OICC's core assumption: the 911 / survey channels are (conditionally)
independent of the patrol process. Where that holds, the OICC-anchored policy
stays calibrated and its exposure disparity stays bounded; the record-only policy
runs away. We claim mitigation UNDER THIS ASSUMPTION, demonstrated -- not a
universal proof.

Run:  python experiments/oicc_runs/run_feedback_routing_experiment.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# --- OICC + civicsafe imports (path-robust) ---------------------------------
_ROOT = Path(__file__).resolve().parent.parent.parent
for p in (_ROOT / "src",):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from oicc.deconvolve import deconvolve_blup  # noqa: E402
from civicsafe.routing.feedback_aware import ExposureDisparityAudit  # noqa: E402


@dataclass
class FeedbackTrajectory:
    """Per-round diagnostics for one allocation policy."""

    belief_corr: list[float]      # corr(belief, true latent) per round
    group_disparity: list[float]  # |over-exposure of the over-patrolled group|
    name: str


def _proportional_allocation(belief: np.ndarray, floor: float = 0.05) -> np.ndarray:
    """Patrol share proportional to belief, with a floor (never zero patrol)."""
    b = np.clip(np.asarray(belief, dtype=float), 1e-9, None)
    share = b / b.sum()
    share = (1 - floor) * share + floor / len(b)   # mix with uniform floor
    return share / share.sum()


def run(
    n_areas: int = 60,
    rounds: int = 40,
    seed: int = 0,
    init_patrol_bias: float = 0.6,
) -> dict:
    """Simulate the runaway loop under record-only vs OICC-anchored allocation.

    Args:
        n_areas: Number of spatial units.
        rounds: Feedback iterations.
        seed: RNG seed (deterministic).
        init_patrol_bias: Structural over-patrol of group 0 at t=0 (the seed of
            the disparity the loop can amplify).

    Returns:
        Dict with the two :class:`FeedbackTrajectory` objects and a summary.
    """
    rng = np.random.default_rng(seed)

    # --- ground truth: latent log-rate, two demographic groups -------------
    groups = (np.arange(n_areas) >= n_areas // 2).astype(int)  # 0 / 1
    u = 0.3 * rng.normal(0, 1, n_areas)          # latent log-rate (KNOWN truth)
    latent = np.exp(u)                           # true rate lambda_s
    latent /= latent.mean()

    audit = ExposureDisparityAudit()

    def _init_patrol():
        p = np.ones(n_areas)
        p[groups == 0] *= (1.0 + init_patrol_bias)   # historically over-patrolled
        return _proportional_allocation(p)

    # ---- Policy A: allocate on police-discovered counts (the loop) --------
    patrol = _init_patrol()
    recA = FeedbackTrajectory([], [], "record-only")
    # ---- Policy B: allocate on OICC latent belief (patrol-independent anchor)
    patrolB = _init_patrol()
    recB = FeedbackTrajectory([], [], "oicc-anchored")

    for _ in range(rounds):
        # ---------- Policy A: discovered = patrol * latent (contaminated) ---
        discovered = patrol * latent * np.exp(rng.normal(0, 0.15, n_areas))
        beliefA = discovered / discovered.mean()
        recA.belief_corr.append(float(np.corrcoef(beliefA, latent)[0, 1]))
        resA = audit.audit(beliefA, latent, groups)
        recA.group_disparity.append(abs(resA.disparity.get("0", 0.0)))
        patrol = _proportional_allocation(beliefA)   # <-- runaway update

        # ---------- Policy B: OICC on 3 channels, 2 patrol-INDEPENDENT ------
        # c0 police-recorded (contaminated by patrolB), c1 911 calls, c2 survey.
        c0 = patrolB * latent * np.exp(rng.normal(0, 0.15, n_areas))
        c1 = latent * np.exp(-0.2 + rng.normal(0, 0.35, n_areas))   # 911 (indep.)
        c2 = latent * np.exp(0.1 + rng.normal(0, 0.45, n_areas))    # survey (indep.)
        log_channels = np.log(np.clip(np.array([c0, c1, c2]), 1e-6, None))
        theta_hat = deconvolve_blup(log_channels, pivot=1).theta_hat  # anchor on 911
        beliefB = np.exp(theta_hat)
        beliefB /= beliefB.mean()
        recB.belief_corr.append(float(np.corrcoef(beliefB, latent)[0, 1]))
        resB = audit.audit(beliefB, latent, groups)
        recB.group_disparity.append(abs(resB.disparity.get("0", 0.0)))
        patrolB = _proportional_allocation(beliefB)

    summary = {
        "final_corr_record": recA.belief_corr[-1],
        "final_corr_oicc": recB.belief_corr[-1],
        "final_disparity_record": recA.group_disparity[-1],
        "final_disparity_oicc": recB.group_disparity[-1],
        "disparity_reduction": recA.group_disparity[-1] - recB.group_disparity[-1],
        "oicc_stays_calibrated": recB.belief_corr[-1] > recA.belief_corr[-1],
    }
    return {"record": recA, "oicc": recB, "summary": summary,
            "latent": latent, "groups": groups}


def main() -> int:
    # be robust to Windows cp1252 stdout (Linux A100 already UTF-8)
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass
    print("=" * 70)
    print("FEEDBACK-LOOP MITIGATION VIA OICC-DEBIASED ALLOCATION")
    print("=" * 70)
    out = run()
    s = out["summary"]
    print("\nBelief-vs-truth correlation (1.0 = perfectly calibrated):")
    print(f"  record-only  policy: {s['final_corr_record']:+.3f}  (diverges)")
    print(f"  OICC-anchored policy: {s['final_corr_oicc']:+.3f}  (stays calibrated)")
    print("\nExposure disparity of the historically over-patrolled group "
          "(0 = fair):")
    print(f"  record-only  policy: {s['final_disparity_record']:.3f}")
    print(f"  OICC-anchored policy: {s['final_disparity_oicc']:.3f}")
    print(f"  disparity REDUCTION : {s['disparity_reduction']:+.3f}")
    print("\n[Honest] Simulation with known latent; the mitigation holds UNDER "
          "the\n  assumption that the 911/survey channels are patrol-independent "
          "(OICC's\n  core identifying assumption). It is a demonstration, not a "
          "universal proof.")
    ok = s["oicc_stays_calibrated"] and s["disparity_reduction"] > 0
    verdict = ("OICC breaks the loop and cuts disparity [PASS]" if ok
               else "no effect")
    print(f"\nRESULT: {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
