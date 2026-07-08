"""Allocation under Observation-Biased Feedback (AOBF) Benchmark.

A closed-loop simulator that exhibits the **"confidently wrong" phase
transition** predicted by :mod:`civicsafe.theory.feedback_law`: as the feedback
gain rises, coverage of the *recorded* process is maintained (and even improves)
while coverage of the *true latent* process collapses.

Design (uses the REAL AdaptiveTemporalECRCCalibrator API):
  * Latent intensity ``lambda_s`` is fixed and unobserved.
  * The model tracks the recorded rate via an EWMA belief.
  * Attention is allocated proportional to the predicted upper interval.
  * Recording is inflated by attention with strength governed by ``gain``.

Two controls make the effect falsifiable:
  * ``feedback_on=False``  -> recording is independent of allocation (no bias).
  * a sweep over ``gain``  -> locates the empirical critical gain kappa*.

If latent coverage stays high with feedback OFF and only collapses with
feedback ON, the phenomenon is causal, not a knob artefact.
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import torch

from civicsafe.calibration.conformal import AdaptiveTemporalECRCCalibrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _belief_to_zinb(belief: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map a scalar belief rate to ZINB (pi, mu, r) parameters for the calibrator.

    We use a light zero-inflation and moderate dispersion; the calibrator only
    needs valid distributional parameters to form CQR scores and quantiles.
    """
    mu = belief.clamp(min=1e-3)
    pi = torch.full_like(mu, 0.05)
    r = torch.full_like(mu, 5.0)
    return pi, mu, r


def simulate_aobf(
    num_steps: int = 60,
    feedback_on: bool = True,
    gain: float = 1.0,
    base_crime_rate: float = 5.0,
    patrol_budget: float = 100.0,
    num_cells: int = 100,
    alpha: float = 0.1,
    seed: int = 0,
    burn_in: int = 15,
) -> dict[str, Any]:
    """Run the AOBF closed-loop simulation.

    Args:
        num_steps: Simulation horizon.
        feedback_on: If True, recorded crime is inflated by patrol allocation.
        gain: Feedback strength (detection amplification). Higher -> stronger loop.
        base_crime_rate: Mean of the fixed latent Poisson intensity.
        patrol_budget: Total attention allocated per step.
        num_cells: Number of spatial cells.
        alpha: Target miscoverage (coverage = 1 - alpha).
        seed: RNG seed.
        burn_in: Steps to discard before averaging (let the loop reach its regime).

    Returns:
        Dict with per-step and burn-in-averaged observed/latent coverage.
    """
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)

    calibrator = AdaptiveTemporalECRCCalibrator(
        alpha=alpha, gamma=0.05, delta=0.1, group_type="geographic",
        k_i=0.005, k_d=0.001, max_width=1e6,  # max_width huge: never abstain in the benchmark
    )
    groups = torch.zeros(num_cells, dtype=torch.long)

    # Fixed, unobserved latent intensity (heterogeneous across cells).
    lam = base_crime_rate * (0.3 + torch.rand(num_cells, generator=gen) * 1.7)

    # Warm-start the calibrator on unbiased draws from the latent process.
    y_hist = torch.poisson(lam.unsqueeze(1).repeat(1, 1), generator=gen).squeeze(1)
    pi0, mu0, r0 = _belief_to_zinb(lam.clone())
    calibrator.fit(y_hist, pi0, mu0, r0, groups=groups)

    belief = lam.clone()
    obs_hist: list[float] = []
    lat_hist: list[float] = []
    width_hist: list[float] = []

    for _ in range(num_steps):
        true_latent = torch.poisson(lam, generator=gen)

        pi, mu, r = _belief_to_zinb(belief)
        interval = calibrator.predict(pi, mu, r, groups=groups)
        lower = interval["lower"].reshape(-1)
        upper = interval["upper"].reshape(-1)

        # Allocate attention proportional to predicted upper bound.
        w = upper / upper.sum().clamp(min=1e-9)
        a_rel = w * num_cells  # 1.0 == average attention
        allocation = w * patrol_budget  # noqa: F841 (kept for interpretability)

        if feedback_on:
            detect = (1.0 + gain * (a_rel - 1.0)).clamp(min=0.1)
        else:
            detect = torch.ones_like(a_rel)
        observed = torch.poisson(lam * detect, generator=gen)

        obs_cov = (((observed >= lower) & (observed <= upper)).float().mean().item())
        lat_cov = (((true_latent >= lower) & (true_latent <= upper)).float().mean().item())
        obs_hist.append(obs_cov)
        lat_hist.append(lat_cov)
        width_hist.append((upper - lower).mean().item())

        belief = 0.8 * belief + 0.2 * observed
        calibrator.update(observed, pi, mu, r, groups=groups)

    def _avg(xs: list[float]) -> float:
        return float(np.mean(xs[burn_in:])) if len(xs) > burn_in else float(np.mean(xs))

    return {
        "observed_coverage": obs_hist,
        "latent_coverage": lat_hist,
        "mean_width": width_hist,
        "avg_observed_coverage": _avg(obs_hist),
        "avg_latent_coverage": _avg(lat_hist),
        "avg_width": _avg(width_hist),
    }


def sweep_gain(
    gains: list[float] | None = None, num_steps: int = 60, seeds: int = 3, **kwargs: Any
) -> None:
    """Sweep the feedback gain and print the phase transition table."""
    if gains is None:
        gains = [0.0, 0.3, 0.6, 1.0, 1.6]
    target = 1.0 - kwargs.get("alpha", 0.1)
    logger.info("Target coverage = %.2f", target)
    print(f"\n{'gain':>5} | {'FEEDBACK ON':^24} | {'FEEDBACK OFF (control)':^24}")
    print(f"{'':>5} | {'obs_cov':>10} {'lat_cov':>12} | {'obs_cov':>10} {'lat_cov':>12}")
    print("-" * 62)
    for g in gains:
        on = np.mean([
            [simulate_aobf(num_steps=num_steps, feedback_on=True, gain=g, seed=s, **kwargs)[k]
             for k in ("avg_observed_coverage", "avg_latent_coverage")]
            for s in range(seeds)
        ], axis=0)
        off = np.mean([
            [simulate_aobf(num_steps=num_steps, feedback_on=False, gain=g, seed=s, **kwargs)[k]
             for k in ("avg_observed_coverage", "avg_latent_coverage")]
            for s in range(seeds)
        ], axis=0)
        print(f"{g:>5.2f} | {on[0]:>10.3f} {on[1]:>12.3f} | {off[0]:>10.3f} {off[1]:>12.3f}")
    print("-" * 62)
    print("Reading: with feedback OFF latent coverage stays ~nominal at all gains;")
    print("with feedback ON it collapses past a critical gain kappa* while observed")
    print("coverage is maintained -> the 'confidently wrong' phase transition.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AOBF Benchmark Simulator")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--sweep", action="store_true", help="Run the gain sweep table")
    args = parser.parse_args()

    if args.sweep:
        sweep_gain(num_steps=args.steps, seeds=args.seeds)
    else:
        on = simulate_aobf(num_steps=args.steps, feedback_on=True, gain=1.0)
        off = simulate_aobf(num_steps=args.steps, feedback_on=False, gain=1.0)
        print(f"Feedback ON : observed={on['avg_observed_coverage']:.3f} "
              f"latent={on['avg_latent_coverage']:.3f}")
        print(f"Feedback OFF: observed={off['avg_observed_coverage']:.3f} "
              f"latent={off['avg_latent_coverage']:.3f}")
