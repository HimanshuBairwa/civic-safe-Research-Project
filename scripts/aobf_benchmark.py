"""Allocation under Observation-Biased Feedback (AOBF) Benchmark.

This script implements a closed-loop simulator to test the robustness of the
CIVIC-SAFE model against observation-biased feedback (the "confidently wrong"
phase transition).

The Reviewer requested a robustness/falsification test:
KEY CONTROL: turn feedback OFF (recording independent of allocation). If latent
coverage stays high with feedback OFF and only collapses with feedback ON, the
phenomenon is causal, not a knob artifact.
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


def simulate_aobf(
    num_steps: int = 10,
    feedback_on: bool = True,
    base_crime_rate: float = 5.0,
    patrol_budget: int = 50,
    num_cells: int = 100,
) -> dict[str, Any]:
    """Run the AOBF closed-loop simulation.

    Args:
        num_steps: Number of simulation steps (horizon).
        feedback_on: If True, observed crime depends on patrol allocation
                     (observation bias). If False, observed crime equals true
                     latent crime (no bias).
        base_crime_rate: Base Poisson rate for true latent crime.
        patrol_budget: Total patrol resources to allocate per step.
        num_cells: Number of spatial cells.

    Returns:
        Dictionary containing simulation results (coverage, error rates).
    """
    logger.info(f"Starting AOBF simulation. Feedback ON: {feedback_on}")
    
    # Initialize calibrator
    calibrator = AdaptiveTemporalECRCCalibrator(
        alpha=0.1, gamma=0.005, delta=0.1, group_type="income",
        k_i=0.001, k_d=0.0005, max_width=100.0
    )
    
    # Dummy groups (just one group for this simple simulation)
    groups = torch.zeros(num_cells, dtype=torch.long)
    
    latent_coverage_history = []
    observed_coverage_history = []
    
    # Pre-train calibrator with some dummy historical data
    logger.info("Pre-training calibrator...")
    hist_predictions = {
        "q_low": torch.ones(num_cells, 1) * (base_crime_rate - 2),
        "q_high": torch.ones(num_cells, 1) * (base_crime_rate + 2),
        "point": torch.ones(num_cells, 1) * base_crime_rate,
    }
    hist_y = torch.poisson(torch.ones(num_cells) * base_crime_rate).unsqueeze(1)
    calibrator.fit(hist_predictions, hist_y, groups)

    # State: our "model's belief" about crime rates
    belief_rate = torch.ones(num_cells) * base_crime_rate

    for step in range(num_steps):
        # 1. True Latent Crime Generation
        true_latent_crime = torch.poisson(torch.ones(num_cells) * base_crime_rate)
        
        # 2. Model Prediction (using our belief)
        # In reality, this would be a full forward pass. Here we just use our belief.
        pred_dict = {
            "q_low": (belief_rate - 2).unsqueeze(1).clamp(min=0),
            "q_high": (belief_rate + 2).unsqueeze(1),
            "point": belief_rate.unsqueeze(1),
        }
        
        # Conformalize predictions
        calib_res = calibrator.predict(pred_dict, groups)
        lower = calib_res["lower"].squeeze()
        upper = calib_res["upper"].squeeze()
        
        # 3. Policy: Allocate patrols based on predicted upper bound
        allocation = torch.zeros(num_cells)
        # Simple proportional allocation
        weights = upper / upper.sum()
        allocation = weights * patrol_budget
        
        # 4. Observation Generation
        if feedback_on:
            # Observation depends on allocation: more patrol -> more recorded crime
            # E.g., baseline discovery rate + patrol bonus
            discovery_rate = 0.5 + 0.5 * (allocation / allocation.max()).clamp(0, 1)
            observed_crime = torch.binomial(true_latent_crime, discovery_rate)
        else:
            # Oracle observation: we see all latent crime regardless of patrol
            observed_crime = true_latent_crime
            
        # 5. Evaluate Coverage
        # Check against OBSERVED crime (what the model calibrates against)
        obs_covered = (observed_crime >= lower) & (observed_crime <= upper)
        obs_coverage = obs_covered.float().mean().item()
        observed_coverage_history.append(obs_coverage)
        
        # Check against LATENT crime (the true safety of the area)
        lat_covered = (true_latent_crime >= lower) & (true_latent_crime <= upper)
        lat_coverage = lat_covered.float().mean().item()
        latent_coverage_history.append(lat_coverage)
        
        # 6. Update Belief & Calibrator
        # The model updates its belief based on observed crime
        belief_rate = 0.8 * belief_rate + 0.2 * observed_crime
        
        # Conformal calibrator updates based on observed miscoverage
        calibrator.update(pred_dict, observed_crime.unsqueeze(1), groups)
        
        logger.debug(
            f"Step {step}: Obs Cov={obs_coverage:.2f}, Latent Cov={lat_coverage:.2f}"
        )

    logger.info("Simulation complete.")
    return {
        "observed_coverage": observed_coverage_history,
        "latent_coverage": latent_coverage_history,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AOBF Benchmark Simulator")
    parser.add_argument("--steps", type=int, default=10, help="Simulation steps")
    args = parser.parse_args()
    
    print("\n--- Running with Feedback ON ---")
    res_on = simulate_aobf(num_steps=args.steps, feedback_on=True)
    print(f"Final Latent Coverage: {res_on['latent_coverage'][-1]:.3f}")
    
    print("\n--- Running with Feedback OFF ---")
    res_off = simulate_aobf(num_steps=args.steps, feedback_on=False)
    print(f"Final Latent Coverage: {res_off['latent_coverage'][-1]:.3f}")
