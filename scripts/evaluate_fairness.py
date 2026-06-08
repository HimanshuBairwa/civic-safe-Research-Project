import argparse
import logging
from pathlib import Path

import torch
import numpy as np
from hydra import compose, initialize

from civicsafe.data.panel import load_panel
from civicsafe.audit.components import (
    CoverageEquityAudit,
    IntervalWidthEquityAudit,
    PointAccuracyEquityAudit,
    CalibrationEquityAudit,
    WinklerEquityAudit,
    AbstentionEquityAudit,
    ReportingBiasSensitivityAudit
)
from civicsafe.audit.stratification import StratificationEngine
from civicsafe.training.metrics import crps_zinb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Note: In a real run, you'd load the model checkpoint and generate these on the test set.
# For demonstration of the audit mechanics without a saved checkpoint, we will use the test set
# targets and mock predictions (or just evaluate the raw data to show the strata).
# To do a real audit, one would load CivicSafeModel.load_from_checkpoint(cfg.checkpoint)

def main(city: str):
    logger.info(f"Running comprehensive Fairness Audit for {city}...")
    
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="training/default", overrides=[f"data={city}"])
    
    data_dir = Path(cfg.data.panel_dir)
    panel_path = data_dir / f"{city}_panel.pt"
    
    if not panel_path.exists():
        logger.error(f"Panel not found at {panel_path}. Run fetch_data.py first.")
        return

    counts, features, _ = load_panel(panel_path)
    S, T, C = counts.shape
    
    logger.info(f"Loaded {city} panel: {S} areas, {T} weeks, {C} categories, {features.shape[-1]} features")
    
    # In the panel, the features tensor is (S, T, F). Let's take the first timestep's features
    # to represent the static demographic strata.
    # We assume the 7 ACS features are roughly: poverty, income, race, etc.
    # Let's stratify by feature 0 (e.g. poverty rate proxy)
    demo_feature = features[:, 0, 0] # Shape: (S,)
    
    logger.info("Stratifying by socio-economic feature 0 (quintiles)...")
    strata = StratificationEngine.quantile_bins(demo_feature, n_bins=5)
    
    logger.info(f"Strata sizes: {[torch.sum(strata == i).item() for i in range(5)]}")
    
    # ---------------------------------------------------------
    # In a full run, we would load the trained model, do a forward pass 
    # to get pi, mu, r, and run the calibrator to get lower/upper/point.
    # For now, this script establishes the wiring for the 7-component audit.
    # ---------------------------------------------------------
    
    # Mocking predictions for the purpose of demonstrating the pipeline wiring
    # We will just perturb the true data to act as "predictions"
    y_test = counts[:, -1, 0].float() # just one timestep and category for demo
    
    # Mock ZINB parameters
    mu_mock = y_test + torch.randn_like(y_test) * 2
    mu_mock = torch.clamp(mu_mock, min=0.1)
    r_mock = torch.ones_like(y_test) * 5.0
    pi_mock = torch.zeros_like(y_test) + 0.1
    
    # Mock Conformal Intervals
    point_mock = mu_mock * (1 - pi_mock)
    lower_mock = torch.clamp(point_mock - 5.0, min=0.0)
    upper_mock = point_mock + 5.0
    
    alpha = 0.1
    
    logger.info("\n--- 1. Coverage Equity Audit ---")
    cov_audit = CoverageEquityAudit(max_coverage_gap=0.10)
    res_cov = cov_audit.evaluate(y_test, point_mock, lower_mock, upper_mock, pi_mock, mu_mock, r_mock, strata, alpha)
    logger.info(res_cov.summary_table())
    
    logger.info("\n--- 2. Interval Width Equity Audit ---")
    width_audit = IntervalWidthEquityAudit(max_width_ratio=1.5)
    res_width = width_audit.evaluate(y_test, point_mock, lower_mock, upper_mock, pi_mock, mu_mock, r_mock, strata, alpha)
    logger.info(res_width.summary_table())
    
    logger.info("\n--- 3. Point Accuracy Equity Audit ---")
    point_audit = PointAccuracyEquityAudit(max_error_ratio=1.5)
    res_point = point_audit.evaluate(y_test, point_mock, lower_mock, upper_mock, pi_mock, mu_mock, r_mock, strata, alpha)
    logger.info(res_point.summary_table())
    
    logger.info("\n--- 4. Calibration Equity Audit ---")
    cal_audit = CalibrationEquityAudit(max_brier_ratio=1.5)
    res_cal = cal_audit.evaluate(y_test, point_mock, lower_mock, upper_mock, pi_mock, mu_mock, r_mock, strata, alpha)
    logger.info(res_cal.summary_table())
    
    logger.info("\n--- 5. Winkler Equity Audit ---")
    wink_audit = WinklerEquityAudit(max_winkler_ratio=1.5)
    res_wink = wink_audit.evaluate(y_test, point_mock, lower_mock, upper_mock, pi_mock, mu_mock, r_mock, strata, alpha)
    logger.info(res_wink.summary_table())
    
    logger.info("\n--- 6. Abstention Equity Audit ---")
    abst_audit = AbstentionEquityAudit()
    res_abst = abst_audit.evaluate(y_test, point_mock, lower_mock, upper_mock, pi_mock, mu_mock, r_mock, strata, alpha)
    logger.info(res_abst.summary_table())
    
    logger.info("\n--- 7. Reporting Bias Sensitivity Audit ---")
    rep_audit = ReportingBiasSensitivityAudit()
    res_rep = rep_audit.evaluate(y_test, point_mock, lower_mock, upper_mock, pi_mock, mu_mock, r_mock, strata, alpha)
    logger.info(res_rep.summary_table())

    logger.info("\n[SUCCESS] Fairness Evaluation pipeline is fully wired.")
    logger.info("Next steps: Hook this up to `CivicSafeModel.load_from_checkpoint()` to audit actual trained runs.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full fairness audit.")
    parser.add_argument("--data", type=str, default="chicago", help="Dataset to run on (chicago, nyc)")
    args = parser.parse_args()
    main(args.data)
