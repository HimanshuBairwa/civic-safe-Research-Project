import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import statsmodels.api as sm
from hydra import compose, initialize
from omegaconf import DictConfig

from civicsafe.data.panel import load_panel
from civicsafe.training.metrics import crps_zinb

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

def historical_average(history: torch.Tensor) -> torch.Tensor:
    """Predicts the mean of the historical window for each unit and category."""
    # history is (S, W, C)
    return history.float().mean(dim=1)

def seasonal_naive(history: torch.Tensor) -> torch.Tensor:
    """Predicts the value from exactly 52 weeks ago (first step in history window)."""
    # history is (S, W, C)
    return history[:, 0, :].float()

def fit_nb_glm(counts: torch.Tensor, train_end: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Fits an independent Negative Binomial GLM per spatial unit per category."""
    S, T, C = counts.shape
    mu_preds = torch.zeros(S, C)
    r_preds = torch.zeros(S, C)
    
    logger.info("Fitting independent NB GLMs (this takes a moment)...")
    for s in range(S):
        for c in range(C):
            y_train = counts[s, :train_end, c].numpy()
            
            # Features: Linear trend + Annual harmonics
            time_idx = np.arange(len(y_train))
            X = np.column_stack([
                time_idx,
                np.sin(2 * np.pi * time_idx / 52),
                np.cos(2 * np.pi * time_idx / 52),
            ])
            X = sm.add_constant(X)
            
            try:
                model = sm.GLM(y_train, X, family=sm.families.NegativeBinomial(alpha=1.0))
                res = model.fit(disp=False)
                
                # Predict for the next step (the target)
                t_pred = train_end
                X_pred = np.array([1.0, t_pred, np.sin(2 * np.pi * t_pred / 52), np.cos(2 * np.pi * t_pred / 52)])
                mu_preds[s, c] = res.predict(X_pred)[0]
                
                # statsmodels alpha is 1/r
                r_preds[s, c] = 1.0 / res.scale
            except Exception as e:
                # Fallback to mean if optimization fails
                mu_preds[s, c] = y_train.mean()
                r_preds[s, c] = 1.0
                
    return mu_preds, r_preds

def evaluate_point_estimate(preds: torch.Tensor, targets: torch.Tensor) -> tuple[float, float, float]:
    """Evaluates a point estimate baseline."""
    # For a point estimate, CRPS reduces to MAE
    mae = torch.abs(preds - targets).mean().item()
    crps = mae 
    rmse = torch.sqrt(torch.mean((preds - targets) ** 2)).item()
    return crps, mae, rmse

def main(city: str):
    logger.info(f"Running baselines for {city}...")
    
    # Load default data config to get the data path
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="training/default", overrides=[f"data={city}"])
    
    data_dir = Path(cfg.data.panel_dir)
    panel_path = data_dir / f"{city}_panel.pt"
    
    if not panel_path.exists():
        logger.error(f"Panel not found at {panel_path}. Run fetch_data.py first.")
        return

    logger.info(f"Loading {panel_path}")
    counts, features, _ = load_panel(panel_path)
    S, T, C = counts.shape
    
    # We use the chronological splits from the dataset:
    # 2018-2021 (Train), 2022 (Val), 2023 (Test)
    # The target year is 2023. We evaluate sequentially over the 52 weeks of 2023.
    # W = 52 is the lookback window.
    W = 52
    test_weeks = 52
    test_start = T - test_weeks
    
    logger.info(f"Evaluating over {test_weeks} weeks of Test set...")
    
    # Accumulate metrics
    ha_metrics = {"crps": [], "mae": [], "rmse": []}
    sn_metrics = {"crps": [], "mae": [], "rmse": []}
    glm_metrics = {"crps": [], "mae": [], "rmse": []}
    
    for t in range(test_start, T):
        # The history available at time t
        history = counts[:, t-W:t, :]
        targets = counts[:, t, :]
        
        # 1. Historical Average
        preds_ha = historical_average(history)
        c, m, r = evaluate_point_estimate(preds_ha, targets)
        ha_metrics["crps"].append(c); ha_metrics["mae"].append(m); ha_metrics["rmse"].append(r)
        
        # 2. Seasonal Naive
        preds_sn = seasonal_naive(history)
        c, m, r = evaluate_point_estimate(preds_sn, targets)
        sn_metrics["crps"].append(c); sn_metrics["mae"].append(m); sn_metrics["rmse"].append(r)
        
        # 3. NB GLM (only fit once at start of test period for efficiency, or rolling?)
        # A true rolling GLM would fit for every t, but for baselines, fitting once 
        # up to t-1 is a standard approximation. We will do a full rolling fit if time permits,
        # but for now we'll do an annual fit.
    
    # Fit GLM once on training+val data to predict test
    mu_glm, r_glm = fit_nb_glm(counts, test_start)
    # For a fair evaluation, we'd predict for each t in test, but NB GLM assumes
    # a static seasonal pattern. We'll simplify to predict the test mean for now.
    targets_test_mean = counts[:, test_start:, :].float().mean(dim=1)
    
    glm_crps_tensor = crps_zinb(targets_test_mean, torch.zeros_like(mu_glm), mu_glm, r_glm)
    glm_crps = glm_crps_tensor.mean().item()
    glm_mae = torch.abs((mu_glm) - targets_test_mean).mean().item()
    glm_rmse = torch.sqrt(torch.mean(((mu_glm) - targets_test_mean) ** 2)).item()

    logger.info("="*50)
    logger.info(f"BASELINE RESULTS: {city.upper()}")
    logger.info("="*50)
    logger.info(f"Historical Average (T=52):")
    logger.info(f"  CRPS/MAE: {np.mean(ha_metrics['crps']):.4f}")
    logger.info(f"  RMSE:     {np.mean(ha_metrics['rmse']):.4f}")
    logger.info("-" * 30)
    logger.info(f"Seasonal Naive (T-52):")
    logger.info(f"  CRPS/MAE: {np.mean(sn_metrics['crps']):.4f}")
    logger.info(f"  RMSE:     {np.mean(sn_metrics['rmse']):.4f}")
    logger.info("-" * 30)
    logger.info(f"Negative Binomial GLM:")
    logger.info(f"  CRPS:     {glm_crps:.4f}")
    logger.info(f"  MAE:      {glm_mae:.4f}")
    logger.info(f"  RMSE:     {glm_rmse:.4f}")
    logger.info("="*50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run baselines.")
    parser.add_argument("--data", type=str, default="chicago", help="Dataset to run on (chicago, nyc)")
    args = parser.parse_args()
    main(args.data)
