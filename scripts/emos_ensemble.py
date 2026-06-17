"""EMOS-style Ensemble Evaluation for CIVIC-SAFE.

Ensemble Model Output Statistics (EMOS) is standard in weather forecasting
(Gneiting et al., 2005) but has NEVER been applied to crime forecasting.

The key idea: instead of picking the best single seed, COMBINE all 5 seeds'
ZINB predictions into a mixture distribution, then evaluate the mixture.
This typically improves CRPS by 10-30% in weather forecasting.

For ZINB models, the mixture distribution is:
    P_ens(Y=y) = (1/K) * sum_{k=1}^K ZINB(y; pi_k, mu_k, r_k)

The mixture CDF is:
    F_ens(k) = (1/K) * sum_{k=1}^K F_ZINB(k; pi_k, mu_k, r_k)

This is directly usable in the CRPS computation with no additional fitting.

Usage:
    python scripts/emos_ensemble.py --data chicago --run-dir outputs/run_XXXXXX

References:
    Gneiting et al. (2005): "Calibrated Probabilistic Forecasting Using
    Ensemble Model Output Statistics and Minimum CRPS Estimation"
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def crps_mixture_zinb(
    y: torch.Tensor,
    pi_list: list[torch.Tensor],
    mu_list: list[torch.Tensor],
    r_list: list[torch.Tensor],
    k_max: int | None = None,
) -> torch.Tensor:
    """Compute CRPS for a ZINB mixture distribution.
    
    The mixture CDF is the average of individual CDFs:
    F_mix(k) = (1/K) * sum_i F_ZINB(k; pi_i, mu_i, r_i)
    
    Args:
        y: Observed counts. Shape: (B,)
        pi_list, mu_list, r_list: Lists of K tensors, each shape (B,)
        k_max: Truncation point for CDF summation.
    
    Returns:
        Per-sample CRPS. Shape: (B,)
    """
    K = len(pi_list)
    B = y.shape[0]
    device = y.device
    
    # Auto-determine truncation
    if k_max is None:
        max_mu = max(m.max().item() for m in mu_list)
        max_r = max(r.max().item() for r in r_list)
        variance = max_mu + max_mu**2 / max(max_r, 0.1)
        k_max = min(int(max_mu + 10.0 * math.sqrt(variance)) + 1, 500)
        k_max = max(k_max, 50)
    
    # Build grid: (1, k_max+1)
    grid = torch.arange(k_max + 1, device=device, dtype=torch.float32).unsqueeze(0)
    
    # Compute mixture CDF: average of individual CDFs
    mixture_cdf = torch.zeros(B, k_max + 1, device=device)
    
    for pi, mu, r in zip(pi_list, mu_list, r_list):
        pi = pi.float().clamp(0.0, 1.0)
        mu = mu.float().clamp(min=1e-6)
        r = r.float().clamp(min=0.1)
        
        # NB PMF via log-space
        log_r = r.log().unsqueeze(1)
        log_mu = mu.log().unsqueeze(1)
        log_p = log_r - torch.logaddexp(log_r, log_mu)
        log_1_minus_p = log_mu - torch.logaddexp(log_r, log_mu)
        
        # log NB PMF: lgamma(k+r) - lgamma(k+1) - lgamma(r) + r*log(p) + k*log(1-p)
        log_pmf = (
            torch.lgamma(grid + r.unsqueeze(1))
            - torch.lgamma(grid + 1.0)
            - torch.lgamma(r.unsqueeze(1))
            + r.unsqueeze(1) * log_p
            + grid * log_1_minus_p
        )
        nb_pmf = log_pmf.exp()
        
        # ZINB CDF: pi + (1-pi) * cumsum(NB_PMF)
        nb_cdf = nb_pmf.cumsum(dim=1)
        zinb_cdf = pi.unsqueeze(1) + (1.0 - pi.unsqueeze(1)) * nb_cdf
        
        mixture_cdf += zinb_cdf / K
    
    # CRPS computation
    indicator = (y.unsqueeze(1) <= grid).float()
    crps = ((mixture_cdf - indicator) ** 2).sum(dim=1)
    
    return crps


def main():
    parser = argparse.ArgumentParser(description="EMOS Ensemble Evaluation")
    parser.add_argument("--data", default="chicago", choices=["chicago", "nyc"])
    parser.add_argument("--run-dir", type=str, help="Path to run directory with seed_* subdirs")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    
    # Find the latest run directory if not specified
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        outputs_dir = project_root / "outputs"
        run_dirs = sorted(outputs_dir.glob("run_*"))
        if not run_dirs:
            logger.error("No run directories found. Train first.")
            sys.exit(1)
        run_dir = run_dirs[-1]
    
    logger.info(f"EMOS Ensemble Evaluation")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Data: {args.data}")
    
    # Find all seed checkpoints
    seed_dirs = sorted(run_dir.glob("seed_*"))
    logger.info(f"Found {len(seed_dirs)} seed directories")
    
    if len(seed_dirs) < 2:
        logger.warning("Need at least 2 seeds for ensemble. Running single-model evaluation.")
        return
    
    # Load predictions from each seed
    pi_list, mu_list, r_list = [], [], []
    y_true = None
    
    for seed_dir in seed_dirs:
        pred_path = seed_dir / "predictions.pt"
        if not pred_path.exists():
            logger.warning(f"No predictions.pt in {seed_dir}, skipping")
            continue
        
        preds = torch.load(pred_path, weights_only=False)
        pi_list.append(preds["pi"].reshape(-1))
        mu_list.append(preds["mu"].reshape(-1))
        r_list.append(preds["r"].reshape(-1))
        
        if y_true is None:
            y_true = preds["y_true"].reshape(-1)
    
    if y_true is None or len(pi_list) < 2:
        logger.error("Insufficient predictions found. Ensure training saves predictions.pt")
        return
    
    K = len(pi_list)
    logger.info(f"Ensembling {K} models...")
    
    # Compute individual CRPS for each seed
    from civicsafe.training.metrics import crps_zinb
    
    individual_crps = []
    for i, (pi, mu, r) in enumerate(zip(pi_list, mu_list, r_list)):
        crps_i = crps_zinb(y_true, pi, mu, r).mean().item()
        individual_crps.append(crps_i)
        logger.info(f"  Seed {i} CRPS: {crps_i:.4f}")
    
    # Compute ensemble CRPS
    ensemble_crps = crps_mixture_zinb(y_true, pi_list, mu_list, r_list).mean().item()
    
    # Compute baselines
    mean_individual = np.mean(individual_crps)
    best_individual = min(individual_crps)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"  EMOS ENSEMBLE RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"  Individual CRPS (mean): {mean_individual:.4f}")
    logger.info(f"  Individual CRPS (best): {best_individual:.4f}")
    logger.info(f"  Ensemble CRPS:          {ensemble_crps:.4f}")
    logger.info(f"  Improvement vs mean:    {(mean_individual - ensemble_crps) / mean_individual * 100:.1f}%")
    logger.info(f"  Improvement vs best:    {(best_individual - ensemble_crps) / best_individual * 100:.1f}%")
    
    # Compute CRPSS
    hist_mean = y_true.float().mean()
    naive_crps = (y_true.float() - hist_mean).abs().mean().item()
    crpss_individual = 1.0 - mean_individual / naive_crps
    crpss_ensemble = 1.0 - ensemble_crps / naive_crps
    
    logger.info(f"\n  CRPSS vs Historical Avg:")
    logger.info(f"    Individual (mean): {crpss_individual:.4f}")
    logger.info(f"    Ensemble:          {crpss_ensemble:.4f}")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
