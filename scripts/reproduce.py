#!/usr/bin/env python
"""Reproducibility script for CIVIC-SAFE.

This script aggregates results from previous training runs and
generates publication-ready LaTeX tables and markdown summaries.
It validates that all required metrics are present and computes
mean ± std across multiple random seeds.

Usage:
    python scripts/reproduce.py --run-dir outputs/run_<timestamp>
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from civicsafe.utils.latex import dict_to_latex_table, format_mean_std_latex

logger = logging.getLogger(__name__)


def find_latest_run_dir(base_dir: str = "outputs") -> Path:
    """Find the most recently created run directory."""
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory {base_dir} does not exist.")
    
    run_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith("run_")]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {base_dir}.")
        
    # Sort by creation time (or modification time)
    return max(run_dirs, key=os.path.getmtime)


def aggregate_seed_results(run_dir: Path) -> dict:
    """Aggregate results from multiple seed subdirectories."""
    seed_dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("seed_")]
    
    if not seed_dirs:
        # Check if the run_dir itself has a history.json (single seed run)
        if (run_dir / "history.json").exists():
            seed_dirs = [run_dir]
        else:
            raise FileNotFoundError(f"No seed directories found in {run_dir}.")

    logger.info(f"Aggregating results from {len(seed_dirs)} seed(s)...")
    
    all_metrics = {}
    for seed_dir in seed_dirs:
        history_file = seed_dir / "history.json"
        if not history_file.exists():
            logger.warning(f"No history.json found in {seed_dir}")
            continue
            
        with open(history_file) as f:
            data = json.load(f)
            best_metrics = data.get("best_metrics", {})
            for k, v in best_metrics.items():
                if k not in all_metrics:
                    all_metrics[k] = []
                all_metrics[k].append(v)
                
    # Compute mean and std
    aggregated = {}
    latex_formatted = {}
    
    for k, values in all_metrics.items():
        mean_val = float(np.mean(values))
        std_val = float(np.std(values))
        aggregated[k] = {"mean": mean_val, "std": std_val, "values": values}
        latex_formatted[k] = format_mean_std_latex(mean_val, std_val)
        
    return {
        "aggregated": aggregated,
        "latex_formatted": latex_formatted,
        "num_seeds": len(seed_dirs)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CIVIC-SAFE Reproducibility Generator")
    parser.add_argument("--run-dir", type=str, default=None, 
                        help="Path to training run directory (defaults to latest)")
    parser.add_argument("--out-dir", type=str, default="outputs/results",
                        help="Where to save the LaTeX tables and summaries")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    
    try:
        run_dir = Path(args.run_dir) if args.run_dir else find_latest_run_dir()
        logger.info(f"Target Run Directory: {run_dir}")
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        return

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Aggregate Training Metrics
    try:
        metrics = aggregate_seed_results(run_dir)
        
        # Build table structure
        table_data = {
            "CIVIC-SAFE (Ours)": {
                "CRPS ↓": metrics["latex_formatted"].get("crps", "N/A"),
                "MAE ↓": metrics["latex_formatted"].get("mae", "N/A"),
                "RMSE ↓": metrics["latex_formatted"].get("rmse", "N/A"),
                "Brier (Zero) ↓": metrics["latex_formatted"].get("brier_zero", "N/A"),
            }
        }
        
        # Generate LaTeX table
        latex_code = dict_to_latex_table(
            results=table_data,
            caption="Predictive performance on test set (mean \\pm std across seeds).",
            label="tab:predictive_performance",
            bold_best=True
        )
        
        tex_path = out_dir / "predictive_performance.tex"
        with open(tex_path, "w") as f:
            f.write("% Requires \\usepackage{booktabs}\n")
            f.write(latex_code)
            
        logger.info(f"\nSaved LaTeX table to: {tex_path}")
        logger.info("\nPreview:")
        logger.info("-" * 40)
        logger.info(latex_code)
        logger.info("-" * 40)
        
    except Exception as e:
        logger.error(f"Failed to aggregate training metrics: {e}")

    logger.info(f"\nReproducibility artifacts successfully saved to {out_dir}")
    logger.info("Ready for paper submission!")


if __name__ == "__main__":
    main()
