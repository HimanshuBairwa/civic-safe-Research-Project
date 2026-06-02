#!/usr/bin/env python
"""CIVIC-SAFE Reproducibility Script.

Generates all publication-ready tables and artifacts required for the
benchmark paper. This script runs lightweight versions of the evaluation
pipeline across different configurations to output:
1. LaTeX tables (for inclusion in paper)
2. JSON summaries (for programmatic consumption)

Usage:
    python scripts/reproduce.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from civicsafe.synthetic.distributions import generate_spatiotemporal_panel
from civicsafe.utils.latex import dict_to_latex_table
from civicsafe.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


def evaluate_calibration_methods(
    output_dir: Path,
    seed: int = 42,
) -> None:
    """Evaluate and compare all 5 conformal calibration methods."""
    from civicsafe.calibration.conformal import (
        ECRCCalibrator,
        EqualizedCoverageCalibrator,
        MondrianConformalCalibrator,
        SplitConformalCalibrator,
        WeightedConformalCalibrator,
    )
    
    logger.info("Evaluating Conformal Calibration Methods...")
    
    # 1. Generate Data
    panel = generate_spatiotemporal_panel(
        num_spatial_units=100,
        num_time_steps=50,
        num_categories=1,
        seed=seed,
    )
    
    # Simulate some pseudo-predictions (since we just want to test calibration)
    # y = mu + noise
    torch.manual_seed(seed)
    mu_true = torch.rand(100) * 10
    pi_true = torch.zeros(100)
    r_true = torch.ones(100) * 2
    
    y_cal = mu_true + torch.randn(100) * 2
    y_cal = torch.clamp(y_cal, min=0).round()
    
    # Stratification for Mondrian/ECRC
    from civicsafe.audit.stratification import StratificationEngine
    strata = StratificationEngine.quantile_bins(mu_true, n_bins=3)
    
    calibrators = {
        "Split CP": SplitConformalCalibrator(alpha=0.1),
        "Weighted CP": WeightedConformalCalibrator(alpha=0.1),
        "Mondrian CP": MondrianConformalCalibrator(alpha=0.1),
        "Equalized CP": EqualizedCoverageCalibrator(alpha=0.1),
        "ECRC (Ours)": ECRCCalibrator(alpha=0.1, delta=0.05),
    }
    
    results = {}
    for name, cal in calibrators.items():
        if isinstance(cal, (MondrianConformalCalibrator, ECRCCalibrator)):
            cal.fit(y_cal, pi_true, mu_true, r_true, groups=strata)
            intervals = cal.predict(pi_true, mu_true, r_true, groups=strata)
        elif isinstance(cal, EqualizedCoverageCalibrator):
            cal.fit(y_cal, pi_true, mu_true, r_true, groups=strata)
            intervals = cal.predict(pi_true, mu_true, r_true)
        else:
            cal.fit(y_cal, pi_true, mu_true, r_true)
            intervals = cal.predict(pi_true, mu_true, r_true)
            
        lower = intervals["lower"]
        upper = intervals["upper"]
        
        coverage = ((y_cal >= lower) & (y_cal <= upper)).float().mean().item()
        width = (upper - lower).mean().item()
        
        results[name] = {
            "Coverage": coverage,
            "Target Cov": 0.900,
            "Avg Width": width,
        }
        
    # Save JSON
    with open(output_dir / "calibration_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Generate LaTeX
    latex_table = dict_to_latex_table(
        data=results,
        caption="Comparison of conformal calibration methods on synthetic panel data. Target coverage is $1-\\alpha=0.90$.",
        label="tab:calibration",
        better={"Coverage": "high", "Avg Width": "low"},
    )
    
    with open(output_dir / "table_calibration.tex", "w") as f:
        f.write(latex_table)
        
    logger.info(f"  Saved calibration results to {output_dir}")


def evaluate_routing_algorithms(
    output_dir: Path,
    seed: int = 42,
) -> None:
    """Compare Tsinghua SSSP router vs Dijkstra."""
    from civicsafe.routing.cost import ParetoCost
    from civicsafe.routing.graph import RoutingGraph
    from civicsafe.routing.tsinghua import DijkstraRouter, TsinghuaRouter
    
    logger.info("Evaluating Routing Algorithms...")
    torch.manual_seed(seed)
    
    n = 100
    positions = torch.rand(n, 2)
    from scipy.spatial import cKDTree
    tree = cKDTree(positions.numpy())
    src_list, dst_list = [], []
    for i in range(n):
        _, neighbors = tree.query(positions[i].numpy(), k=5)
        for j in neighbors:
            if j != i:
                src_list.extend([i, int(j)])
                dst_list.extend([int(j), i])

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    graph = RoutingGraph.from_edge_index(edge_index, n, positions)
    
    upper = torch.rand(n) * 10
    lower = torch.zeros(n)
    graph.inject_predictions(upper, lower)
    
    cost_fn = ParetoCost(w_dist=0.3, w_risk=0.7)
    tsinghua = TsinghuaRouter(graph, cost_fn)
    dijkstra = DijkstraRouter(graph, cost_fn)
    
    results = {}
    test_pairs = [(0, 10), (0, 50), (0, 99)]
    
    for src, dst in test_pairs:
        t_res = tsinghua.shortest_path(src, dst)
        d_res = dijkstra.shortest_path(src, dst)
        
        results[f"{src} $\\rightarrow$ {dst}"] = {
            "Dijkstra Cost": d_res.total_cost,
            "Tsinghua Cost": t_res.total_cost,
            "Cost Match": float(abs(t_res.total_cost - d_res.total_cost) < 1e-4),
            "Frontier Reductions": t_res.frontier_reductions,
        }
        
    # Save JSON
    with open(output_dir / "routing_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Generate LaTeX
    latex_table = dict_to_latex_table(
        data=results,
        caption="Comparison of classical Dijkstra vs. Tsinghua 2025 SSSP algorithm on 100-node graph.",
        label="tab:routing",
    )
    
    with open(output_dir / "table_routing.tex", "w") as f:
        f.write(latex_table)
        
    logger.info(f"  Saved routing results to {output_dir}")


def main() -> None:
    """Run all reproducibility evaluations."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    
    seed_everything(42)
    output_dir = Path("outputs/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("  CIVIC-SAFE Reproducibility Script")
    logger.info("=" * 60)
    
    evaluate_calibration_methods(output_dir)
    evaluate_routing_algorithms(output_dir)
    
    logger.info("=" * 60)
    logger.info(f"  All reproducibility artifacts generated in {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
