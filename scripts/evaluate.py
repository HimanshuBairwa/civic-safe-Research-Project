#!/usr/bin/env python
"""CIVIC-SAFE full pipeline evaluation.

Runs the complete pipeline: synthetic data → model → calibration →
audit → routing, demonstrating every module working together.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --num-nodes 77 --alpha 0.1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def run_evaluation(
    num_nodes: int = 20,
    num_time_steps: int = 52,
    num_categories: int = 3,
    alpha: float = 0.1,
    seed: int = 42,
    output_dir: str = "outputs/eval",
) -> dict:
    """Run the full CIVIC-SAFE evaluation pipeline.

    Args:
        num_nodes: Number of spatial units.
        num_time_steps: Number of time steps.
        num_categories: Number of crime categories.
        alpha: Nominal miscoverage level for conformal prediction.
        seed: Random seed.
        output_dir: Output directory for results.

    Returns:
        Dictionary with all evaluation results.
    """
    from civicsafe.utils.seeding import seed_everything

    seed_everything(seed)
    results: dict = {"config": {
        "num_nodes": num_nodes,
        "num_time_steps": num_time_steps,
        "num_categories": num_categories,
        "alpha": alpha,
        "seed": seed,
    }}

    logger.info("=" * 60)
    logger.info("  CIVIC-SAFE Full Pipeline Evaluation")
    logger.info("=" * 60)

    # ── Step 1: Synthetic Data ──────────────────────────────────
    logger.info("\n[1/6] Generating synthetic spatiotemporal panel...")
    from civicsafe.synthetic.distributions import generate_spatiotemporal_panel

    panel = generate_spatiotemporal_panel(
        num_spatial_units=num_nodes,
        num_time_steps=num_time_steps,
        num_categories=num_categories,
        seed=seed,
    )
    counts = panel["counts"]  # (S, T, C)
    S, T, C = counts.shape
    logger.info(f"  Panel shape: {S} spatial × {T} time × {C} categories")
    results["data"] = {"shape": [S, T, C], "total_crimes": int(counts.sum().item())}

    # ── Step 2: Model Forward Pass ──────────────────────────────
    logger.info("\n[2/6] Running model forward pass...")
    from civicsafe.models.civicsafe_model import CivicSafeModel
    from civicsafe.models.graph import build_adjacency_from_synthetic

    model = CivicSafeModel(
        num_features=panel["features"].shape[-1],
        hidden_dim=64,
        spatial_layers=1,
        spatial_heads=2,
        temporal_layers=1,
        temporal_heads=2,
        temporal_ff_dim=128,
        num_categories=C,
        max_seq_len=min(T, 52),
    )
    model.eval()
    graph = build_adjacency_from_synthetic(num_nodes=S, seed=seed, knn_k=4)

    # Simulate a single forward pass with features
    window = min(T, 12)
    x_feat = panel["features"][:, :window, :]  # (S, W, F)

    with torch.no_grad():
        out = model(
            x_feat,
            edge_index_queen=graph["queen"],
            edge_index_knn=graph.get("knn"),
        )

    pi = out["pi"]  # (S, C)
    mu = out["mu"]
    r = out["r"]
    logger.info(f"  π range: [{pi.min():.3f}, {pi.max():.3f}]")
    logger.info(f"  μ range: [{mu.min():.3f}, {mu.max():.3f}]")
    logger.info(f"  r range: [{r.min():.3f}, {r.max():.3f}]")

    num_params = sum(p.numel() for p in model.parameters())
    results["model"] = {"parameters": num_params, "pi_mean": float(pi.mean()), "mu_mean": float(mu.mean())}

    # ── Step 3: Conformal Calibration ───────────────────────────
    logger.info("\n[3/6] Running conformal calibration (Split CP)...")
    from civicsafe.calibration.conformal import SplitConformalCalibrator

    # Use last timestep as calibration target
    y_cal = counts[:, -1, 0].float()  # Category 0
    pi_cal = pi[:, 0]
    mu_cal = mu[:, 0]
    r_cal = r[:, 0]

    calibrator = SplitConformalCalibrator(alpha=alpha)
    calibrator.fit(y_cal, pi_cal, mu_cal, r_cal)
    intervals = calibrator.predict(pi_cal, mu_cal, r_cal)

    lower = intervals["lower"]
    upper = intervals["upper"]
    point = intervals["point"]

    # Check coverage
    covered = ((y_cal >= lower) & (y_cal <= upper)).float().mean().item()
    avg_width = (upper - lower).mean().item()

    logger.info(f"  Coverage: {covered:.3f} (target: {1 - alpha:.3f})")
    logger.info(f"  Average interval width: {avg_width:.2f}")
    logger.info(f"  Threshold: {calibrator.threshold:.4f}")

    results["calibration"] = {
        "method": "split_cp",
        "coverage": round(covered, 4),
        "target_coverage": round(1 - alpha, 4),
        "avg_width": round(avg_width, 2),
        "threshold": round(calibrator.threshold, 4),
    }

    # ── Step 4: Equity Audit ────────────────────────────────────
    logger.info("\n[4/6] Running equity audit...")
    from civicsafe.audit.components import CoverageEquityAudit, PointAccuracyEquityAudit
    from civicsafe.audit.stratification import StratificationEngine

    strata = StratificationEngine.quantile_bins(mu_cal, n_bins=3)

    cov_audit = CoverageEquityAudit(max_coverage_gap=0.15)
    cov_result = cov_audit.evaluate(y_cal, point, lower, upper, pi_cal, mu_cal, r_cal, strata, alpha)
    logger.info(f"  Coverage equity: {'PASS' if cov_result.passes_threshold else 'FAIL'}")
    logger.info(f"    Max gap from target: {cov_result.disparity_metrics.get('max_gap_from_target', 'N/A')}")

    acc_audit = PointAccuracyEquityAudit()
    acc_result = acc_audit.evaluate(y_cal, point, lower, upper, pi_cal, mu_cal, r_cal, strata, alpha)
    logger.info(f"  Point accuracy equity: {'PASS' if acc_result.passes_threshold else 'FAIL'}")

    results["audit"] = {
        "coverage_equity": cov_result.passes_threshold,
        "point_accuracy_equity": acc_result.passes_threshold,
        "disparity_metrics": {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in cov_result.disparity_metrics.items()},
    }

    # ── Step 5: Advisory Routing ────────────────────────────────
    logger.info("\n[5/6] Running advisory safe-route (Tsinghua SSSP)...")
    from civicsafe.routing.engine import AdvisoryRoutingEngine

    engine = AdvisoryRoutingEngine.from_adjacency(
        edge_index=graph["queen"],
        num_nodes=S,
        upper_bounds=upper,
        lower_bounds=lower,
        peak_threshold=upper.max().item() + 10,  # Allow all routes for demo
        budget_threshold=1000.0,
    )

    # Find a route
    src, dst = 0, min(S - 1, 10)
    route = engine.safe_route(src, dst, raise_on_abstention=False)

    logger.info(f"  Route {src} → {dst}: {route.path}")
    logger.info(f"  Total cost: {route.total_cost:.4f}")
    logger.info(f"  Abstain: {route.abstention_verdict.should_abstain}")
    logger.info(f"  Peak uncertainty: {route.abstention_verdict.peak_width:.4f}")
    logger.info(f"  Algorithm: {route.algorithm}")

    # Compare with Dijkstra
    comparison = engine.compare_algorithms(src, dst)
    logger.info(f"  Tsinghua vs Dijkstra cost match: {comparison['cost_match']}")

    results["routing"] = {
        "source": src,
        "target": dst,
        "path": route.path,
        "total_cost": round(route.total_cost, 4),
        "should_abstain": route.abstention_verdict.should_abstain,
        "peak_uncertainty": round(route.abstention_verdict.peak_width, 4),
        "algorithm": route.algorithm,
        "matches_dijkstra": comparison["cost_match"],
        "frontier_reductions": comparison["tsinghua_frontier_reductions"],
    }

    # ── Step 6: Summary ─────────────────────────────────────────
    logger.info("\n[6/6] Saving results...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results_file = output_path / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"  Results saved to: {results_file}")

    # ── Final Summary ───────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("  CIVIC-SAFE EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"  Data:        {S} nodes × {T} time × {C} categories")
    logger.info(f"  Model:       {num_params:,} parameters")
    logger.info(f"  Coverage:    {covered:.3f} (target {1 - alpha:.3f})")
    logger.info(f"  Audit:       Coverage={'PASS' if cov_result.passes_threshold else 'FAIL'}, "
                f"Accuracy={'PASS' if acc_result.passes_threshold else 'FAIL'}")
    logger.info(f"  Route:       {src}→{dst} via {route.algorithm} "
                f"({'safe' if not route.abstention_verdict.should_abstain else 'ABSTAIN'})")
    logger.info(f"  Dijkstra:    {'MATCHES' if comparison['cost_match'] else 'DIFFERS'}")
    logger.info("=" * 60)

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="CIVIC-SAFE evaluation")
    parser.add_argument("--num-nodes", type=int, default=20)
    parser.add_argument("--num-time-steps", type=int, default=52)
    parser.add_argument("--num-categories", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/eval")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_evaluation(
        num_nodes=args.num_nodes,
        num_time_steps=args.num_time_steps,
        num_categories=args.num_categories,
        alpha=args.alpha,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
