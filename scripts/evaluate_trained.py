#!/usr/bin/env python
"""CIVIC-SAFE production test-set evaluation for trained model checkpoints.

Rolling one-step-ahead evaluation on the 2023 test set (weeks 260-313).
For each test week t, the model receives features from [t-52, t) and
predicts counts at week t.  Outputs ZINB parameters (pi, mu, r) are
converted to point forecasts via  ŷ = (1 - π) · μ.

Metrics computed:
  - Overall:       MAE, RMSE, MAPE, CRPS, Brier-zero
  - Per-category:  MAE, RMSE, MAPE  (violent / property / drug)
  - Per-spatial:   MAE per spatial unit  (best / worst areas)
  - Conformal:     90 % coverage & avg width on test set
  - LaTeX table:   ready for paper inclusion

Usage:
    python scripts/evaluate_trained.py --checkpoint outputs/run_XXX/seed_42/best.pt
    python scripts/evaluate_trained.py --checkpoint outputs/run_XXX/seed_42/best.pt --data nyc
    python scripts/evaluate_trained.py --checkpoint auto --data chicago --alpha 0.1
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CATEGORY_NAMES = {0: "violent", 1: "property", 2: "drug"}

# Chronological split boundaries (2018-2023, 52 weeks/year)
START_YEAR = 2018
VAL_YEAR = 2022
TEST_YEAR = 2023
WEEKS_PER_YEAR = 52
WINDOW_SIZE = 52

VAL_START_WEEK = (VAL_YEAR - START_YEAR) * WEEKS_PER_YEAR   # 208
TEST_START_WEEK = (TEST_YEAR - START_YEAR) * WEEKS_PER_YEAR  # 260


# ───────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ───────────────────────────────────────────────────────────────────
def discover_checkpoint(data_name: str) -> Path:
    """Auto-discover the latest checkpoint for this dataset in outputs/.
    
    Searches dataset-specific directories first (``run_{data_name}_*``),
    then falls back to generic ``run_*`` for backward compatibility.
    """
    outputs_dir = PROJECT_ROOT / "outputs"
    if not outputs_dir.exists():
        raise FileNotFoundError(f"No outputs directory at {outputs_dir}")

    # Priority 1: dataset-specific run directories
    dataset_prefix = f"run_{data_name}_"
    run_dirs = sorted(
        [d for d in outputs_dir.iterdir()
         if d.is_dir() and d.name.startswith(dataset_prefix)],
        key=lambda p: p.name,
    )
    
    # Priority 2: generic run_* directories (backward compat)
    if not run_dirs:
        run_dirs = sorted(
            [d for d in outputs_dir.iterdir()
             if d.is_dir() and d.name.startswith("run_")],
            key=lambda p: p.name,
        )

    # Search for checkpoints in run directories first
    for run_dir in reversed(run_dirs):  # most recent first
        candidates = sorted(run_dir.glob("seed_*/best.pt"))
        if candidates:
            chosen = candidates[0]
            logger.info(f"  Auto-discovered checkpoint: {chosen}")
            return chosen

    # Fallback: any .pt file in outputs
    candidates: list[Path] = []
    for ext in ("*.pt", "*.pth"):
        candidates.extend(outputs_dir.rglob(ext))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint files (*.pt, *.pth) found under {outputs_dir}. "
            f"Train a model first with: python scripts/train.py"
        )

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    chosen = candidates[0]
    logger.info(f"  Auto-discovered checkpoint: {chosen}")
    return chosen


def load_checkpoint(
    checkpoint_path: str, data_name: str
) -> tuple[dict[str, Any], Path]:
    """Load a checkpoint, handling 'auto' discovery.

    Returns:
        (state_dict, resolved_path)
    """
    if checkpoint_path.lower() == "auto":
        ckpt_path = discover_checkpoint(data_name)
    else:
        ckpt_path = Path(checkpoint_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    logger.info(f"  Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Handle both formats:
    #   1. Full trainer checkpoint: {"model_state_dict": ..., "ema_state_dict": ..., ...}
    #   2. Raw state_dict: {"input_proj.weight": ..., ...}
    if isinstance(ckpt, dict):
        # Prefer EMA weights if available (they're usually better)
        if "ema_state_dict" in ckpt:
            logger.info("  Using EMA model weights from checkpoint")
            state_dict = ckpt["ema_state_dict"]
            # AveragedModel wraps keys with "module." prefix
            cleaned = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                cleaned[new_key] = v
            state_dict = cleaned
        elif "model_state_dict" in ckpt:
            logger.info("  Using model_state_dict from checkpoint")
            state_dict = ckpt["model_state_dict"]
        else:
            # Assume it IS the state_dict directly
            state_dict = ckpt
    else:
        raise ValueError(f"Unexpected checkpoint type: {type(ckpt)}")

    return state_dict, ckpt_path


# ───────────────────────────────────────────────────────────────────
# Data loading & normalization
# ───────────────────────────────────────────────────────────────────
def load_data(data_name: str) -> tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Load panel and graph data.

    Returns:
        (counts, features, edge_queen, edge_knn)
        counts:  (S, T, C)  raw crime counts
        features: (S, T, F) z-score normalised features
        edge_queen: (2, E)  queen contiguity edges
        edge_knn:   (2, E)  or None
    """
    panel_path = PROJECT_ROOT / "data" / "processed" / f"{data_name}_panel.pt"
    graph_path = PROJECT_ROOT / "data" / "processed" / f"{data_name}_graph.pt"

    if not panel_path.exists():
        raise FileNotFoundError(
            f"Panel data not found: {panel_path}\n"
            f"Run: python scripts/fetch_data.py  to download and preprocess data."
        )
    if not graph_path.exists():
        raise FileNotFoundError(
            f"Graph data not found: {graph_path}\n"
            f"Run: python scripts/fetch_data.py  to download and preprocess data."
        )

    logger.info(f"  Loading panel from {panel_path}")
    panel = torch.load(panel_path, weights_only=False)
    counts = panel["counts"]      # (S, T, C)
    features = panel["features"]  # (S, T, F)
    S, T, C = counts.shape
    F = features.shape[-1]
    logger.info(f"  Panel: {S} spatial × {T} time × {C} categories, {F} features")

    # Z-score normalize (same as train.py)
    feat_mean = features.mean(dim=(0, 1), keepdim=True)
    feat_std = features.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
    features = (features - feat_mean) / feat_std

    logger.info(f"  Loading graph from {graph_path}")
    graph = torch.load(graph_path, weights_only=False)
    edge_queen = graph["queen"]
    edge_knn = graph.get("knn", None)

    # Validate graph-panel alignment
    max_node = edge_queen.max().item()
    if edge_knn is not None:
        max_node = max(max_node, edge_knn.max().item())
    if max_node >= S:
        raise ValueError(
            f"Graph-panel mismatch: max node index {max_node} >= panel spatial dim {S}. "
            f"Re-run 'python scripts/fetch_data.py'."
        )

    return counts, features, edge_queen, edge_knn


# ───────────────────────────────────────────────────────────────────
# Model construction
# ───────────────────────────────────────────────────────────────────
def build_model(num_features: int, num_categories: int) -> "torch.nn.Module":
    """Build CivicSafeModel with default architecture hyperparameters."""
    from civicsafe.models.civicsafe_model import CivicSafeModel

    model = CivicSafeModel(
        num_features=num_features,
        hidden_dim=128,
        spatial_layers=2,
        spatial_heads=4,
        temporal_layers=2,
        temporal_heads=4,
        temporal_ff_dim=512,
        num_categories=num_categories,
        max_seq_len=WINDOW_SIZE,
    )
    return model


# ───────────────────────────────────────────────────────────────────
# Rolling evaluation
# ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def rolling_evaluate(
    model: torch.nn.Module,
    counts: Tensor,
    features: Tensor,
    edge_queen: Tensor,
    edge_knn: Tensor | None,
    start_week: int,
    end_week: int,
    window_size: int,
    device: torch.device,
) -> dict[str, Tensor]:
    """Run rolling one-step-ahead evaluation.

    For each target week t in [start_week, end_week):
        input = features[:, t-window_size : t, :]
        predict → pi, mu, r for week t
        ground truth = counts[:, t, :]

    Args:
        model: Trained CivicSafeModel in eval mode.
        counts: (S, T, C) full panel counts.
        features: (S, T, F) normalised features.
        edge_queen, edge_knn: Graph edges.
        start_week, end_week: Target week range [start, end).
        window_size: Lookback window length.
        device: Compute device.

    Returns:
        Dict with stacked tensors:
            y_true: (N_weeks, S, C)
            pi, mu, r: (N_weeks, S, C)
    """
    model.eval()
    edge_queen = edge_queen.to(device)
    if edge_knn is not None:
        edge_knn = edge_knn.to(device)

    all_y, all_pi, all_mu, all_r = [], [], [], []

    T_total = counts.shape[1]
    actual_end = min(end_week, T_total)

    n_steps = actual_end - start_week
    logger.info(f"  Rolling evaluation: {n_steps} weeks [{start_week}, {actual_end})")

    for t in range(start_week, actual_end):
        if t - window_size < 0:
            logger.warning(f"  Skipping week {t}: insufficient history")
            continue

        # Input features for this step: (S, W, F)
        x_feat = features[:, t - window_size : t, :].to(device)

        output = model(x_feat, edge_queen, edge_knn)

        all_y.append(counts[:, t, :].cpu())           # (S, C)
        all_pi.append(output["pi"].cpu())              # (S, C)
        all_mu.append(output["mu"].cpu())              # (S, C)
        all_r.append(output["r"].cpu())                # (S, C)

        if (t - start_week + 1) % 10 == 0 or t == actual_end - 1:
            logger.info(f"    Evaluated week {t} ({t - start_week + 1}/{n_steps})")

    return {
        "y_true": torch.stack(all_y),  # (N, S, C)
        "pi": torch.stack(all_pi),     # (N, S, C)
        "mu": torch.stack(all_mu),     # (N, S, C)
        "r": torch.stack(all_r),       # (N, S, C)
    }


# ───────────────────────────────────────────────────────────────────
# Metrics computation
# ───────────────────────────────────────────────────────────────────
def compute_metrics(
    y_true: Tensor, pi: Tensor, mu: Tensor, r: Tensor
) -> dict[str, Any]:
    """Compute comprehensive metrics.

    Args:
        y_true, pi, mu, r: each (N, S, C)

    Returns:
        Nested dict with overall, per_category, per_spatial metrics.
    """
    from civicsafe.training.metrics import (
        brier_zero_inflation,
        crps_zinb,
        mae_zinb,
        rmse_zinb,
    )

    N, S, C = y_true.shape
    y_hat = (1.0 - pi.clamp(0, 1)) * mu.clamp(min=0)  # point forecast

    # ── Overall metrics (flatten all dims) ──
    y_flat = y_true.reshape(-1).float()
    pi_flat = pi.reshape(-1).float()
    mu_flat = mu.reshape(-1).float()
    r_flat = r.reshape(-1).float()
    yhat_flat = y_hat.reshape(-1).float()

    overall_mae = (y_flat - yhat_flat).abs().mean().item()
    overall_rmse = ((y_flat - yhat_flat) ** 2).mean().sqrt().item()
    overall_crps = crps_zinb(y_flat, pi_flat, mu_flat, r_flat).mean().item()
    overall_brier = brier_zero_inflation(y_flat, pi_flat).item()

    # MAPE: only where y > 0 to avoid division by zero
    mask_nonzero = y_flat > 0
    if mask_nonzero.any():
        overall_mape = (
            ((y_flat[mask_nonzero] - yhat_flat[mask_nonzero]).abs()
             / y_flat[mask_nonzero])
            .mean().item() * 100.0
        )
    else:
        overall_mape = float("nan")

    results: dict[str, Any] = {
        "overall": {
            "mae": round(overall_mae, 4),
            "rmse": round(overall_rmse, 4),
            "mape_pct": round(overall_mape, 2),
            "crps": round(overall_crps, 4),
            "brier_zero": round(overall_brier, 6),
            "n_test_weeks": N,
            "n_spatial_units": S,
            "n_categories": C,
        }
    }

    # ── Per-category metrics ──
    per_cat: dict[str, dict[str, float]] = {}
    for c in range(C):
        cat_name = CATEGORY_NAMES.get(c, f"category_{c}")
        yc = y_true[:, :, c].reshape(-1).float()
        yhat_c = y_hat[:, :, c].reshape(-1).float()
        pi_c = pi[:, :, c].reshape(-1).float()
        mu_c = mu[:, :, c].reshape(-1).float()
        r_c = r[:, :, c].reshape(-1).float()

        cat_mae = (yc - yhat_c).abs().mean().item()
        cat_rmse = ((yc - yhat_c) ** 2).mean().sqrt().item()

        mask_c = yc > 0
        if mask_c.any():
            cat_mape = (
                ((yc[mask_c] - yhat_c[mask_c]).abs() / yc[mask_c])
                .mean().item() * 100.0
            )
        else:
            cat_mape = float("nan")

        cat_crps = crps_zinb(yc, pi_c, mu_c, r_c).mean().item()
        cat_brier = brier_zero_inflation(yc, pi_c).item()

        per_cat[cat_name] = {
            "mae": round(cat_mae, 4),
            "rmse": round(cat_rmse, 4),
            "mape_pct": round(cat_mape, 2),
            "crps": round(cat_crps, 4),
            "brier_zero": round(cat_brier, 6),
        }
    results["per_category"] = per_cat

    # ── Per-spatial-unit MAE (identify best / worst predicted areas) ──
    spatial_mae: list[float] = []
    for s in range(S):
        ys = y_true[:, s, :].reshape(-1).float()
        yhat_s = y_hat[:, s, :].reshape(-1).float()
        spatial_mae.append((ys - yhat_s).abs().mean().item())

    sorted_idx = np.argsort(spatial_mae)
    n_show = min(5, S)
    results["per_spatial"] = {
        "best_units": [
            {"unit": int(sorted_idx[i]), "mae": round(spatial_mae[sorted_idx[i]], 4)}
            for i in range(n_show)
        ],
        "worst_units": [
            {"unit": int(sorted_idx[-(i + 1)]), "mae": round(spatial_mae[sorted_idx[-(i + 1)]], 4)}
            for i in range(n_show)
        ],
        "mean_spatial_mae": round(float(np.mean(spatial_mae)), 4),
        "std_spatial_mae": round(float(np.std(spatial_mae)), 4),
    }

    return results


# ───────────────────────────────────────────────────────────────────
# Conformal calibration on cal → evaluate coverage on test
# ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def conformal_evaluation(
    model: torch.nn.Module,
    counts: Tensor,
    features: Tensor,
    edge_queen: Tensor,
    edge_knn: Tensor | None,
    alpha: float,
    device: torch.device,
) -> dict[str, Any]:
    """Calibrate on dedicated calibration set, evaluate coverage on test set.

    Returns dict with calibration results.
    """
    from civicsafe.calibration.conformal import ECRCCalibrator

    CAL_START_WEEK = VAL_START_WEEK + (WEEKS_PER_YEAR // 2)

    # Run rolling eval on calibration set
    logger.info("  Running conformal: calibrating on calibration set (2022 H2)...")
    cal_results = rolling_evaluate(
        model, counts, features, edge_queen, edge_knn,
        start_week=CAL_START_WEEK,
        end_week=TEST_START_WEEK,
        window_size=WINDOW_SIZE,
        device=device,
    )

    # Run rolling eval on test set
    logger.info("  Running conformal: evaluating on test set...")
    test_results = rolling_evaluate(
        model, counts, features, edge_queen, edge_knn,
        start_week=TEST_START_WEEK,
        end_week=counts.shape[1],
        window_size=WINDOW_SIZE,
        device=device,
    )

    # Flatten cal results for calibration
    y_cal = cal_results["y_true"].reshape(-1).float()
    pi_cal = cal_results["pi"].reshape(-1).float()
    mu_cal = cal_results["mu"].reshape(-1).float()
    r_cal = cal_results["r"].reshape(-1).float()

    # Use ECRC (Equalized Conditional Risk Control) calibrator
    calibrator = ECRCCalibrator(alpha=alpha, delta=0.05)
    
    # Stratify calibration points by population density or total pop (feature index 0)
    # to form "groups" for ECRC coverage guarantees
    # We grab features for the calibration and test periods
    feat_cal = features[:, CAL_START_WEEK : TEST_START_WEEK, 0].transpose(0, 1).flatten()
    q_bins = torch.quantile(feat_cal, torch.tensor([0.2, 0.4, 0.6, 0.8], device=feat_cal.device))
    groups_cal = torch.bucketize(feat_cal, q_bins)

    calibrator.fit(y_cal, pi_cal, mu_cal, r_cal, groups=groups_cal)

    # Predict intervals on test set
    y_test = test_results["y_true"].reshape(-1).float()
    pi_test = test_results["pi"].reshape(-1).float()
    mu_test = test_results["mu"].reshape(-1).float()
    r_test = test_results["r"].reshape(-1).float()
    
    feat_test = features[:, TEST_START_WEEK : counts.shape[1], 0].transpose(0, 1).flatten()
    groups_test = torch.bucketize(feat_test, q_bins)

    intervals = calibrator.predict(pi_test, mu_test, r_test, groups=groups_test)
    lower = intervals["lower"]
    upper = intervals["upper"]

    # Coverage
    covered = ((y_test >= lower) & (y_test <= upper)).float()
    coverage = covered.mean().item()
    avg_width = (upper - lower).mean().item()
    median_width = (upper - lower).median().item()

    conf_results = {
        "method": "split_conformal",
        "alpha": alpha,
        "target_coverage": round(1 - alpha, 4),
        "test_coverage": round(coverage, 4),
        "avg_interval_width": round(avg_width, 2),
        "median_interval_width": round(median_width, 2),
        "threshold": round(calibrator.threshold, 4),
        "n_cal": int(y_val.shape[0]),
        "n_test": int(y_test.shape[0]),
    }

    # Per-category coverage
    N, S, C = test_results["y_true"].shape
    for c in range(C):
        cat_name = CATEGORY_NAMES.get(c, f"cat_{c}")
        yc = test_results["y_true"][:, :, c].reshape(-1).float()
        pic = test_results["pi"][:, :, c].reshape(-1).float()
        muc = test_results["mu"][:, :, c].reshape(-1).float()
        rc = test_results["r"][:, :, c].reshape(-1).float()
        ivals = calibrator.predict(pic, muc, rc)
        cov_c = ((yc >= ivals["lower"]) & (yc <= ivals["upper"])).float().mean().item()
        conf_results[f"coverage_{cat_name}"] = round(cov_c, 4)

    return conf_results


# ───────────────────────────────────────────────────────────────────
# LaTeX table generation
# ───────────────────────────────────────────────────────────────────
def generate_latex_table(metrics: dict[str, Any]) -> str:
    """Generate a publication-ready LaTeX results table."""
    overall = metrics["overall"]
    per_cat = metrics.get("per_category", {})

    lines = [
        r"\begin{table}[htbp]",
        r"  \centering",
        r"  \caption{CIVIC-SAFE test set evaluation results (2023, rolling one-step-ahead)}",
        r"  \label{tab:test_results}",
        r"  \begin{tabular}{l r r r r r}",
        r"    \toprule",
        r"    & \textbf{MAE} & \textbf{RMSE} & \textbf{MAPE (\%)} & \textbf{CRPS} & \textbf{Brier} \\",
        r"    \midrule",
        f"    Overall "
        f"& {overall['mae']:.4f} "
        f"& {overall['rmse']:.4f} "
        f"& {overall['mape_pct']:.2f} "
        f"& {overall['crps']:.4f} "
        f"& {overall['brier_zero']:.4f} \\\\",
    ]

    if per_cat:
        lines.append(r"    \midrule")
        for cat_name, cat_metrics in per_cat.items():
            lines.append(
                f"    {cat_name.capitalize()} "
                f"& {cat_metrics['mae']:.4f} "
                f"& {cat_metrics['rmse']:.4f} "
                f"& {cat_metrics['mape_pct']:.2f} "
                f"& {cat_metrics['crps']:.4f} "
                f"& {cat_metrics['brier_zero']:.4f} \\\\"
            )

    lines.extend([
        r"    \bottomrule",
        r"  \end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ───────────────────────────────────────────────────────────────────
# Main evaluation pipeline
# ───────────────────────────────────────────────────────────────────
def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the full evaluation pipeline."""
    t_start = time.time()

    logger.info("=" * 70)
    logger.info("  CIVIC-SAFE — Production Test-Set Evaluation")
    logger.info("=" * 70)

    # ── 1. Load data ──
    logger.info("\n[1/5] Loading data...")
    counts, features, edge_queen, edge_knn = load_data(args.data)
    S, T, C = counts.shape
    F = features.shape[-1]

    # ── 2. Build model & load checkpoint ──
    logger.info("\n[2/5] Loading model checkpoint...")
    state_dict, ckpt_path = load_checkpoint(args.checkpoint, args.data)
    model = build_model(num_features=F, num_categories=C)

    # Try to load state_dict (handle possible key mismatches gracefully)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning(f"  Missing keys in checkpoint: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        logger.warning(f"  Unexpected keys in checkpoint: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model loaded: {num_params:,} parameters")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    logger.info(f"  Device: {device}")

    # ── 3. Rolling test-set evaluation ──
    logger.info("\n[3/5] Running rolling test-set evaluation...")
    T_total = counts.shape[1]
    test_out = rolling_evaluate(
        model, counts, features, edge_queen, edge_knn,
        start_week=TEST_START_WEEK,
        end_week=T_total,
        window_size=WINDOW_SIZE,
        device=device,
    )

    # ── 4. Compute metrics ──
    logger.info("\n[4/5] Computing metrics...")
    metrics = compute_metrics(
        test_out["y_true"], test_out["pi"], test_out["mu"], test_out["r"]
    )

    # Add metadata
    metrics["metadata"] = {
        "data": args.data,
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "num_parameters": num_params,
        "test_start_week": TEST_START_WEEK,
        "test_end_week": T_total,
        "window_size": WINDOW_SIZE,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # ── 4b. Conformal calibration & coverage ──
    logger.info("\n[4b/5] Running conformal calibration (val→test)...")
    try:
        conf_results = conformal_evaluation(
            model, counts, features, edge_queen, edge_knn,
            alpha=args.alpha, device=device,
        )
        metrics["conformal"] = conf_results
    except Exception as e:
        logger.warning(f"  Conformal evaluation failed: {e}")
        metrics["conformal"] = {"error": str(e)}

    # ── 5. Save results ──
    logger.info("\n[5/5] Saving results...")
    output_dir = PROJECT_ROOT / "outputs" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{args.data}_test_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"  Results saved to: {output_file}")

    # ── LaTeX table ──
    latex = generate_latex_table(metrics)
    latex_file = output_dir / f"{args.data}_results_table.tex"
    with open(latex_file, "w", encoding="utf-8") as f:
        f.write(latex)
    logger.info(f"  LaTeX table saved to: {latex_file}")

    # ── Summary ──
    elapsed = time.time() - t_start
    overall = metrics["overall"]

    logger.info("\n" + "=" * 70)
    logger.info("  TEST SET RESULTS SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Data:          {args.data} ({S} spatial × {C} categories)")
    logger.info(f"  Test weeks:    {metrics['overall']['n_test_weeks']}")
    logger.info(f"  Checkpoint:    {ckpt_path.name}")
    logger.info(f"  ─────────────────────────────────────────────────")
    logger.info(f"  MAE:           {overall['mae']:.4f}")
    logger.info(f"  RMSE:          {overall['rmse']:.4f}")
    logger.info(f"  MAPE:          {overall['mape_pct']:.2f}%")
    logger.info(f"  CRPS:          {overall['crps']:.4f}")
    logger.info(f"  Brier(zero):   {overall['brier_zero']:.6f}")

    if "conformal" in metrics and "test_coverage" in metrics["conformal"]:
        conf = metrics["conformal"]
        logger.info(f"  ─────────────────────────────────────────────────")
        logger.info(f"  Coverage:      {conf['test_coverage']:.4f} (target {conf['target_coverage']:.4f})")
        logger.info(f"  Avg width:     {conf['avg_interval_width']:.2f}")

    per_cat = metrics.get("per_category", {})
    if per_cat:
        logger.info(f"  ─────────────────────────────────────────────────")
        for cat_name, cm in per_cat.items():
            logger.info(
                f"  {cat_name:12s}:  MAE={cm['mae']:.4f}  RMSE={cm['rmse']:.4f}  MAPE={cm['mape_pct']:.1f}%"
            )

    logger.info(f"  ─────────────────────────────────────────────────")
    logger.info(f"  Elapsed:       {elapsed:.1f}s")
    logger.info("=" * 70)

    # Print LaTeX table
    logger.info("\nLaTeX table:\n" + latex)

    return metrics


# ───────────────────────────────────────────────────────────────────
# CLI
# ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="CIVIC-SAFE production test-set evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
    # Evaluate with auto-discovered checkpoint
    python scripts/evaluate_trained.py --checkpoint auto --data chicago

    # Evaluate specific checkpoint
    python scripts/evaluate_trained.py --checkpoint outputs/run_123/seed_42/best.pt

    # Evaluate on NYC data with custom alpha
    python scripts/evaluate_trained.py --checkpoint path/to/model.pt --data nyc --alpha 0.05
""",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt/.pth), or 'auto' to discover latest",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="chicago",
        choices=["chicago", "nyc"],
        help="City dataset to evaluate on (default: chicago)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Conformal prediction miscoverage level (default: 0.1 = 90%% coverage)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_evaluation(args)


if __name__ == "__main__":
    main()
