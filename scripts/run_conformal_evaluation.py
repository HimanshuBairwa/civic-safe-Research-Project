#!/usr/bin/env python
"""CIVIC-SAFE Phase 5: Conformal Calibration + Coverage Evaluation Pipeline.

This is the heart of novelty claims [N1] and [N2]. It transforms a trained
ZINB forecasting model into a conformal prediction system with:
  (a) Provable marginal coverage guarantees (1-α)
  (b) Conditional coverage audits stratified by demographic quartile
  (c) Adaptive temporal correction for non-exchangeability (ACI)
  (d) Full fairness audit with pass/fail against pre-registered thresholds

Pipeline stages:
  1. Load trained checkpoint + data panel
  2. Run model inference on calibration set (2022 H2) to collect ZINB params
  3. Fit ALL calibration methods (SplitCP, WeightedCP, Mondrian, ECRC, AdaptiveECRC)
  4. Run model inference on test set (2023) with rolling one-step-ahead
  5. Produce calibrated prediction intervals for each method
  6. Compute coverage, width, CRPS, CRPSS vs baselines, demographic disparity
  7. Serialize calibration objects + audit report to disk

Usage:
    python scripts/run_conformal_evaluation.py --data chicago
    python scripts/run_conformal_evaluation.py --data nyc --alpha 0.1
    python scripts/run_conformal_evaluation.py --data chicago --checkpoint outputs/run_XXX

References:
    - Romano, Patterson, Candès (2019): Conformalized Quantile Regression
    - Gibbs & Candès (2021): Adaptive Conformal Inference Under Distribution Shift
    - Feldman et al. (2021): Achieving Risk Control via Online Learning
    - Tibshirani et al. (2019): Conformal Prediction Under Covariate Shift
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import Tensor

# ───────────────────────────────────────────────────────────────────
# Project setup
# ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from civicsafe.audit.feedback_loop import compute_all_feedback_metrics
from civicsafe.calibration.conformal import (
    AdaptiveTemporalECRCCalibrator,
    ECRCCalibrator,
    EqualizedCoverageCalibrator,
    MondrianConformalCalibrator,
    RandomizedSplitConformalCalibrator,
    SplitConformalCalibrator,
    VarianceScaledConformalCalibrator,
    WeightedConformalCalibrator,
)
from civicsafe.calibration.emos import crps_decomposition
from civicsafe.calibration.ensemble_evaluator import (
    combine_ensemble_outputs,
    resolve_ensemble_checkpoints,
    rolling_panel_inference,
    select_checkpoint_weight_sets,
)
from civicsafe.calibration.policies import (
    DEFAULT_MAX_ABSTENTION,
    DEFAULT_MAX_DISPARITY,
    assess_forecasting_gate,
    select_best_calibrator,
)
from civicsafe.calibration.recalibration import recalibrate_and_evaluate
from civicsafe.calibration.significance import compare_forecasts
from civicsafe.models.civicsafe_model import CivicSafeModel
from civicsafe.models.dataset import CrimeWindowDataset, create_chronological_splits
from civicsafe.training.metrics import compute_all_metrics, crps_zinb, pit_values

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────
CATEGORY_NAMES = {0: "violent", 1: "property", 2: "drug"}
ALPHA_DEFAULT = 0.1  # 90% coverage target
COVERAGE_DISPARITY_THRESHOLD = DEFAULT_MAX_DISPARITY

# Pre-registered kill criteria
class KillCriterionTriggered(Exception):
    """Raised when a pre-registered quality threshold is violated."""
    pass


# ───────────────────────────────────────────────────────────────────
# Checkpoint Discovery
# ───────────────────────────────────────────────────────────────────
def resolve_checkpoints(
    checkpoint_path: str | Path | None, data_name: str
) -> list[Path]:
    """Resolve a checkpoint file, seed directory, run directory, or auto."""
    return resolve_ensemble_checkpoints(
        checkpoint_path,
        data_name=data_name,
        outputs_dir=PROJECT_ROOT / "outputs",
    )


def discover_checkpoint(data_name: str) -> Path:
    """Auto-discover the first seed in the preferred evaluation run."""
    return discover_all_checkpoints(data_name)[0]


def discover_all_checkpoints(data_name: str) -> list[Path]:
    """Auto-discover all seed checkpoints in the preferred evaluation run."""
    checkpoints = resolve_checkpoints("auto", data_name)
    logger.info(
        f"  Found {len(checkpoints)} seed checkpoint(s) in "
        f"{checkpoints[0].parent.parent.name}"
    )
    for checkpoint in checkpoints:
        logger.info(f"    {checkpoint.parent.name}/{checkpoint.name}")
    return checkpoints


# Model Loading
# ───────────────────────────────────────────────────────────────────
def load_model_from_checkpoint(
    checkpoint_path: Path,
    num_features: int,
    num_categories: int,
    config: dict[str, Any],
    device: str = "cpu",
    weights: str = "raw",
) -> CivicSafeModel:
    """Load a CivicSafeModel from a training checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        num_features: Number of input features (F dimension).
        num_categories: Number of crime categories (C dimension).
        config: Model configuration dictionary.
        device: Target device.
        weights: Which parameter set to load — "raw" (model_state_dict) or
            "ema" (ema_state_dict). Choose via validation, not test.

    Returns:
        Loaded model in eval mode.
    """
    model_cfg = config.get("model", {})
    spatial_cfg = model_cfg.get("spatial", {})
    temporal_cfg = model_cfg.get("temporal", {})

    # Load the checkpoint BEFORE constructing the model: the trainer records an
    # `arch` fingerprint describing the model it actually trained, and some of
    # those toggles change forward() behaviour without changing any parameter
    # shape. `level_anchor` is the dangerous case -- an anchored and an
    # unanchored head have byte-identical state dicts, so a mismatch loads
    # cleanly under strict=True and then silently rescales every mu. Rebuilding
    # from the config alone would reintroduce exactly that class of bug (cf. the
    # raw-vs-ema divergence documented below, which had the same root cause:
    # this script inferring what evaluate_trained.py was told).
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arch = checkpoint.get("arch", {}) if isinstance(checkpoint, dict) else {}
    ablations = model_cfg.get("ablations", {})

    def _toggle(name: str, default: bool = True) -> bool:
        """Prefer the checkpoint's recorded architecture over the config."""
        if name in arch:
            return bool(arch[name])
        return bool(ablations.get(name, default))

    model = CivicSafeModel(
        num_features=num_features,
        hidden_dim=arch.get("hidden_dim", spatial_cfg.get("hidden_dim", 128)),
        spatial_layers=spatial_cfg.get("num_layers", 2),
        spatial_heads=spatial_cfg.get("num_heads", 4),
        temporal_layers=temporal_cfg.get("num_layers", 2),
        temporal_heads=temporal_cfg.get("num_heads", 4),
        temporal_ff_dim=temporal_cfg.get("dim_feedforward", 512),
        num_categories=num_categories,
        max_seq_len=temporal_cfg.get("max_seq_len", 52),
        use_gnn=_toggle("use_gnn"),
        use_transformer=_toggle("use_transformer"),
        zero_inflation=_toggle("zero_inflation"),
        level_anchor=_toggle("level_anchor", default=False),
    )

    # Handle different checkpoint formats.
    #
    # `weights` selects which parameter set to evaluate:
    #   "raw"  -> model_state_dict (online SGD weights)
    #   "ema"  -> ema_state_dict   (Polyak-averaged weights)
    #
    # This MUST be stated explicitly rather than left to key-order luck. This
    # script used to read model_state_dict while evaluate_trained.py read
    # ema_state_dict, so the two reported irreconcilable CRPS for the SAME
    # checkpoint (Chicago seed_1024: 3.30 raw vs 4.63 ema).
    #
    # Note the EMA weights here are not trustworthy by default: the trainer uses
    # decay=0.999 (≈1000-step horizon) but only runs ~10 optimizer steps/epoch,
    # so EMA barely converges within a run and retains a large fraction of the
    # initial snapshot. Pick the weight set on VALIDATION data, never on test.
    if weights == "ema":
        if "ema_state_dict" not in checkpoint:
            raise KeyError(f"{checkpoint_path} has no 'ema_state_dict'.")
        state_dict = checkpoint["ema_state_dict"]
        weights_source = "ema_state_dict"
    elif weights == "raw":
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            weights_source = "model_state_dict"
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            weights_source = "state_dict"
        else:
            state_dict = checkpoint
            weights_source = "raw_toplevel"
    else:
        raise ValueError(f"weights must be 'raw' or 'ema', got {weights!r}")

    # Handle EMA model state dicts (AveragedModel wraps keys with 'module.'
    # and adds a non-parameter 'n_averaged' buffer that the bare model lacks).
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        if key == "n_averaged":
            continue
        clean_key = key.replace("module.", "")
        cleaned_state_dict[clean_key] = value

    # strict=True: a silent key mismatch here would leave layers at random init
    # and still "work", quietly reporting garbage metrics. Fail loudly instead.
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        raise RuntimeError(
            f"Checkpoint {checkpoint_path} is missing {len(missing)} model keys "
            f"(first few: {list(missing)[:5]}). Refusing to evaluate a partially "
            f"initialised model."
        )
    if unexpected:
        logger.warning(f"  Unexpected keys ignored: {list(unexpected)[:5]}")
    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"  Loaded model: {num_params:,} parameters from {checkpoint_path.name} "
        f"[weights={weights_source}]"
    )
    return model


# ───────────────────────────────────────────────────────────────────
# Rolling Inference
# ───────────────────────────────────────────────────────────────────
@torch.inference_mode()
def run_rolling_inference(
    model: CivicSafeModel,
    dataset: CrimeWindowDataset,
    edge_queen: Tensor,
    edge_knn: Tensor | None,
    device: str = "cpu",
) -> dict[str, Tensor]:
    """Run rolling one-step-ahead inference on a dataset split.

    For each window in the dataset, the model produces ZINB parameters
    (pi, mu, r) for the next timestep across all spatial units and categories.

    Args:
        model: Trained CivicSafeModel in eval mode.
        dataset: CrimeWindowDataset (cal or test split).
        edge_queen: Queen contiguity edges. Shape: (2, E_q)
        edge_knn: KNN edges. Shape: (2, E_k) or None.
        device: Computation device.

    Returns:
        Dictionary with tensors:
            y: Ground-truth counts. Shape: (N_windows, S, C)
            pi: Zero-inflation probs. Shape: (N_windows, S, C)
            mu: NB means. Shape: (N_windows, S, C)
            r: NB dispersions. Shape: (N_windows, S, C)
    """
    return rolling_panel_inference(
        model,
        counts=dataset.counts,
        features=dataset.features,
        edge_queen=edge_queen,
        edge_knn=edge_knn,
        target_weeks=dataset.valid_targets,
        window_size=dataset.window_size,
        device=device,
    )


# ───────────────────────────────────────────────────────────────────
# Demographic Group Assignment
# ───────────────────────────────────────────────────────────────────
def load_demographic_groups(
    data_name: str,
    num_spatial_units: int,
) -> Tensor:
    """Load demographic quartile assignments for spatial units.

    Uses median household income from the pre-computed demographics CSV
    to assign each spatial unit to one of 4 income quartiles.

    Args:
        data_name: 'chicago' or 'nyc'.
        num_spatial_units: Expected number of spatial units (S).

    Returns:
        Integer group labels. Shape: (S,) with values in {0, 1, 2, 3}.
    """
    import pandas as pd

    demo_path = PROJECT_ROOT / "data" / "processed" / f"{data_name}_demographics.csv"

    if demo_path.exists():
        df = pd.read_csv(demo_path)
        # Look for median household income column
        income_col = None
        for col in df.columns:
            if "median" in col.lower() and "income" in col.lower():
                income_col = col
                break
            if "B19013_001E" in col:
                income_col = col
                break

        if income_col is not None and len(df) >= num_spatial_units:
            # Align rows to the panel's spatial index rather than trusting CSV
            # row order. `spatial_unit` ids are not guaranteed to be 0..S-1 in
            # order (NYC precinct ids start 1, 5, ...), so positional slicing
            # can assign a unit's income to the wrong spatial cell.
            if "spatial_unit" in df.columns:
                df = df.sort_values("spatial_unit").reset_index(drop=True)
            incomes = np.asarray(
                df[income_col].values[:num_spatial_units], dtype=np.float64
            )

            # Replace NaN/non-positive with the median of the valid entries.
            valid_mask = np.isfinite(incomes) & (incomes > 0)
            n_imputed = int((~valid_mask).sum())
            if valid_mask.sum() > 0:
                median_income = float(np.median(incomes[valid_mask]))
                incomes[~valid_mask] = median_income

            # Quartile assignment via RANK, not value-bin edges.
            #
            # np.digitize on raw values collapses ties into a single bin: every
            # unit imputed to the median lands on the same side of the p50 edge,
            # which on NYC produced Q1=20, Q2=4, Q3=34, Q4=20. A 4-unit group is
            # only 4*26*3 = 312 calibration points, inflating the ECRC Hoeffding
            # slack to eps = sqrt(ln(2/delta_g)/(2*n_g)) = 0.0902. That drove
            # adjusted_alpha to its 0.01 floor => 98.7% coverage at width 34
            # instead of the target 90%. Ranking splits ties evenly so all four
            # groups stay ~S/4 and eps stays comparable across groups and cities.
            order = np.argsort(np.argsort(incomes, kind="stable"), kind="stable")
            quartiles = np.minimum(
                (order * 4) // max(len(incomes), 1), 3
            ).astype(np.int64)

            counts_per_q = [int(np.sum(quartiles == k)) for k in range(4)]
            logger.info(
                f"  Demographic groups from {demo_path.name}: "
                f"Q1={counts_per_q[0]}, Q2={counts_per_q[1]}, "
                f"Q3={counts_per_q[2]}, Q4={counts_per_q[3]}"
                + (f" ({n_imputed} imputed to median)" if n_imputed else "")
            )
            if n_imputed:
                logger.warning(
                    f"  {n_imputed}/{num_spatial_units} units had missing/invalid "
                    f"income and were imputed to the median. Income-quartile "
                    f"fairness groups for those units are not meaningful — report "
                    f"this caveat alongside any disparity number."
                )
            return torch.tensor(quartiles, dtype=torch.long)

    # Fallback: equal-size geographic groups
    logger.warning(
        f"  Demographics file not found or incomplete. "
        f"Using geographic group assignment (spatial unit index mod 4)."
    )
    return torch.arange(num_spatial_units, dtype=torch.long) % 4


# ───────────────────────────────────────────────────────────────────
# Baseline CRPS Computation (HA + Seasonal-Naive)
# ───────────────────────────────────────────────────────────────────
def compute_baseline_crps(
    counts: Tensor,
    test_start: int = 260,
    train_end: int = 208,
    window_size: int = 52,
) -> dict[str, float]:
    """Compute CRPS for the naive baselines: Historical Average and Seasonal Naive.

    Historical Average comes in two flavours and they are NOT interchangeable:

      rolling (reported as ``ha_crps``, the honest one)
          For test week t, predict the mean of weeks [t-52, t). Updates every
          week, so it tracks level drift. This matches baselines.py:174
          (``item["input_counts"].mean(dim=1)``) and is what a reviewer means
          by "historical average".

      frozen (reported as ``ha_crps_frozen``, for transparency only)
          A single mean over the training period [0, 208), held constant across
          all 53 test weeks. Cannot track drift.

    On Chicago the gap is large -- rolling 2.9322 vs frozen 3.8781 -- because
    the panel level moves substantially over a 53-week horizon. Skill scores
    were previously computed against the frozen variant, which inflated CRPSS
    vs HA from a genuine loss into an apparent 16.7% win. Both are emitted now
    and the skill score uses the rolling one; a model that cannot beat a
    trailing mean has no forecasting claim to make.

    Seasonal Naive: predict Y(t-52) = same week last year. Unaffected by this
    distinction (4.4008 here vs 4.4013 in baselines.py -- the residual is the
    window-alignment convention, not a definitional disagreement), which is
    what localizes the discrepancy to the HA definition.

    All point-prediction baselines are scored through the same ZINB-CRPS with
    pi=0, r=1000 (the Poisson limit), so the comparison isolates the
    conditional mean rather than rewarding the model for its richer
    distributional form.

    Args:
        counts: Full crime count tensor. Shape: (S, T, C)
        test_start: First week of test set.
        train_end: Last week of training set (exclusive).
        window_size: Trailing window for the rolling HA. Must match the
            model's input window so the two see identical history.

    Returns:
        Dictionary with 'ha_crps' (rolling), 'ha_crps_frozen', and
        'seasonal_naive_crps'.
    """
    train_counts = counts[:, :train_end, :].float()  # (S, train_T, C)
    test_counts = counts[:, test_start:, :].float()   # (S, test_T, C)

    S, test_T, C = test_counts.shape
    y_flat = test_counts.reshape(-1)

    # Shared ZINB-CRPS parameters for point-prediction baselines
    pi_zero = torch.zeros_like(y_flat)
    r_large = torch.full_like(y_flat, 1000.0)  # r→∞ gives Poisson

    # --- Baseline 1a: Historical Average, ROLLING (the honest baseline) ---
    # Predict each test week from the mean of the preceding `window_size`
    # weeks. Built by stacking per-week trailing means so it is obvious that
    # only past data is used -- no test week contributes to its own prediction.
    rolling_means = torch.stack(
        [
            counts[:, t - window_size : t, :].float().mean(dim=1)  # (S, C)
            for t in range(test_start, test_start + test_T)
        ],
        dim=1,
    )  # (S, test_T, C)
    mu_ha_rolling = rolling_means.reshape(-1).clamp(min=0.01)
    ha_crps_rolling = crps_zinb(y_flat, pi_zero, mu_ha_rolling, r_large).mean().item()

    # --- Baseline 1b: Historical Average, FROZEN (reported for transparency) ---
    hist_mean = train_counts.mean(dim=1, keepdim=True)  # (S, 1, C)
    mu_ha_frozen = hist_mean.expand_as(test_counts).reshape(-1).clamp(min=0.01)
    ha_crps_frozen = crps_zinb(y_flat, pi_zero, mu_ha_frozen, r_large).mean().item()

    # --- Baseline 2: Seasonal Naive (Y(t-52) = same week last year) ---
    # For test week t (starting at test_start=260), seasonal prediction = Y(t-52)
    seasonal_start = test_start - 52  # = 208 (start of val year)
    seasonal_counts = counts[:, seasonal_start:seasonal_start + test_T, :].float()  # (S, test_T, C)
    mu_sn = seasonal_counts.reshape(-1).clamp(min=0.01)
    sn_crps = crps_zinb(y_flat, pi_zero, mu_sn, r_large).mean().item()

    return {
        "ha_crps": ha_crps_rolling,
        "ha_crps_frozen": ha_crps_frozen,
        "seasonal_naive_crps": sn_crps,
    }


# ───────────────────────────────────────────────────────────────────
# Coverage Computation
# ───────────────────────────────────────────────────────────────────
def compute_coverage_metrics(
    y: Tensor,
    lower: Tensor,
    upper: Tensor,
    groups: Tensor | None = None,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """Compute comprehensive coverage metrics for prediction intervals.

    Abstentions (NaN bounds) are EXCLUDED from coverage and reported separately.
    A calibrator that declines to issue an interval has made no claim, so it can
    be neither right nor wrong about that cell. Folding abstentions into the
    denominator as misses is the more dangerous of the two mistakes: `y >= nan`
    is False, so an unmasked abstention silently reads as a miscoverage. With
    the old absolute `max_width=100.0` default this drove marginal coverage to
    0.0000 on high-count cells while every interval was in fact sound.

    `marginal_coverage` is therefore coverage CONDITIONAL on issuing an
    interval, and must always be read alongside `abstention_rate` -- a method
    can buy coverage by abstaining, and the pair makes that visible.

    Args:
        y: Ground-truth counts. Shape: (N,)
        lower: Lower bounds. Shape: (N,)
        upper: Upper bounds. Shape: (N,)
        groups: Demographic group labels. Shape: (N,) or None.
        alpha: Nominal miscoverage level.

    Returns:
        Dictionary with coverage metrics.
    """
    issued = ~(torch.isnan(lower) | torch.isnan(upper))
    n_total = int(issued.numel())
    n_issued = int(issued.sum().item())

    covered = ((y >= lower) & (y <= upper)).float()
    width = (upper - lower).float()

    if n_issued == 0:
        logger.warning(
            f"  compute_coverage_metrics: ALL {n_total} cells abstained; "
            "coverage is undefined. Reporting NaN rather than 0.0 so this "
            "cannot be mistaken for a total-miscoverage result."
        )
        nan = float("nan")
        result: dict[str, Any] = {
            "marginal_coverage": nan,
            "mean_width": nan,
            "median_width": nan,
            "target_coverage": 1.0 - alpha,
            "coverage_gap": nan,
            "abstention_rate": 1.0,
            "n_issued": 0,
            "n_total": n_total,
        }
        if groups is not None:
            result["per_group"] = {}
            result["coverage_disparity"] = nan
        return result

    cov_issued = covered[issued]
    w_issued = width[issued]

    result = {
        "marginal_coverage": cov_issued.mean().item(),
        "mean_width": w_issued.mean().item(),
        "median_width": w_issued.median().item(),
        "target_coverage": 1.0 - alpha,
        "coverage_gap": cov_issued.mean().item() - (1.0 - alpha),
        # Fraction of cells where no interval was issued. Read WITH coverage.
        "abstention_rate": 1.0 - (n_issued / n_total),
        "n_issued": n_issued,
        "n_total": n_total,
    }
    if result["abstention_rate"] > 0.0:
        logger.warning(
            f"  {n_total - n_issued}/{n_total} cells abstained "
            f"({result['abstention_rate']:.1%}). Coverage "
            f"{result['marginal_coverage']:.4f} is CONDITIONAL on the "
            f"{n_issued} issued intervals, not over the full test set."
        )

    # Per-category coverage (if data has category structure)
    # Per-group coverage
    if groups is not None:
        group_coverages = {}
        unique_groups = groups.unique().tolist()  # type: ignore[no-untyped-call]
        for g in unique_groups:
            mask = groups == g
            if mask.sum() > 0:
                scored = mask & issued
                n_g = int(mask.sum().item())
                n_g_issued = int(scored.sum().item())
                # Abstention disparity is audit component 6, so the per-group
                # rate is a reported quantity in its own right, not just a
                # denominator correction.
                entry: dict[str, Any] = {
                    "n_samples": n_g,
                    "n_issued": n_g_issued,
                    "abstention_rate": 1.0 - (n_g_issued / n_g),
                }
                if n_g_issued == 0:
                    entry["coverage"] = float("nan")
                    entry["mean_width"] = float("nan")
                else:
                    entry["coverage"] = covered[scored].mean().item()
                    entry["mean_width"] = width[scored].mean().item()
                group_coverages[f"group_{g}"] = entry
        result["per_group"] = group_coverages

        # Coverage disparity: max - min across groups. Groups that abstained
        # entirely have no coverage to compare and are excluded here; their
        # absence is visible in per_group abstention_rate.
        all_coverages = [
            v["coverage"]
            for v in group_coverages.values()
            if not math.isnan(v["coverage"])
        ]
        abst = [v["abstention_rate"] for v in group_coverages.values()]
        if len(abst) >= 2:
            result["abstention_disparity"] = max(abst) - min(abst)
        if len(all_coverages) >= 2:
            result["coverage_disparity"] = max(all_coverages) - min(all_coverages)
            result["min_group_coverage"] = min(all_coverages)
            result["max_group_coverage"] = max(all_coverages)
        else:
            result["coverage_disparity"] = 0.0

    return result


# ───────────────────────────────────────────────────────────────────
# Main Evaluation Pipeline
# ───────────────────────────────────────────────────────────────────
def run_conformal_evaluation(
    data_name: str,
    checkpoint_path: str | None = None,
    alpha: float = ALPHA_DEFAULT,
    device: str = "cpu",
    weights: str = "auto",
) -> dict[str, Any]:
    """Execute the complete conformal calibration + evaluation pipeline.

    Args:
        data_name: Dataset name ('chicago' or 'nyc').
        checkpoint_path: Path to checkpoint, or None for auto-discovery.
        alpha: Nominal miscoverage level (default 0.1 for 90% coverage).
        device: Computation device.
        weights: Checkpoint parameter set — "auto" (pick by validation CRPS),
            "raw" (model_state_dict) or "ema" (ema_state_dict).

    Returns:
        Complete evaluation results dictionary.
    """
    t_start = time.time()

    logger.info("=" * 70)
    logger.info("  CIVIC-SAFE Phase 5: Conformal Calibration + Coverage Evaluation")
    logger.info(f"  Dataset: {data_name} | Alpha: {alpha} | Device: {device}")
    logger.info("=" * 70)

    # ─── Step 1: Load data ───
    logger.info("\n[1/7] Loading data panel and graph...")
    panel_path = PROJECT_ROOT / "data" / "processed" / f"{data_name}_panel.pt"
    graph_path = PROJECT_ROOT / "data" / "processed" / f"{data_name}_graph.pt"

    if not panel_path.exists():
        raise FileNotFoundError(
            f"Panel not found at {panel_path}. Run: python scripts/fetch_data.py"
        )

    panel = torch.load(panel_path, weights_only=False)
    counts = panel["counts"]   # (S, T, C)
    features = panel["features"]  # (S, T, F)
    S, T, C = counts.shape
    F = features.shape[-1]

    # Normalize features using training-only statistics (no data leakage)
    norm_stats_path = PROJECT_ROOT / 'data' / 'processed' / f'{data_name}_norm_stats.pt'
    if norm_stats_path.exists():
        norm_stats = torch.load(norm_stats_path, weights_only=False)
        feat_mean = norm_stats['mean']
        feat_std = norm_stats['std']
        logger.info('  Loaded normalization stats from training')
    else:
        # Fallback: compute from training period only
        train_end = 208
        train_features = features[:, :train_end, :]
        feat_mean = train_features.mean(dim=(0, 1), keepdim=True)
        feat_std = train_features.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
        logger.info('  Computed normalization from training period (no saved stats)')
    features = (features - feat_mean) / feat_std

    graph = torch.load(graph_path, weights_only=False)
    edge_queen = graph["queen"]
    edge_knn = graph.get("knn")

    logger.info(f"  Panel: {S} spatial × {T} weeks × {C} categories, {F} features")

    # ─── Step 2: Create chronological splits ───
    logger.info("\n[2/7] Creating chronological splits...")
    splits = create_chronological_splits(counts, features)
    val_dataset = splits["val"]
    cal_dataset = splits["cal"]
    test_dataset = splits["test"]
    logger.info(f"  Validation set: {len(val_dataset)} windows")
    logger.info(f"  Calibration set: {len(cal_dataset)} windows")
    logger.info(f"  Test set: {len(test_dataset)} windows")

    # ─── Step 3: Load trained model ───
    logger.info("\n[3/7] Loading trained model...")
    config_dir = PROJECT_ROOT / "configs"
    config: dict[str, Any] = {}
    for cfg_file in [
        config_dir / "model" / "spatiotemporal_zinb.yaml",
        config_dir / "training" / "default.yaml",
    ]:
        if cfg_file.exists():
            with open(cfg_file, encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    config.update(loaded)

    all_ckpts = resolve_checkpoints(checkpoint_path, data_name)
    def _load_weight_candidate(path: Path, candidate: str) -> CivicSafeModel:
        return load_model_from_checkpoint(
            path,
            F + C,
            C,
            config,
            device,
            weights=candidate,
        )

    def _score_weight_candidate(model: CivicSafeModel) -> float:
        probe_results = run_rolling_inference(
            model,
            val_dataset,
            edge_queen,
            edge_knn,
            device,
        )
        return float(
            crps_zinb(
                probe_results["y"].reshape(-1),
                probe_results["pi"].reshape(-1),
                probe_results["mu"].reshape(-1),
                probe_results["r"].reshape(-1),
            ).mean().item()
        )

    selected_weights, val_scores_per_checkpoint = select_checkpoint_weight_sets(
        all_ckpts,
        requested=weights,
        load_model=_load_weight_candidate,
        score_model=_score_weight_candidate,
    )

    K = len(all_ckpts)
    logger.info(f"\n[3-5/7] Ensemble inference with {K} seed(s)...")
    logger.info(
        "  Per-seed weight sets: "
        f"{[selected_weights[checkpoint] for checkpoint in all_ckpts]}"
    )

    cal_results_list: list[dict[str, Tensor]] = []
    test_results_list: list[dict[str, Tensor]] = []

    for i, ckpt in enumerate(all_ckpts):
        logger.info(f"\n  --- Seed {i+1}/{K}: {ckpt.parent.name}/{ckpt.name} ---")
        model_i = load_model_from_checkpoint(
            ckpt, F + C, C, config, device, weights=selected_weights[ckpt]
        )

        cal_res = run_rolling_inference(model_i, cal_dataset, edge_queen, edge_knn, device)
        test_res = run_rolling_inference(model_i, test_dataset, edge_queen, edge_knn, device)

        cal_results_list.append(cal_res)
        test_results_list.append(test_res)

        seed_crps = crps_zinb(
            test_res["y"].reshape(-1), test_res["pi"].reshape(-1),
            test_res["mu"].reshape(-1), test_res["r"].reshape(-1)
        ).mean().item()
        logger.info(f"    Individual CRPS: {seed_crps:.4f}")

        del model_i
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cal_results, test_results, ensemble_info = combine_ensemble_outputs(
        cal_results_list,
        test_results_list,
        target_key="y",
        category_wise=True,
        entropy_lambda=0.005,
        holdout_fraction=0.30,
        min_holdout_improvement=0.0025,
    )
    if K > 1:
        per_seed_crps = ensemble_info["per_seed_test_crps"]
        ensemble_crps = float(ensemble_info["equal_weight_test_crps"])
        emos_crps = float(ensemble_info["learned_weight_test_crps"])
        emos_info = {
            "weights": ensemble_info["emos_weights"],
            "category_weights": ensemble_info["emos_category_weights"],
            "improvement_pct": 100.0
            * (ensemble_crps - emos_crps)
            / max(ensemble_crps, 1e-12),
            "fallback_used": ensemble_info["emos_fallback_used"],
            "fallback_by_category": ensemble_info["emos_fallback_by_category"],
            "holdout_improvement_pct": ensemble_info[
                "emos_holdout_improvement_pct"
            ],
        }
        aleatoric = float(ensemble_info["aleatoric_uncertainty"])
        epistemic = float(ensemble_info["epistemic_uncertainty"])
        logger.info(
            "  Category-conditioned EMOS: equal CRPS %.4f -> %.4f; "
            "holdout fallback=%s",
            ensemble_crps,
            emos_crps,
            emos_info["fallback_used"],
        )

    logger.info(
        f"\n  Calibration: {cal_results['y'].shape[0]} windows x "
        f"{S} spatial x {C} categories = "
        f"{cal_results['y'].numel()} total observations"
    )
    logger.info(
        f"  Test: {test_results['y'].shape[0]} windows x "
        f"{S} spatial x {C} categories = "
        f"{test_results['y'].numel()} total observations"
    )

    # ─── Step 6: Fit calibrators + evaluate ───
    logger.info("\n[6/7] Fitting conformal calibrators and evaluating coverage...")

    # Flatten calibration data
    y_cal = cal_results["y"].reshape(-1)     # (N_cal * S * C,)
    pi_cal = cal_results["pi"].reshape(-1)
    mu_cal = cal_results["mu"].reshape(-1)
    r_cal = cal_results["r"].reshape(-1)

    # Flatten test data
    y_test = test_results["y"].reshape(-1)
    pi_test = test_results["pi"].reshape(-1)
    mu_test = test_results["mu"].reshape(-1)
    r_test = test_results["r"].reshape(-1)

    # Demographic groups (expanded across windows and categories)
    spatial_groups = load_demographic_groups(data_name, S)
    n_cal_windows = cal_results["y"].shape[0]
    n_test_windows = test_results["y"].shape[0]

    # Expand groups: (S,) → (N_windows, S, C) → (N_windows * S * C,)
    groups_cal = spatial_groups.unsqueeze(0).unsqueeze(-1).expand(
        n_cal_windows, S, C
    ).reshape(-1)
    groups_test = spatial_groups.unsqueeze(0).unsqueeze(-1).expand(
        n_test_windows, S, C
    ).reshape(-1)

    # ─── Grouping axes ───
    # Until the category axis was added, EVERY calibrator here conditioned on
    # the demographic axis alone. That leaves per-category coverage
    # uncontrolled, and it is not uniform: property undercovers (Chicago 0.859,
    # NYC 0.863) while drug overcovers (0.990 / 0.956).
    #
    # The cause is category-specific overconfidence in the ZINB head, not the
    # marginal-vs-conditional gap. On a perfectly specified model, split CP
    # reproduces drug OVER-coverage (0.958 — you cannot build a discrete
    # interval on a near-zero series that covers exactly 90%) but puts property
    # at 0.909, on target. Inflating predicted r on property alone reproduces
    # the observed pattern (property 0.771, violent 0.969, drug 0.996).
    # Calibration/test drift is ruled out: property FALLS between the windows
    # (Chicago -4.8%, NYC -8.1%), which would over-cover, not under-cover.
    #
    # A single additive threshold cannot repair this — the same count offset is
    # a large relative widening for drug and a negligible one for property —
    # which is why every demographic-grouped method inherits the same spread.
    # Conditioning Mondrian on category does repair it (property 0.734 → 0.902
    # on the synthetic analogue, coverage spread 0.266 → 0.051) because each
    # category then gets its own threshold. Grouping on the demographic axis
    # does NOT help (0.734 → 0.737): the axis has to match the axis the defect
    # lives on.
    cat_index = torch.arange(C, dtype=torch.long)
    cat_cal = cat_index.view(1, 1, C).expand(n_cal_windows, S, C).reshape(-1)
    cat_test = cat_index.view(1, 1, C).expand(n_test_windows, S, C).reshape(-1)

    # Interaction axis: demographic quartile x category, id = demo * C + cat.
    grouping_axes: dict[str, tuple[Tensor, Tensor]] = {
        "demographic": (groups_cal, groups_test),
        "category": (cat_cal, cat_test),
        "demo_x_category": (groups_cal * C + cat_cal, groups_test * C + cat_test),
    }

    # ─── Fit ALL calibration methods ───
    calibrator_configs: dict[str, Any] = {
        "split_cp": SplitConformalCalibrator(alpha=alpha),
        # Exact-guarantee counterpart to split_cp. The integer CQR score is
        # degenerate on this panel (>90% of scores <= 0, threshold pins to 0.0,
        # correction becomes the identity), so split_cp's reported coverage is
        # the RAW ZINB interval. This conformalizes the randomized PIT instead.
        # Seeded so the run is reproducible; the seed is recorded in metadata.
        "randomized_split_cp": RandomizedSplitConformalCalibrator(
            alpha=alpha, seed=0
        ),
        "weighted_cp": WeightedConformalCalibrator(alpha=alpha, decay_rate=0.05),
        "mondrian": MondrianConformalCalibrator(alpha=alpha, min_group_size=20),
        "mondrian_category": MondrianConformalCalibrator(alpha=alpha, min_group_size=20),
        "mondrian_demo_x_category": MondrianConformalCalibrator(
            alpha=alpha, min_group_size=20
        ),
        "equalized_coverage": EqualizedCoverageCalibrator(alpha=alpha, lambda_eq=1.0),
        "variance_scaled_split_cp": VarianceScaledConformalCalibrator(
            alpha=alpha
        ),
        "ecrc": ECRCCalibrator(
            alpha=alpha,
            delta=0.05,
            group_type="demographic",
            bound="exact_binomial",
            score_type="variance_scaled",
        ),
        # NOTE: adaptive_ecrc is deliberately NOT listed here. Fitted and then
        # asked to predict without any update() call, its per-group alpha_t
        # never moves off the initial base_alpha, which equals ECRC's
        # adjusted_alpha -- so it returned output byte-identical to ecrc while
        # being advertised as a distinct "adaptive" method. The genuine
        # temporal evaluation is adaptive_ecrc_rolling below, which calls
        # update() week by week.
    }

    # Which axis each method CONDITIONS ON. Reporting is always on the
    # demographic axis (below) so disparity stays comparable across methods.
    calibration_axis = {
        "mondrian": "demographic",
        "mondrian_category": "category",
        "mondrian_demo_x_category": "demo_x_category",
        "equalized_coverage": "demographic",
        "ecrc": "demographic",
    }

    all_coverage_results: dict[str, Any] = {}

    for method_name, calibrator in calibrator_configs.items():
        logger.info(f"\n  ─── {method_name.upper()} ───")

        # Fit — on whichever axis this method conditions on.
        fit_kwargs: dict[str, Any] = {}
        if method_name in calibration_axis:
            fit_kwargs["groups"] = grouping_axes[calibration_axis[method_name]][0]
        if method_name == "weighted_cp":
            # Decay by WEEK, not by flat array position.
            #
            # y_cal is (N_windows, S, C) flattened, so the default
            # time_deltas=arange(n,0,-1) treats each (unit, category) cell as its
            # own time step: deltas run to ~6000 and exp(-0.05*6000) underflows,
            # clamping 97% of points to min_weight. Effective sample size
            # collapsed to ~42 of 6006, and the "weighted" method silently
            # degenerated to uniform weights — which is exactly why weighted_cp
            # reported numbers bit-identical to split_cp. Weeks are the axis
            # along which non-stationarity actually occurs.
            n_cal_windows = cal_results["y"].shape[0]
            week_index = torch.arange(n_cal_windows, dtype=torch.float32)
            # Most recent calibration week has delta 0.
            week_delta = (n_cal_windows - 1) - week_index          # (N,)
            fit_kwargs["time_deltas"] = (
                week_delta.view(-1, 1, 1)
                .expand(n_cal_windows, S, C)
                .reshape(-1)
            )

        try:
            calibrator.fit(y_cal, pi_cal, mu_cal, r_cal, **fit_kwargs)
        except Exception as e:
            logger.error(f"  {method_name} fitting failed: {e}")
            all_coverage_results[method_name] = {"error": str(e)}
            continue

        # Predict
        predict_kwargs: dict[str, Any] = {}
        if method_name in ("mondrian", "mondrian_category",
                           "mondrian_demo_x_category", "ecrc"):
            predict_kwargs["groups"] = grouping_axes[calibration_axis[method_name]][1]

        try:
            intervals = calibrator.predict(pi_test, mu_test, r_test, **predict_kwargs)
        except Exception as e:
            logger.error(f"  {method_name} prediction failed: {e}")
            all_coverage_results[method_name] = {"error": str(e)}
            continue

        # Compute coverage metrics.
        # Reported on the DEMOGRAPHIC axis for every method regardless of what
        # it calibrated on, so `coverage_disparity` stays comparable — a
        # category-conditioned method reporting category disparity would look
        # artificially good against a demographic-conditioned one.
        coverage = compute_coverage_metrics(
            y_test, intervals["lower"], intervals["upper"],
            groups=groups_test, alpha=alpha,
        )
        coverage["calibration_axis"] = calibration_axis.get(method_name, "none")

        # Degeneracy flag: on a discrete panel the integer CQR threshold can pin
        # to exactly 0, making the correction the identity and the "calibrated"
        # coverage nothing but the raw ZINB interval. Record it per method so a
        # reader of the JSON can tell a real calibration from a no-op without
        # re-deriving the threshold.
        thr = getattr(calibrator, "_threshold", None)
        if thr is not None:
            coverage["calibration_threshold"] = float(thr)
            coverage["calibration_is_degenerate"] = bool(abs(float(thr)) < 1e-9)

        # For the randomized calibrator, also record the coverage that actually
        # carries the exact finite-sample guarantee. The integer interval still
        # overcovers (lattice ceiling), so BOTH numbers are reported: this one
        # is what the theory certifies, `marginal_coverage` is what an analyst
        # acts on. Quoting only one of them would misstate the result.
        if isinstance(calibrator, RandomizedSplitConformalCalibrator):
            coverage["pit_space_coverage"] = calibrator.coverage_in_pit_space(
                y_test, pi_test, mu_test, r_test
            )
            coverage["pit_band"] = [
                float(calibrator._lo_level),  # type: ignore[arg-type]
                float(calibrator._hi_level),  # type: ignore[arg-type]
            ]
            coverage["pit_randomization_seed"] = calibrator.seed

        # Per-category coverage
        per_cat: dict[str, dict[str, float]] = {}
        for c_idx, c_name in CATEGORY_NAMES.items():
            cat_mask = torch.zeros(y_test.shape[0], dtype=torch.bool)
            # Every C-th element belongs to category c_idx
            # Pattern: for shape (N_windows, S, C) flattened, category c is at indices c, C+c, 2C+c, ...
            cat_mask[c_idx::C] = True
            if cat_mask.sum() > 0:
                cat_covered = ((y_test[cat_mask] >= intervals["lower"][cat_mask]) &
                               (y_test[cat_mask] <= intervals["upper"][cat_mask])).float()
                cat_width = (intervals["upper"][cat_mask] - intervals["lower"][cat_mask]).float()
                per_cat[c_name] = {
                    "coverage": cat_covered.mean().item(),
                    "mean_width": cat_width.mean().item(),
                    "n_samples": int(cat_mask.sum().item()),
                }
        coverage["per_category"] = per_cat

        all_coverage_results[method_name] = coverage

        # Log summary
        logger.info(
            f"  Coverage: {coverage['marginal_coverage']:.4f} "
            f"(target: {1-alpha:.2f}) | "
            f"Width: {coverage['mean_width']:.2f} | "
            f"Disparity: {coverage.get('coverage_disparity', 0):.4f}"
        )
        if coverage.get("calibration_is_degenerate"):
            logger.warning(
                "  ^ threshold pinned to 0.0 — this coverage is the RAW ZINB "
                "interval, not a calibrated one (see randomized_split_cp)"
            )
        if "pit_space_coverage" in coverage:
            logger.info(
                f"  PIT-space coverage: {coverage['pit_space_coverage']:.4f} "
                f"(target {1-alpha:.2f}) — this is the exact guarantee; the "
                f"integer interval above is conservative by construction"
            )
        # Category spread is the diagnostic that motivated the category axis;
        # surface it in the log instead of leaving it buried in the JSON.
        if per_cat:
            cat_covs = {k: v["coverage"] for k, v in per_cat.items()}
            spread = max(cat_covs.values()) - min(cat_covs.values())
            logger.info(
                "  Per-category: "
                + " | ".join(f"{k} {v:.4f}" for k, v in cat_covs.items())
                + f"  (spread {spread:.4f})"
            )

    # ─── Rolling Adaptive ECRC (the REAL adaptive evaluation) ───
    # The loop above evaluates Adaptive ECRC on the full test set without
    # calling update() — making the "adaptive" claim non-functional.
    # Here we implement the CORRECT rolling evaluation that processes
    # the test set window-by-window, calling update() after each window.
    logger.info("\n  ─── ROLLING ADAPTIVE ECRC (week-by-week) ───")
    
    rolling_calibrator = AdaptiveTemporalECRCCalibrator(
        alpha=alpha, gamma=0.05, delta=0.05, group_type="demographic"
    )
    rolling_calibrator.fit(y_cal, pi_cal, mu_cal, r_cal, groups=groups_cal)
    
    # Process test data window-by-window
    # test_results["y"] shape: (N_windows, S, C)
    N_test_windows = test_results["y"].shape[0]
    rolling_coverages = []
    rolling_widths = []
    rolling_alpha_history: dict[int, list[float]] = {}  # Track alpha_t per group
    
    all_rolling_lower = []
    all_rolling_upper = []
    
    for w in range(N_test_windows):
        # Extract this window's data: (S, C) → flatten to (S*C,)
        y_w = test_results["y"][w].reshape(-1)
        pi_w = test_results["pi"][w].reshape(-1)
        mu_w = test_results["mu"][w].reshape(-1)
        r_w = test_results["r"][w].reshape(-1)
        groups_w = spatial_groups.unsqueeze(-1).expand(S, C).reshape(-1)
        
        # Predict intervals using CURRENT adaptive alpha_t values
        intervals_w = rolling_calibrator.predict(
            pi_w, mu_w, r_w, groups=groups_w
        )
        all_rolling_lower.append(intervals_w["lower"])
        all_rolling_upper.append(intervals_w["upper"])
        
        # Compute this window's coverage
        covered_w = ((y_w >= intervals_w["lower"]) & 
                     (y_w <= intervals_w["upper"])).float()
        width_w = (intervals_w["upper"] - intervals_w["lower"]).float()
        rolling_coverages.append(covered_w.mean().item())
        rolling_widths.append(width_w.mean().item())
        
        # Record alpha_t history before update
        for g_idx, a_t in rolling_calibrator._alpha_t.items():
            if g_idx not in rolling_alpha_history:
                rolling_alpha_history[g_idx] = []
            rolling_alpha_history[g_idx].append(a_t)
        
        # UPDATE: This is the critical step that makes it adaptive!
        rolling_calibrator.update(
            y_w, pi_w, mu_w, r_w, groups=groups_w
        )
    
    # Concatenate all rolling predictions for aggregate metrics
    rolling_lower_all = torch.cat(all_rolling_lower)
    rolling_upper_all = torch.cat(all_rolling_upper)
    
    rolling_coverage_overall = compute_coverage_metrics(
        y_test, rolling_lower_all, rolling_upper_all,
        groups=groups_test, alpha=alpha,
    )

    # Persist the same category breakdown as every non-rolling calibrator so
    # the publication table never has to substitute metrics from a different
    # method. Abstentions are excluded from coverage and width denominators,
    # matching compute_coverage_metrics() above.
    rolling_per_cat: dict[str, dict[str, float | int]] = {}
    rolling_issued = ~(
        torch.isnan(rolling_lower_all) | torch.isnan(rolling_upper_all)
    )
    rolling_covered = (
        (y_test >= rolling_lower_all) & (y_test <= rolling_upper_all)
    ).float()
    rolling_width = (rolling_upper_all - rolling_lower_all).float()
    for c_idx, c_name in CATEGORY_NAMES.items():
        cat_mask = torch.zeros(y_test.shape[0], dtype=torch.bool)
        cat_mask[c_idx::C] = True
        scored = cat_mask & rolling_issued
        n_samples = int(cat_mask.sum().item())
        n_issued = int(scored.sum().item())
        if n_samples == 0:
            continue
        rolling_per_cat[c_name] = {
            "coverage": (
                rolling_covered[scored].mean().item()
                if n_issued > 0
                else float("nan")
            ),
            "mean_width": (
                rolling_width[scored].mean().item()
                if n_issued > 0
                else float("nan")
            ),
            "n_samples": n_samples,
            "n_issued": n_issued,
            "abstention_rate": 1.0 - (n_issued / n_samples),
        }
    rolling_coverage_overall["per_category"] = rolling_per_cat
    
    # Add rolling-specific metrics
    rolling_coverage_overall["per_window_coverage"] = rolling_coverages
    rolling_coverage_overall["per_window_width"] = rolling_widths
    rolling_coverage_overall["alpha_convergence"] = {
        str(g): alphas for g, alphas in rolling_alpha_history.items()
    }
    rolling_coverage_overall["final_alpha_t"] = {
        str(g): a for g, a in rolling_calibrator._alpha_t.items()
    }
    
    all_coverage_results["adaptive_ecrc_rolling"] = rolling_coverage_overall
    
    logger.info(
        f"  Rolling Adaptive ECRC Coverage: {rolling_coverage_overall['marginal_coverage']:.4f} "
        f"(target: {1-alpha:.2f}) | "
        f"Width: {rolling_coverage_overall['mean_width']:.2f} | "
        f"Disparity: {rolling_coverage_overall.get('coverage_disparity', 0):.4f}"
    )
    logger.info(
        f"  Coverage convergence: window 1-5 avg={np.mean(rolling_coverages[:5]):.4f}, "
        f"last 5 avg={np.mean(rolling_coverages[-5:]):.4f}"
    )

    # ─── Compute baseline CRPS and CRPSS ───
    logger.info("\n  ─── CRPS SKILL SCORE ───")
    baselines = compute_baseline_crps(counts)
    ha_crps = baselines["ha_crps"]                 # rolling — the honest one
    ha_crps_frozen = baselines["ha_crps_frozen"]   # reported for transparency
    sn_crps = baselines["seasonal_naive_crps"]
    model_crps = crps_zinb(y_test, pi_test, mu_test, r_test).mean().item()

    # CRPSS against the ROLLING historical average. On this panel the rolling
    # mean is the STRONGEST naive baseline, not the weakest -- it beats
    # seasonal-naive by ~33% because week-to-week level is far more predictive
    # than same-week-last-year. Gating only on seasonal-naive would let a model
    # that loses to a trailing mean report a passing skill score.
    crpss_ha = 1.0 - (model_crps / ha_crps) if ha_crps > 0 else 0.0
    crpss_ha_frozen = (
        1.0 - (model_crps / ha_crps_frozen) if ha_crps_frozen > 0 else 0.0
    )
    # CRPSS against Seasonal Naive
    crpss_sn = 1.0 - (model_crps / sn_crps) if sn_crps > 0 else 0.0

    logger.info(f"  Baseline CRPS (HA, rolling {test_dataset.window_size}w): {ha_crps:.4f}  <- honest HA")
    logger.info(f"  Baseline CRPS (HA, frozen train mean):  {ha_crps_frozen:.4f}  (transparency only)")
    logger.info(f"  Baseline CRPS (seasonal naive):         {sn_crps:.4f}")
    logger.info(f"  Model CRPS:                             {model_crps:.4f}")
    logger.info(f"  CRPSS vs HA (rolling):                  {crpss_ha:+.4f}")
    logger.info(f"  CRPSS vs HA (frozen):                   {crpss_ha_frozen:+.4f}")
    logger.info(f"  CRPSS vs Seasonal Naive:                {crpss_sn:+.4f}")

    # Retain the conservative minimum skill score as a descriptive metric.
    # The forecasting claim gate itself is evaluated below, after the DM and
    # block-bootstrap evidence against the rolling HA has been computed.
    crpss_primary = min(crpss_ha, crpss_sn)
    logger.info(
        f"  Conservative min CRPSS:                {crpss_primary:+.4f} "
        f"(binding: {'rolling HA' if crpss_ha < crpss_sn else 'seasonal naive'})"
    )

    # ─── Post-hoc Recalibration ───
    logger.info("\n  ─── POST-HOC RECALIBRATION ───")
    
    (pi_recal, mu_recal, r_recal), recal_metrics = recalibrate_and_evaluate(
        y_cal, pi_cal, mu_cal, r_cal,
        y_test, pi_test, mu_test, r_test,
        method="affine", lr=0.01, max_iter=500,
    )
    logger.info(
        f"  CRPS improvement: {recal_metrics['test_crps_before']:.4f} → "
        f"{recal_metrics['test_crps_after']:.4f} "
        f"({recal_metrics['test_improvement_pct']:.2f}% improvement)"
    )
    logger.info(f"  Learned params: {recal_metrics['learned_params']}")
    
    # Recompute CRPSS with recalibrated predictions
    model_crps_recal = recal_metrics['test_crps_after']
    crpss_ha_recal = 1.0 - (model_crps_recal / ha_crps) if ha_crps > 0 else 0.0
    crpss_sn_recal = 1.0 - (model_crps_recal / sn_crps) if sn_crps > 0 else 0.0
    logger.info(f"  CRPSS vs HA (recalibrated):   {crpss_ha_recal:.4f}")
    logger.info(f"  CRPSS vs SN (recalibrated):   {crpss_sn_recal:.4f}")

    # ─── Per-Category CRPSS Breakdown ───
    logger.info("\n  ─── PER-CATEGORY CRPSS ───")
    per_cat_crpss = {}
    # Rolling HA per (week, unit, category), aligned to the model's own target
    # weeks. Shared across categories so it is computed once and sliced, and so
    # it provably matches the aggregate rolling HA above.
    rolling_ha_3d = torch.stack(
        [
            counts[:, t - test_dataset.window_size : t, :].float().mean(dim=1)  # (S, C)
            for t in test_dataset.valid_targets[:test_results["y"].shape[0]]
        ],
        dim=0,
    )  # (N_windows, S, C)
    for c_idx in range(C):
        cat_name = CATEGORY_NAMES.get(c_idx, f"cat_{c_idx}")
        # Extract per-category data
        y_c = test_results["y"][:, :, c_idx].reshape(-1)
        pi_c = test_results["pi"][:, :, c_idx].reshape(-1)
        mu_c = test_results["mu"][:, :, c_idx].reshape(-1)
        r_c = test_results["r"][:, :, c_idx].reshape(-1)
        model_crps_c = crps_zinb(y_c, pi_c, mu_c, r_c).mean().item()
        # Baseline: rolling HA per category (same definition as the headline)
        ha_mu_c = rolling_ha_3d[:, :, c_idx].reshape(-1).clamp(min=0.01)
        ha_crps_c = crps_zinb(
            y_c, torch.zeros_like(y_c), ha_mu_c, torch.full_like(y_c, 1000.0)
        ).mean().item()
        crpss_c = 1.0 - (model_crps_c / ha_crps_c) if ha_crps_c > 0 else 0.0
        per_cat_crpss[cat_name] = {
            "model_crps": model_crps_c, "ha_crps": ha_crps_c, "crpss": crpss_c
        }
        logger.info(f"  {cat_name:10s}: CRPS={model_crps_c:.4f}, HA={ha_crps_c:.4f}, CRPSS={crpss_c:+.4f}")

    # ─── PIT Histogram (Calibration Diagnostic) ───
    logger.info("\n  ─── PIT HISTOGRAM ───")
    pit = pit_values(y_test, pi_test, mu_test, r_test).numpy()
    n_bins = 10
    pit_hist, pit_edges = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    pit_freq = pit_hist / pit_hist.sum()
    uniform_freq = 1.0 / n_bins
    pit_deviation = np.abs(pit_freq - uniform_freq).max()
    logger.info(f"  PIT bins: {pit_freq.tolist()}")
    logger.info(f"  Uniform reference: {uniform_freq:.4f}")
    logger.info(f"  Max deviation from uniform: {pit_deviation:.4f}")
    # Chi-squared test for uniformity.
    #
    # Interpret with care and report the EFFECT SIZE, not just the p-value. The
    # test assumes i.i.d. draws, but PIT values here are one per
    # (week, unit, category) cell and are strongly correlated in space and time,
    # so the effective sample size is far below n. At n>12k even a ~1% deviation
    # from uniform returns p<1e-10. Reporting "miscalibrated" off that p-value
    # alone overstates the problem: a max bin deviation of ~0.015 against a 0.10
    # reference is a well-calibrated forecast by any practical standard.
    from scipy import stats as sp_stats
    chi2_stat, chi2_p = sp_stats.chisquare(pit_hist)
    n_pit = int(np.sum(pit_hist))
    # Cramer's V for a 1-D goodness-of-fit table: sqrt(chi2 / (n * df)).
    n_bins = len(pit_hist)
    cramers_v = float(np.sqrt(chi2_stat / (n_pit * (n_bins - 1)))) if n_pit else 0.0
    logger.info(f"  Chi-squared test: stat={chi2_stat:.2f}, p={chi2_p:.4f} (n={n_pit})")
    logger.info(f"  Effect size (Cramer's V): {cramers_v:.4f}  [<0.1 = negligible]")
    if chi2_p > 0.05:
        logger.info("  ✅ PIT histogram is consistent with uniform (well-calibrated)")
    elif pit_deviation < 0.02 and cramers_v < 0.1:
        logger.info(
            "  ✅ PIT deviation is negligible in magnitude "
            f"(max {pit_deviation:.4f} vs {uniform_freq:.2f} reference). "
            "Chi-squared rejects only because n is large and PIT cells are "
            "spatiotemporally correlated — not evidence of practical miscalibration."
        )
    else:
        logger.info(
            f"  ⚠️ PIT histogram deviates from uniform "
            f"(max dev {pit_deviation:.4f}, V={cramers_v:.4f})"
        )

    # ─── Compute point forecast metrics on test set ───
    test_metrics = compute_all_metrics(y_test, pi_test, mu_test, r_test)
    logger.info(f"  Test MAE: {test_metrics['mae']:.4f}")
    logger.info(f"  Test RMSE: {test_metrics['rmse']:.4f}")
    logger.info(f"  Test Brier: {test_metrics['brier_zero']:.4f}")

    # ─── Feedback Loop Index + Bias Amplification Score ───
    logger.info("\n  ─── FEEDBACK LOOP ANALYSIS ───")
    
    # Point predictions: E[Y] = (1-pi)*mu
    y_pred_point = ((1.0 - pi_test) * mu_test)
    
    # Historical mean per spatial unit (training period) as trend baseline
    train_means = counts[:, :208, :].float().mean(dim=1)  # (S, C)
    # Expand to match test shape: (S, C) → (N_windows, S, C) → (N*S*C,)
    historical_trend = train_means.unsqueeze(0).expand(
        n_test_windows, S, C
    ).reshape(-1)
    
    feedback_metrics = compute_all_feedback_metrics(
        y_pred=y_pred_point,
        y_true=y_test,
        groups=groups_test,
        counts_historical=historical_trend,
    )
    
    asc_agg = feedback_metrics["asc"]["aggregate"]
    bas_agg = feedback_metrics["bas"]["aggregate"]
    logger.info(f"  ASC — mean: {asc_agg['mean_asc']:.4f}, "
                f"max: {asc_agg['max_asc']:.4f}, min: {asc_agg['min_asc']:.4f}")
    logger.info(f"  BAS — mean |BAS|: {bas_agg['mean_abs_bas']:.4f}, "
                f"max |BAS|: {bas_agg['max_abs_bas']:.4f}")
    
    # DAD metric
    dad_result = feedback_metrics.get("dad", {})
    if dad_result:
        dad_agg = dad_result.get("aggregate", {})
        logger.info(f"  DAD — mean disparity: {dad_agg.get('mean_alloc_disparity', float('nan')):.4f}, "
                    f"max disparity: {dad_agg.get('max_alloc_disparity', float('nan')):.4f}")
    
    if abs(asc_agg['mean_asc']) < 0.1:
        logger.info("  ✅ Model predictions are trend-neutral (low feedback loop risk)")
    elif asc_agg['mean_asc'] > 0.1:
        logger.info("  ⚠️ Model predictions may amplify historical trends")
    else:
        logger.info("  ✅ Model predictions counteract historical trends (corrective)")

    # ─── CRPS Decomposition (Hersbach 2000) ───
    logger.info("\n  ─── CRPS DECOMPOSITION (Hersbach 2000) ───")
    crps_decomp = crps_decomposition(y_test, pi_test, mu_test, r_test)

    # ─── Statistical Significance (Diebold-Mariano + Block Bootstrap) ───
    logger.info("\n  ─── STATISTICAL SIGNIFICANCE ───")
    # Compute per-timestep CRPS for model and baseline
    crps_per_obs_model = crps_zinb(y_test, pi_test, mu_test, r_test)
    
    # Historical Average baseline CRPS per observation.
    # Must use the SAME rolling definition as compute_baseline_crps, or the
    # DM test certifies superiority over the frozen straw man while the skill
    # score is quoted against the rolling one -- two different baselines
    # reported side by side as if they were the same comparison.
    #
    # Week indices come from the dataset itself rather than hardcoded splits:
    # `valid_targets` is precisely the sequence of target weeks that produced
    # y_test, in order, so the baseline is aligned with the model's
    # predictions by construction instead of by an assumption about 260.
    ha_pred_expanded = torch.stack(
        [
            counts[:, t - test_dataset.window_size : t, :].float().mean(dim=1)  # (S, C)
            for t in test_dataset.valid_targets[:n_test_windows]
        ],
        dim=0,
    ).reshape(-1)  # (n_windows*S*C,)
    # Poisson approx CRPS for HA: pi=0, mu=ha_pred, r=1000 (NB->Poisson)
    crps_per_obs_ha = crps_zinb(
        y_test,
        torch.zeros_like(y_test),
        ha_pred_expanded.clamp(min=0.01),
        torch.full_like(y_test, 1000.0),
    )
    
    # Aggregate into per-window CRPS for DM test (need temporal sequence)
    obs_per_window = S * C
    n_windows_actual = y_test.shape[0] // obs_per_window
    if n_windows_actual >= 10:
        crps_model_windows = crps_per_obs_model.reshape(
            n_windows_actual, obs_per_window
        ).mean(dim=1)
        crps_ha_windows = crps_per_obs_ha.reshape(
            n_windows_actual, obs_per_window
        ).mean(dim=1)
        
        significance_results = compare_forecasts(
            crps_model_windows, crps_ha_windows,
            baseline_name="Historical Average",
        )
        logger.info(f"  {significance_results['summary']}")
    else:
        significance_results = {"note": "Insufficient windows for DM test (need >= 10)"}
        logger.info("  ⚠️ Too few test windows for DM test")

    # ─── Step 7: Compile and save results ───
    forecasting_gate = assess_forecasting_gate(crpss_ha, significance_results)
    gate_payload = forecasting_gate.as_dict()
    dm_p_text = (
        f"{forecasting_gate.dm_p_value:.6f}"
        if forecasting_gate.dm_p_value is not None
        else "unavailable"
    )
    bootstrap_p_text = (
        f"{forecasting_gate.bootstrap_p_value:.6f}"
        if forecasting_gate.bootstrap_p_value is not None
        else "unavailable"
    )
    dm_stat_text = (
        f"{forecasting_gate.dm_stat:.6f}"
        if forecasting_gate.dm_stat is not None
        else "unavailable"
    )
    dm_ci_text = (
        f"[{forecasting_gate.dm_ci[0]:.6f}, "
        f"{forecasting_gate.dm_ci[1]:.6f}]"
        if forecasting_gate.dm_ci is not None
        else "unavailable"
    )
    bootstrap_ci_text = (
        f"[{forecasting_gate.bootstrap_ci[0]:.6f}, "
        f"{forecasting_gate.bootstrap_ci[1]:.6f}]"
        if forecasting_gate.bootstrap_ci is not None
        else "unavailable"
    )
    logger.info("\n  --- FORECASTING CLAIM GATE ---")
    logger.info(f"  Rule: {gate_payload['rule']}")
    logger.info(f"  CRPSS vs rolling HA: {crpss_ha:+.6f}")
    logger.info(
        f"  Diebold-Mariano: statistic={dm_stat_text}, p={dm_p_text}, "
        f"95% CI={dm_ci_text}"
    )
    logger.info(
        f"  Block bootstrap: p={bootstrap_p_text}, "
        f"95% CI={bootstrap_ci_text}"
    )
    if forecasting_gate.passed:
        logger.info("  GATE MET: positive CRPSS vs HA with p < 0.05")
    else:
        logger.warning(
            "  GATE NOT MET: forecasting superiority requires positive CRPSS "
            "vs rolling HA and DM or block-bootstrap p < 0.05"
        )

    calibrator_selection = select_best_calibrator(
        all_coverage_results,
        alpha=alpha,
        max_disparity=COVERAGE_DISPARITY_THRESHOLD,
        max_abstention=DEFAULT_MAX_ABSTENTION,
    )

    logger.info("\n[7/7] Compiling results and saving to disk...")

    # Dataset hash for reproducibility
    panel_hash = hashlib.md5(
        counts.numpy().tobytes()[:10000]  # First 10KB for speed
    ).hexdigest()[:12]

    results = {
        "metadata": {
            "dataset": data_name,
            "checkpoint": str(all_ckpts[0].parent.parent) if K > 1 else str(all_ckpts[0]),
            "checkpoints": [str(checkpoint) for checkpoint in all_ckpts],
            "num_ensemble_seeds": K,
            "weights_source": (
                weights
                if weights != "auto"
                else "per_seed_validation"
            ),
            "weights_source_per_seed": {
                checkpoint.parent.name: selected_weights[checkpoint]
                for checkpoint in all_ckpts
            },
            "weight_selection_val_crps_per_seed": {
                checkpoint.parent.name: {
                    name: round(score, 6)
                    for name, score in val_scores_per_checkpoint[checkpoint].items()
                }
                for checkpoint in all_ckpts
            },
            "alpha": alpha,
            "timestamp": datetime.now().isoformat(),
            "panel_hash": panel_hash,
            "spatial_units": S,
            "time_steps": T,
            "categories": C,
            "cal_set_size": len(cal_dataset),
            "test_set_size": len(test_dataset),
            "total_cal_observations": int(y_cal.numel()),
            "total_test_observations": int(y_test.numel()),
        },
        "point_forecast_metrics": test_metrics,
        "skill_scores": {
            "baseline_crps_ha": ha_crps,
            "baseline_crps_ha_frozen": ha_crps_frozen,
            "crpss_vs_ha_frozen": crpss_ha_frozen,
            "baseline_crps_seasonal_naive": sn_crps,
            "model_crps": model_crps,
            "crpss_vs_ha": crpss_ha,
            "crpss_vs_seasonal_naive": crpss_sn,
            # Retained as a conservative descriptive statistic across the naive
            # family. The inferential gate below is specifically defined against
            # the rolling HA, whose aligned loss series supports DM/bootstrap.
            "crpss": crpss_primary,
            "crpss_binding_baseline": (
                "ha_rolling" if crpss_ha < crpss_sn else "seasonal_naive"
            ),
            "forecasting_gate_passed": forecasting_gate.passed,
            # Compatibility alias for downstream consumers of older JSON.
            "crpss_passes_threshold": forecasting_gate.passed,
        },
        "per_category_crpss": per_cat_crpss,
        "calibration_diagnostics": {
            "pit_histogram": pit_freq.tolist(),
            "pit_bin_edges": pit_edges.tolist(),
            "pit_max_deviation": float(pit_deviation),
            "pit_chi2_stat": float(chi2_stat),
            "pit_chi2_pvalue": float(chi2_p),
            "pit_is_uniform": bool(chi2_p > 0.05),
        },
        "coverage_results": all_coverage_results,
        "recalibration": {
            **recal_metrics,
            "crpss_ha_recalibrated": crpss_ha_recal,
            "crpss_sn_recalibrated": crpss_sn_recal,
        },
        "feedback_loop_analysis": feedback_metrics,
        "crps_decomposition": crps_decomp,
        "statistical_significance": significance_results,
        "forecasting_gate": gate_payload,
        "calibrator_selection": calibrator_selection.as_dict(),
    }
    
    # Add ensemble-specific results if applicable
    if K > 1:
        results["ensemble"] = {
            **ensemble_info,
            "mean_seed_crps": float(np.mean(per_seed_crps)),
            "ensemble_improvement": float(
                1.0 - emos_crps / max(np.mean(per_seed_crps), 1e-12)
            ),
            "emos_improvement_pct": emos_info["improvement_pct"],
        }

    # Save results
    output_dir = PROJECT_ROOT / "outputs" / "conformal_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"{data_name}_conformal_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"  Results saved: {results_path}")

    # Save calibration objects
    calibration_path = output_dir / f"{data_name}_calibrators.pt"
    torch.save(
        {
            "calibrators": {
                name: cal for name, cal in calibrator_configs.items()
            },
            "metadata": results["metadata"],
        },
        calibration_path,
    )
    logger.info(f"  Calibrators saved: {calibration_path}")

    # ─── Generate Audit Report ───
    _generate_audit_report(results, output_dir / f"{data_name}_audit_report.md")

    # ─── Exit Criteria Check ───
    logger.info("\n" + "=" * 70)
    logger.info("  EXIT CRITERIA CHECK")
    logger.info("=" * 70)

    # Render the centralized decision made before serialization. Console, JSON,
    # audit report, and tests therefore use one policy rather than copied rules.
    candidates = {
        m: c for m, c in all_coverage_results.items()
        if isinstance(c, dict) and "marginal_coverage" in c
    }
    best_method = calibrator_selection.selected_method
    logger.info(f"  Selection rule: {calibrator_selection.selection_rule}")
    if calibrator_selection.fallback_used:
        logger.warning(
            "  FALLBACK SELECTION: no calibrator satisfied coverage >= "
            f"{calibrator_selection.coverage_floor:.4f} and demographic "
            f"disparity <= {calibrator_selection.max_disparity:.4f}. "
            f"Selected {best_method!r} by minimum width among comparable methods."
        )
    elif best_method is None:
        logger.error(
            "  No calibrator could be selected: every candidate was explicitly "
            "ineligible, abstained too heavily, or had non-finite coverage/width."
        )

    if candidates:
        logger.info(
            f"  {'method':<24} {'coverage':>9} {'width':>8} {'disparity':>10} "
            f"{'abstain':>8}  status"
        )
        for m in sorted(
            candidates,
            key=lambda k: candidates[k].get("mean_width") or float("inf"),
        ):
            c = candidates[m]
            cov_raw = c.get("marginal_coverage")
            width_raw = c.get("mean_width")
            disparity_raw = c.get("coverage_disparity")
            abstention_raw = c.get("abstention_rate", 0.0)
            cov_m = float(cov_raw) if cov_raw is not None else float("nan")
            width_m = float(width_raw) if width_raw is not None else float("nan")
            disparity_m = (
                float(disparity_raw)
                if disparity_raw is not None
                else float("nan")
            )
            abst_m = (
                float(abstention_raw)
                if abstention_raw is not None
                else float("nan")
            )
            if m in calibrator_selection.eligible_methods:
                status = "ELIGIBLE"
            else:
                reasons = calibrator_selection.rejected_reasons.get(m, ())
                status = "; ".join(reasons) if reasons else "not comparable"
            logger.info(
                f"  {m:<24} {cov_m:>9.4f} {width_m:>8.2f} "
                f"{disparity_m:>10.4f} {abst_m:>8.2%}  {status}"
                + ("  <-- selected" if m == best_method else "")
            )

    if best_method:
        best_results = all_coverage_results[best_method]
        logger.info(f"  Best calibrator: {best_method}")
        logger.info(f"  Marginal coverage: {best_results['marginal_coverage']:.4f}")
        disparity = best_results.get("coverage_disparity", 0)
        logger.info(f"  Coverage disparity: {disparity:.4f}")
        logger.info(f"  CRPSS vs HA: {crpss_ha:.4f}")
        logger.info(f"  CRPSS vs Seasonal Naive: {crpss_sn:.4f}")

        # Check all pre-registered exit criteria against the same policy values.
        passed_all = not calibrator_selection.fallback_used
        if calibrator_selection.fallback_used:
            logger.warning(
                "  CALIBRATOR POLICY FAILED: selected method is a fallback, "
                "not a fully eligible headline calibrator."
            )

        if best_results["marginal_coverage"] < calibrator_selection.coverage_floor:
            logger.warning(
                "  COVERAGE BELOW FLOOR: "
                f"{best_results['marginal_coverage']:.4f} < "
                f"{calibrator_selection.coverage_floor:.4f}"
            )
            passed_all = False

        if disparity > calibrator_selection.max_disparity:
            logger.warning(
                "  COVERAGE DISPARITY EXCEEDS THRESHOLD: "
                f"{disparity:.4f} > {calibrator_selection.max_disparity:.4f}"
            )
            passed_all = False

        abstention = best_results.get("abstention_rate", 0.0)
        if abstention > calibrator_selection.max_abstention:
            logger.warning(
                "  ABSTENTION EXCEEDS THRESHOLD: "
                f"{abstention:.2%} > {calibrator_selection.max_abstention:.2%}"
            )
            passed_all = False

        if not forecasting_gate.passed:
            logger.warning(
                "  FORECASTING GATE FAILED: CRPSS vs rolling HA must be "
                "positive with DM or block-bootstrap p < 0.05."
            )
            passed_all = False

        # Overcoverage is not a pass. Intervals far wider than needed are
        # useless operationally and a reviewer will read >97% at alpha=0.1 as
        # a calibration failure, not a strength.
        if best_results["marginal_coverage"] > (1.0 - alpha) + 0.03:
            logger.warning(
                "  SUBSTANTIAL OVERCOVERAGE: "
                f"{best_results['marginal_coverage']:.4f} >> {1 - alpha:.2f} target "
                f"(width {best_results.get('mean_width', float('nan')):.2f}). "
                f"Intervals are wider than necessary; check group sizes feeding "
                f"the ECRC Hoeffding slack."
            )
            passed_all = False

        if passed_all:
            logger.info("  ALL EXIT CRITERIA PASSED")
        else:
            logger.info("  SOME EXIT CRITERIA FAILED (see warnings above)")

    elapsed = time.time() - t_start
    logger.info(f"\n  Pipeline complete in {elapsed:.1f}s")

    return results


# ───────────────────────────────────────────────────────────────────
# Audit Report Generation
# ───────────────────────────────────────────────────────────────────
def _format_optional_number(value: Any, digits: int = 6) -> str:
    """Format a finite audit value without raising on missing evidence."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "unavailable"
    if not math.isfinite(number):
        return "unavailable"
    return f"{number:.{digits}f}"


def _format_audit_interval(value: Any) -> str:
    """Format a two-sided confidence interval for the Markdown audit."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return "unavailable"
    lower = _format_optional_number(value[0])
    upper = _format_optional_number(value[1])
    if "unavailable" in {lower, upper}:
        return "unavailable"
    return f"[{lower}, {upper}]"


def _generate_audit_report(results: dict[str, Any], output_path: Path) -> None:
    """Generate a comprehensive markdown audit report.

    Args:
        results: Complete evaluation results dictionary.
        output_path: Path to write the markdown report.
    """
    meta = results["metadata"]
    metrics = results["point_forecast_metrics"]
    skill = results["skill_scores"]
    coverage = results["coverage_results"]
    significance = results.get("statistical_significance", {})
    gate = results.get("forecasting_gate")
    if not isinstance(gate, dict):
        gate = assess_forecasting_gate(
            skill.get("crpss_vs_ha"), significance
        ).as_dict()
    selection = results.get("calibrator_selection")
    if not isinstance(selection, dict):
        selection = select_best_calibrator(
            coverage,
            alpha=float(meta.get("alpha", ALPHA_DEFAULT)),
            max_disparity=COVERAGE_DISPARITY_THRESHOLD,
            max_abstention=DEFAULT_MAX_ABSTENTION,
        ).as_dict()

    gate_passed = bool(gate.get("passed", False))
    gate_status = "PASS" if gate_passed else "FAIL"
    selection_method = selection.get("selected_method")
    selection_rule = selection.get("selection_rule", "unavailable")

    lines = [
        f"# CIVIC-SAFE Conformal Prediction Audit Report",
        f"",
        f"**Dataset:** {meta['dataset']}  ",
        f"**Timestamp:** {meta['timestamp']}  ",
        f"**Alpha (miscoverage):** {meta['alpha']}  ",
        f"**Checkpoint:** `{Path(meta['checkpoint']).name}`  ",
        f"**Panel hash:** `{meta['panel_hash']}`  ",
        f"",
        f"## Point Forecast Metrics (Test Set — 2023)",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| CRPS | {metrics['crps']:.4f} |",
        f"| MAE | {metrics['mae']:.4f} |",
        f"| RMSE | {metrics['rmse']:.4f} |",
        f"| Brier (zero-inflation) | {metrics['brier_zero']:.4f} |",
        f"",
        f"## CRPS Skill Score",
        f"",
        f"| Component | Value |",
        f"|-----------|-------|",
        f"| Baseline CRPS (Historical Average, rolling) | {skill.get('baseline_crps_ha', skill.get('baseline_crps', 'N/A')):.4f} |",
        f"| Baseline CRPS (Historical Average, frozen) | {skill.get('baseline_crps_ha_frozen', float('nan')):.4f} |",
        f"| Baseline CRPS (Seasonal Naive) | {skill.get('baseline_crps_seasonal_naive', 'N/A')} |",
        f"| Model CRPS | {skill['model_crps']:.4f} |",
        f"| CRPSS vs HA (rolling) | {skill.get('crpss_vs_ha', skill.get('crpss', 0)):+.4f} |",
        f"| CRPSS vs Seasonal Naive | {skill.get('crpss_vs_seasonal_naive', skill.get('crpss', 0)):+.4f} |",
        f"| **CRPSS (min over naive family)** | **{skill.get('crpss', 0):+.4f}** |",
        f"| Binding baseline | {skill.get('crpss_binding_baseline', 'N/A')} |",
        f"| Forecasting claim gate | **{gate_status}** |",
        f"",
        f"## Forecasting Claim Gate",
        f"",
        f"| Evidence | Value |",
        f"|----------|-------|",
        f"| Rule | {gate.get('rule', 'unavailable')} |",
        f"| CRPSS vs rolling HA | {_format_optional_number(gate.get('crpss_ha'))} |",
        f"| DM statistic | {_format_optional_number(gate.get('dm_stat'))} |",
        f"| DM p-value | {_format_optional_number(gate.get('dm_p_value'))} |",
        f"| DM 95% CI | {_format_audit_interval(gate.get('dm_ci'))} |",
        f"| Block-bootstrap p-value | {_format_optional_number(gate.get('bootstrap_p_value'))} |",
        f"| Block-bootstrap 95% CI | {_format_audit_interval(gate.get('bootstrap_ci'))} |",
        f"| **Decision** | **{gate_status}** |",
        f"",
        f"## Calibrator Selection",
        f"",
        f"**Rule:** {selection_rule}  ",
        f"**Selected method:** `{selection_method or 'none'}`  ",
        f"**Fallback used:** {bool(selection.get('fallback_used', False))}  ",
        f"",
        f"## Coverage Results by Calibration Method",
        f"",
    ]

    # Table header
    lines.append(
        "| Method | Marginal Coverage | Target | Mean Width | "
        "Demographic Disparity | Abstention | Policy Status |"
    )
    lines.append(
        "|--------|:-----------------:|:------:|:----------:|"
        ":---------------------:|:----------:|---------------|"
    )

    for method, cov in coverage.items():
        if isinstance(cov, dict) and "marginal_coverage" in cov:
            mc = cov["marginal_coverage"]
            target = cov["target_coverage"]
            width = cov["mean_width"]
            disp = cov.get("coverage_disparity", 0)
            abstention = cov.get("abstention_rate", 0.0)
            eligible_methods = selection.get("eligible_methods", [])
            reasons = selection.get("rejected_reasons", {}).get(method, [])
            if method in eligible_methods:
                policy_status = "eligible"
            elif reasons:
                policy_status = "; ".join(str(reason) for reason in reasons)
            else:
                policy_status = "not comparable"
            if method == selection_method:
                selected_label = "selected fallback" if selection.get("fallback_used") else "selected"
                policy_status = f"{selected_label}; {policy_status}"
            lines.append(
                f"| {method} | {mc:.4f} | {target:.2f} | {width:.2f} | "
                f"{disp:.4f} | {abstention:.2%} | {policy_status} |"
            )
        elif isinstance(cov, dict) and "error" in cov:
            lines.append(f"| {method} | ERROR | - | - | - | - | error |")

    lines.append("")

    # The report consumes the persisted centralized decision; it never reruns a
    # second, potentially divergent selection rule.
    best_method = selection_method if isinstance(selection_method, str) else None
    best_cov_val = (
        coverage[best_method]["marginal_coverage"]
        if best_method and best_method in coverage
        else 0.0
    )

    if best_method and "per_category" in coverage[best_method]:
        lines.append(f"### Per-Category Coverage ({best_method})")
        lines.append("")
        lines.append(f"| Category | Coverage | Width | N |")
        lines.append(f"|----------|:--------:|:-----:|--:|")
        for cat_name, cat_data in coverage[best_method]["per_category"].items():
            lines.append(
                f"| {cat_name} | {cat_data['coverage']:.4f} | "
                f"{cat_data['mean_width']:.2f} | {cat_data['n_samples']} |"
            )
        lines.append("")

    if best_method and "per_group" in coverage[best_method]:
        lines.append(f"### Per-Demographic-Quartile Coverage ({best_method})")
        lines.append("")
        lines.append(f"| Group | Coverage | Width | N |")
        lines.append(f"|-------|:--------:|:-----:|--:|")
        for group_name, group_data in coverage[best_method]["per_group"].items():
            lines.append(
                f"| {group_name} | {group_data['coverage']:.4f} | "
                f"{group_data['mean_width']:.2f} | {group_data['n_samples']} |"
            )
        lines.append("")

    if best_method and best_method in coverage:
        selected_cov = coverage[best_method]
        selection_sentence = (
            f"The selected calibrator ({best_method}) achieves "
            f"{best_cov_val:.1%} marginal coverage with mean prediction "
            f"interval width {selected_cov['mean_width']:.2f} counts and a "
            "maximum cross-group coverage disparity of "
            f"{selected_cov.get('coverage_disparity', 0):.4f}."
        )
    else:
        selection_sentence = (
            "No calibrator was selected because every method was fundamentally "
            "incomparable under the pre-specified policy."
        )

    # Paper-ready paragraph
    lines.extend([
        f"## Methods Paragraph (Paper-Ready)",
        f"",
        f"We apply Conformalized Quantile Regression (CQR; Romano et al., 2019) ",
        f"to the ZINB predictive distribution, computing non-conformity scores ",
        f"$s_i = \\max(\\hat{{q}}_{{\\alpha/2}}(X_i) - Y_i, Y_i - \\hat{{q}}_{{1-\\alpha/2}}(X_i))$ ",
        f"on a held-out calibration set (2022 H2, {meta['cal_set_size']} windows, ",
        f"{meta['total_cal_observations']} observations). The calibration threshold ",
        f"$\\hat{{q}}$ is chosen as the $\\lceil (1-\\alpha)(1+1/n) \\rceil$-th empirical ",
        f"quantile of the scores, guaranteeing finite-sample marginal coverage ",
        f"$P(Y \\in [L, U]) \\geq 1-\\alpha$ under exchangeability. To correct for ",
        f"temporal non-exchangeability, we additionally implement Adaptive Conformal ",
        f"Inference (ACI; Gibbs & Candès, 2021) with per-demographic-quartile tracking, ",
        f"achieving asymptotic conditional coverage $P(Y \\in C(X) | G=g) \\to 1-\\alpha$ ",
        f"for each income quartile $g$. On the 2023 test set "
        f"({meta['test_set_size']} windows), {selection_sentence}",
        f"",
        f"## Ablation TODO Registry (Table 2)",
        f"",
        f"- [ ] ACI gamma sensitivity: γ ∈ {{0.01, 0.05, 0.1, adaptive-PI}}",
        f"- [ ] Calibration set size: 13 vs 26 vs 52 weeks",
        f"- [ ] Group granularity: geographic (S groups) vs demographic (4 quartiles)",
        f"- [ ] CQR vs ABS vs RAPS non-conformity score functions",
        f"- [ ] ECRC delta sensitivity: δ ∈ {{0.01, 0.05, 0.1}}",
        f"- [ ] Cross-city transfer: calibrate on Chicago, test on NYC",
    ])

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"  Audit report saved: {output_path}")


# ───────────────────────────────────────────────────────────────────
# CLI Entry Point
# ───────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="CIVIC-SAFE Phase 5: Conformal Calibration + Evaluation"
    )
    parser.add_argument(
        "--data", type=str, default="chicago",
        choices=["chicago", "nyc"],
        help="Dataset to evaluate (default: chicago)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help=(
            "Checkpoint file or run directory containing seed_*/best.pt "
            "(default: auto-discover the preferred anchored run)"
        )
    )
    parser.add_argument(
        "--alpha", type=float, default=0.1,
        help="Nominal miscoverage level (default: 0.1 for 90%% coverage)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (default: auto-detect cuda/cpu)"
    )
    parser.add_argument(
        "--weights", type=str, default="auto",
        choices=["auto", "raw", "ema"],
        help=(
            "Which checkpoint parameter set to evaluate. 'auto' (default) picks "
            "whichever scores lower CRPS on the 2022-H1 VALIDATION split, so the "
            "choice never touches the test set. 'raw'=model_state_dict, "
            "'ema'=ema_state_dict."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    results = run_conformal_evaluation(
        data_name=args.data,
        checkpoint_path=args.checkpoint,
        alpha=args.alpha,
        device=device,
        weights=args.weights,
    )

    # Final summary
    skill = results["skill_scores"]
    gate = results.get("forecasting_gate", {})
    selection = results.get("calibrator_selection", {})
    logger.info("\n" + "=" * 70)
    logger.info("  FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  CRPSS vs HA: {skill.get('crpss_vs_ha', skill['crpss']):.4f}")
    logger.info(f"  CRPSS vs Seasonal Naive: {skill.get('crpss_vs_seasonal_naive', skill['crpss']):.4f}")
    logger.info(
        f"  Forecasting gate: {'PASS' if gate.get('passed') else 'FAIL'} | "
        f"DM={_format_optional_number(gate.get('dm_stat'))}, "
        f"DM p={_format_optional_number(gate.get('dm_p_value'))}, "
        f"bootstrap p={_format_optional_number(gate.get('bootstrap_p_value'))}"
    )
    logger.info(
        f"  Selected calibrator: {selection.get('selected_method', 'none')} | "
        f"fallback={bool(selection.get('fallback_used', False))}"
    )
    for method, cov in results["coverage_results"].items():
        if isinstance(cov, dict) and "marginal_coverage" in cov:
            logger.info(
                f"  {method}: coverage={cov['marginal_coverage']:.4f}, "
                f"width={cov['mean_width']:.2f}, "
                f"disparity={cov.get('coverage_disparity', 0):.4f}"
            )


if __name__ == "__main__":
    main()
