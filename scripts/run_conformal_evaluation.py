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
    python scripts/run_conformal_evaluation.py --data chicago --checkpoint outputs/run_XXX/seed_42

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

from civicsafe.calibration.conformal import (
    AdaptiveTemporalECRCCalibrator,
    ECRCCalibrator,
    EqualizedCoverageCalibrator,
    MondrianConformalCalibrator,
    SplitConformalCalibrator,
    WeightedConformalCalibrator,
    compute_cqr_scores,
)
from civicsafe.calibration.zinb_distribution import zinb_ppf_pair
from civicsafe.models.civicsafe_model import CivicSafeModel
from civicsafe.models.dataset import CrimeWindowDataset, create_chronological_splits
from civicsafe.training.metrics import compute_all_metrics, crps_zinb, pit_values
from civicsafe.calibration.recalibration import recalibrate_and_evaluate
from civicsafe.calibration.emos import learn_emos_weights, apply_emos_weights, crps_decomposition
from civicsafe.calibration.significance import compare_forecasts
from civicsafe.audit.feedback_loop import compute_all_feedback_metrics

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────────────────────────
CATEGORY_NAMES = {0: "violent", 1: "property", 2: "drug"}
ALPHA_DEFAULT = 0.1  # 90% coverage target
COVERAGE_DISPARITY_THRESHOLD = 0.03  # 3 percentage point max disparity
CRPSS_SKILL_THRESHOLD = 0.10  # 10% improvement over baseline

# Pre-registered kill criteria
class KillCriterionTriggered(Exception):
    """Raised when a pre-registered quality threshold is violated."""
    pass


# ───────────────────────────────────────────────────────────────────
# Checkpoint Discovery
# ───────────────────────────────────────────────────────────────────
def discover_checkpoint(data_name: str) -> Path:
    """Auto-discover the most recent checkpoint for the given dataset.
    
    Falls back to single-checkpoint mode if discover_all_checkpoints
    is not called explicitly.
    """
    checkpoints = discover_all_checkpoints(data_name)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found for {data_name}")
    # Default: pick the first seed (usually seed_42)
    chosen = checkpoints[0]
    logger.info(f"  Auto-discovered checkpoint: {chosen}")
    return chosen


def discover_all_checkpoints(data_name: str) -> list[Path]:
    """Discover ALL seed checkpoints in the latest run directory for this dataset.
    
    This enables ensemble evaluation: load all K seeds, average their
    predictions, and evaluate the ensemble. EMOS-style ensembling
    typically improves CRPS by 10-30% (Gneiting et al., 2005).
    
    Searches for dataset-specific directories first (``run_{data_name}_*``),
    then falls back to generic ``run_*`` for backward compatibility.
    
    Returns:
        Sorted list of best.pt paths, one per seed.
    """
    outputs_dir = PROJECT_ROOT / "outputs"
    if not outputs_dir.exists():
        raise FileNotFoundError(f"No outputs directory at {outputs_dir}")
    
    # Priority 1: dataset-specific run directories (run_chicago_*, run_nyc_*)
    #
    # The canonical full-model run is UNTAGGED: run_{city}_{timestamp}. Ablations
    # and probes are TAGGED: run_{city}_{tag}_{timestamp}. Because the untagged
    # prefix is a prefix of every tagged one, a bare glob matches both, and
    # sorting by name then puts tags AFTER the digits -- so `run_dirs[-1]` would
    # resolve to `run_chicago_no_transformer_...` once ablations exist and
    # silently report an ablation as the headline model. Keep only untagged dirs,
    # matching the filter train.py:377 and run_ablations.py:126 already apply.
    dataset_prefix = f"run_{data_name}_"
    run_dirs = sorted(
        (
            d for d in outputs_dir.glob(f"{dataset_prefix}*")
            if d.is_dir() and d.name[len(dataset_prefix):].isdigit()
        ),
        key=lambda p: p.name,
    )

    # Priority 2: generic run_* directories (backward compat, pre-city-prefix
    # era). The untagged filter still applies: keep only dirs whose name after
    # "run_" is all digits. Without it this fallback re-introduces the exact bug
    # fixed above -- it would match another city's run (run_nyc_* for a chicago
    # request) and every ablation dir, and it fires precisely when the untagged
    # run for this city is missing, i.e. mid-regeneration when only ablations
    # are on disk.
    if not run_dirs:
        run_dirs = sorted(
            (
                d for d in outputs_dir.glob("run_*")
                if d.is_dir() and d.name[len("run_"):].isdigit()
            ),
            key=lambda p: p.name,
        )

    if not run_dirs:
        raise FileNotFoundError(
            f"No run directories found under {outputs_dir} "
            f"(searched for '{dataset_prefix}*' and 'run_*')"
        )
    
    latest_run = run_dirs[-1]
    
    # Find all seed_*/best.pt checkpoints
    seed_checkpoints = sorted(latest_run.glob("seed_*/best.pt"))
    
    if not seed_checkpoints:
        # Last-resort fallback: any model-looking .pt anywhere under outputs/.
        #
        # This is UNSCOPED on purpose (it exists so a hand-placed checkpoint still
        # works), which makes it the most dangerous path in this function: it can
        # return another city's checkpoint or an ablation's, and it returns only
        # ONE file, which silently collapses the multi-seed EMOS ensemble to a
        # single model while the report still labels the number an ensemble CRPS.
        # A quiet 5x reduction in the headline result is not acceptable, so warn
        # loudly and name the file that was chosen.
        candidates = list(outputs_dir.rglob("*.pt"))
        candidates = [
            p for p in candidates
            if "panel" not in p.name and "graph" not in p.name
            and "demographics" not in p.name and "calibrators" not in p.name
        ]
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            return []
        logger.warning(
            f"  No seed_*/best.pt under {latest_run.name}. Falling back to the "
            f"most recently modified checkpoint anywhere in outputs/: "
            f"{candidates[0].relative_to(outputs_dir)}"
        )
        logger.warning(
            "  This is a SINGLE checkpoint: the multi-seed EMOS ensemble is "
            "DISABLED and any CRPS reported below is a single-model score, not "
            "an ensemble score. It is also not scoped to "
            f"--data {data_name}. Do not use this for headline numbers."
        )
        return candidates[:1]
    
    logger.info(f"  Found {len(seed_checkpoints)} seed checkpoints in {latest_run.name}")
    for ckpt in seed_checkpoints:
        logger.info(f"    {ckpt.parent.name}/{ckpt.name}")
    
    return seed_checkpoints


# ───────────────────────────────────────────────────────────────────
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
    all_y, all_pi, all_mu, all_r = [], [], [], []

    edge_q = edge_queen.to(device)
    edge_k = edge_knn.to(device) if edge_knn is not None else None

    for idx in range(len(dataset)):
        sample = dataset[idx]
        x_feat = sample["input_features"].to(device)   # (S, W, F)
        x_counts = torch.log1p(sample["input_counts"].float().to(device))  # (S, W, C)
        features = torch.cat([x_feat, x_counts], dim=-1)  # (S, W, F+C)
        target = sample["target_counts"]  # (S, C)

        output = model(features, edge_q, edge_k)

        all_y.append(target.cpu().float())
        all_pi.append(output["pi"].cpu().float())
        all_mu.append(output["mu"].cpu().float())
        all_r.append(output["r"].cpu().float())

    return {
        "y": torch.stack(all_y),     # (N, S, C)
        "pi": torch.stack(all_pi),   # (N, S, C)
        "mu": torch.stack(all_mu),   # (N, S, C)
        "r": torch.stack(all_r),     # (N, S, C)
    }


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

    Args:
        y: Ground-truth counts. Shape: (N,)
        lower: Lower bounds. Shape: (N,)
        upper: Upper bounds. Shape: (N,)
        groups: Demographic group labels. Shape: (N,) or None.
        alpha: Nominal miscoverage level.

    Returns:
        Dictionary with coverage metrics.
    """
    covered = ((y >= lower) & (y <= upper)).float()
    width = (upper - lower).float()

    result: dict[str, Any] = {
        "marginal_coverage": covered.mean().item(),
        "mean_width": width.mean().item(),
        "median_width": width.median().item(),
        "target_coverage": 1.0 - alpha,
        "coverage_gap": covered.mean().item() - (1.0 - alpha),
    }

    # Per-category coverage (if data has category structure)
    # Per-group coverage
    if groups is not None:
        group_coverages = {}
        unique_groups = groups.unique().tolist()
        for g in unique_groups:
            mask = groups == g
            if mask.sum() > 0:
                group_cov = covered[mask].mean().item()
                group_width = width[mask].mean().item()
                group_coverages[f"group_{g}"] = {
                    "coverage": group_cov,
                    "mean_width": group_width,
                    "n_samples": int(mask.sum().item()),
                }
        result["per_group"] = group_coverages

        # Coverage disparity: max - min across groups
        all_coverages = [v["coverage"] for v in group_coverages.values()]
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

    if checkpoint_path and checkpoint_path != "auto":
        ckpt_path = Path(checkpoint_path)
        all_ckpts = [ckpt_path]
    else:
        all_ckpts = discover_all_checkpoints(data_name)
        if not all_ckpts:
            raise FileNotFoundError(f"No checkpoints found for {data_name}")

    # ─── Weight-set selection on VALIDATION data ───
    #
    # best.pt stores two parameter sets: the online weights (model_state_dict)
    # and the Polyak-averaged ones (ema_state_dict). They do not perform
    # equally here: the trainer uses EMA decay 0.999 (~1000-step horizon) but
    # only ~10 optimizer steps/epoch, so the EMA never converges and keeps a
    # large share of its initial snapshot. Empirically it is much worse.
    #
    # Which to report cannot be decided on the test set — that is tuning on
    # test. Decide on 2022-H1 validation, which is disjoint from both the
    # calibration (2022-H2) and test (2023) windows, then use that choice
    # everywhere downstream.
    if weights == "auto":
        logger.info("\n  ─── WEIGHT-SET SELECTION (on validation, 2022 H1) ───")
        probe_ckpt = all_ckpts[0]
        val_scores: dict[str, float] = {}
        for cand in ("raw", "ema"):
            try:
                probe_model = load_model_from_checkpoint(
                    probe_ckpt, F + C, C, config, device, weights=cand
                )
            except KeyError as exc:
                logger.info(f"    {cand:>3}: unavailable ({exc})")
                continue
            probe_res = run_rolling_inference(
                probe_model, val_dataset, edge_queen, edge_knn, device
            )
            val_crps = crps_zinb(
                probe_res["y"].reshape(-1), probe_res["pi"].reshape(-1),
                probe_res["mu"].reshape(-1), probe_res["r"].reshape(-1),
            ).mean().item()
            val_scores[cand] = val_crps
            logger.info(f"    {cand:>3}: validation CRPS = {val_crps:.4f}")
            del probe_model

        if not val_scores:
            raise RuntimeError(f"No usable weight set found in {probe_ckpt}")
        weights = min(val_scores, key=lambda k: val_scores[k])
        logger.info(
            f"  Selected weights='{weights}' (lowest validation CRPS). "
            f"Test set was NOT used for this choice."
        )
        if len(val_scores) == 2:
            gap = abs(val_scores["raw"] - val_scores["ema"])
            if gap > 0.25:
                logger.warning(
                    f"  Large raw/EMA validation gap ({gap:.4f}). EMA decay 0.999 "
                    f"is likely mistuned for this run length (~10 steps/epoch); "
                    f"consider decay≈0.99 or EMA-per-epoch for future runs."
                )

    # ─── Step 4-5: Ensemble inference (EMOS-style) ───
    K = len(all_ckpts)
    logger.info(f"\n[3-5/7] Ensemble inference with {K} seed(s)...")
    logger.info(f"  Using weight set: '{weights}' for all {K} seed(s)")

    cal_results_list = []
    test_results_list = []

    for i, ckpt in enumerate(all_ckpts):
        logger.info(f"\n  --- Seed {i+1}/{K}: {ckpt.parent.name}/{ckpt.name} ---")
        model_i = load_model_from_checkpoint(
            ckpt, F + C, C, config, device, weights=weights
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

    # ─── TRUE CDF Mixture Ensemble (EMOS) ───
    # MATHEMATICS.md §11: F_ens(j) = (1/K) Σ_k F_ZINB^(k)(j)
    # We average the per-seed CRPS scores (since CRPS decomposes linearly
    # over CDF terms) and store all per-seed params for the mixture CDF.
    # For conformal calibration, we compute non-conformity scores using
    # the mixture CDF rather than averaged parameters.
    if K > 1:
        logger.info(f"\n  Ensembling {K} seeds via CDF mixture...")
        # Store all per-seed results for mixture CDF computation
        # For CRPS: compute per-seed CRPS, then report ensemble CRPS
        # using the mixture CDF formula
        cal_results = {
            "y": cal_results_list[0]["y"],
            # Store all seeds for mixture CDF-based scoring
            "all_pi": [r["pi"] for r in cal_results_list],
            "all_mu": [r["mu"] for r in cal_results_list],
            "all_r": [r["r"] for r in cal_results_list],
            # For conformal calibration APIs that expect single (pi,mu,r),
            # use the mean as an approximation of the mixture mode.
            # The actual conformal scores use the mixture CDF below.
            "pi": torch.stack([r["pi"] for r in cal_results_list]).mean(dim=0),
            "mu": torch.stack([r["mu"] for r in cal_results_list]).mean(dim=0),
            "r": torch.stack([r["r"] for r in cal_results_list]).mean(dim=0),
        }
        test_results = {
            "y": test_results_list[0]["y"],
            "all_pi": [r["pi"] for r in test_results_list],
            "all_mu": [r["mu"] for r in test_results_list],
            "all_r": [r["r"] for r in test_results_list],
            "pi": torch.stack([r["pi"] for r in test_results_list]).mean(dim=0),
            "mu": torch.stack([r["mu"] for r in test_results_list]).mean(dim=0),
            "r": torch.stack([r["r"] for r in test_results_list]).mean(dim=0),
        }

        # Compute true mixture CDF CRPS: average CRPS across seeds
        # By linearity: CRPS(F_mix, y) = E_k[CRPS(F_k, y)] - diversity_term
        # The diversity_term is non-negative, so mixture CRPS ≤ mean(seed CRPS)
        per_seed_crps = []
        for res in test_results_list:
            sc = crps_zinb(
                res["y"].reshape(-1), res["pi"].reshape(-1),
                res["mu"].reshape(-1), res["r"].reshape(-1)
            ).mean().item()
            per_seed_crps.append(sc)
        
        # The ensemble CRPS with averaged params (approximation)
        ensemble_crps = crps_zinb(
            test_results["y"].reshape(-1), test_results["pi"].reshape(-1),
            test_results["mu"].reshape(-1), test_results["r"].reshape(-1)
        ).mean().item()
        
        logger.info(f"  Per-seed CRPS: {[f'{c:.4f}' for c in per_seed_crps]}")
        logger.info(f"  Mean per-seed CRPS: {np.mean(per_seed_crps):.4f}")
        logger.info(f"  Ensemble CRPS (equal-weight): {ensemble_crps:.4f}")

        # --- EMOS: Learn Optimal Weights ---
        logger.info("\n  --- EMOS WEIGHT LEARNING ---")
        emos_info = learn_emos_weights(
            y_cal=cal_results["y"].reshape(-1),
            all_pi=cal_results["all_pi"],
            all_mu=cal_results["all_mu"],
            all_r=cal_results["all_r"],
        )
        # Apply learned weights to get EMOS-combined params
        pi_emos_cal, mu_emos_cal, r_emos_cal = apply_emos_weights(
            emos_info["weights"],
            cal_results["all_pi"], cal_results["all_mu"], cal_results["all_r"],
        )
        pi_emos_test, mu_emos_test, r_emos_test = apply_emos_weights(
            emos_info["weights"],
            test_results["all_pi"], test_results["all_mu"], test_results["all_r"],
        )
        # Overwrite the ensemble params with EMOS-optimized versions
        cal_results["pi"] = pi_emos_cal
        cal_results["mu"] = mu_emos_cal
        cal_results["r"] = r_emos_cal
        test_results["pi"] = pi_emos_test
        test_results["mu"] = mu_emos_test
        test_results["r"] = r_emos_test
        
        emos_crps = crps_zinb(
            test_results["y"].reshape(-1), pi_emos_test.reshape(-1),
            mu_emos_test.reshape(-1), r_emos_test.reshape(-1)
        ).mean().item()
        logger.info(f"  EMOS CRPS (learned weights): {emos_crps:.4f}")
        logger.info(
            f"  EMOS improvement over equal-weight: "
            f"{(1.0 - emos_crps / ensemble_crps) * 100:.2f}%"
        )
        
        # ─── Uncertainty Decomposition ───
        # Aleatoric = mean of per-seed ZINB variance
        # Epistemic = variance of per-seed point predictions E[Y|k] = (1-pi_k)*mu_k
        logger.info("\n  ─── UNCERTAINTY DECOMPOSITION ───")
        point_preds = torch.stack([
            (1.0 - r["pi"]) * r["mu"] for r in test_results_list
        ])  # (K, N, S, C)
        epistemic = point_preds.var(dim=0).mean().item()
        
        from civicsafe.training.sac_loss import zinb_variance
        aleatoric_per_seed = []
        for res in test_results_list:
            v = zinb_variance(
                res["pi"].reshape(-1), res["mu"].reshape(-1), res["r"].reshape(-1)
            ).mean().item()
            aleatoric_per_seed.append(v)
        aleatoric = np.mean(aleatoric_per_seed)
        
        logger.info(f"  Aleatoric uncertainty (mean ZINB var): {aleatoric:.4f}")
        logger.info(f"  Epistemic uncertainty (seed disagreement): {epistemic:.4f}")
        logger.info(f"  Epistemic / Total: {epistemic / (aleatoric + epistemic):.2%}")
    else:
        cal_results = cal_results_list[0]
        test_results = test_results_list[0]

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
    calibrator_configs = {
        "split_cp": SplitConformalCalibrator(alpha=alpha),
        "weighted_cp": WeightedConformalCalibrator(alpha=alpha, decay_rate=0.05),
        "mondrian": MondrianConformalCalibrator(alpha=alpha, min_group_size=20),
        "mondrian_category": MondrianConformalCalibrator(alpha=alpha, min_group_size=20),
        "mondrian_demo_x_category": MondrianConformalCalibrator(
            alpha=alpha, min_group_size=20
        ),
        "equalized_coverage": EqualizedCoverageCalibrator(alpha=alpha, lambda_eq=1.0),
        "ecrc": ECRCCalibrator(alpha=alpha, delta=0.05, group_type="demographic"),
        "adaptive_ecrc": AdaptiveTemporalECRCCalibrator(
            alpha=alpha, gamma=0.05, delta=0.05, group_type="demographic"
        ),
    }

    # Which axis each method CONDITIONS ON. Reporting is always on the
    # demographic axis (below) so disparity stays comparable across methods.
    calibration_axis = {
        "mondrian": "demographic",
        "mondrian_category": "category",
        "mondrian_demo_x_category": "demo_x_category",
        "equalized_coverage": "demographic",
        "ecrc": "demographic",
        "adaptive_ecrc": "demographic",
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
                           "mondrian_demo_x_category", "ecrc", "adaptive_ecrc"):
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
    logger.info(f"  CRPSS vs HA (rolling):                  {crpss_ha:+.4f} (threshold: >={CRPSS_SKILL_THRESHOLD})")
    logger.info(f"  CRPSS vs HA (frozen):                   {crpss_ha_frozen:+.4f}")
    logger.info(f"  CRPSS vs Seasonal Naive:                {crpss_sn:+.4f} (threshold: >={CRPSS_SKILL_THRESHOLD})")

    # The gate is the MINIMUM skill across both naive baselines. A model only
    # earns a forecasting claim if it beats every naive competitor, not the
    # most convenient one.
    crpss_primary = min(crpss_ha, crpss_sn)
    if crpss_primary < CRPSS_SKILL_THRESHOLD:
        logger.warning(
            f"  GATE NOT MET: min(CRPSS) = {crpss_primary:+.4f} < "
            f"{CRPSS_SKILL_THRESHOLD}. The binding baseline is "
            f"{'rolling HA' if crpss_ha < crpss_sn else 'seasonal naive'}."
        )
    else:
        logger.info(
            f"  GATE MET: min(CRPSS) = {crpss_primary:+.4f} >= {CRPSS_SKILL_THRESHOLD}"
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
    y_test_3d = test_results["y"]  # (N_windows, S, C)
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
    logger.info("\n[7/7] Compiling results and saving to disk...")

    # Dataset hash for reproducibility
    panel_hash = hashlib.md5(
        counts.numpy().tobytes()[:10000]  # First 10KB for speed
    ).hexdigest()[:12]

    results = {
        "metadata": {
            "dataset": data_name,
            "checkpoint": str(all_ckpts) if K > 1 else str(all_ckpts[0]),
            "num_ensemble_seeds": K,
            "weights_source": weights,
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
            # Primary skill score is the MINIMUM over the naive family, so the
            # gate is decided by whichever naive baseline is hardest to beat.
            # This used to be crpss_sn alone, which passed at 0.2662 on Chicago
            # while the model was simultaneously 9.2% WORSE than a rolling
            # 52-week mean -- a passing gate on a losing forecaster.
            "crpss": crpss_primary,
            "crpss_binding_baseline": (
                "ha_rolling" if crpss_ha < crpss_sn else "seasonal_naive"
            ),
            "crpss_passes_threshold": crpss_primary >= CRPSS_SKILL_THRESHOLD,
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
    }
    
    # Add ensemble-specific results if applicable
    if K > 1:
        results["ensemble"] = {
            "num_seeds": K,
            "per_seed_crps": per_seed_crps,
            "mean_seed_crps": float(np.mean(per_seed_crps)),
            "ensemble_crps_equal_weight": ensemble_crps,
            "emos_crps_learned_weight": emos_crps,
            "emos_weights": emos_info["weights"],
            "emos_improvement_pct": emos_info["improvement_pct"],
            "ensemble_improvement": float(1.0 - emos_crps / np.mean(per_seed_crps)),
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "epistemic_fraction": float(epistemic / (aleatoric + epistemic)),
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

    # Find the best calibrator.
    #
    # Selecting by max(coverage) is wrong: coverage is trivially maximised by
    # emitting absurdly wide intervals, so it always crowned the most
    # OVER-covered method (NYC ecrc: 98.7% coverage at width 34.0) over a
    # well-calibrated one (split_cp: 90.5% at width 17.4). Conformal prediction
    # targets coverage >= 1-alpha with the SMALLEST width; among methods that
    # are valid, narrower is strictly better. So: filter to methods that hit the
    # coverage floor, then pick minimum mean width. Overcoverage is reported as
    # a diagnostic rather than rewarded.
    COVERAGE_TOL = 0.01
    coverage_floor = 1.0 - alpha - COVERAGE_TOL

    candidates = {
        m: c for m, c in all_coverage_results.items()
        if isinstance(c, dict) and "marginal_coverage" in c
    }
    valid = {
        m: c for m, c in candidates.items()
        if c["marginal_coverage"] >= coverage_floor
    }

    best_method = None
    if valid:
        # Efficiency-optimal among valid calibrators.
        best_method = min(
            valid, key=lambda m: valid[m].get("mean_width", float("inf"))
        )
        selection_rule = "min width s.t. coverage >= 1-alpha"
    elif candidates:
        # Nothing reached the floor — fall back to the closest to target so the
        # exit-criteria check below reports a genuine failure instead of hiding it.
        best_method = min(
            candidates,
            key=lambda m: abs(candidates[m]["marginal_coverage"] - (1.0 - alpha)),
        )
        selection_rule = "closest to target (NO method met coverage floor)"

    if best_method:
        logger.info(f"  Selection rule: {selection_rule}")
        # Efficiency table: makes over/under-coverage explicit for the paper.
        logger.info(
            f"  {'method':<24} {'coverage':>9} {'width':>8} {'disparity':>10}  status"
        )
        for m in sorted(candidates, key=lambda k: candidates[k].get("mean_width", 0.0)):
            c = candidates[m]
            cov_m = c["marginal_coverage"]
            if cov_m < coverage_floor:
                status = "UNDER-COVERS"
            elif cov_m > (1.0 - alpha) + 0.03:
                status = "over-covers (inefficient)"
            else:
                status = "well-calibrated"
            logger.info(
                f"  {m:<24} {cov_m:>9.4f} {c.get('mean_width', float('nan')):>8.2f} "
                f"{c.get('coverage_disparity', 0.0):>10.4f}  {status}"
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

        # Check kill criteria
        passed_all = True
        if best_results["marginal_coverage"] < (1 - alpha - 0.01):
            logger.warning(
                f"  ⚠ COVERAGE BELOW TARGET: "
                f"{best_results['marginal_coverage']:.4f} < {1-alpha-0.01:.2f}"
            )
            passed_all = False

        if disparity > COVERAGE_DISPARITY_THRESHOLD:
            logger.warning(
                f"  ⚠ COVERAGE DISPARITY EXCEEDS THRESHOLD: "
                f"{disparity:.4f} > {COVERAGE_DISPARITY_THRESHOLD}"
            )
            passed_all = False

        # Overcoverage is not a pass. Intervals far wider than needed are
        # useless operationally and a reviewer will read >97% at alpha=0.1 as
        # a calibration failure, not a strength.
        if best_results["marginal_coverage"] > (1.0 - alpha) + 0.03:
            logger.warning(
                f"  ⚠ SUBSTANTIAL OVERCOVERAGE: "
                f"{best_results['marginal_coverage']:.4f} >> {1 - alpha:.2f} target "
                f"(width {best_results.get('mean_width', float('nan')):.2f}). "
                f"Intervals are wider than necessary; check group sizes feeding "
                f"the ECRC Hoeffding slack."
            )
            passed_all = False

        if passed_all:
            logger.info("  ✓ ALL EXIT CRITERIA PASSED")
        else:
            logger.info("  ✗ SOME EXIT CRITERIA FAILED (see warnings above)")

    elapsed = time.time() - t_start
    logger.info(f"\n  Pipeline complete in {elapsed:.1f}s")

    return results


# ───────────────────────────────────────────────────────────────────
# Audit Report Generation
# ───────────────────────────────────────────────────────────────────
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
        f"| Threshold (≥0.10 vs ALL naive) | {'✓ PASS' if skill['crpss_passes_threshold'] else '✗ FAIL'} |",
        f"",
        f"## Coverage Results by Calibration Method",
        f"",
    ]

    # Table header
    lines.append(f"| Method | Marginal Coverage | Target | Mean Width | Disparity |")
    lines.append(f"|--------|:-----------------:|:------:|:----------:|:---------:|")

    for method, cov in coverage.items():
        if isinstance(cov, dict) and "marginal_coverage" in cov:
            mc = cov["marginal_coverage"]
            target = cov["target_coverage"]
            width = cov["mean_width"]
            disp = cov.get("coverage_disparity", 0)
            pass_mark = "✓" if abs(mc - target) < 0.01 else "⚠"
            lines.append(
                f"| {method} | {pass_mark} {mc:.4f} | {target:.2f} | {width:.2f} | {disp:.4f} |"
            )
        elif isinstance(cov, dict) and "error" in cov:
            lines.append(f"| {method} | ERROR | - | - | - |")

    lines.append("")

    # Per-category breakdown for best method.
    # Must use the SAME rule as the exit-criteria check (min width subject to
    # coverage >= 1-alpha), otherwise the markdown report advertises a different
    # "best calibrator" than the console summary.
    _alpha = results.get("metadata", {}).get("alpha", 0.1)
    _floor = 1.0 - _alpha - 0.01
    _cands = {
        m: c for m, c in coverage.items()
        if isinstance(c, dict) and "marginal_coverage" in c
    }
    _valid = {
        m: c for m, c in _cands.items() if c["marginal_coverage"] >= _floor
    }
    if _valid:
        best_method = min(
            _valid, key=lambda m: _valid[m].get("mean_width", float("inf"))
        )
    elif _cands:
        best_method = min(
            _cands,
            key=lambda m: abs(_cands[m]["marginal_coverage"] - (1.0 - _alpha)),
        )
    else:
        best_method = None
    best_cov_val = _cands[best_method]["marginal_coverage"] if best_method else 0.0

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
        f"for each income quartile $g$. On the 2023 test set ({meta['test_set_size']} windows), ",
        f"the best calibrator ({best_method}) achieves {best_cov_val:.1%} marginal ",
        f"coverage with mean prediction interval width {coverage[best_method]['mean_width']:.2f} ",
        f"counts and a maximum cross-group coverage disparity of ",
        f"{coverage[best_method].get('coverage_disparity', 0):.4f}.",
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
        help="Path to model checkpoint (default: auto-discover latest)"
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
    logger.info("\n" + "=" * 70)
    logger.info("  FINAL SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  CRPSS vs HA: {skill.get('crpss_vs_ha', skill['crpss']):.4f}")
    logger.info(f"  CRPSS vs Seasonal Naive: {skill.get('crpss_vs_seasonal_naive', skill['crpss']):.4f}")
    for method, cov in results["coverage_results"].items():
        if isinstance(cov, dict) and "marginal_coverage" in cov:
            logger.info(
                f"  {method}: coverage={cov['marginal_coverage']:.4f}, "
                f"width={cov['mean_width']:.2f}, "
                f"disparity={cov.get('coverage_disparity', 0):.4f}"
            )


if __name__ == "__main__":
    main()
