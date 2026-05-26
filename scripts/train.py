#!/usr/bin/env python
"""CIVIC-SAFE training entry point.

Multi-seed experiment runner with Hydra configuration management.

Usage:
    # Full run (5 seeds, 100 epochs, default config)
    python scripts/train.py

    # Quick smoke test (1 seed, 2 epochs)
    python scripts/train.py training.epochs=2 training.num_seeds=1

    # Override model size
    python scripts/train.py model.spatial.hidden_dim=256

    # Use NYC data instead of Chicago
    python scripts/train.py data=nyc
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def run_single_seed(
    seed: int,
    config: dict,
    output_dir: Path,
) -> dict:
    """Train and evaluate with a single random seed.

    Args:
        seed: Random seed for this run.
        config: Full configuration dictionary.
        output_dir: Output directory for this seed's results.

    Returns:
        Dictionary with training history and best metrics.
    """
    from civicsafe.models.civicsafe_model import CivicSafeModel
    from civicsafe.models.dataset import CrimeWindowDataset, create_chronological_splits
    from civicsafe.models.graph import build_adjacency_from_synthetic
    from civicsafe.synthetic.distributions import generate_spatiotemporal_panel
    from civicsafe.training.trainer import Trainer
    from civicsafe.utils.seeding import seed_everything

    # --- Seed everything ---
    seed_everything(seed)
    logger.info(f"=" * 60)
    logger.info(f"  SEED {seed}")
    logger.info(f"=" * 60)

    # --- Configuration extraction ---
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    spatial_cfg = model_cfg.get("spatial", {})
    temporal_cfg = model_cfg.get("temporal", {})
    mixture_cfg = model_cfg.get("feature_mixture", {})
    zinb_cfg = model_cfg.get("zinb", {})

    # --- Data loading ---
    # TODO: Replace with real data pipeline once data download completes
    # For now, use synthetic data for development/testing
    logger.info("  Generating synthetic spatiotemporal panel...")
    panel = generate_spatiotemporal_panel(
        num_spatial_units=77,
        num_time_steps=52 * 6,  # 6 years of weekly data
        num_categories=9,
        seed=seed,
    )
    counts = panel["counts"]
    features = panel["features"]
    S, T, C = counts.shape
    F = features.shape[-1]

    # --- Graph construction ---
    graph = build_adjacency_from_synthetic(num_nodes=S, seed=seed, knn_k=8)

    # --- Chronological splits ---
    splits = create_chronological_splits(
        counts,
        features,
        start_year=2018,
        end_year=2023,
        val_year=2022,
        test_year=2023,
        window_size=temporal_cfg.get("max_seq_len", 52),
    )

    # --- Create DataLoaders with graph data ---
    edge_queen = graph["queen"]
    edge_knn = graph.get("knn")

    def collate_with_graph(batch: list[dict]) -> dict:
        """Collate function that adds graph edges to each batch."""
        collated = {
            "input_features": torch.stack([b["input_features"] for b in batch]),
            "input_counts": torch.stack([b["input_counts"] for b in batch]),
            "target_counts": torch.stack([b["target_counts"] for b in batch]),
            "edge_queen": edge_queen,
        }
        if edge_knn is not None:
            collated["edge_knn"] = edge_knn
        return collated

    dl_cfg = train_cfg.get("dataloader", {})
    num_workers = dl_cfg.get("num_workers", 4)

    train_loader = torch.utils.data.DataLoader(
        splits["train"],
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=dl_cfg.get("pin_memory", True),
        persistent_workers=num_workers > 0 and dl_cfg.get("persistent_workers", True),
        collate_fn=collate_with_graph,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        splits["val"],
        batch_size=train_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=dl_cfg.get("pin_memory", True),
        persistent_workers=num_workers > 0 and dl_cfg.get("persistent_workers", True),
        collate_fn=collate_with_graph,
    )

    # --- Model ---
    model = CivicSafeModel(
        num_features=F,
        hidden_dim=spatial_cfg.get("hidden_dim", 128),
        spatial_layers=spatial_cfg.get("num_layers", 2),
        spatial_heads=spatial_cfg.get("num_heads", 4),
        temporal_layers=temporal_cfg.get("num_layers", 2),
        temporal_heads=temporal_cfg.get("num_heads", 4),
        temporal_ff_dim=temporal_cfg.get("dim_feedforward", 512),
        num_categories=C,
        max_seq_len=temporal_cfg.get("max_seq_len", 52),
    )
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {num_params:,}")

    # --- Device ---
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Train ---
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        output_dir=output_dir / f"seed_{seed}",
    )

    results = trainer.fit()
    logger.info(
        f"  Seed {seed} complete: "
        f"best CRPS = {results['best_metrics']['crps']:.4f}, "
        f"MAE = {results['best_metrics']['mae']:.4f}, "
        f"RMSE = {results['best_metrics']['rmse']:.4f}"
    )

    return results


def main() -> None:
    """Main entry point for multi-seed training."""
    import yaml
    
    # Fix PyTorch CuBLAS determinism warning for A100 GPUs
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # --- Setup logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Load config ---
    project_root = Path(__file__).resolve().parent.parent
    config_dir = project_root / "configs"

    # Load and merge configs
    config: dict = {}
    for cfg_file in [
        config_dir / "model" / "spatiotemporal_zinb.yaml",
        config_dir / "training" / "default.yaml",
    ]:
        if cfg_file.exists():
            with open(cfg_file) as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    config.update(loaded)

    # --- Parse CLI overrides (simple key=value) ---
    for arg in sys.argv[1:]:
        if "=" in arg:
            key, value = arg.split("=", 1)
            # Navigate nested keys
            parts = key.split(".")
            target = config
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            # Auto-convert types
            try:
                target[parts[-1]] = int(value)
            except ValueError:
                try:
                    target[parts[-1]] = float(value)
                except ValueError:
                    if value.lower() in ("true", "false"):
                        target[parts[-1]] = value.lower() == "true"
                    else:
                        target[parts[-1]] = value

    # --- Training config ---
    train_cfg = config.get("training", {})
    num_seeds = train_cfg.get("num_seeds", 5)
    seeds = train_cfg.get("seeds", [42, 137, 256, 512, 1024])[:num_seeds]

    output_dir = Path("outputs") / f"run_{int(time.time())}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"CIVIC-SAFE Training — {num_seeds} seed(s): {seeds}")
    logger.info(f"Output directory: {output_dir}")

    # --- W&B init (optional) ---
    try:
        import wandb

        # Default to disabled so it never blocks execution with prompts.
        mode = os.environ.get("WANDB_MODE", "disabled")
        wandb.init(
            project="civicsafe",
            config=config,
            dir=str(output_dir),
            mode=mode,
        )
    except Exception as e:
        logger.warning(f"W&B initialization skipped or failed ({e}). Logging to console only.")

    # --- Multi-seed training ---
    all_results = []
    for seed in seeds:
        result = run_single_seed(seed, config, output_dir)
        all_results.append(result)

    # --- Aggregate results across seeds ---
    import numpy as np

    metric_names = ["crps", "mae", "rmse", "brier_zero"]
    logger.info("\n" + "=" * 60)
    logger.info("  AGGREGATE RESULTS (mean ± std across seeds)")
    logger.info("=" * 60)
    for metric in metric_names:
        values = [r["best_metrics"][metric] for r in all_results]
        mean = np.mean(values)
        std = np.std(values)
        logger.info(f"  {metric:>12s}: {mean:.4f} ± {std:.4f}")

    # --- W&B summary ---
    try:
        import wandb

        if wandb.run is not None:
            for metric in metric_names:
                values = [r["best_metrics"][metric] for r in all_results]
                wandb.summary[f"best_{metric}_mean"] = np.mean(values)
                wandb.summary[f"best_{metric}_std"] = np.std(values)
            wandb.finish()
    except ImportError:
        pass

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
