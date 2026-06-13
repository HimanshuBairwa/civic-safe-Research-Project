#!/usr/bin/env python
"""CIVIC-SAFE: Quick model checkpoint generator for conformal evaluation testing.

Runs a minimal 2-epoch training to produce a valid checkpoint file,
enabling the conformal evaluation pipeline to be tested end-to-end
on CPU before deploying to GPU for full 200-epoch training.

Usage:
    python scripts/quick_train_checkpoint.py --data chicago
    python scripts/quick_train_checkpoint.py --data nyc
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from civicsafe.models.civicsafe_model import CivicSafeModel
from civicsafe.models.dataset import create_chronological_splits
from civicsafe.models.zinb_loss import ZINBLoss
from civicsafe.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Quick checkpoint generator")
    parser.add_argument("--data", type=str, default="chicago", choices=["chicago", "nyc"])
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    data_name = args.data
    seed_everything(42)

    # Load data
    panel_path = PROJECT_ROOT / "data" / "processed" / f"{data_name}_panel.pt"
    graph_path = PROJECT_ROOT / "data" / "processed" / f"{data_name}_graph.pt"

    if not panel_path.exists():
        logger.error(f"Panel not found: {panel_path}. Run: python scripts/fetch_data.py")
        sys.exit(1)

    panel = torch.load(panel_path, weights_only=False)
    counts = panel["counts"]
    features = panel["features"]
    S, T, C = counts.shape
    F = features.shape[-1]

    # Normalize features
    feat_mean = features.mean(dim=(0, 1), keepdim=True)
    feat_std = features.std(dim=(0, 1), keepdim=True).clamp(min=1e-6)
    features = (features - feat_mean) / feat_std

    graph = torch.load(graph_path, weights_only=False)
    edge_queen = graph["queen"]
    edge_knn = graph.get("knn")

    logger.info(f"Data: {S} spatial x {T} weeks x {C} categories, {F} features")

    # Splits
    splits = create_chronological_splits(counts, features)
    train_ds = splits["train"]

    # Load model config
    config: dict = {}
    for cfg_file in [
        PROJECT_ROOT / "configs" / "model" / "spatiotemporal_zinb.yaml",
        PROJECT_ROOT / "configs" / "training" / "default.yaml",
    ]:
        if cfg_file.exists():
            with open(cfg_file, encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    config.update(loaded)

    model_cfg = config.get("model", {})
    spatial_cfg = model_cfg.get("spatial", {})
    temporal_cfg = model_cfg.get("temporal", {})

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
    logger.info(f"Model: {num_params:,} parameters")

    device = "cpu"
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    loss_fn = ZINBLoss(reduction="mean")

    # Quick training loop
    model.train()
    for epoch in range(args.epochs):
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0

        # Process a subset of windows (max 10 for speed)
        n_windows = min(len(train_ds), 10)
        for idx in range(n_windows):
            sample = train_ds[idx]
            feat = sample["input_features"].to(device)  # (S, W, F)
            target = sample["target_counts"].to(device)  # (S, C)

            output = model(feat, edge_queen, edge_knn)

            pi = output["pi"].float()
            mu = output["mu"].float()
            r = output["r"].float()
            y = target.float()

            loss = loss_fn(y.reshape(-1), pi.reshape(-1), mu.reshape(-1), r.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0
        logger.info(f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f} | {elapsed:.1f}s")

    # Save checkpoint
    output_dir = PROJECT_ROOT / "outputs" / f"run_{int(time.time())}" / "seed_42"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pt"

    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.epochs - 1,
        "metrics": {"crps": avg_loss, "mae": 0.0, "rmse": 0.0, "brier_zero": 0.0},
    }, checkpoint_path)

    logger.info(f"Checkpoint saved: {checkpoint_path}")
    logger.info(f"Now run: python scripts/run_conformal_evaluation.py --data {data_name}")


if __name__ == "__main__":
    main()
