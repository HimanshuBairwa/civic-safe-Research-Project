"""Core Trainer for CIVIC-SAFE spatiotemporal ZINB forecaster.

Native PyTorch training loop with:
  - BFloat16 mixed precision (A100-optimized, no GradScaler needed)
  - EMA model averaging (decay=0.999)
  - Per-step cosine warmup scheduler
  - Configurable loss: CRPS-direct / ZINB NLL / blended + λ·diversity penalty
  - Gradient clipping (global norm)
  - Non-blocking GPU transfers
  - Atomic checkpoint saves
  - W&B / console logging

Why NOT PyTorch Lightning?
  The ZINB loss requires explicit float32 precision islands inside the
  autocast context, and the MFFM diversity loss needs custom weighting
  that changes across training phases. A native loop gives full control.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from civicsafe.models.zinb_loss import ZINBLoss
from civicsafe.training.early_stopping import EarlyStopping
from civicsafe.training.metrics import compute_all_metrics, crps_zinb
from civicsafe.training.sac_loss import sac_loss
from civicsafe.training.scheduler import CosineWarmupScheduler
from civicsafe.utils.seeding import seed_everything

logger = logging.getLogger(__name__)


def _worker_init_fn(worker_id: int) -> None:
    """Seed DataLoader workers for reproducibility."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


class Trainer:
    """Training orchestrator for CivicSafeModel.

    Handles the complete training lifecycle: optimizer setup, training loop,
    validation, early stopping, checkpointing, and multi-seed execution.

    Args:
        model: The CivicSafeModel to train.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        config: Training configuration dictionary (from Hydra).
        device: Target device ('cuda', 'cpu', or specific GPU).
        output_dir: Directory for checkpoints and logs.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any],
        config: dict[str, Any],
        device: torch.device | str = "cuda",
        output_dir: Path | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.output_dir = output_dir or Path("outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- Configuration ---
        train_cfg = config.get("training", config)
        self.epochs = train_cfg.get("epochs", 100)
        self.gradient_clip_norm = train_cfg.get("gradient_clip_norm", 1.0)
        self.use_mixed_precision = train_cfg.get("mixed_precision", True)
        self.diversity_lambda = train_cfg.get("diversity_lambda", 0.1)

        # r-floor regularization (Opus formula: per-cell penalty)
        # Prevents ZINB dispersion collapse that degrades CRPS while MAE improves
        self.r_reg_lambda = train_cfg.get("r_reg_lambda", 0.1)
        self.r_reg_floor = train_cfg.get("r_reg_floor", 0.5)

        # Loss function selection: 'nll', 'crps', 'blended', or 'sac'
        # CRPS-direct training eliminates the train-eval metric mismatch
        # that causes negative CRPSS when training on NLL.
        # SAC = Sharpness-Aware Calibration (CRPS + sharpness + r-reg)
        self.loss_fn = train_cfg.get("loss_fn", "crps")
        self.crps_blend_alpha = train_cfg.get("crps_blend_alpha", 0.5)
        self.sac_lambda_sharpness = train_cfg.get("sac_lambda_sharpness", 0.1)

        # --- Model ---
        self.model = model.to(self.device)

        # --- Loss ---
        self.zinb_loss = ZINBLoss(reduction="mean")

        # --- Optimizer (AdamW) ---
        opt_cfg = train_cfg.get("optimizer", {})
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_cfg.get("lr", 1e-3),
            weight_decay=opt_cfg.get("weight_decay", 1e-2),
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
        )

        # --- Scheduler (per-step cosine warmup) ---
        sched_cfg = train_cfg.get("scheduler", {})
        steps_per_epoch = len(train_loader)
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_epochs=sched_cfg.get("warmup_epochs", 5),
            total_epochs=self.epochs,
            steps_per_epoch=max(steps_per_epoch, 1),
            min_lr=sched_cfg.get("min_lr", 1e-6),
        )

        # --- EMA model (decay=0.999, uses PyTorch native API) ---
        try:
            from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

            self.ema_model: nn.Module | None = AveragedModel(
                self.model,
                multi_avg_fn=get_ema_multi_avg_fn(decay=0.999),  # type: ignore[no-untyped-call]
                use_buffers=True,
            )
        except (ImportError, TypeError, AttributeError):
            # Fallback for older PyTorch versions
            self.ema_model = None
            logger.warning("EMA not available (requires PyTorch 2.0+). Skipping.")

        # --- Early stopping ---
        es_cfg = train_cfg.get("early_stopping", {})
        self.early_stopping = EarlyStopping(
            patience=es_cfg.get("patience", 10),
            min_delta=es_cfg.get("min_delta", 1e-4),
            mode=es_cfg.get("mode", "min"),
        )

        # --- Data ---
        self.train_loader = train_loader
        self.val_loader = val_loader

        # --- Mixed precision context ---
        self._amp_enabled = self.use_mixed_precision and self.device.type == "cuda"
        # BFloat16 on A100: same dynamic range as float32, no GradScaler needed
        self._amp_dtype = torch.bfloat16 if self._amp_enabled else torch.float32

        # --- Tracking ---
        self._global_step = 0
        self._best_metrics: dict[str, float] = {}

    def fit(self) -> dict[str, Any]:
        """Execute the full training loop.

        Returns:
            Dictionary with training history and best metrics.
        """
        logger.info(
            f"Starting training: {self.epochs} epochs, "
            f"device={self.device}, amp={'bf16' if self._amp_enabled else 'off'}"
        )

        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_crps": [],
            "val_mae": [],
            "val_rmse": [],
            "val_brier": [],
            "lr": [],
        }

        for epoch in range(self.epochs):
            t0 = time.time()

            # --- Train ---
            train_loss = self._train_epoch(epoch)
            history["train_loss"].append(train_loss)

            # --- Validate (use EMA model if available) ---
            eval_model = self.ema_model if self.ema_model is not None else self.model
            val_metrics = self._val_epoch(eval_model)
            history["val_crps"].append(val_metrics["crps"])
            history["val_mae"].append(val_metrics["mae"])
            history["val_rmse"].append(val_metrics["rmse"])
            history["val_brier"].append(val_metrics["brier_zero"])
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            elapsed = time.time() - t0

            # --- Log ---
            logger.info(
                f"Epoch {epoch:3d}/{self.epochs} | "
                f"loss={train_loss:.4f} | "
                f"CRPS={val_metrics['crps']:.4f} | "
                f"MAE={val_metrics['mae']:.4f} | "
                f"RMSE={val_metrics['rmse']:.4f} | "
                f"Brier={val_metrics['brier_zero']:.4f} | "
                f"LR={self.optimizer.param_groups[0]['lr']:.2e} | "
                f"{elapsed:.1f}s"
            )

            # --- W&B logging (if available) ---
            try:
                import wandb

                if wandb.run is not None:
                    wandb.log(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            **{f"val_{k}": v for k, v in val_metrics.items()},
                            "lr": self.optimizer.param_groups[0]["lr"],
                        },
                        step=self._global_step,
                    )
            except ImportError:
                pass

            # --- Early stopping check ---
            should_stop = self.early_stopping.step(
                metric=val_metrics["crps"],
                epoch=epoch,
                model=eval_model,
            )

            # --- Save checkpoint on improvement ---
            if self.early_stopping.counter == 0:  # Just improved
                ckpt_path = self.output_dir / "best.pt"
                self.save_checkpoint(ckpt_path, epoch, val_metrics)

            if should_stop:
                logger.info(
                    f"Early stopping triggered at epoch {epoch}. "
                    f"Best CRPS = {self.early_stopping.best_score:.6f} "
                    f"at epoch {self.early_stopping.best_epoch}."
                )
                break

        # --- Restore best weights ---
        target_model = self.ema_model if self.ema_model is not None else self.model
        self.early_stopping.restore_best_weights(target_model)
        self._best_metrics = self._val_epoch(target_model)

        return {
            "history": history,
            "best_epoch": self.early_stopping.best_epoch,
            "best_metrics": self._best_metrics,
        }

    def _train_epoch(self, epoch: int) -> float:
        """Run a single training epoch.

        Args:
            epoch: Current epoch index.

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            loss = self._train_step(batch)
            total_loss += loss
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _train_step(self, batch: dict[str, Tensor]) -> float:
        """Execute a single training step.

        The CrimeWindowDataset returns samples with shape (S, W, F) where
        S = all spatial units. The DataLoader batches these into (B, S, W, F).
        Since the GNN operates on the full spatial graph per sample, we
        process each batch element individually and average the losses.

        Args:
            batch: Dictionary with keys from CrimeWindowDataset.

        Returns:
            Loss value for this step (float, detached).
        """
        # Move to GPU with non-blocking transfers (pinned memory)
        features = batch["input_features"].to(self.device, non_blocking=True)
        target_counts = batch["target_counts"].to(self.device, non_blocking=True)

        # Edge indices (shared across batch — graph is the same)
        edge_queen = batch.get("edge_queen")
        edge_knn = batch.get("edge_knn")
        if edge_queen is not None:
            edge_queen = edge_queen.to(self.device, non_blocking=True)
        if edge_knn is not None:
            edge_knn = edge_knn.to(self.device, non_blocking=True)

        B = features.shape[0]
        total_loss = torch.tensor(0.0, device=self.device)

        for i in range(B):
            feat_i = features[i]  # (S, W, F)
            target_i = target_counts[i]  # (S, C)

            # --- Forward pass with mixed precision ---
            with torch.amp.autocast(  # type: ignore[attr-defined]
                device_type=self.device.type,
                dtype=self._amp_dtype,
                enabled=self._amp_enabled,
            ):
                output = self.model(feat_i, edge_queen, edge_knn)

            # --- ZINB loss ALWAYS in float32 (critical for numerical stability) ---
            pi = output["pi"].float()
            mu = output["mu"].float()
            r = output["r"].float()
            y = target_i.float()

            # --- Primary loss computation ---
            if self.loss_fn == "nll":
                primary_loss = self.zinb_loss(
                    y.reshape(-1), pi.reshape(-1), mu.reshape(-1), r.reshape(-1)
                )
            elif self.loss_fn == "crps":
                # CRPS-direct training: eliminates train-eval metric mismatch
                # The CDF summation is fully differentiable through ZINB params
                primary_loss = crps_zinb(
                    y.reshape(-1), pi.reshape(-1), mu.reshape(-1), r.reshape(-1)
                ).mean()
            elif self.loss_fn == "blended":
                # Blended: α·CRPS + (1-α)·NLL for smooth transition
                nll_loss = self.zinb_loss(
                    y.reshape(-1), pi.reshape(-1), mu.reshape(-1), r.reshape(-1)
                )
                crps_loss = crps_zinb(
                    y.reshape(-1), pi.reshape(-1), mu.reshape(-1), r.reshape(-1)
                ).mean()
                primary_loss = (
                    self.crps_blend_alpha * crps_loss
                    + (1.0 - self.crps_blend_alpha) * nll_loss
                )
            elif self.loss_fn == "sac":
                # SAC: Sharpness-Aware Calibration (novel contribution)
                # Unified objective: CRPS + λ_s·log(1+Var) + λ_r·r_penalty
                # Implements Gneiting & Raftery (2007) 'max sharpness s.t. calibration'
                primary_loss, sac_diag = sac_loss(
                    y.reshape(-1), pi.reshape(-1), mu.reshape(-1), r.reshape(-1),
                    lambda_sharpness=self.sac_lambda_sharpness,
                    lambda_r_reg=self.r_reg_lambda,
                    r_reg_floor=self.r_reg_floor,
                )
            else:
                raise ValueError(f"Unknown loss_fn: {self.loss_fn}")

            # r-floor regularization (Opus formula: per-cell then average)
            # Penalizes each cell where r < r_floor individually, preventing
            # heavy-tailed cells from collapsing while batch mean stays safe.
            # Ref: Conflict A resolution — Opus > Fable formulation
            r_penalty = torch.nn.functional.relu(
                self.r_reg_floor - r.reshape(-1)
            ).mean()

            # Diversity regularization from MFFM
            div_loss = output.get(
                "diversity_loss", torch.tensor(0.0, device=self.device)
            )
            total_loss = (
                total_loss
                + primary_loss
                + self.diversity_lambda * div_loss
                + self.r_reg_lambda * r_penalty
            )

        # Average over batch
        total_loss = total_loss / B

        # --- Backward ---
        self.optimizer.zero_grad(set_to_none=True)
        total_loss.backward()  # type: ignore[no-untyped-call]

        # Gradient clipping (before optimizer step)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=self.gradient_clip_norm
        )

        # --- Optimizer step ---
        self.optimizer.step()

        # --- Scheduler step (per-batch, not per-epoch) ---
        self.scheduler.step()

        # --- EMA update ---
        if self.ema_model is not None:
            self.ema_model.update_parameters(self.model)  # type: ignore[operator]

        self._global_step += 1

        return total_loss.detach().item()

    @torch.inference_mode()
    def _val_epoch(self, model: nn.Module) -> dict[str, float]:
        """Run validation and compute all metrics.

        Args:
            model: Model to evaluate (may be EMA model).

        Returns:
            Dictionary of metric names → values.
        """
        model.eval()

        all_y: list[Tensor] = []
        all_pi: list[Tensor] = []
        all_mu: list[Tensor] = []
        all_r: list[Tensor] = []

        for batch in self.val_loader:
            features = batch["input_features"].to(self.device, non_blocking=True)
            target_counts = batch["target_counts"].to(self.device, non_blocking=True)

            edge_queen = batch.get("edge_queen")
            edge_knn = batch.get("edge_knn")
            if edge_queen is not None:
                edge_queen = edge_queen.to(self.device, non_blocking=True)
            if edge_knn is not None:
                edge_knn = edge_knn.to(self.device, non_blocking=True)

            B = features.shape[0]
            for i in range(B):
                output = model(features[i], edge_queen, edge_knn)

                all_y.append(target_counts[i].cpu().float().reshape(-1))
                all_pi.append(output["pi"].cpu().float().reshape(-1))
                all_mu.append(output["mu"].cpu().float().reshape(-1))
                all_r.append(output["r"].cpu().float().reshape(-1))

        # Concatenate all batches
        y = torch.cat(all_y)
        pi = torch.cat(all_pi)
        mu = torch.cat(all_mu)
        r = torch.cat(all_r)

        return compute_all_metrics(y, pi, mu, r)

    def save_checkpoint(self, path: Path, epoch: int, metrics: dict[str, Any]) -> None:
        """Save a training checkpoint with atomic write.

        Args:
            path: File path for the checkpoint.
            epoch: Current epoch number.
            metrics: Current validation metrics.
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "global_step": self._global_step,
        }

        if self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

        # Atomic save: write to tmp then rename
        tmp_path = path.with_suffix(".tmp")
        torch.save(checkpoint, tmp_path)
        tmp_path.rename(path)

        logger.debug(f"Checkpoint saved: {path}")
