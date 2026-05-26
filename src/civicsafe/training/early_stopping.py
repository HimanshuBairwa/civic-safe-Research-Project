"""Early stopping with best-model tracking for ZINB training.

Monitors a validation metric (default: CRPS) and triggers training
termination when improvement stalls. Integrates with the existing
SHA-256 checkpointing system for robust model persistence.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Patience-based early stopping with best-weight restoration.

    Args:
        patience: Number of epochs without improvement before stopping.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' if lower metric is better (CRPS, NLL), 'max' if higher.
        checkpoint_dir: Directory for saving the best model checkpoint.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "min",
        checkpoint_dir: Path | None = None,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.checkpoint_dir = checkpoint_dir

        self._best_score: float | None = None
        self._counter: int = 0
        self._best_epoch: int = -1
        self._best_state: dict[str, Any] | None = None
        self._stopped: bool = False

        # For 'min' mode, improvement means score decreased by at least min_delta.
        # For 'max' mode, improvement means score increased by at least min_delta.
        self._is_better: Callable[[float, float], bool]
        if mode == "min":
            self._is_better = lambda new, best: new < best - min_delta
        else:
            self._is_better = lambda new, best: new > best + min_delta

    @property
    def best_score(self) -> float | None:
        """Best metric value seen so far."""
        return self._best_score

    @property
    def best_epoch(self) -> int:
        """Epoch index of the best metric value."""
        return self._best_epoch

    @property
    def should_stop(self) -> bool:
        """Whether the patience budget has been exhausted."""
        return self._stopped

    @property
    def counter(self) -> int:
        """Number of epochs since last improvement."""
        return self._counter

    def step(
        self,
        metric: float,
        epoch: int,
        model: torch.nn.Module,
    ) -> bool:
        """Record a new metric observation.

        Args:
            metric: Validation metric value for this epoch.
            epoch: Current epoch index (0-based).
            model: The model whose state_dict to snapshot on improvement.

        Returns:
            True if training should stop (patience exhausted).
        """
        if self._best_score is None or self._is_better(metric, self._best_score):
            # Improvement detected
            self._best_score = metric
            self._best_epoch = epoch
            self._counter = 0

            # Snapshot best weights (deep copy to CPU to avoid GPU memory waste)
            self._best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            logger.info(
                f"  EarlyStopping: new best {self._metric_name} = {metric:.6f} "
                f"at epoch {epoch}"
            )
        else:
            self._counter += 1
            logger.info(
                f"  EarlyStopping: no improvement for {self._counter}/{self.patience} "
                f"epochs (best = {self._best_score:.6f})"
            )
            if self._counter >= self.patience:
                self._stopped = True
                logger.warning(
                    f"  EarlyStopping: patience exhausted at epoch {epoch}. "
                    f"Best was epoch {self._best_epoch}."
                )

        return self._stopped

    def restore_best_weights(self, model: torch.nn.Module) -> None:
        """Load the best-epoch weights back into the model.

        Args:
            model: Model to load weights into.
        """
        if self._best_state is not None:
            model.load_state_dict(self._best_state)
            logger.info(
                f"  Restored best weights from epoch {self._best_epoch} "
                f"({self._metric_name} = {self._best_score:.6f})"
            )
        else:
            logger.warning("  No best state to restore (step() was never called).")

    def reset(self) -> None:
        """Reset the stopping state for a new training run (new seed)."""
        self._best_score = None
        self._counter = 0
        self._best_epoch = -1
        self._best_state = None
        self._stopped = False

    @property
    def _metric_name(self) -> str:
        return "metric"
