"""Per-step cosine warmup learning-rate scheduler.

Linear warmup from 0 to peak LR, then cosine decay to min_lr.
Stepped per *batch* (not per epoch) for smoother loss curves,
which is critical for ZINB optimization stability.

Reference: Loshchilov & Hutter (2019), "Decoupled Weight Decay
Regularization" — cosine schedule variant.
"""

from __future__ import annotations

import math
from typing import Any

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def create_cosine_warmup_scheduler(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr: float = 1e-6,
) -> LambdaLR:
    """Create a per-step cosine warmup scheduler.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of steps for linear warmup (0 → peak LR).
        total_steps: Total training steps (warmup + cosine decay).
        min_lr: Minimum learning rate at end of cosine decay.

    Returns:
        A LambdaLR scheduler that should be stepped every training batch.
    """
    base_lr = optimizer.defaults["lr"]

    # Prevent division by zero if min_lr >= base_lr
    min_lr_ratio = min(min_lr / max(base_lr, 1e-10), 1.0)

    def lr_lambda(current_step: int) -> float:
        """Compute LR multiplier for a given step."""
        if current_step < warmup_steps:
            # Linear warmup: 0 → 1
            return current_step / max(1, warmup_steps)

        # Cosine decay: 1 → min_lr_ratio
        progress = (current_step - warmup_steps) / max(
            1, total_steps - warmup_steps
        )
        progress = min(progress, 1.0)  # Clamp for steps beyond total
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))

        return float(max(min_lr_ratio, cosine_factor))

    return LambdaLR(optimizer, lr_lambda)


class CosineWarmupScheduler:
    """Convenience wrapper around create_cosine_warmup_scheduler.

    Computes warmup_steps and total_steps from epoch-level configs
    and the number of batches per epoch.

    Args:
        optimizer: The optimizer to schedule.
        warmup_epochs: Number of warmup epochs.
        total_epochs: Total training epochs.
        steps_per_epoch: Number of batches per epoch.
        min_lr: Minimum learning rate.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        steps_per_epoch: int,
        min_lr: float = 1e-6,
    ) -> None:
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        self.scheduler = create_cosine_warmup_scheduler(
            optimizer,
            warmup_steps=self.warmup_steps,
            total_steps=self.total_steps,
            min_lr=min_lr,
        )

    def step(self) -> None:
        """Step the scheduler (call after every training batch)."""
        self.scheduler.step()

    def get_last_lr(self) -> list[Any]:
        """Return the last computed learning rate."""
        return self.scheduler.get_last_lr()

    def state_dict(self) -> dict[str, Any]:
        """Return scheduler state for checkpointing."""
        return self.scheduler.state_dict()  # type: ignore[no-untyped-call,no-any-return]

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore scheduler state from checkpoint."""
        self.scheduler.load_state_dict(state)  # type: ignore[no-untyped-call]
