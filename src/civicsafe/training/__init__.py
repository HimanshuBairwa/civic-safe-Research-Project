"""Training infrastructure for CIVIC-SAFE spatiotemporal forecaster.

Provides the complete training lifecycle: metrics computation, early stopping,
learning-rate scheduling, and the core Trainer class for ZINB-based
spatiotemporal GNN training.
"""

from __future__ import annotations

from civicsafe.training.early_stopping import EarlyStopping
from civicsafe.training.metrics import (
    brier_zero_inflation,
    crps_zinb,
    mae_zinb,
    rmse_zinb,
)
from civicsafe.training.scheduler import CosineWarmupScheduler
from civicsafe.training.trainer import Trainer

__all__: list[str] = [
    "crps_zinb",
    "mae_zinb",
    "rmse_zinb",
    "brier_zero_inflation",
    "EarlyStopping",
    "CosineWarmupScheduler",
    "Trainer",
]
