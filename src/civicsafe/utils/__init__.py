"""Public API surface for civicsafe.utils."""

from civicsafe.utils.checkpointing import (
    CheckpointData,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from civicsafe.utils.exceptions import (
    CheckpointCorruptionError,
    CivicSafeError,
    DataValidationError,
    KillCriterionTriggered,
    NumericalInstabilityError,
)
from civicsafe.utils.logging import log_metrics, setup_logger
from civicsafe.utils.numerics import (
    LOG_FLOOR,
    NUMERICAL_EPS,
    clamp_probabilities,
    log_sum_exp,
    safe_divide,
    safe_log,
)
from civicsafe.utils.seeding import get_seed_state, seed_everything, set_seed_state

__all__: list[str] = [
    # seeding
    "seed_everything",
    "get_seed_state",
    "set_seed_state",
    # exceptions
    "CivicSafeError",
    "KillCriterionTriggered",
    "NumericalInstabilityError",
    "DataValidationError",
    "CheckpointCorruptionError",
    # numerics
    "NUMERICAL_EPS",
    "LOG_FLOOR",
    "safe_log",
    "safe_divide",
    "log_sum_exp",
    "clamp_probabilities",
    # checkpointing
    "CheckpointData",
    "save_checkpoint",
    "load_checkpoint",
    "find_latest_checkpoint",
    # logging
    "setup_logger",
    "log_metrics",
]
