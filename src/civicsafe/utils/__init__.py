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
    "LOG_FLOOR",
    # numerics
    "NUMERICAL_EPS",
    "CheckpointCorruptionError",
    # checkpointing
    "CheckpointData",
    # exceptions
    "CivicSafeError",
    "DataValidationError",
    "KillCriterionTriggered",
    "NumericalInstabilityError",
    "clamp_probabilities",
    "find_latest_checkpoint",
    "get_seed_state",
    "load_checkpoint",
    "log_metrics",
    "log_sum_exp",
    "safe_divide",
    "safe_log",
    "save_checkpoint",
    # seeding
    "seed_everything",
    "set_seed_state",
    # logging
    "setup_logger",
]
