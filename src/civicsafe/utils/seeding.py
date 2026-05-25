"""Reproducibility seeding: seed all PRNG sources and capture/restore state."""

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed all PRNG sources and enforce deterministic execution.

    Args:
        seed: Non-negative integer seed value. Valid range: [0, 2**32 - 1].
    """
    assert seed >= 0, f"seed={seed} is negative; must be >= 0"

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def get_seed_state() -> dict[str, Any]:
    """Capture full PRNG state from all frameworks for checkpointing.

    Returns:
        Dictionary with keys 'python', 'numpy', 'torch_cpu', and optionally
        'torch_cuda' mapping to their respective PRNG states.
    """
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_seed_state(state: dict[str, Any]) -> None:
    """Restore PRNG state from a previously captured dictionary.

    Args:
        state: Dictionary produced by :func:`get_seed_state`.
    """
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])
