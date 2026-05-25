"""
Shared pytest fixtures for the CIVIC-SAFE Phase 0 test suite.

Provides reusable fixtures for device selection, deterministic seeding,
synthetic spatiotemporal panel data, and temporary checkpoint directories.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor


@pytest.fixture
def default_seed() -> int:
    """Return the canonical reproducibility seed used across all tests."""
    return 42


@pytest.fixture
def device() -> torch.device:
    """Return CUDA device when available, otherwise fall back to CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def tiny_panel() -> dict[str, Tensor]:
    """Generate a small spatiotemporal panel with known dimensions for testing.

    Returns a dict whose keys/values match the output contract of
    ``generate_spatiotemporal_panel`` with:
        num_spatial_units = 5
        num_time_steps    = 10
        num_categories    = 2
        num_features      = 3
        seed              = 42
    """
    from civicsafe.synthetic.distributions import generate_spatiotemporal_panel

    return generate_spatiotemporal_panel(
        num_spatial_units=5,
        num_time_steps=10,
        num_categories=2,
        num_features=3,
        seed=42,
    )


@pytest.fixture
def tmp_checkpoint_dir(tmp_path: Path) -> Path:
    """Create and return a temporary directory for checkpoint round-trip tests."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir
