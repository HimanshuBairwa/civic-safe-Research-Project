"""
Tests for civicsafe.utils.checkpointing — save/load round-trip, SHA-256
integrity verification, find_latest_checkpoint, and required-field checks.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from civicsafe.utils.checkpointing import (
    CheckpointData,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from civicsafe.utils.exceptions import CheckpointCorruptionError


# ---------------------------------------------------------------------------
# Helpers — tiny model + optimizer + checkpoint factory
# ---------------------------------------------------------------------------

def _build_tiny_model_and_optimizer() -> tuple[nn.Module, optim.Optimizer]:
    """Create a trivially small Linear model and Adam optimizer."""
    model = nn.Linear(in_features=4, out_features=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    return model, optimizer


def _make_checkpoint_data(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    loss: float = 0.1234,
) -> CheckpointData:
    """Build a CheckpointData dict from model/optimizer state."""
    return CheckpointData(
        epoch=epoch,
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=None,
        metrics={"loss": loss},
        seed_state={"python": None, "numpy": None, "torch_cpu": torch.random.get_rng_state()},
        config={"lr": 1e-3},
    )


# ---------------------------------------------------------------------------
# Round-trip: save → load → verify equality
# ---------------------------------------------------------------------------

def test_save_load_roundtrip(tmp_checkpoint_dir: Path) -> None:
    """Saving then loading a checkpoint must reproduce every stored field."""
    model, optimizer = _build_tiny_model_and_optimizer()
    checkpoint_data = _make_checkpoint_data(model, optimizer, epoch=5, loss=0.1234)

    saved_path = save_checkpoint(checkpoint_data, tmp_checkpoint_dir, epoch=5)

    loaded = load_checkpoint(saved_path)

    assert loaded["epoch"] == 5, "Epoch mismatch after round-trip"
    assert loaded["metrics"]["loss"] == pytest.approx(0.1234), "Loss mismatch after round-trip"

    # Verify every parameter tensor in the model state dict
    original_state = model.state_dict()
    for param_name, param_tensor in loaded["model_state_dict"].items():
        assert torch.equal(param_tensor, original_state[param_name]), (
            f"model_state_dict['{param_name}'] differs after round-trip"
        )


# ---------------------------------------------------------------------------
# SHA-256 corruption detection
# ---------------------------------------------------------------------------

def test_sha256_verification(tmp_checkpoint_dir: Path) -> None:
    """Corrupting the checkpoint file on disk must raise CheckpointCorruptionError."""
    model, optimizer = _build_tiny_model_and_optimizer()
    checkpoint_data = _make_checkpoint_data(model, optimizer, epoch=1, loss=0.5)

    saved_path = save_checkpoint(checkpoint_data, tmp_checkpoint_dir, epoch=1)

    # Corrupt the .pt file by appending garbage bytes
    with open(saved_path, "ab") as fh:
        fh.write(b"\x00\xff" * 128)

    with pytest.raises(CheckpointCorruptionError):
        load_checkpoint(saved_path)


# ---------------------------------------------------------------------------
# find_latest_checkpoint — multiple epochs
# ---------------------------------------------------------------------------

def test_find_latest_checkpoint(tmp_checkpoint_dir: Path) -> None:
    """find_latest_checkpoint must return the highest-epoch path."""
    model, optimizer = _build_tiny_model_and_optimizer()

    for epoch_number in (1, 3, 7):
        checkpoint_data = _make_checkpoint_data(
            model, optimizer, epoch=epoch_number, loss=float(epoch_number)
        )
        save_checkpoint(checkpoint_data, tmp_checkpoint_dir, epoch=epoch_number)

    latest_path = find_latest_checkpoint(tmp_checkpoint_dir)
    assert latest_path is not None, "find_latest_checkpoint returned None with 3 checkpoints"
    assert "0007" in latest_path.name, (
        f"Expected latest checkpoint to be epoch 7, got {latest_path.name}"
    )


def test_find_latest_empty_dir(tmp_checkpoint_dir: Path) -> None:
    """An empty directory must make find_latest_checkpoint return None."""
    latest_path = find_latest_checkpoint(tmp_checkpoint_dir)
    assert latest_path is None, (
        f"Expected None for empty dir, got {latest_path}"
    )


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------

_REQUIRED_CHECKPOINT_FIELDS = frozenset({
    "epoch",
    "model_state_dict",
    "optimizer_state_dict",
    "scheduler_state_dict",
    "metrics",
    "seed_state",
    "config",
})


def test_checkpoint_contains_all_fields(tmp_checkpoint_dir: Path) -> None:
    """Every saved checkpoint must contain all required metadata fields."""
    model, optimizer = _build_tiny_model_and_optimizer()
    checkpoint_data = _make_checkpoint_data(model, optimizer, epoch=10, loss=0.42)

    saved_path = save_checkpoint(checkpoint_data, tmp_checkpoint_dir, epoch=10)
    loaded = load_checkpoint(saved_path)

    missing_fields = _REQUIRED_CHECKPOINT_FIELDS - set(loaded.keys())
    assert not missing_fields, (
        f"Checkpoint is missing required fields: {missing_fields}"
    )
