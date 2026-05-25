"""Checkpoint save/resume with SHA-256 corruption detection."""

import hashlib
from pathlib import Path
from typing import Any, TypedDict

import torch

from civicsafe.utils.exceptions import CheckpointCorruptionError

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CHECKPOINT_GLOB: str = "checkpoint_epoch_*.pt"
"""Glob pattern used to discover checkpoint files in a directory."""

SHA256_SUFFIX: str = ".sha256"
"""Sidecar file extension storing the hex digest of the checkpoint."""


class CheckpointData(TypedDict):
    """Typed schema for everything persisted in a checkpoint."""

    epoch: int
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    scheduler_state_dict: dict[str, Any] | None
    metrics: dict[str, float]
    seed_state: dict[str, Any]
    config: dict[str, Any]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sha256_of_file(filepath: Path) -> str:
    """Compute hex SHA-256 digest of *filepath* in streaming fashion."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as file_handle:
        while chunk := file_handle.read(1 << 16):  # 64 KiB blocks
            hasher.update(chunk)
    return hasher.hexdigest()


def _checkpoint_path(directory: Path, epoch: int) -> Path:
    """Build the canonical checkpoint filename for a given epoch."""
    return directory / f"checkpoint_epoch_{epoch:04d}.pt"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def save_checkpoint(
    checkpoint_data: CheckpointData,
    directory: Path,
    epoch: int,
) -> Path:
    """Persist a checkpoint to disk with a SHA-256 sidecar.

    Args:
        checkpoint_data: Full training state conforming to :class:`CheckpointData`.
        directory: Target directory (created if absent).
        epoch: Epoch number, used to format the filename.

    Returns:
        Path to the saved ``.pt`` file.
    """
    directory.mkdir(parents=True, exist_ok=True)
    filepath: Path = _checkpoint_path(directory, epoch)

    torch.save(dict(checkpoint_data), filepath)

    digest: str = _sha256_of_file(filepath)
    filepath.with_suffix(filepath.suffix + SHA256_SUFFIX).write_text(
        digest, encoding="utf-8"
    )
    return filepath


def load_checkpoint(path: Path) -> CheckpointData:
    """Load a checkpoint and verify its SHA-256 integrity.

    Args:
        path: Path to the ``.pt`` checkpoint file.

    Returns:
        Deserialized :class:`CheckpointData`.

    Raises:
        CheckpointCorruptionError: If the sidecar hash does not match.
        FileNotFoundError: If the checkpoint or sidecar is missing.
    """
    sidecar: Path = path.with_suffix(path.suffix + SHA256_SUFFIX)
    expected_hash: str = sidecar.read_text(encoding="utf-8").strip()
    actual_hash: str = _sha256_of_file(path)

    if actual_hash != expected_hash:
        raise CheckpointCorruptionError(str(path))

    loaded: dict[str, Any] = torch.load(path, weights_only=False)
    from typing import cast
    return cast(CheckpointData, loaded)


def find_latest_checkpoint(directory: Path) -> Path | None:
    """Return the highest-epoch checkpoint in *directory*, or None.

    Args:
        directory: Directory to scan for checkpoint files.

    Returns:
        Path to the latest checkpoint, or ``None`` if *directory* is empty
        or contains no matching files.
    """
    if not directory.is_dir():
        return None

    candidates: list[Path] = sorted(directory.glob(CHECKPOINT_GLOB))
    if not candidates:
        return None
    return candidates[-1]
