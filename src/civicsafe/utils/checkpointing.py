"""Checkpoint persistence and evaluation-time checkpoint resolution."""

import hashlib
import logging
from pathlib import Path
from typing import Any, TypedDict, cast

import torch

from civicsafe.utils.exceptions import CheckpointCorruptionError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CHECKPOINT_GLOB: str = "checkpoint_epoch_*.pt"
"""Glob pattern used to discover checkpoint files in a directory."""

SHA256_SUFFIX: str = ".sha256"
"""Sidecar file extension storing the hex digest of the checkpoint."""

_ABLATION_RUN_TAGS = frozenset(
    {
        "nb_only",
        "no_level_anchor",
        "no_r_reg",
        "no_sharpness",
    }
)
"""Run tags that do not change the architecture fingerprint enough to detect."""


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


def _seed_sort_key(path: Path) -> tuple[int, str]:
    """Sort ``seed_<integer>`` directories numerically and deterministically."""
    seed_name = path.parent.name
    if seed_name.startswith("seed_") and seed_name[5:].isdigit():
        return int(seed_name[5:]), str(path)
    return 2**63 - 1, str(path)


def _checkpoints_in_directory(directory: Path) -> list[Path]:
    """Return evaluation checkpoints represented by a run or seed directory."""
    direct = directory / "best.pt"
    if direct.is_file():
        return [direct]
    return sorted(
        (path for path in directory.glob("seed_*/best.pt") if path.is_file()),
        key=_seed_sort_key,
    )


def _read_arch_fingerprint(path: Path) -> dict[str, Any]:
    """Read a checkpoint architecture fingerprint, tolerating legacy files."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        logger.debug("Could not inspect checkpoint metadata at %s: %s", path, exc)
        return {}

    if not isinstance(checkpoint, dict):
        return {}
    arch = checkpoint.get("arch", {})
    return dict(arch) if isinstance(arch, dict) else {}


def _parse_dataset_run_name(name: str, data_name: str) -> tuple[str, int] | None:
    """Parse ``run_<city>_[tag_]timestamp`` into ``(tag, timestamp)``."""
    prefix = f"run_{data_name}_"
    if not name.startswith(prefix):
        return None

    remainder = name[len(prefix) :]
    parts = remainder.split("_")
    if not parts or not parts[-1].isdigit():
        return None
    return "_".join(parts[:-1]), int(parts[-1])


def _parse_legacy_run_name(name: str) -> tuple[str, int] | None:
    """Parse the pre-city layout ``run_<timestamp>``."""
    remainder = name.removeprefix("run_")
    if name.startswith("run_") and remainder.isdigit():
        return "", int(remainder)
    return None


def _is_ablation_tag(tag: str) -> bool:
    """Return whether a run tag identifies an ablation, probe, or test run."""
    lowered = tag.lower()
    return (
        lowered.startswith("no_")
        or lowered.startswith("loss_")
        or lowered.startswith(("ablation", "probe", "debug", "test"))
        or lowered in _ABLATION_RUN_TAGS
    )


def _is_ablated_architecture(arch: dict[str, Any]) -> bool:
    """Detect architecture ablations recorded in a trainer fingerprint."""
    return any(
        key in arch and not bool(arch[key])
        for key in ("use_gnn", "use_transformer", "zero_inflation")
    )


def _rank_run(
    run_dir: Path,
    checkpoints: list[Path],
    tag: str,
    timestamp: int,
) -> tuple[int, int, str] | None:
    """Rank a candidate run, excluding ablations before selection."""
    if _is_ablation_tag(tag):
        return None

    architectures = [_read_arch_fingerprint(path) for path in checkpoints]
    if any(_is_ablated_architecture(arch) for arch in architectures):
        return None

    # A recorded level anchor is stronger evidence than the directory tag.
    # The tag remains a compatibility signal for partially written or older
    # anchor runs whose checkpoint metadata cannot be inspected.
    has_level_anchor = any(arch.get("level_anchor") is True for arch in architectures)
    anchor_priority = 2 if has_level_anchor else int(tag.lower() == "anchor")
    return anchor_priority, timestamp, run_dir.name


def discover_evaluation_checkpoints(outputs_dir: Path, data_name: str) -> list[Path]:
    """Discover every seed checkpoint in the best eligible run for a dataset.

    Dataset-scoped runs are considered before legacy ``run_<timestamp>`` runs.
    Eligible runs are ranked by recorded ``level_anchor=True`` first, then by
    an ``anchor`` run tag, then by timestamp. Known ablations and architectures
    with disabled core components are never eligible.
    """
    if not outputs_dir.is_dir():
        raise FileNotFoundError(f"No outputs directory at {outputs_dir}")

    candidates: list[tuple[tuple[int, int, str], Path, list[Path]]] = []
    for run_dir in outputs_dir.iterdir():
        if not run_dir.is_dir():
            continue
        parsed = _parse_dataset_run_name(run_dir.name, data_name)
        if parsed is None:
            continue
        checkpoints = _checkpoints_in_directory(run_dir)
        if not checkpoints:
            continue
        tag, timestamp = parsed
        rank = _rank_run(run_dir, checkpoints, tag, timestamp)
        if rank is not None:
            candidates.append((rank, run_dir, checkpoints))

    # Legacy runs predate dataset-scoped directory names. They are considered
    # only when no city-specific run is available, avoiding cross-city picks.
    if not candidates:
        for run_dir in outputs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            parsed = _parse_legacy_run_name(run_dir.name)
            if parsed is None:
                continue
            checkpoints = _checkpoints_in_directory(run_dir)
            if not checkpoints:
                continue
            tag, timestamp = parsed
            rank = _rank_run(run_dir, checkpoints, tag, timestamp)
            if rank is not None:
                candidates.append((rank, run_dir, checkpoints))

    if not candidates:
        raise FileNotFoundError(
            f"No eligible seed_*/best.pt checkpoints found for {data_name!r} "
            f"under {outputs_dir}."
        )

    _, selected_run, selected_checkpoints = max(candidates, key=lambda item: item[0])
    logger.info(
        "Auto-discovered %d checkpoint(s) in %s",
        len(selected_checkpoints),
        selected_run,
    )
    return selected_checkpoints


def resolve_evaluation_checkpoints(
    checkpoint: str | Path | None,
    *,
    data_name: str,
    outputs_dir: Path,
) -> list[Path]:
    """Resolve an explicit checkpoint file, run directory, seed directory, or auto.

    ``None`` and the string ``"auto"`` select the best eligible run. A run
    directory expands to every ``seed_*/best.pt`` member, while a seed directory
    containing ``best.pt`` resolves to that single file.
    """
    if checkpoint is None or str(checkpoint).lower() == "auto":
        return discover_evaluation_checkpoints(outputs_dir, data_name)

    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path not found: {path}")
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"Checkpoint path is neither a file nor directory: {path}")

    checkpoints = _checkpoints_in_directory(path)
    if not checkpoints:
        raise FileNotFoundError(
            f"Checkpoint directory {path} contains neither best.pt nor "
            "seed_*/best.pt files."
        )
    return checkpoints


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
    return cast("CheckpointData", loaded)


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
