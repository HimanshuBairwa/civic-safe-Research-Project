"""
Tests for civicsafe.utils.checkpointing — save/load round-trip, SHA-256
integrity verification, find_latest_checkpoint, and required-field checks.
"""

from __future__ import annotations

# Path is used at runtime below (script loading, isinstance checks), not just in
# annotations, so it cannot live under TYPE_CHECKING.
from pathlib import Path

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
        seed_state={
            "python": None,
            "numpy": None,
            "torch_cpu": torch.random.get_rng_state(),
        },
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
    assert loaded["metrics"]["loss"] == pytest.approx(
        0.1234
    ), "Loss mismatch after round-trip"

    # Verify every parameter tensor in the model state dict
    original_state = model.state_dict()
    for param_name, param_tensor in loaded["model_state_dict"].items():
        assert torch.equal(
            param_tensor, original_state[param_name]
        ), f"model_state_dict['{param_name}'] differs after round-trip"


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
    assert (
        latest_path is not None
    ), "find_latest_checkpoint returned None with 3 checkpoints"
    assert (
        "0007" in latest_path.name
    ), f"Expected latest checkpoint to be epoch 7, got {latest_path.name}"


def test_find_latest_empty_dir(tmp_checkpoint_dir: Path) -> None:
    """An empty directory must make find_latest_checkpoint return None."""
    latest_path = find_latest_checkpoint(tmp_checkpoint_dir)
    assert latest_path is None, f"Expected None for empty dir, got {latest_path}"


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------

_REQUIRED_CHECKPOINT_FIELDS = frozenset(
    {
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
        "scheduler_state_dict",
        "metrics",
        "seed_state",
        "config",
    }
)


def test_checkpoint_contains_all_fields(tmp_checkpoint_dir: Path) -> None:
    """Every saved checkpoint must contain all required metadata fields."""
    model, optimizer = _build_tiny_model_and_optimizer()
    checkpoint_data = _make_checkpoint_data(model, optimizer, epoch=10, loss=0.42)

    saved_path = save_checkpoint(checkpoint_data, tmp_checkpoint_dir, epoch=10)
    loaded = load_checkpoint(saved_path)

    missing_fields = _REQUIRED_CHECKPOINT_FIELDS - set(loaded.keys())
    assert (
        not missing_fields
    ), f"Checkpoint is missing required fields: {missing_fields}"


# ---------------------------------------------------------------------------
# Run-directory discovery: untagged (canonical) vs tagged (ablation) runs
# ---------------------------------------------------------------------------
#
# train.py names the canonical full-model run `run_{city}_{timestamp}` and every
# ablation/probe `run_{city}_{tag}_{timestamp}`. The untagged prefix is a prefix
# of every tagged one, and name-sorting puts letters after digits, so an
# unfiltered `run_chicago_*` search resolves its "latest" entry to an ablation
# once ablations exist -- reporting e.g. the no_transformer variant as the
# headline model. train.py:377 and run_ablations.py:126 filter with .isdigit();
# these tests pin the same filter into the three consumer scripts that lacked it
# (evaluate_trained, run_conformal_evaluation, emos_ensemble).


def _make_run_dirs(root: Path, names: list[str]) -> None:
    """Create run directories each holding one seed checkpoint."""
    for name in names:
        seed_dir = root / name / "seed_42"
        seed_dir.mkdir(parents=True)
        (seed_dir / "best.pt").touch()


@pytest.fixture()
def outputs_with_ablations(tmp_path: Path) -> Path:
    """An outputs/ dir where a tagged ablation sorts AFTER the canonical run."""
    _make_run_dirs(
        tmp_path,
        [
            "run_chicago_1785214452",  # canonical full model
            "run_chicago_no_gatv2_1785300000",  # ablation, sorts later by name
            "run_chicago_no_transformer_1785300001",  # ablation, sorts last
            "run_nyc_1785346837",  # other city
        ],
    )
    return tmp_path


def _latest_untagged(outputs_dir: Path, city: str) -> Path:
    """The resolution rule under test, mirroring the three fixed scripts."""
    prefix = f"run_{city}_"
    candidates = sorted(
        (
            d for d in outputs_dir.glob(f"{prefix}*")
            if d.is_dir() and d.name[len(prefix) :].isdigit()
        ),
        key=lambda p: p.name,
    )
    return candidates[-1]


def test_discovery_skips_tagged_ablation_runs(outputs_with_ablations: Path) -> None:
    """The canonical untagged run wins even when tagged dirs sort after it."""
    assert (
        _latest_untagged(outputs_with_ablations, "chicago").name
        == "run_chicago_1785214452"
    )


def test_unfiltered_discovery_would_pick_an_ablation(
    outputs_with_ablations: Path,
) -> None:
    """Negative control: without the filter the bug is real, not hypothetical."""
    unfiltered = sorted(
        outputs_with_ablations.glob("run_chicago_*"), key=lambda p: p.name
    )
    assert unfiltered[-1].name == "run_chicago_no_transformer_1785300001"


def test_discovery_is_scoped_to_the_requested_city(
    outputs_with_ablations: Path,
) -> None:
    """A bare run_* glob would return NYC for a Chicago request ('nyc' > 'chi')."""
    assert _latest_untagged(outputs_with_ablations, "nyc").name == "run_nyc_1785346837"
    bare = sorted(outputs_with_ablations.glob("run_*"), key=lambda p: p.name)
    assert bare[-1].name.startswith("run_nyc_")


def test_latest_untagged_run_wins_among_several(tmp_path: Path) -> None:
    """Among multiple untagged runs, the newest timestamp is chosen."""
    _make_run_dirs(
        tmp_path,
        ["run_chicago_1785214452", "run_chicago_1785999999", "run_chicago_1785300000"],
    )
    assert _latest_untagged(tmp_path, "chicago").name == "run_chicago_1785999999"


# ---------------------------------------------------------------------------
# The REAL discovery functions, including the Priority-2 fallback
#
# The tests above pin the resolution *rule*. These load the actual scripts and
# drive their real entry points, so a future edit to either script is caught
# even if it diverges from the rule reimplemented above.
# ---------------------------------------------------------------------------
def _load_script(name: str):
    """Import a file in scripts/ as a module (they are not an installed package)."""
    import importlib.util

    scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
    spec = importlib.util.spec_from_file_location(name, scripts_dir / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def project_with_ablations(tmp_path: Path) -> Path:
    """A PROJECT_ROOT whose outputs/ holds the canonical run plus ablations.

    Nested under its own directory (rather than renaming ``tmp_path``) because
    ``tmp_path.parent`` is shared across tests in a session -- renaming into it
    makes the second test to run collide with the first.
    """
    root = tmp_path / "root"
    (root / "outputs").mkdir(parents=True)
    _make_run_dirs(
        root / "outputs",
        [
            "run_chicago_1785214452",  # canonical full model
            "run_chicago_no_gatv2_1785300000",  # ablation, sorts later by name
            "run_chicago_no_transformer_1785300001",  # ablation, sorts last
            "run_nyc_1785346837",  # other city
        ],
    )
    return root


@pytest.mark.parametrize(
    ("script", "fn_name"),
    [
        ("evaluate_trained", "discover_checkpoint"),
        ("run_conformal_evaluation", "discover_all_checkpoints"),
    ],
)
def test_real_discovery_skips_ablations(
    project_with_ablations: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: str,
    fn_name: str,
) -> None:
    """Both real entry points resolve the canonical untagged run, not an ablation."""
    mod = _load_script(script)
    monkeypatch.setattr(mod, "PROJECT_ROOT", project_with_ablations)

    result = getattr(mod, fn_name)("chicago")
    paths = [result] if isinstance(result, Path) else result
    assert paths, f"{script}.{fn_name} found nothing"
    for p in paths:
        assert p.parent.parent.name == "run_chicago_1785214452", (
            f"{script}.{fn_name} resolved {p.parent.parent.name}"
        )


@pytest.fixture()
def legacy_outputs(tmp_path: Path) -> Path:
    """Pre-city-prefix layout: run_<digits> plus legacy ablations and a city run.

    This is what triggers the Priority-2 fallback -- there is no
    ``run_chicago_<digits>`` at all, so dataset-scoped discovery finds nothing.
    """
    nested = tmp_path / "root"
    (nested / "outputs").mkdir(parents=True)
    _make_run_dirs(
        nested / "outputs",
        [
            "run_1785100000",  # legacy full model, older
            "run_1785200000",  # legacy full model, newest -> must win
            "run_no_gatv2_1785300000",  # legacy ablation, sorts after by name
            "run_nyc_1785400000",  # another city, sorts last of all
        ],
    )
    return nested


@pytest.mark.parametrize(
    ("script", "fn_name"),
    [
        ("evaluate_trained", "discover_checkpoint"),
        ("run_conformal_evaluation", "discover_all_checkpoints"),
    ],
)
def test_fallback_path_still_excludes_tags_and_other_cities(
    legacy_outputs: Path, monkeypatch: pytest.MonkeyPatch, script: str, fn_name: str
) -> None:
    """Priority-2 must not re-introduce the bug Priority-1 fixes.

    The fallback fires exactly when no untagged run exists for the requested
    city -- i.e. mid-regeneration, when ablations may be the only dirs on disk.
    An unfiltered ``run_*`` glob there returns ``run_nyc_*`` for a Chicago
    request, because 'nyc' sorts after both digits and 'no_gatv2'.
    """
    mod = _load_script(script)
    monkeypatch.setattr(mod, "PROJECT_ROOT", legacy_outputs)

    result = getattr(mod, fn_name)("chicago")
    paths = [result] if isinstance(result, Path) else result
    assert paths, f"{script}.{fn_name} found nothing in the legacy layout"
    for p in paths:
        assert p.parent.parent.name == "run_1785200000", (
            f"{script}.{fn_name} fell through to {p.parent.parent.name}"
        )

