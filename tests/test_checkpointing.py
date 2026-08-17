"""
Tests for civicsafe.utils.checkpointing — save/load round-trip, SHA-256
integrity verification, find_latest_checkpoint, and required-field checks.
"""

from __future__ import annotations

# Path is used at runtime below (script loading, isinstance checks), not just in
# annotations, so it cannot live under TYPE_CHECKING.
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from civicsafe.utils.checkpointing import (
    CheckpointData,
    discover_evaluation_checkpoints,
    find_latest_checkpoint,
    load_checkpoint,
    resolve_evaluation_checkpoints,
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


def _make_checkpoint_run(
    root: Path,
    name: str,
    *,
    seeds: tuple[int, ...] = (42,),
    arch: dict | None = None,
) -> Path:
    """Create a run with minimally loadable metadata checkpoints."""
    run_dir = root / name
    for seed in seeds:
        seed_dir = run_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        torch.save({"arch": arch or {}}, seed_dir / "best.pt")
    return run_dir


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


# ---------------------------------------------------------------------------
# Architecture-aware discovery and explicit directory expansion
# ---------------------------------------------------------------------------


def test_auto_discovery_prefers_level_anchor_over_newer_legacy_run(
    tmp_path: Path,
) -> None:
    """A recorded level anchor outranks a newer unanchored checkpoint run."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    _make_checkpoint_run(
        outputs,
        "run_chicago_anchor_1786718493",
        seeds=(42, 137, 256),
        arch={"level_anchor": True},
    )
    _make_checkpoint_run(
        outputs,
        "run_chicago_1786999999",
        seeds=(42,),
        arch={},
    )

    checkpoints = discover_evaluation_checkpoints(outputs, "chicago")

    assert [path.parent.name for path in checkpoints] == [
        "seed_42",
        "seed_137",
        "seed_256",
    ]
    assert all(
        path.parent.parent.name == "run_chicago_anchor_1786718493"
        for path in checkpoints
    )


def test_auto_discovery_prefers_recorded_anchor_without_anchor_tag(
    tmp_path: Path,
) -> None:
    """Checkpoint metadata is stronger than the directory naming convention."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    _make_checkpoint_run(
        outputs,
        "run_nyc_1786812069",
        arch={"level_anchor": True},
    )
    _make_checkpoint_run(
        outputs,
        "run_nyc_1786999999",
        arch={},
    )

    checkpoints = discover_evaluation_checkpoints(outputs, "nyc")

    assert checkpoints[0].parent.parent.name == "run_nyc_1786812069"


def test_auto_discovery_uses_newest_run_among_recorded_anchors(
    tmp_path: Path,
) -> None:
    """Once runs are architecture-qualified, the newest timestamp wins."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    _make_checkpoint_run(
        outputs,
        "run_chicago_anchor_1786000000",
        arch={"level_anchor": True},
    )
    _make_checkpoint_run(
        outputs,
        "run_chicago_1787000000",
        arch={"level_anchor": True},
    )

    checkpoints = discover_evaluation_checkpoints(outputs, "chicago")

    assert checkpoints[0].parent.parent.name == "run_chicago_1787000000"


def test_auto_discovery_excludes_anchor_named_architecture_ablation(
    tmp_path: Path,
) -> None:
    """An anchor-looking tag cannot make a disabled-core architecture eligible."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    _make_checkpoint_run(
        outputs,
        "run_chicago_1786000000",
        arch={"level_anchor": True},
    )
    _make_checkpoint_run(
        outputs,
        "run_chicago_anchor_1787000000",
        arch={"level_anchor": True, "use_transformer": False},
    )

    checkpoints = discover_evaluation_checkpoints(outputs, "chicago")

    assert checkpoints[0].parent.parent.name == "run_chicago_1786000000"


def test_resolve_run_directory_expands_all_seed_checkpoints(tmp_path: Path) -> None:
    """Explicit run directories must expand to every numeric seed deterministically."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    run_dir = _make_checkpoint_run(
        outputs,
        "run_chicago_anchor_1786718493",
        seeds=(1024, 42, 256, 137, 512),
        arch={"level_anchor": True},
    )

    checkpoints = resolve_evaluation_checkpoints(
        run_dir,
        data_name="chicago",
        outputs_dir=outputs,
    )

    assert [path.parent.name for path in checkpoints] == [
        "seed_42",
        "seed_137",
        "seed_256",
        "seed_512",
        "seed_1024",
    ]


@pytest.mark.parametrize("use_seed_directory", [False, True])
def test_resolve_explicit_file_or_seed_directory(
    tmp_path: Path, use_seed_directory: bool
) -> None:
    """Single-file evaluation remains supported alongside run ensembles."""
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    run_dir = _make_checkpoint_run(
        outputs,
        "run_chicago_anchor_1786718493",
        arch={"level_anchor": True},
    )
    checkpoint = run_dir / "seed_42" / "best.pt"
    checkpoint_input = checkpoint.parent if use_seed_directory else checkpoint

    resolved = resolve_evaluation_checkpoints(
        checkpoint_input,
        data_name="chicago",
        outputs_dir=outputs,
    )

    assert resolved == [checkpoint]


@pytest.mark.parametrize(
    ("script", "fn_name"),
    [
        ("evaluate_trained", "resolve_checkpoints"),
        ("run_conformal_evaluation", "resolve_checkpoints"),
    ],
)
def test_both_evaluators_accept_explicit_run_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    script: str,
    fn_name: str,
) -> None:
    """Both CLI backends expand a run directory before any torch.load call."""
    project_root = tmp_path / "project"
    outputs = project_root / "outputs"
    outputs.mkdir(parents=True)
    run_dir = _make_checkpoint_run(
        outputs,
        "run_chicago_anchor_1786718493",
        seeds=(42, 137),
        arch={"level_anchor": True},
    )
    module = _load_script(script)
    monkeypatch.setattr(module, "PROJECT_ROOT", project_root)

    checkpoints = getattr(module, fn_name)(str(run_dir), "chicago")

    assert [path.parent.name for path in checkpoints] == ["seed_42", "seed_137"]


def test_empty_explicit_run_directory_has_actionable_error(tmp_path: Path) -> None:
    """An empty directory must fail before it can reach torch.load(directory)."""
    outputs = tmp_path / "outputs"
    run_dir = outputs / "run_chicago_anchor_1786718493"
    run_dir.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match=r"seed_\*/best\.pt"):
        resolve_evaluation_checkpoints(
            run_dir,
            data_name="chicago",
            outputs_dir=outputs,
        )


def test_campaign_discovers_anchor_run_for_both_evaluators(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The campaign resolves an anchored run instead of delegating stale auto picks."""
    project_root = tmp_path / "project"
    outputs = project_root / "outputs"
    outputs.mkdir(parents=True)
    anchor_run = _make_checkpoint_run(
        outputs,
        "run_chicago_anchor_1786718493",
        seeds=(42, 137),
        arch={"level_anchor": True},
    )
    _make_checkpoint_run(
        outputs,
        "run_chicago_1786999999",
        arch={},
    )
    campaign = _load_script("run_full_campaign")
    monkeypatch.setattr(campaign, "ROOT", project_root)

    assert campaign._discover_evaluation_run("chicago") == anchor_run


def test_evaluate_trained_runs_every_seed_and_records_ensemble(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run-directory evaluation must load every seed and emit ensemble metadata."""
    project_root = tmp_path / "project"
    outputs = project_root / "outputs"
    outputs.mkdir(parents=True)
    run_dir = _make_checkpoint_run(
        outputs,
        "run_chicago_anchor_1786718493",
        seeds=(42, 137),
        arch={"level_anchor": True},
    )
    module = _load_script("evaluate_trained")
    monkeypatch.setattr(module, "PROJECT_ROOT", project_root)

    loaded_paths: list[Path] = []

    class _FakeModel:
        def __init__(self, marker: float) -> None:
            self.marker = marker

        def load_state_dict(self, _state, strict: bool = False):
            del strict
            return [], []

        def to(self, _device):
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter(())

    def _fake_load_checkpoint(path, _data_name, weights="auto"):
        del weights
        checkpoint = Path(path)
        loaded_paths.append(checkpoint)
        seed = int(checkpoint.parent.name.removeprefix("seed_"))
        return {}, checkpoint, "raw_toplevel", {"seed_marker": seed}

    def _fake_build_model(num_features, num_categories, arch=None):
        del num_features, num_categories
        return _FakeModel(float((arch or {})["seed_marker"]))

    def _fake_rolling(model, *_args, start_week, **_kwargs):
        marker = model.marker / 1000.0
        return {
            "y_true": torch.ones(1, 1, 1),
            "pi": torch.full((1, 1, 1), 0.1),
            "mu": torch.full((1, 1, 1), 1.0 + marker),
            "r": torch.full((1, 1, 1), 2.0),
            "week_idx": torch.tensor([start_week]),
        }

    monkeypatch.setattr(
        module,
        "load_data",
        lambda _data: (
            torch.zeros(1, 313, 1),
            torch.zeros(1, 313, 1),
            torch.empty(2, 0, dtype=torch.long),
            None,
        ),
    )
    monkeypatch.setattr(module, "load_checkpoint", _fake_load_checkpoint)
    monkeypatch.setattr(module, "build_model", _fake_build_model)
    monkeypatch.setattr(module, "rolling_evaluate", _fake_rolling)
    monkeypatch.setattr(
        module,
        "conformal_evaluation",
        lambda *_args, **_kwargs: {
            "test_coverage": 0.9,
            "target_coverage": 0.9,
            "avg_interval_width": 2.0,
        },
    )

    output_path = tmp_path / "ensemble_results.json"
    args = SimpleNamespace(
        data="chicago",
        checkpoint=str(run_dir),
        weights="raw",
        alpha=0.1,
        output=str(output_path),
    )
    results = module.run_evaluation(args)

    assert [path.parent.name for path in loaded_paths] == ["seed_42", "seed_137"]
    assert results["metadata"]["num_ensemble_seeds"] == 2
    assert len(results["ensemble"]["emos_weights"]) == 2
    assert output_path.is_file()
