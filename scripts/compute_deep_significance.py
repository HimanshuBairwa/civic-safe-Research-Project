#!/usr/bin/env python
"""Post-training paired significance tests against deep baselines.

This utility consumes persisted *per-week* CRPS series.  It never opens a
checkpoint and never trains a model.  CIVIC-SAFE's series may be stored in a
conformal JSON block, an evaluation JSON block, or a JSON/NPZ sidecar.  Deep
baseline series are read from every city-qualified seed file and averaged by
absolute week index before comparison.

The script is deliberately fail-closed for a missing CIVIC-SAFE loss series:
an aggregate CRPS scalar or a violent-crime policy panel cannot support a
paired all-category DM test.  In that case an auditable ``status=unavailable``
sidecar is written and the process exits successfully, so publication table
generation can still report the exact source limitation rather than silently
inventing p-values.

Usage::

    python scripts/compute_deep_significance.py
    python scripts/compute_deep_significance.py --data chicago --data nyc
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from civicsafe.calibration.significance import compare_forecasts  # noqa: E402

logger = logging.getLogger(__name__)
MIN_OVERLAP = 10
DEEP_MODELS = ("LSTM_NB", "TFT_ZINB", "GraphWaveNet", "STZINB_GNN")


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return None
    return value if isinstance(value, dict) else None


def _series(block: Any) -> tuple[list[int], list[float]] | None:
    """Extract and validate an absolute-week CRPS series from a JSON block."""
    if not isinstance(block, dict):
        return None
    weeks = block.get("week_index", block.get("weeks"))
    values = block.get("crps", block.get("crps_values"))
    if weeks is None or values is None:
        return None
    try:
        weeks = [int(x) for x in weeks]
        values = [float(x) for x in values]
    except (TypeError, ValueError):
        return None
    if len(weeks) != len(values) or len(weeks) < MIN_OVERLAP:
        return None
    if len(set(weeks)) != len(weeks) or not all(math.isfinite(x) for x in values):
        return None
    return weeks, values


def _find_series_in_json(blob: dict[str, Any] | None) -> tuple[list[int], list[float]] | None:
    if not blob:
        return None
    for key in ("per_week", "per_week_crps", "weekly_crps", "crps_by_week"):
        found = _series(blob.get(key))
        if found is not None:
            return found
    # Some artifacts put the series under metrics/results.
    for key in ("metrics", "results", "overall"):
        nested = blob.get(key)
        if isinstance(nested, dict):
            found = _find_series_in_json(nested)
            if found is not None:
                return found
    return None


def load_civicsafe_series(data: str, results_dir: Path) -> tuple[list[int], list[float]] | None:
    """Load CIVIC-SAFE's all-category weekly CRPS from persisted artifacts."""
    candidates = (
        results_dir / "conformal_evaluation" / f"{data}_conformal_results.json",
        results_dir / "evaluation" / f"{data}_test_results.json",
        results_dir / "conformal_evaluation" / f"{data}_per_week_crps.json",
        results_dir / "evaluation" / f"{data}_per_week_crps.json",
    )
    for path in candidates:
        found = _find_series_in_json(_load_json(path))
        if found is not None:
            return found

    # Optional NPZ sidecars are intentionally explicit: the policy panel is
    # violent-only and cannot be mistaken for an all-category CRPS series.
    for path in (
        results_dir / "conformal_evaluation" / f"{data}_crps.npz",
        results_dir / "conformal_evaluation" / f"{data}_per_week.npz",
    ):
        if not path.exists():
            continue
        try:
            with np.load(path, allow_pickle=False) as loaded:
                for key in ("civicsafe_crps", "per_week_crps", "crps"):
                    if key not in loaded.files:
                        continue
                    values = np.asarray(loaded[key], dtype=float).reshape(-1)
                    if values.size < MIN_OVERLAP or not np.isfinite(values).all():
                        continue
                    weeks = loaded["week_index"] if "week_index" in loaded.files else np.arange(260, 260 + values.size)
                    return [int(x) for x in np.asarray(weeks).reshape(-1)], values.tolist()
        except (OSError, ValueError) as exc:
            logger.warning("Could not read %s: %s", path, exc)
    return None


def load_deep_series(data: str, results_dir: Path) -> dict[str, tuple[list[int], list[float]]]:
    """Load and aggregate each deep baseline's per-week CRPS over seed files."""
    baseline_dir = results_dir / "baselines"
    files = sorted(baseline_dir.glob(f"{data}_deep_baselines_seed*.json"))
    if not files:
        canonical = _load_json(baseline_dir / f"{data}_deep_baselines.json")
        files = []
        blobs = [canonical] if canonical else []
    else:
        blobs = [_load_json(path) for path in files]

    by_model: dict[str, list[tuple[list[int], list[float]]]] = {name: [] for name in DEEP_MODELS}
    for blob in blobs:
        if not blob:
            continue
        for name in DEEP_MODELS:
            found = _series(blob.get(name, {}).get("per_week")) if isinstance(blob.get(name), dict) else None
            if found is not None:
                by_model[name].append(found)

    out: dict[str, tuple[list[int], list[float]]] = {}
    for name, runs in by_model.items():
        if not runs:
            continue
        week_sets = [set(weeks) for weeks, _ in runs]
        common = sorted(set.intersection(*week_sets))
        if len(common) < MIN_OVERLAP:
            logger.warning("%s/%s has only %d common weeks", data, name, len(common))
            continue
        aligned = []
        for weeks, values in runs:
            mapping = dict(zip(weeks, values))
            aligned.append([mapping[week] for week in common])
        out[name] = (common, np.mean(np.asarray(aligned, dtype=float), axis=0).tolist())
    return out


def align_on_weeks(
    ours: tuple[list[int], list[float]],
    baseline: tuple[list[int], list[float]],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Inner-join two series by absolute week index, in chronological order."""
    left = dict(zip(ours[0], ours[1]))
    right = dict(zip(baseline[0], baseline[1]))
    weeks = sorted(set(left) & set(right))
    return (
        torch.tensor([left[w] for w in weeks], dtype=torch.float64),
        torch.tensor([right[w] for w in weeks], dtype=torch.float64),
        weeks,
    )


def compute_city(data: str, results_dir: Path, *, bootstrap_replicates: int = 10_000) -> dict[str, Any]:
    ours = load_civicsafe_series(data, results_dir)
    deep = load_deep_series(data, results_dir)
    if ours is None:
        return {
            "status": "unavailable",
            "data": data,
            "reason": (
                "No all-category CIVIC-SAFE per-week CRPS series was found. "
                "The aggregate CRPS and violent-only policy NPZ cannot support "
                "a paired deep-baseline test."
            ),
            "baselines_available": sorted(deep),
        }

    comparisons: dict[str, Any] = {}
    skipped: dict[str, str] = {}
    for name, series in sorted(deep.items()):
        ours_t, base_t, weeks = align_on_weeks(ours, series)
        if len(weeks) < MIN_OVERLAP:
            skipped[name] = f"only {len(weeks)} overlapping weeks"
            continue
        # ``compare_forecasts`` owns the repository's fixed 10,000-resample
        # protocol.  Keep the CLI knob for forward-compatible sidecars, but do
        # not pass unsupported arguments to older library versions.
        result = compare_forecasts(ours_t, base_t, baseline_name=name)
        result["n_weeks_tested"] = len(weeks)
        result["week_range"] = [weeks[0], weeks[-1]]
        result["mean_crps_ours"] = float(ours_t.mean())
        result["mean_crps_baseline"] = float(base_t.mean())
        comparisons[name] = result

    return {
        "status": "ok" if comparisons else "unavailable",
        "data": data,
        "n_baselines_tested": len(comparisons),
        "comparisons": comparisons,
        "skipped": skipped,
        "sign_convention": "mean_diff = CRPS_CIVIC_SAFE - CRPS_baseline; negative is better",
        "source": "persisted per-week CRPS; no model retraining",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="append", choices=("chicago", "nyc"), default=None)
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    output_dir = Path(args.results_dir) / "significance"
    output_dir.mkdir(parents=True, exist_ok=True)
    for data in args.data or ["chicago", "nyc"]:
        result = compute_city(data, Path(args.results_dir), bootstrap_replicates=args.bootstrap_replicates)
        path = output_dir / f"{data}_deep_significance.json"
        path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info("%s deep significance: %s -> %s", data, result["status"], path)


if __name__ == "__main__":
    main()
