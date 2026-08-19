#!/usr/bin/env python
"""Zero-shot cross-city transfer diagnostics for frozen CIVIC-SAFE outputs.

This script evaluates a source-city forecast artifact on a target-city panel;
it does not retrain or fine-tune a model.  NPZ/JSON artifacts may contain
``y``, ``pi``, ``mu``, ``r`` and optional source calibration arrays.  The
conformal comparison uses a split calibrator fitted only on the source
calibration portion and then applied to the target city.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from civicsafe.calibration.conformal import SplitConformalCalibrator
from civicsafe.training.metrics import crps_zinb, mae_zinb, rmse_zinb

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load(path: Path, city: str) -> dict[str, Any] | None:
    candidates = [
        path,
        path / f"{city}_transfer.npz",
        path / f"{city}_predictions.npz",
        path / "cross_city" / f"{city}_transfer.npz",
    ]
    for candidate in candidates:
        if not candidate.exists() or candidate.is_dir():
            continue
        if candidate.suffix.lower() == ".npz":
            with np.load(candidate, allow_pickle=False) as loaded:
                return {key: loaded[key] for key in loaded.files}
        if candidate.suffix.lower() == ".json":
            value = json.loads(candidate.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
    return None


def _tensor(data: dict[str, Any], *names: str) -> torch.Tensor | None:
    for name in names:
        if name in data:
            return torch.as_tensor(data[name], dtype=torch.float32).reshape(-1)
    return None


def _required(data: dict[str, Any], names: tuple[str, ...], label: str) -> torch.Tensor:
    value = _tensor(data, *names)
    if value is None:
        raise KeyError(f"Missing {label}; expected one of {names}")
    return value


def _summary(y: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor, r: torch.Tensor) -> dict[str, float]:
    return {
        "crps": float(crps_zinb(y, pi, mu, r).mean().item()),
        "mae": float(mae_zinb(y, pi, mu).item()),
        "rmse": float(rmse_zinb(y, pi, mu).item()),
    }


def evaluate_transfer(
    source_city: str,
    target_city: str,
    source: dict[str, Any],
    target: dict[str, Any],
    *,
    alpha: float = 0.1,
) -> dict[str, Any]:
    """Evaluate frozen source predictions before/after source-fitted CP."""
    source_y = _required(source, ("y", "actual", "y_target"), "source observations")
    source_pi = _required(source, ("pi", "source_pi"), "source pi")
    source_mu = _required(source, ("mu", "source_mu", "point_prediction"), "source mu")
    source_r = _required(source, ("r", "source_r"), "source r")
    target_y = _required(target, ("y", "actual", "y_target"), "target observations")
    target_pi = _required(target, ("pi", "target_pi"), "target pi")
    target_mu = _required(target, ("mu", "target_mu", "point_prediction"), "target mu")
    target_r = _required(target, ("r", "target_r"), "target r")
    n = min(target_y.numel(), target_pi.numel(), target_mu.numel(), target_r.numel())
    target_y, target_pi, target_mu, target_r = (v[:n] for v in (target_y, target_pi, target_mu, target_r))

    before = _summary(target_y, target_pi, target_mu, target_r)
    calibration_y = _tensor(source, "calibration_y", "y_cal")
    calibration_pi = _tensor(source, "calibration_pi", "pi_cal")
    calibration_mu = _tensor(source, "calibration_mu", "mu_cal")
    calibration_r = _tensor(source, "calibration_r", "r_cal")
    if calibration_y is None:
        calibration_y, calibration_pi, calibration_mu, calibration_r = source_y, source_pi, source_mu, source_r
    if any(v is None for v in (calibration_pi, calibration_mu, calibration_r)):
        raise KeyError("Source artifact must include calibration pi/mu/r when calibration_y is supplied")

    calibrator = SplitConformalCalibrator(alpha=alpha)
    calibrator.fit(calibration_y, calibration_pi, calibration_mu, calibration_r)
    interval = calibrator.predict(target_pi, target_mu, target_r)
    covered = ((target_y >= interval["lower"]) & (target_y <= interval["upper"])).float()
    after = {
        **before,
        "coverage": float(covered.mean().item()),
        "mean_interval_width": float((interval["upper"] - interval["lower"]).mean().item()),
        "conformal_threshold": float(calibrator.threshold),
    }
    return {
        "source_city": source_city,
        "target_city": target_city,
        "status": "ok",
        "before_conformal": before,
        "after_conformal": after,
        "before": before,
        "after": after,
        "raw": before,
        "calibrated": after,
        "crps_change_pct": 0.0,
        "calibration_source": "source_city",
    }


cross_city_transfer = evaluate_transfer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "cross_city_transfer.json"))
    parser.add_argument("--alpha", type=float, default=0.1)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = Path(args.results_dir)
    reports: list[dict[str, Any]] = []
    for source_city, target_city in (("chicago", "nyc"), ("nyc", "chicago")):
        source = _load(root, source_city)
        target = _load(root, target_city)
        if source is None or target is None:
            reports.append({
                "source_city": source_city,
                "target_city": target_city,
                "status": "unavailable",
                "reason": "raw cross-city prediction artifacts not found",
            })
            continue
        try:
            reports.append(evaluate_transfer(source_city, target_city, source, target, alpha=args.alpha))
        except (KeyError, ValueError, RuntimeError) as exc:
            reports.append({
                "source_city": source_city,
                "target_city": target_city,
                "status": "unavailable",
                "reason": str(exc),
            })
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"reports": reports}, indent=2), encoding="utf-8")
    logger.info("Cross-city diagnostic written to %s", out)


if __name__ == "__main__":
    main()
