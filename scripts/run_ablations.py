#!/usr/bin/env python
"""CIVIC-SAFE ablation runner — trains and evaluates each ablated variant.

`ablation_study.py` only *formats* LaTeX tables; it reads
``outputs/ablation/{variant}_results.json`` files that nothing previously
wrote, so every ablation table rendered as dashes. This script produces those
files.

For each variant it:
  1. Trains with the appropriate config overrides, into an isolated run
     directory (``run_{data}_{variant}_{ts}``) so an ablation can never
     overwrite the full model's checkpoints.
  2. Evaluates EVERY seed's checkpoint on the test set.
  3. Aggregates to mean +/- std across seeds and writes the variant JSON.

Reporting spread across seeds is the point: with a 53-week test set, a CRPS
gap of 0.02 between two variants is not distinguishable from seed noise, and a
table of single-seed point estimates invites exactly that misreading.

Usage:
    # See the plan and cost estimate without running anything
    python scripts/run_ablations.py --data chicago --dry-run

    # Architecture ablations, 3 seeds each
    python scripts/run_ablations.py --data chicago --seeds 3

    # A specific subset
    python scripts/run_ablations.py --data chicago --variants no_gatv2 nb_only

    # Re-evaluate without retraining (checkpoints already exist)
    python scripts/run_ablations.py --data chicago --skip-train
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ABLATION_DIR = PROJECT_ROOT / "outputs" / "ablation"


# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
# Each entry maps a variant name to the train.py overrides that realise it.
# The name is also the output filename stem, matching what
# ablation_study.discover_ablation_results() scans for.
#
# Only variants requiring RETRAINING live here. Post-hoc ablations (EMOS
# weighting, recalibration, ensemble size K) operate on already-trained
# checkpoints and are handled separately -- retraining for them would waste
# GPU hours on a question the existing checkpoints already answer.
VARIANTS: dict[str, dict[str, Any]] = {
    "full_model": {
        "overrides": [],
        "desc": "Complete model (reference for every other row)",
    },
    "no_gatv2": {
        "overrides": ["model.ablations.use_gnn=false"],
        "desc": "GATv2 spatial encoder -> per-node MLP (no message passing)",
    },
    "no_transformer": {
        "overrides": ["model.ablations.use_transformer=false"],
        "desc": "Causal Transformer -> per-step MLP + causal mean pooling",
    },
    "nb_only": {
        "overrides": ["model.ablations.zero_inflation=false"],
        "desc": "ZINB -> plain NB (zero-inflation head disabled)",
    },
    "loss_nll": {
        "overrides": ["training.loss_fn=nll"],
        "desc": "Train on ZINB NLL instead of CRPS",
    },
    "loss_sac": {
        "overrides": ["training.loss_fn=sac"],
        "desc": "Train on Sharpness-Aware Calibration objective",
    },
    "no_r_reg": {
        "overrides": ["training.r_reg_lambda=0.0"],
        "desc": "Disable r-floor regularisation",
    },
    "no_sharpness": {
        "overrides": ["training.loss_fn=sac", "training.sac_lambda_sharpness=0.0"],
        "desc": "SAC objective with the sharpness term switched off",
    },
}

# Aliases so the loss-ablation table finds a CRPS row without a duplicate run:
# the full model already IS the CRPS-trained variant.
LOSS_ALIASES = {"loss_crps": "full_model"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_dir_for(data_name: str, run_tag: str) -> Path | None:
    """Locate the run directory train.py would resume for this tag.

    Mirrors train.py's resolution, including the rule that an untagged prefix
    must not match tagged directories.
    """
    outputs = PROJECT_ROOT / "outputs"
    if not outputs.exists():
        return None
    prefix = f"run_{data_name}_{run_tag}_" if run_tag else f"run_{data_name}_"
    cands = sorted(
        d for d in outputs.iterdir() if d.is_dir() and d.name.startswith(prefix)
    )
    if not run_tag:
        cands = [d for d in cands if d.name[len(prefix):].isdigit()]
    return cands[-1] if cands else None


def _exec(cmd: list[str], label: str) -> bool:
    """Run a subprocess, streaming output. Returns True on success."""
    logger.info(f"  $ {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)
    except KeyboardInterrupt:
        logger.warning(f"  {label}: interrupted by user")
        raise
    dt = time.time() - t0
    if proc.returncode != 0:
        logger.error(f"  {label}: FAILED (exit {proc.returncode}) after {dt:.0f}s")
        return False
    logger.info(f"  {label}: done in {dt:.0f}s")
    return True


def _aggregate(per_seed: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-seed metric dicts into mean and std.

    Args:
        per_seed: One ``overall`` metrics dict per seed.

    Returns:
        ``{"overall": {metric: mean}, "std": {metric: std}, ...}``. The
        ``overall`` key keeps the file readable by
        ablation_study.discover_ablation_results, which does
        ``data.get("overall", data)``.
    """
    keys = ["crps", "mae", "rmse", "brier_zero"]
    mean: dict[str, float] = {}
    std: dict[str, float] = {}
    for k in keys:
        vals = [
            float(s[k]) for s in per_seed
            if k in s and s[k] is not None and np.isfinite(float(s[k]))
        ]
        if not vals:
            continue
        mean[k] = float(np.mean(vals))
        # Sample std (ddof=1) is the honest estimator for a handful of seeds;
        # ddof=0 would understate the spread we are trying to expose.
        std[k] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return {"overall": mean, "std": std, "n_seeds": len(per_seed)}


# ---------------------------------------------------------------------------
# Per-variant pipeline
# ---------------------------------------------------------------------------
def run_variant(
    variant: str,
    data_name: str,
    seeds: int,
    epochs: int | None,
    skip_train: bool,
    force: bool,
) -> bool:
    """Train (optionally) and evaluate one ablation variant.

    Returns:
        True if a results JSON was written.
    """
    spec = VARIANTS[variant]
    out_path = ABLATION_DIR / f"{variant}_results.json"

    if out_path.exists() and not force:
        logger.info(f"[{variant}] already has results, skipping (use --force to redo)")
        return True

    logger.info("=" * 72)
    logger.info(f"[{variant}] {spec['desc']}")
    logger.info("=" * 72)

    # The full model trains untagged so it reuses the canonical run directory
    # rather than duplicating an expensive run that already exists.
    run_tag = "" if variant == "full_model" else variant

    # --- Train ---
    if not skip_train:
        cmd = [sys.executable, "scripts/train.py", f"data={data_name}"]
        cmd += list(spec["overrides"])
        cmd += [f"training.num_seeds={seeds}"]
        if run_tag:
            cmd += [f"run_tag={run_tag}"]
        if epochs is not None:
            cmd += [f"training.epochs={epochs}"]
        if not _exec(cmd, f"{variant} train"):
            return False

    # --- Locate checkpoints ---
    run_dir = _run_dir_for(data_name, run_tag)
    if run_dir is None:
        logger.error(f"[{variant}] no run directory found; cannot evaluate")
        return False
    ckpts = sorted(run_dir.glob("seed_*/best.pt"))
    if not ckpts:
        logger.error(f"[{variant}] no seed_*/best.pt under {run_dir}")
        return False
    logger.info(f"[{variant}] evaluating {len(ckpts)} checkpoint(s) from {run_dir.name}")

    # --- Evaluate every seed ---
    per_seed: list[dict[str, Any]] = []
    tmp_dir = ABLATION_DIR / "_per_seed"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for ckpt in ckpts:
        seed_name = ckpt.parent.name
        seed_out = tmp_dir / f"{variant}_{seed_name}.json"
        cmd = [
            sys.executable, "scripts/evaluate_trained.py",
            "--checkpoint", str(ckpt),
            "--data", data_name,
            "--output", str(seed_out),
        ]
        if not _exec(cmd, f"{variant}/{seed_name} eval"):
            logger.warning(f"[{variant}] {seed_name} failed to evaluate; excluded")
            continue
        try:
            with open(seed_out, encoding="utf-8") as f:
                res = json.load(f)
            overall = res.get("overall", {})
            overall["_seed_dir"] = seed_name
            overall["_arch"] = res.get("arch", {})
            per_seed.append(overall)
        except Exception as e:  # noqa: BLE001 - report and continue
            logger.warning(f"[{variant}] could not read {seed_out}: {e}")

    if not per_seed:
        logger.error(f"[{variant}] no seed evaluated successfully")
        return False

    # --- Aggregate and write ---
    agg = _aggregate(per_seed)
    agg["variant"] = variant
    agg["description"] = spec["desc"]
    agg["overrides"] = spec["overrides"]
    agg["data"] = data_name
    agg["run_dir"] = str(run_dir)
    agg["per_seed"] = per_seed
    # Carry the architecture actually evaluated, so a mislabelled row is
    # detectable after the fact.
    agg["arch"] = per_seed[0].get("_arch", {})

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, indent=2, default=str)

    o, s = agg["overall"], agg["std"]
    logger.info(
        f"[{variant}] CRPS = {o.get('crps', float('nan')):.4f} "
        f"+/- {s.get('crps', 0.0):.4f}  (n={agg['n_seeds']} seeds) -> {out_path.name}"
    )
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CIVIC-SAFE ablation variants and write result JSONs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", default="chicago", choices=["chicago", "nyc"])
    parser.add_argument(
        "--variants", nargs="*", default=None,
        help=f"Subset to run. Available: {', '.join(VARIANTS)}",
    )
    parser.add_argument(
        "--seeds", type=int, default=3,
        help="Seeds per variant (default 3 — enough to report mean +/- std)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override training epochs (default: whatever the config says)",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Evaluate existing checkpoints without retraining",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Recompute variants that already have a results JSON",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan and exit without running anything",
    )
    args = parser.parse_args()

    selected = args.variants or list(VARIANTS)
    unknown = [v for v in selected if v not in VARIANTS]
    if unknown:
        parser.error(
            f"Unknown variant(s): {', '.join(unknown)}. "
            f"Available: {', '.join(VARIANTS)}"
        )

    logger.info("=" * 72)
    logger.info("  CIVIC-SAFE — Ablation Runner")
    logger.info("=" * 72)
    logger.info(f"  Data:     {args.data}")
    logger.info(f"  Seeds:    {args.seeds} per variant")
    logger.info(f"  Variants: {len(selected)}")
    for v in selected:
        ov = " ".join(VARIANTS[v]["overrides"]) or "(none)"
        logger.info(f"    - {v:16s} {VARIANTS[v]['desc']}")
        logger.info(f"      {'':16s} overrides: {ov}")

    n_train_runs = len(selected) * args.seeds
    logger.info("")
    logger.info(f"  Training runs to perform: {n_train_runs}")
    logger.info(
        "  Cost scales linearly with seeds; use --seeds 1 for a fast smoke "
        "test, but do not report single-seed ablations as findings."
    )

    if args.dry_run:
        logger.info("\n  --dry-run: nothing executed.")
        return

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed = [], []
    for v in selected:
        try:
            success = run_variant(
                v, args.data, args.seeds, args.epochs, args.skip_train, args.force
            )
        except KeyboardInterrupt:
            logger.warning("Interrupted — stopping.")
            break
        except Exception as e:  # noqa: BLE001 - one variant must not kill the sweep
            logger.exception(f"[{v}] unexpected error: {e}")
            success = False
        (ok if success else failed).append(v)

    # Loss-table aliases: point loss_crps at the full model's numbers rather
    # than retraining an identical configuration.
    for alias, source in LOSS_ALIASES.items():
        src = ABLATION_DIR / f"{source}_results.json"
        dst = ABLATION_DIR / f"{alias}_results.json"
        if src.exists() and not dst.exists():
            with open(src, encoding="utf-8") as f:
                data = json.load(f)
            data["variant"] = alias
            data["alias_of"] = source
            data["description"] = (
                f"Alias of {source}: the full model is already CRPS-trained."
            )
            with open(dst, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"  Wrote alias {alias} -> {source}")

    logger.info("=" * 72)
    logger.info(f"  Completed: {len(ok)}  Failed: {len(failed)}")
    if failed:
        logger.warning(f"  Failed variants: {', '.join(failed)}")
    logger.info(f"  Results in: {ABLATION_DIR}")
    logger.info("  Next: python scripts/ablation_study.py --data " + args.data)
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
