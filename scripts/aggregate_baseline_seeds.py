#!/usr/bin/env python
"""Aggregate multi-seed baseline runs into mean +/- std, and build the fair
head-to-head table against CIVIC-SAFE.

The fairness problem this solves
--------------------------------
CIVIC-SAFE is reported as 5 seeds combined by an EMOS mixture. The deep
baselines were run once, at one seed. Comparing an ensemble against a single
model attributes to the *architecture* a margin that is partly just
*ensembling* -- and a reviewer will say so.

There are only two honest fixes: ensemble the baselines too, or report
CIVIC-SAFE's single-model number next to its ensemble number. This script does
both, so the paper can state plainly how much of the margin is architecture and
how much is ensembling:

    Baseline           CRPS (mean +/- std over seeds)      n
    CIVIC-SAFE (1 seed)  ...                               5
    CIVIC-SAFE (EMOS)    ...                               1

Inputs
------
  outputs/baselines/{data}_deep_baselines_seed{S}.json   (one per seed)
  outputs/ablation/_per_seed/full_model_seed_*.json      (CIVIC-SAFE seeds)

Usage:
    # after: for s in 42 137 256; do python scripts/deep_baselines.py \
    #            data=chicago epochs=200 seed=$s; done
    python scripts/aggregate_baseline_seeds.py --data chicago
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

METRICS = ["crps", "mae", "rmse", "brier_zero"]


def _load(path: Path) -> dict[str, Any] | None:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]
    except Exception as e:  # noqa: BLE001
        logger.warning(f"  Could not read {path.name}: {e}")
        return None


def collect_baseline_seeds(data: str) -> dict[str, list[dict[str, Any]]]:
    """Group per-seed baseline runs by model name."""
    bl_dir = PROJECT_ROOT / "outputs" / "baselines"
    files = sorted(bl_dir.glob(f"{data}_deep_baselines_seed*.json"))
    if not files:
        logger.warning(
            f"  No seed-stamped baseline files for {data}. "
            f"Run: python scripts/deep_baselines.py data={data} epochs=200 seed=<S>"
        )
    per_model: dict[str, list[dict[str, Any]]] = {}
    for fp in files:
        blob = _load(fp)
        if not blob:
            continue
        seed = blob.get("_meta", {}).get("seed")
        for name, m in blob.items():
            if name.startswith("_") or not isinstance(m, dict) or "crps" not in m:
                continue
            m = dict(m)
            m["_seed"] = seed
            m["_file"] = fp.name
            per_model.setdefault(name, []).append(m)
    for name, runs in per_model.items():
        logger.info(f"  {name:<18} {len(runs)} seed(s): {[r['_seed'] for r in runs]}")
    return per_model


def collect_civicsafe_seeds(data: str) -> list[dict[str, Any]]:
    """CIVIC-SAFE per-seed evaluations, written by run_ablations.py."""
    d = PROJECT_ROOT / "outputs" / "ablation" / "_per_seed"
    runs: list[dict[str, Any]] = []
    qualified = sorted(d.glob(f"{data}_full_model_seed_*.json"))
    qualified += sorted(d.glob(f"full_model_{data}_seed_*.json"))
    candidates = qualified or sorted(d.glob("full_model_seed_*.json"))
    for fp in candidates:
        blob = _load(fp)
        if blob and "overall" in blob:
            metadata = blob.get("metadata", {})
            recorded_city = blob.get("data") or blob.get("dataset")
            if isinstance(metadata, dict):
                recorded_city = recorded_city or metadata.get("data") or metadata.get("dataset")
            if recorded_city is not None and str(recorded_city).lower() != data:
                logger.warning("  Ignoring cross-city seed file %s (%s)", fp.name, recorded_city)
                continue
            # Unqualified, metadata-free legacy files caused the NYC/Chicago
            # duplication bug.  They remain acceptable only for Chicago, where
            # those historical filenames originated.
            if not qualified and recorded_city is None and data != "chicago":
                logger.warning("  Ignoring ambiguous seed file for %s: %s", data, fp.name)
                continue
            o = dict(blob["overall"])
            o["_file"] = fp.name
            runs.append(o)
    if not runs:
        # The post-training conformal artifact is the authoritative source for
        # the completed five-seed ensemble.  Its per-seed test CRPS values are
        # persisted under ``ensemble.per_seed_test_crps`` and are already
        # selected using validation-only raw/EMA decisions.  Use these values
        # instead of accidentally mixing another city's legacy ablation files.
        fp = PROJECT_ROOT / "outputs" / "conformal_evaluation" / f"{data}_conformal_results.json"
        blob = _load(fp)
        per_seed = (
            blob.get("ensemble", {}).get("per_seed_test_crps")
            if isinstance(blob, dict) and isinstance(blob.get("ensemble"), dict)
            else None
        )
        if isinstance(per_seed, list) and per_seed:
            for index, value in enumerate(per_seed):
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    continue
                if np.isfinite(number):
                    runs.append({"crps": number, "_file": fp.name, "_seed": index})
            if runs:
                logger.info("  Loaded %d CIVIC-SAFE seed scores from %s", len(runs), fp.name)

    if not runs:
        # Last-resort legacy single-run evaluation.  This path is intentionally
        # explicit because it is not a seed-matched comparison.
        fp = PROJECT_ROOT / "outputs" / "evaluation" / f"{data}_test_results.json"
        blob = _load(fp)
        if blob and "overall" in blob:
            o = dict(blob["overall"])
            o["_file"] = fp.name
            runs.append(o)
            logger.warning(
                "  Only the single canonical evaluation was found for CIVIC-SAFE. "
                "Run `python scripts/run_ablations.py --data "
                f"{data} --variants full_model --skip-train` to get per-seed numbers."
            )
    return runs


def summarise(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """mean / std / n per metric. Sample std (ddof=1) -- with 3-5 seeds the
    population std would understate the spread we are trying to expose."""
    out: dict[str, Any] = {"n_seeds": len(runs)}
    for k in METRICS:
        vals = [
            float(r[k]) for r in runs
            if k in r and r[k] is not None and np.isfinite(float(r[k]))
        ]
        if not vals:
            continue
        out[k] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        out[f"{k}_values"] = [round(v, 6) for v in vals]
    return out


def make_latex(res: dict[str, Any]) -> str:
    data = res["data"]
    rows = []
    for name, s in sorted(
        res["baselines"].items(), key=lambda kv: kv[1].get("crps", 9e9), reverse=True
    ):
        if "crps" not in s:
            continue
        n = s["n_seeds"]
        pm = (
            rf"{s['crps']:.4f} $\pm$ {s['crps_std']:.4f}"
            if n > 1 else f"{s['crps']:.4f}"
        )
        rows.append(
            f"{name.replace('_', chr(92) + '_')} & {pm} & {n} \\\\"
        )

    cs = res.get("civicsafe_single", {})
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Seed-matched comparison ({data.title()}). Baselines and "
        rf"CIVIC-SAFE are both reported as mean $\pm$ sample std over seeds, so "
        rf"the architecture margin is separated from the ensembling gain. "
        rf"$n$ is the number of seeds.}}",
        rf"\label{{tab:seed_matched_{data}}}",
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Model & CRPS & $n$ \\",
        r"\midrule",
        *rows,
        r"\midrule",
    ]
    if cs.get("crps") is not None:
        n = cs["n_seeds"]
        pm = (
            rf"{cs['crps']:.4f} $\pm$ {cs['crps_std']:.4f}"
            if n > 1 else f"{cs['crps']:.4f}"
        )
        lines.append(rf"CIVIC-SAFE (single model) & {pm} & {n} \\")
    if res.get("civicsafe_ensemble") is not None:
        lines.append(
            rf"\textbf{{CIVIC-SAFE (EMOS ensemble)}} & "
            rf"\textbf{{{res['civicsafe_ensemble']:.4f}}} & 1 \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="chicago", choices=["chicago", "nyc"])
    p.add_argument("--ensemble-crps", type=float, default=None,
                   help="EMOS ensemble CRPS from emos_ensemble.py, for the table")
    args = p.parse_args()

    logger.info("=" * 72)
    logger.info(f"  Seed-matched baseline aggregation -- {args.data}")
    logger.info("=" * 72)

    logger.info("  Baselines:")
    per_model = collect_baseline_seeds(args.data)
    baselines = {name: summarise(runs) for name, runs in per_model.items()}

    logger.info("  CIVIC-SAFE:")
    cs_runs = collect_civicsafe_seeds(args.data)
    cs = summarise(cs_runs)
    logger.info(f"    {len(cs_runs)} seed(s)")

    res: dict[str, Any] = {
        "data": args.data,
        "baselines": baselines,
        "civicsafe_single": cs,
        "civicsafe_ensemble": args.ensemble_crps,
        "note": (
            "Baselines and CIVIC-SAFE single-model are both mean +/- sample std "
            "over seeds. The ensemble row is a single EMOS mixture over "
            "CIVIC-SAFE seeds; compare it to the single-model row to see how "
            "much of the margin is ensembling rather than architecture."
        ),
    }

    out_dir = PROJECT_ROOT / "outputs" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.data}_seed_matched.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    logger.info(f"  Saved: {out_file}")

    tex = out_dir / f"{args.data}_seed_matched_table.tex"
    with open(tex, "w", encoding="utf-8") as f:
        f.write(make_latex(res))
    logger.info(f"  LaTeX: {tex}")

    # --- Console table ---
    print()
    print("=" * 70)
    print(f"  SEED-MATCHED COMPARISON -- {args.data.upper()}")
    print("=" * 70)
    print(f"  {'Model':<24}{'CRPS':>10}{'+/- std':>10}{'seeds':>7}")
    print("-" * 70)
    for name, s in sorted(baselines.items(), key=lambda kv: kv[1].get("crps", 9e9),
                          reverse=True):
        if "crps" not in s:
            continue
        print(f"  {name:<24}{s['crps']:>10.4f}{s.get('crps_std', 0.0):>10.4f}"
              f"{s['n_seeds']:>7}")
    print("-" * 70)
    if "crps" in cs:
        print(f"  {'CIVIC-SAFE (single)':<24}{cs['crps']:>10.4f}"
              f"{cs.get('crps_std', 0.0):>10.4f}{cs['n_seeds']:>7}")
    if args.ensemble_crps is not None:
        print(f"  {'CIVIC-SAFE (EMOS)':<24}{args.ensemble_crps:>10.4f}"
              f"{'--':>10}{1:>7}")
        if "crps" in cs:
            gain = cs["crps"] - args.ensemble_crps
            print(f"\n  Ensembling accounts for {gain:+.4f} CRPS "
                  f"({100 * gain / cs['crps']:+.1f}%) of the total margin.")
    print("=" * 70)
    n_single = {s["n_seeds"] for s in baselines.values()}
    if n_single and max(n_single) < 2:
        print("  WARNING: baselines have 1 seed each. Their std is unknown, so any\n"
              "  margin over them cannot be separated from seed noise. Run more\n"
              f"  seeds: for s in 42 137 256; do python scripts/deep_baselines.py \\\n"
              f"      data={args.data} epochs=200 seed=$s; done")


if __name__ == "__main__":
    main()
