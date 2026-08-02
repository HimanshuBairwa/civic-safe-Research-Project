#!/usr/bin/env python
"""Diebold-Mariano significance tests: CIVIC-SAFE vs every baseline.

The main results table previously reported CRPS point estimates with no
uncertainty. On a ~53-week test set that is not enough to claim a win: a
0.02 CRPS gap can easily be within week-to-week noise, and a reviewer is
entitled to ask "is that difference real?".

This script answers that question properly:

  1. Loads the per-week CRPS series that ``evaluate_trained.py``,
     ``baselines.py`` and ``deep_baselines.py`` now export.
  2. JOINS each pair on absolute week index -- never on position. Two
     forecasters can cover different week sets, and a positional zip would
     silently compare week 261 against week 262.
  3. Runs Diebold-Mariano with Newey-West HAC variance (autocorrelation-robust)
     plus a moving-block bootstrap as a distribution-free second opinion.
  4. Applies Benjamini-Hochberg across the family of baselines, because
     testing against 9 baselines at alpha=0.05 yields a ~37% chance of at
     least one spurious "win" if left uncorrected.

Sign convention: ``mean_diff = CRPS_ours - CRPS_baseline``, so NEGATIVE means
CIVIC-SAFE is better.

Usage:
    python scripts/significance_tests.py --data chicago
    python scripts/significance_tests.py --data nyc --alpha 0.05
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from civicsafe.calibration.significance import (  # noqa: E402
    benjamini_hochberg,
    compare_forecasts,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# significance._validate_crps_pair rejects anything shorter.
MIN_OVERLAP = 10


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)  # type: ignore[no-any-return]
    except Exception as e:  # noqa: BLE001 - a corrupt file must not kill the run
        logger.warning(f"  Could not read {path.name}: {e}")
        return None


def _as_series(block: dict[str, Any] | None) -> tuple[list[int], list[float]] | None:
    """Extract (week_index, crps) from a per_week block, if it is usable."""
    if not isinstance(block, dict):
        return None
    crps = block.get("crps")
    weeks = block.get("week_index")
    if not crps or not weeks or len(crps) != len(weeks):
        return None
    return [int(w) for w in weeks], [float(c) for c in crps]


def load_model_series(data: str) -> tuple[list[int], list[float]] | None:
    """CIVIC-SAFE's own per-week CRPS from the evaluation output."""
    path = PROJECT_ROOT / "outputs" / "evaluation" / f"{data}_test_results.json"
    res = _load_json(path)
    if res is None:
        logger.error(f"  Missing {path} -- run evaluate_trained.py --data {data} first.")
        return None
    series = _as_series(res.get("per_week"))
    if series is None:
        logger.error(
            f"  {path.name} has no usable per_week block. It was produced before "
            f"per-week export existed; re-run evaluate_trained.py --data {data}."
        )
    return series


def load_baseline_series(data: str) -> dict[str, tuple[list[int], list[float]]]:
    """Per-week CRPS for every baseline that exported one."""
    out: dict[str, tuple[list[int], list[float]]] = {}

    # Classical baselines -> sidecar JSON keyed by model name.
    classical = _load_json(
        PROJECT_ROOT / "outputs" / "baselines" / f"{data}_baselines_per_week.json"
    )
    if classical:
        for name, block in classical.items():
            s = _as_series(block)
            if s is not None:
                out[name] = s
    else:
        logger.warning(
            f"  No classical per-week file for {data} "
            f"(run: python scripts/baselines.py data={data})"
        )

    # Deep baselines -> per_week nested inside each model's metrics dict.
    deep = _load_json(
        PROJECT_ROOT / "outputs" / "baselines" / f"{data}_deep_baselines.json"
    )
    if deep:
        for name, metrics in deep.items():
            # "_meta" and friends are run metadata, not forecasters.
            if name.startswith("_") or not isinstance(metrics, dict):
                continue
            s = _as_series(metrics.get("per_week"))
            if s is not None:
                out[name] = s
            else:
                logger.warning(
                    f"  Deep baseline '{name}' has no per_week block; "
                    f"re-run deep_baselines.py to include it."
                )
    else:
        logger.warning(
            f"  No deep-baseline file for {data} "
            f"(run: python scripts/deep_baselines.py data={data})"
        )

    return out


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------
def align_on_weeks(
    ours: tuple[list[int], list[float]],
    theirs: tuple[list[int], list[float]],
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Restrict both series to the weeks they have in common.

    Returns:
        (crps_ours, crps_theirs, weeks) -- the tensors are 1-D and equal length,
        ordered by increasing week.

    Comparing forecasters on different weeks is not a comparison at all, so the
    join is on the week label rather than on list position.
    """
    w_ours, c_ours = ours
    w_theirs, c_theirs = theirs
    map_ours = dict(zip(w_ours, c_ours))
    map_theirs = dict(zip(w_theirs, c_theirs))
    common = sorted(set(map_ours) & set(map_theirs))
    a = torch.tensor([map_ours[w] for w in common], dtype=torch.float64)
    b = torch.tensor([map_theirs[w] for w in common], dtype=torch.float64)
    return a, b, common


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------
def run_comparisons(data: str, alpha: float) -> dict[str, Any]:
    logger.info("=" * 72)
    logger.info(f"  Significance testing -- CIVIC-SAFE vs baselines -- {data}")
    logger.info("=" * 72)

    ours = load_model_series(data)
    if ours is None:
        raise SystemExit(1)
    logger.info(f"  CIVIC-SAFE per-week CRPS: {len(ours[0])} weeks "
                f"[{min(ours[0])}, {max(ours[0])}]")

    baselines = load_baseline_series(data)
    if not baselines:
        logger.error("  No baseline per-week series found. Nothing to test against.")
        raise SystemExit(1)
    logger.info(f"  Baselines with per-week series: {len(baselines)}")

    comparisons: dict[str, Any] = {}
    skipped: dict[str, str] = {}

    for name in sorted(baselines):
        a, b, weeks = align_on_weeks(ours, baselines[name])
        if len(weeks) < MIN_OVERLAP:
            reason = (
                f"only {len(weeks)} overlapping weeks "
                f"(need >= {MIN_OVERLAP} for a meaningful test)"
            )
            logger.warning(f"  [{name}] skipped: {reason}")
            skipped[name] = reason
            continue

        n_ours, n_theirs = len(ours[0]), len(baselines[name][0])
        if len(weeks) < max(n_ours, n_theirs):
            # Not fatal, but the reader must know the test used a subset.
            logger.warning(
                f"  [{name}] partial overlap: testing on {len(weeks)} weeks "
                f"(ours={n_ours}, baseline={n_theirs}) -- weeks outside the "
                f"intersection are excluded."
            )

        try:
            res = compare_forecasts(a, b, baseline_name=name)
        except Exception as e:  # noqa: BLE001 - one bad baseline must not stop the rest
            logger.warning(f"  [{name}] comparison failed: {e}")
            skipped[name] = str(e)
            continue

        res["n_weeks_tested"] = len(weeks)
        res["week_range"] = [int(weeks[0]), int(weeks[-1])]
        res["mean_crps_ours"] = round(float(a.mean()), 6)
        res["mean_crps_baseline"] = round(float(b.mean()), 6)
        # CRPSS: skill score relative to this baseline. Positive = we improve.
        mb = float(b.mean())
        res["crpss_pct"] = (
            round(100.0 * (1.0 - float(a.mean()) / mb), 2) if mb > 0 else None
        )
        comparisons[name] = res

        d = res["dm"]
        logger.info(
            f"  [{name:<18}] ours={res['mean_crps_ours']:.4f} "
            f"base={res['mean_crps_baseline']:.4f} "
            f"diff={d['mean_diff']:+.4f} DM={d['dm_stat']:+.3f} "
            f"p={d['p_value']:.4g} T={len(weeks)}"
        )

    if not comparisons:
        logger.error("  Every comparison was skipped; no results to report.")
        raise SystemExit(1)

    # --- Benjamini-Hochberg across the family of baselines ---
    #
    # Correction is applied over the DM p-values; the bootstrap p-values are
    # corrected separately so each testing procedure stays internally
    # consistent (mixing the two families would not control anything).
    names = list(comparisons)
    dm_p = [comparisons[n]["dm"]["p_value"] for n in names]
    boot_p = [comparisons[n]["bootstrap"]["p_value"] for n in names]

    dm_adj = benjamini_hochberg(dm_p, alpha=alpha)
    boot_adj = benjamini_hochberg(boot_p, alpha=alpha)

    for i, n in enumerate(names):
        better = comparisons[n]["dm"]["mean_diff"] < 0
        comparisons[n]["dm_bh"] = dm_adj[i]
        comparisons[n]["bootstrap_bh"] = boot_adj[i]
        # "Significant" alone is direction-blind: a significant result where we
        # are WORSE is not a win. Record the conjunction explicitly.
        comparisons[n]["significant_win"] = bool(dm_adj[i]["significant"] and better)
        comparisons[n]["significant_loss"] = bool(
            dm_adj[i]["significant"] and not better
        )

    n_win = sum(c["significant_win"] for c in comparisons.values())
    n_loss = sum(c["significant_loss"] for c in comparisons.values())
    n_tie = len(comparisons) - n_win - n_loss

    out: dict[str, Any] = {
        "data": data,
        "alpha": alpha,
        "correction": "benjamini-hochberg",
        "sign_convention": "mean_diff = CRPS_ours - CRPS_baseline; negative = ours better",
        "n_baselines_tested": len(comparisons),
        "n_significant_wins": n_win,
        "n_significant_losses": n_loss,
        "n_not_significant": n_tie,
        "comparisons": comparisons,
        "skipped": skipped,
    }
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def make_latex(res: dict[str, Any]) -> str:
    """LaTeX table with CRPS, skill score, CI, and BH-adjusted p-values."""
    data = res["data"]
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Diebold--Mariano tests, CIVIC-SAFE vs.\ baselines ({data.title()}). "
        rf"$\Delta$CRPS is CIVIC-SAFE minus baseline, so negative favours our model. "
        rf"$p_{{\mathrm{{BH}}}}$ is Benjamini--Hochberg adjusted across "
        rf"{res['n_baselines_tested']} baselines at FDR "
        rf"$\alpha={res['alpha']}$. HAC variance uses Newey--West with "
        rf"$h=\lfloor T^{{1/3}} \rfloor$.}}",
        rf"\label{{tab:significance_{data}}}",
        r"\begin{tabular}{lrrrrrc}",
        r"\toprule",
        r"Baseline & CRPS & $\Delta$CRPS & 95\% CI & DM & $p_{\mathrm{BH}}$ & Sig. \\",
        r"\midrule",
    ]
    ordered = sorted(
        res["comparisons"].items(), key=lambda kv: kv[1]["dm"]["mean_diff"]
    )
    for name, c in ordered:
        dm = c["dm"]
        p_bh = c["dm_bh"]["adjusted_p"]
        if c["significant_win"]:
            mark = r"$\checkmark$"
        elif c["significant_loss"]:
            mark = r"$\times$"
        else:
            mark = r"--"
        p_str = r"$<$0.001" if p_bh < 1e-3 else f"{p_bh:.3f}"
        safe = name.replace("_", r"\_")
        lines.append(
            f"{safe} & {c['mean_crps_baseline']:.4f} & {dm['mean_diff']:+.4f} & "
            f"[{dm['ci_lower']:+.4f}, {dm['ci_upper']:+.4f}] & "
            f"{dm['dm_stat']:+.2f} & {p_str} & {mark} \\\\"
        )
    first = next(iter(res["comparisons"].values()))
    lines += [
        r"\midrule",
        rf"\textbf{{CIVIC-SAFE}} & \textbf{{{first['mean_crps_ours']:.4f}}} & "
        r"-- & -- & -- & -- & -- \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def print_summary(res: dict[str, Any]) -> None:
    print()
    print("=" * 92)
    print(f"  SIGNIFICANCE: CIVIC-SAFE vs BASELINES -- {res['data'].upper()}")
    print("=" * 92)
    print(
        f"  {'Baseline':<20}{'CRPS':>9}{'dCRPS':>10}{'DM':>9}"
        f"{'p_raw':>10}{'p_BH':>10}{'T':>5}  verdict"
    )
    print("-" * 92)
    ordered = sorted(
        res["comparisons"].items(), key=lambda kv: kv[1]["dm"]["mean_diff"]
    )
    for name, c in ordered:
        dm = c["dm"]
        if c["significant_win"]:
            verdict = "WIN (sig.)"
        elif c["significant_loss"]:
            verdict = "LOSS (sig.)"
        elif dm["mean_diff"] < 0:
            verdict = "better, n.s."
        else:
            verdict = "worse, n.s."
        print(
            f"  {name:<20}{c['mean_crps_baseline']:>9.4f}{dm['mean_diff']:>+10.4f}"
            f"{dm['dm_stat']:>+9.2f}{dm['p_value']:>10.4g}"
            f"{c['dm_bh']['adjusted_p']:>10.4g}{c['n_weeks_tested']:>5}  {verdict}"
        )
    print("-" * 92)
    ours = next(iter(res["comparisons"].values()))["mean_crps_ours"]
    print(f"  CIVIC-SAFE CRPS: {ours:.4f}")
    print(
        f"  Significant wins: {res['n_significant_wins']}/"
        f"{res['n_baselines_tested']}   losses: {res['n_significant_losses']}   "
        f"not significant: {res['n_not_significant']}   "
        f"(BH FDR alpha={res['alpha']})"
    )
    if res["skipped"]:
        print(f"  Skipped: {', '.join(res['skipped'])}")
    print("=" * 92)
    print(
        "  Note: 'not significant' means the test cannot distinguish the two on\n"
        "  this test set. Report it as such -- it is not evidence of a win."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diebold-Mariano tests: CIVIC-SAFE vs baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--data", default="chicago", choices=["chicago", "nyc"])
    parser.add_argument(
        "--alpha", type=float, default=0.05, help="BH false discovery rate"
    )
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    res = run_comparisons(args.data, args.alpha)

    out_dir = PROJECT_ROOT / "outputs" / "significance"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = Path(args.output) if args.output else out_dir / f"{args.data}_significance.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    logger.info(f"  Results saved to: {out_file}")

    tex_file = out_file.with_name(f"{out_file.stem}_table.tex")
    with open(tex_file, "w", encoding="utf-8") as f:
        f.write(make_latex(res))
    logger.info(f"  LaTeX table saved to: {tex_file}")

    print_summary(res)


if __name__ == "__main__":
    main()
