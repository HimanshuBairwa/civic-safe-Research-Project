#!/usr/bin/env python
"""Post-training decision simulation for CIVIC-SAFE resource allocation.

The simulator consumes saved prediction artifacts when available.  It never
loads an optimizer or changes a checkpoint.  Artifacts are accepted as NPZ,
JSON, or a mapping passed directly to :func:`simulate_city`.  A typical NPZ
contains ``y_violent``, ``point_violent``, ``rolling_ha_violent``,
``conformal_upper_violent`` and ``demographic_group`` with shape ``(T, S)``
for the first four arrays and ``(S,)`` for the last one.

When only the compact conformal result JSON is present, the CLI writes a
diagnostic ``status=unavailable`` record instead of inventing model-level
predictions.  This keeps publication outputs auditable and makes missing DVC
artifacts explicit.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BUDGETS = (20, 50, 100)
POLICY_NAMES = (
    "naive_ha",
    "point_prediction",
    "unconstrained_conformal",
    "civic_safe_oicc",
)


@dataclass(frozen=True)
class PolicyMetrics:
    city: str
    policy: str
    budget: int
    violent_hit_rate: float
    demographic_overallocation_ratio: float
    idle_wasted_resource_ratio: float
    allocation_disparity: float
    disadvantaged_allocation_share: float
    disadvantaged_incident_share: float
    weeks: int
    spatial_units: int


def _as_panel(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must have shape (weeks, spatial_units), got {arr.shape}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _normalise_inputs(data: dict[str, Any]) -> dict[str, np.ndarray]:
    aliases = {
        "actual_violent": ("actual_violent", "y_violent", "y", "violent_actual"),
        "point_prediction": ("point_prediction", "point_violent", "mu_violent", "mu"),
        "rolling_ha": ("rolling_ha", "rolling_ha_violent", "ha_violent", "ha"),
        "conformal_upper": (
            "conformal_upper",
            "conformal_upper_violent",
            "upper_violent",
            "upper",
        ),
    }
    out: dict[str, np.ndarray] = {}
    for canonical, candidates in aliases.items():
        for candidate in candidates:
            if candidate in data:
                out[canonical] = _as_panel(data[candidate], canonical)
                break
        if canonical not in out:
            raise KeyError(f"Missing required policy input: one of {candidates}")

    weeks, units = out["actual_violent"].shape
    for key, arr in out.items():
        if arr.shape != (weeks, units):
            raise ValueError(f"{key} has shape {arr.shape}; expected {(weeks, units)}")

    group = data.get("demographic_group", data.get("groups", data.get("income_quartile")))
    if group is None:
        raise KeyError("Missing demographic_group/groups/income_quartile")
    group_arr = np.asarray(group)
    if group_arr.ndim == 2:
        group_arr = group_arr[0]
    if group_arr.ndim != 1 or group_arr.size != units:
        raise ValueError("demographic_group must have shape (spatial_units,)")
    out["demographic_group"] = np.nan_to_num(group_arr.astype(float), nan=0.0)
    return out


def _largest_remainder(scores: np.ndarray, budget: int) -> np.ndarray:
    """Convert non-negative scores into integer resource counts summing to B."""
    scores = np.nan_to_num(np.maximum(scores, 0.0), nan=0.0)
    if budget <= 0 or scores.size == 0:
        return np.zeros(scores.size, dtype=int)
    if scores.sum() <= 0.0:
        scores = np.ones_like(scores)
    raw = scores / scores.sum() * int(budget)
    allocation = np.floor(raw).astype(int)
    remainder = int(budget - allocation.sum())
    if remainder > 0:
        order = np.argsort(-(raw - allocation), kind="stable")
        allocation[order[:remainder]] += 1
    return allocation


def allocate_resources(scores: np.ndarray, budget: int) -> np.ndarray:
    """Public wrapper for deterministic largest-remainder allocation."""
    return _largest_remainder(np.asarray(scores, dtype=float), int(budget))


def _fair_allocation(
    scores: np.ndarray,
    groups: np.ndarray,
    budget: int,
    *,
    disadvantaged_label: float,
    tolerance: float,
) -> np.ndarray:
    """Allocate using scores while enforcing a demographic share band.

    The band is defined from the available-unit share, which is observable at
    decision time.  Incident shares are used only for the post-hoc audit.
    """
    disadvantaged = groups == disadvantaged_label
    n_disadv = int(disadvantaged.sum())
    if n_disadv == 0 or n_disadv == groups.size:
        return _largest_remainder(scores, budget)
    target_share = n_disadv / groups.size
    lo = max(0, int(np.ceil(budget * target_share * (1.0 - tolerance))))
    hi = min(budget, int(np.floor(budget * target_share * (1.0 + tolerance))))
    target = int(np.clip(round(budget * target_share), lo, hi))
    alloc = np.zeros(groups.size, dtype=int)
    idx_d = np.flatnonzero(disadvantaged)
    idx_o = np.flatnonzero(~disadvantaged)
    alloc[idx_d] = _largest_remainder(scores[idx_d], target)
    alloc[idx_o] = _largest_remainder(scores[idx_o], budget - target)
    return alloc


def _audit_allocation(
    actual: np.ndarray,
    allocation: np.ndarray,
    groups: np.ndarray,
    *,
    disadvantaged_label: float,
) -> tuple[float, float, float, float, float, float]:
    incident_total = float(actual.sum())
    allocated_total = float(allocation.sum())
    hit = float(actual[allocation > 0].sum() / incident_total) if incident_total > 0 else 0.0
    disadvantaged = groups == disadvantaged_label
    incident_disadv = float(actual[disadvantaged].sum())
    alloc_disadv = float(allocation[disadvantaged].sum())
    incident_share = incident_disadv / incident_total if incident_total > 0 else 0.0
    alloc_share = alloc_disadv / allocated_total if allocated_total > 0 else 0.0
    ratio = alloc_share / incident_share if incident_share > 1e-12 else float("inf")
    disparity = abs(alloc_share - incident_share)
    idle = float(allocation[actual <= 0].sum() / allocated_total) if allocated_total > 0 else 0.0
    return hit, ratio, idle, disparity, alloc_share, incident_share


def simulate_city(
    city: str,
    data: dict[str, Any],
    *,
    budgets: Iterable[int] = DEFAULT_BUDGETS,
    fairness_tolerance: float = 0.03,
) -> list[PolicyMetrics]:
    """Simulate all policies for a city and return one row per budget/policy."""
    arrays = _normalise_inputs(data)
    groups = arrays.pop("demographic_group")
    disadvantaged_label = float(np.nanmin(groups))
    outputs: list[PolicyMetrics] = []
    score_map = {
        "naive_ha": arrays["rolling_ha"],
        "point_prediction": arrays["point_prediction"],
        "unconstrained_conformal": arrays["conformal_upper"],
        "civic_safe_oicc": arrays["conformal_upper"],
    }
    for budget in budgets:
        if budget < 0:
            raise ValueError("budgets must be non-negative")
        per_policy = {name: [] for name in POLICY_NAMES}
        for week in range(arrays["actual_violent"].shape[0]):
            actual = arrays["actual_violent"][week]
            for policy, scores in score_map.items():
                if policy == "civic_safe_oicc":
                    alloc = _fair_allocation(
                        scores[week], groups, int(budget),
                        disadvantaged_label=disadvantaged_label,
                        tolerance=fairness_tolerance,
                    )
                else:
                    alloc = _largest_remainder(scores[week], int(budget))
                per_policy[policy].append(
                    _audit_allocation(actual, alloc, groups, disadvantaged_label=disadvantaged_label)
                )
        for policy in POLICY_NAMES:
            values = np.asarray(per_policy[policy], dtype=float)
            outputs.append(PolicyMetrics(
                city=city,
                policy=policy,
                budget=int(budget),
                violent_hit_rate=float(values[:, 0].mean()),
                demographic_overallocation_ratio=float(values[:, 1][np.isfinite(values[:, 1])].mean()) if np.isfinite(values[:, 1]).any() else float("nan"),
                idle_wasted_resource_ratio=float(values[:, 2].mean()),
                allocation_disparity=float(values[:, 3].mean()),
                disadvantaged_allocation_share=float(values[:, 4].mean()),
                disadvantaged_incident_share=float(values[:, 5].mean()),
                weeks=arrays["actual_violent"].shape[0],
                spatial_units=arrays["actual_violent"].shape[1],
            ))
    return outputs


run_policy_simulation = simulate_city


def _load_artifact(city: str, path: Path) -> dict[str, Any] | None:
    candidates = [
        path,
        # ``main`` receives the project-level outputs directory by default;
        # conformal evaluation artifacts live in this nested directory.
        path / "conformal_evaluation" / f"{city}_predictions.npz",
        path / f"{city}_policy_inputs.npz",
        path / "policy" / f"{city}_policy_inputs.npz",
        path / "evaluation" / f"{city}_policy_inputs.npz",
        path / f"{city}_predictions.npz",
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


def _write_latex(rows: list[PolicyMetrics], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Decision-theoretic violent-crime resource allocation under three budgets.}",
        r"\label{tab:policy-simulation}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"City & Policy & $B$ & Hit rate (\%) & Over-allocation ratio & Idle ratio (\%) \\",
        r"\midrule",
    ]
    for city in sorted({row.city for row in rows}):
        city_label = "NYC" if city.lower() == "nyc" else city.title()
        lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{city_label}}}}} \\")
        lines.append(r"\midrule")
        for row in (item for item in rows if item.city == city):
            lines.append(
                f"{city_label} & {row.policy.replace('_', ' ')} & {row.budget} & "
                f"{100 * row.violent_hit_rate:.2f} & {row.demographic_overallocation_ratio:.3f} & "
                f"{100 * row.idle_wasted_resource_ratio:.2f} " + r"\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_figure(rows: list[PolicyMetrics], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Embed fonts as TrueType subsets rather than matplotlib's default Type 3,
    # which IEEE PDF eXpress rejects. Glyph storage only; the plot is unchanged.
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    policies = list(POLICY_NAMES)
    labels = [p.replace("_", " ").title() for p in policies]
    for city in sorted({r.city for r in rows}):
        city_rows = [r for r in rows if r.city == city]
        for ax, field, ylabel in (
            (axes[0], "violent_hit_rate", "Violent incident hit rate"),
            (axes[1], "allocation_disparity", "Absolute allocation/incident share gap"),
        ):
            for policy, label in zip(policies, labels, strict=True):
                points = [r for r in city_rows if r.policy == policy]
                points.sort(key=lambda r: r.budget)
                ax.plot([r.budget for r in points], [getattr(r, field) for r in points], marker="o", label=f"{city.title()} {label}")
            ax.set_xlabel("Weekly resource budget")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", action="append", choices=["chicago", "nyc"], default=None)
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--budget", type=int, action="append", dest="budgets")
    parser.add_argument("--fairness-tolerance", type=float, default=0.03)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cities = args.data or ["chicago", "nyc"]
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    rows: list[PolicyMetrics] = []
    unavailable: list[dict[str, str]] = []
    for city in cities:
        artifact = _load_artifact(city, results_dir)
        if artifact is None:
            unavailable.append({"city": city, "status": "unavailable", "reason": "raw prediction artifact not found"})
            logger.warning("%s: raw policy artifact not found; no fabricated policy metrics written", city)
            continue
        try:
            rows.extend(simulate_city(city, artifact, budgets=args.budgets or DEFAULT_BUDGETS, fairness_tolerance=args.fairness_tolerance))
        except (KeyError, ValueError) as exc:
            unavailable.append({"city": city, "status": "unavailable", "reason": str(exc)})
            logger.warning("%s: invalid policy artifact: %s", city, exc)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"status": "ok" if rows else "unavailable", "rows": [asdict(r) for r in rows], "unavailable": unavailable}
    (output_dir / "policy_simulation_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if rows:
        _write_latex(rows, output_dir / "tables" / "table7_policy_simulation.tex")
        _write_figure(rows, output_dir / "figures" / "fig9_policy_tradeoff.pdf")
        _write_figure(rows, output_dir / "figures" / "fig9_policy_tradeoff.png")
    logger.info("Policy simulation status: %s", payload["status"])


if __name__ == "__main__":
    main()
