"""LaTeX table generation utilities for publication-ready results.

Converts evaluation metrics and audit results into perfectly formatted
LaTeX tables suitable for NeurIPS, ICML, or ICLR submissions.
Includes support for bolding best results, adding confidence intervals,
and proper multi-row formatting.
"""

from __future__ import annotations

from typing import Any


def format_metric(mean: float, std: float | None = None, is_best: bool = False, decimals: int = 3) -> str:
    """Format a metric with optional std and bolding for LaTeX."""
    if std is not None:
        text = f"{mean:.{decimals}f} \\pm {std:.{decimals}f}"
    else:
        text = f"{mean:.{decimals}f}"
    
    if is_best:
        return f"\\mathbf{{{text}}}"
    return text


def generate_calibration_table(results: list[dict[str, Any]], caption: str = "Conformal Calibration Results", label: str = "tab:calibration") -> str:
    """Generate a LaTeX table comparing calibration methods.
    
    Args:
        results: List of dictionaries with keys: method, coverage, target_coverage, avg_width
    """
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\begin{tabular}{l c c c}",
        "\\toprule",
        "\\textbf{Calibration Method} & \\textbf{Target Coverage} & \\textbf{Empirical Coverage} & \\textbf{Avg. Interval Width} \\\\",
        "\\midrule"
    ]
    
    # Find best width among those that meet target coverage
    valid_widths = [r["avg_width"] for r in results if r["coverage"] >= r["target_coverage"] - 0.05]
    best_width = min(valid_widths) if valid_widths else None
    
    method_names = {
        "split_cp": "Split Conformal",
        "weighted_cp": "Weighted Conformal",
        "mondrian_cp": "Mondrian Conformal",
        "equalized_coverage": "Equalized Coverage",
        "ecrc": "ECRC (Ours)"
    }
    
    for r in results:
        method = method_names.get(r["method"], r["method"])
        target = f"{r['target_coverage']:.2f}"
        cov = f"{r['coverage']:.3f}"
        
        # Highlight coverage if it meets target
        if r["coverage"] >= r["target_coverage"]:
            cov = f"\\mathbf{{{cov}}}"
            
        width = f"{r['avg_width']:.2f}"
        if best_width is not None and abs(r["avg_width"] - best_width) < 1e-5:
            width = f"\\mathbf{{{width}}}"
            
        lines.append(f"{method} & {target} & {cov} & {width} \\\\")
        
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def generate_routing_table(comparison: dict[str, Any], caption: str = "Routing Algorithm Comparison", label: str = "tab:routing") -> str:
    """Generate a LaTeX table comparing routing algorithms."""
    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\begin{tabular}{l c c}",
        "\\toprule",
        "\\textbf{Metric} & \\textbf{Dijkstra (Baseline)} & \\textbf{Tsinghua SSSP (Ours)} \\\\",
        "\\midrule"
    ]
    
    lines.append(f"Total Cost & {comparison['dijkstra_cost']:.4f} & {comparison['tsinghua_cost']:.4f} \\\\")
    lines.append(f"Nodes Settled & {comparison['dijkstra_settled']} & {comparison['tsinghua_settled']} \\\\")
    lines.append(f"Frontier Reductions & N/A & {comparison['tsinghua_frontier_reductions']} \\\\")
    lines.append(f"Optimal Path Match & \\multicolumn{{2}}{{c}}{{{'Yes' if comparison['cost_match'] else 'No'}}} \\\\")
    
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)
