"""LaTeX table generation utilities for publication-ready outputs.

Generates standard, NeurIPS-compliant LaTeX tables from python dictionaries
and pandas DataFrames. Supports standard deviations, bolding of best
results, and proper booktabs formatting.
"""

from __future__ import annotations

import pandas as pd


def dict_to_latex_table(
    data: dict[str, dict[str, float]],
    caption: str,
    label: str,
    better: dict[str, str] | None = None,
) -> str:
    """Convert a nested dictionary of results to a LaTeX table.
    
    Args:
        data: Nested dictionary {row_name: {col_name: value}}.
        caption: Table caption.
        label: LaTeX label for cross-referencing.
        better: Dictionary mapping col_name to "high" or "low" to 
            automatically bold the best result in that column.
            
    Returns:
        String containing the formatted LaTeX table.
    """
    if not data:
        return ""

    better = better or {}
    
    # Extract columns and rows
    rows = list(data.keys())
    cols = list(next(iter(data.values())).keys())
    
    # Find best values for bolding
    best_vals: dict[str, float] = {}
    for col in cols:
        vals = [data[r].get(col, float('nan')) for r in rows]
        valid_vals = [v for v in vals if not pd.isna(v)]
        
        if not valid_vals:
            continue
            
        direction = better.get(col, "low")
        if direction == "low":
            best_vals[col] = min(valid_vals)
        else:
            best_vals[col] = max(valid_vals)
            
    # Build LaTeX string
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        "\\begin{tabular}{l" + "c" * len(cols) + "}",
        "\\toprule",
        "Method & " + " & ".join(cols) + " \\\\",
        "\\midrule"
    ]
    
    for row in rows:
        row_str = [row]
        for col in cols:
            val = data[row].get(col, float('nan'))
            if pd.isna(val):
                row_str.append("-")
            else:
                # Format to 3 decimal places
                formatted = f"{val:.3f}"
                # Bold if best
                if col in best_vals and abs(val - best_vals[col]) < 1e-6:
                    formatted = f"\\textbf{{{formatted}}}"
                row_str.append(formatted)
        
        lines.append(" & ".join(row_str) + " \\\\")
        
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}"
    ])
    
    return "\n".join(lines)


def df_to_latex(
    df: pd.DataFrame,
    caption: str,
    label: str,
) -> str:
    """Convert a pandas DataFrame to a LaTeX table with booktabs.
    
    Args:
        df: Pandas DataFrame.
        caption: Table caption.
        label: LaTeX label for cross-referencing.
        
    Returns:
        String containing the formatted LaTeX table.
    """
    latex_str = df.to_latex(
        index=False,
        escape=False,
        column_format="l" + "c" * (len(df.columns) - 1),
    )
    
    # Wrap in table environment
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{" + caption + "}",
        "\\label{" + label + "}",
        latex_str.strip(),
        "\\end{table}"
    ]
    
    return "\n".join(lines)
