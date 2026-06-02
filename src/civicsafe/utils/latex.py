"""LaTeX table generation utilities for publication-ready outputs.

This module provides utilities to convert Python dictionaries and evaluation
results directly into well-formatted LaTeX tables, ready to be copy-pasted
into a NeurIPS/ICLR/ICML paper template.
"""

from __future__ import annotations

import pandas as pd


def dict_to_latex_table(
    results: dict[str, dict[str, float | str]],
    caption: str,
    label: str,
    bold_best: bool = True,
    metric_directions: dict[str, str] | None = None,
) -> str:
    """Convert a nested dictionary of results into a LaTeX table.

    Args:
        results: Format ``{"Row Name": {"Metric 1": value, ...}}``.
        caption: Table caption for LaTeX.
        label: Table reference label for LaTeX.
        bold_best: If True, bold the best value in each column.
        metric_directions: Dict mapping metric names to "min" or "max"
            to determine which value to bold.

    Returns:
        A string containing the LaTeX table code (using booktabs).
    """
    df = pd.DataFrame.from_dict(results, orient="index")
    
    if metric_directions is None:
        metric_directions = {}

    # Format dataframe to strings with bolding
    formatted_df = pd.DataFrame(index=df.index, columns=df.columns)
    
    for col in df.columns:
        direction = metric_directions.get(col, "min")  # default assume lower is better
        
        try:
            # Check if column is numeric
            numeric_col = pd.to_numeric(df[col])
            best_val = numeric_col.min() if direction == "min" else numeric_col.max()
            
            for idx in df.index:
                val = numeric_col[idx]
                # Format to 4 decimal places if float
                val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
                
                if bold_best and val == best_val:
                    formatted_df.loc[idx, col] = f"\\textbf{{{val_str}}}"
                else:
                    formatted_df.loc[idx, col] = val_str
        except ValueError:
            # Not numeric, just copy as string
            formatted_df[col] = df[col].astype(str)

    # Generate LaTeX using pandas built-in, but customize for booktabs
    latex_code = formatted_df.to_latex(
        escape=False,
        column_format="l" + "c" * len(df.columns),
        caption=caption,
        label=label,
    )

    # Clean up pandas LaTeX output for better publication quality
    # Requires \usepackage{booktabs} in the LaTeX document
    latex_code = latex_code.replace("\\toprule", "\\toprule")
    latex_code = latex_code.replace("\\midrule", "\\midrule")
    latex_code = latex_code.replace("\\bottomrule", "\\bottomrule")

    return latex_code


def format_mean_std_latex(mean: float, std: float, decimals: int = 4) -> str:
    """Format mean and standard deviation into a LaTeX string.
    
    Example: 15.67 ± 0.02 -> $15.67 \\pm 0.02$
    """
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$"
