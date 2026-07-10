"""Publication-grade figure style for OICC (NeurIPS / Nature conventions).

- Okabe-Ito colorblind-safe categorical palette (validated: worst adjacent CVD
  deltaE ~18, well above the 12 target).
- cividis for sequential/heatmap magnitude (perceptually uniform AND CVD-safe).
- recessive spines/grid, thin marks, tight sans-serif type, 300 DPI.

Import `apply_style()` once, use PALETTE[i] for series i (fixed order, never
cycled), SEQ_CMAP for magnitude heatmaps.
"""
from __future__ import annotations

import matplotlib as mpl

# Okabe-Ito (fixed order; assign by entity, never by rank)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
           "#F0E442", "#000000"]
BLUE, VERMILLION, GREEN, PURPLE, ORANGE, SKY, YELLOW, BLACK = PALETTE
INK = "#222222"       # primary text
MUTED = "#6b6b6b"     # secondary text / axes
SEQ_CMAP = "cividis"  # sequential magnitude (CVD-safe, perceptually uniform)
DIV_CMAP = "RdBu_r"   # diverging (signed) -- gray-ish midpoint


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.titleweight": "bold",
        "axes.edgecolor": MUTED,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#dddddd",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "text.color": INK,
        "figure.titlesize": 12,
        "figure.titleweight": "bold",
    })


def panel_label(ax, text: str, dx: float = -0.08, dy: float = 1.06) -> None:
    """Add a bold (a)/(b) panel label in axis-fraction coords (Nature style)."""
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="right", color="#000000")
