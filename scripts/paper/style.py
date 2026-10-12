"""One look for every paper figure: palette, type, marks, and how a figure is saved.

The categorical order is fixed and assigned by entity, never by rank, so a series keeps its
colour across figures. Checked with the dataviz palette validator (light surface): every
adjacent pair clears the CVD and normal-vision floors; yellow, aqua and magenta sit under 3:1
contrast, so figures that use them carry direct labels or a table beside them.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DPI = 300
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
REFERENCE = "#8a8984"  # the control / reference series: neutral, never a categorical slot
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.edgecolor": INK_SECONDARY,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "lines.linewidth": 1.6,
        "lines.markersize": 4,
        "legend.frameon": False,
        "legend.fontsize": 7,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)


def save(fig: plt.Figure, path: Path) -> Path:
    """Write `fig` as a 300 dpi PNG, tight, and close it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path
