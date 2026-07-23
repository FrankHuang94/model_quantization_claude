"""Shared matplotlib styling for the model-quantization reference database.

Import this at the top of every chart script:

    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from chartstyle import apply_style, PALETTE, save

All charts use a consistent, print-legible, brand-neutral palette and are
saved at 150 DPI PNG into assets/charts/.
"""
import os
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from matplotlib import font_manager  # noqa: F401

# ---------------------------------------------------------------------------
# Palette — brand-neutral, colour-blind-aware ordering, works on white ground.
# ---------------------------------------------------------------------------
PALETTE = {
    "ink":       "#1b2733",   # near-black text / axes
    "muted":     "#5b6b7a",   # secondary text, gridlines
    "grid":      "#dfe4ea",
    "blue":      "#2f6db3",
    "teal":      "#2a9d8f",
    "amber":     "#e9a13b",
    "red":       "#d1495b",
    "purple":    "#8360a8",
    "green":     "#5c8a3a",
    "slate":     "#6c7a89",
    "cyan":      "#4ba3c3",
    "brown":     "#a07a55",
}

# ordered cycle for categorical series
SERIES = [
    PALETTE["blue"], PALETTE["amber"], PALETTE["teal"], PALETTE["red"],
    PALETTE["purple"], PALETTE["green"], PALETTE["cyan"], PALETTE["brown"],
    PALETTE["slate"],
]

# maturity colour semantics reused across the DB
MATURITY_COLORS = {
    "production": PALETTE["green"],
    "sdk":        PALETTE["amber"],
    "research":   PALETTE["red"],
}

CHART_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "charts")
)


def apply_style():
    """Apply the global rcParams. Call once per script before plotting."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": PALETTE["muted"],
        "axes.labelcolor": PALETTE["ink"],
        "axes.titlecolor": PALETTE["ink"],
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.8,
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "text.color": PALETTE["ink"],
        "font.size": 10.5,
        "font.family": "DejaVu Sans",
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save(fig, name):
    """Save a figure into assets/charts/ as <name>.png and return the path."""
    os.makedirs(CHART_DIR, exist_ok=True)
    path = os.path.join(CHART_DIR, name + ".png")
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    print("wrote", os.path.relpath(path))
    return path
