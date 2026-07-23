"""Section 09 charts: Snapdragon/Hexagon precision-support roadmap; format-introduction timeline."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from chartstyle import apply_style, PALETTE, save
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
apply_style()

df = pd.read_csv(os.path.join(DATA, "09_qualcomm.csv"))
formats = ["FP16", "INT16", "INT8", "INT4", "FP8", "INT2"]
fcolor = {"FP16": PALETTE["slate"], "INT16": PALETTE["cyan"], "INT8": PALETTE["blue"],
          "INT4": PALETTE["teal"], "FP8": PALETTE["amber"], "INT2": PALETTE["red"]}

# ---------------------------------------------------------------------------
# Chart 1: precision-support matrix across Snapdragon generations (dot grid)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 6))
for yi, (_, r) in enumerate(df.iterrows()):
    present = set(str(r["precisions"]).split(";"))
    for xi, f in enumerate(formats):
        has = f in present
        ax.scatter(xi, yi, s=260 if has else 60,
                   color=fcolor[f] if has else PALETTE["grid"],
                   edgecolor="white", linewidth=1.2, zorder=3)
ax.set_yticks(range(len(df))); ax.set_yticklabels(
    [f"{r.platform}\n({r.year})" for _, r in df.iterrows()], fontsize=8)
ax.set_xticks(range(len(formats))); ax.set_xticklabels(formats)
ax.invert_yaxis()
ax.set_title("Qualcomm Hexagon NPU precision support by Snapdragon generation")
ax.set_xlabel("Precision format (filled = supported)")
ax.grid(False)
save(fig, "09_qualcomm_precision_matrix")

# ---------------------------------------------------------------------------
# Chart 2: format-introduction timeline (when each format first appeared)
# ---------------------------------------------------------------------------
intro = {"INT8/INT16": 2019, "FP16": 2020, "INT4": 2022, "INT2": 2025, "FP8": 2025}
fig, ax = plt.subplots(figsize=(10, 5))
items = sorted(intro.items(), key=lambda x: x[1])
for i, (f, y) in enumerate(items):
    ax.plot([y, 2026], [i, i], color=PALETTE["teal"], lw=6, alpha=0.35, solid_capstyle="round")
    ax.scatter(y, i, s=160, color=PALETTE["blue"], edgecolor="white", zorder=3)
    ax.annotate(f"{f} — first in {y}", (y, i), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=9, color=PALETTE["ink"])
ax.set_yticks([]); ax.set_xlim(2018.5, 2027.5)
ax.set_xlabel("Year first supported in shipping Snapdragon NPU")
ax.set_title("Qualcomm Hexagon: precision-format introduction timeline")
save(fig, "09_qualcomm_format_timeline")
print("done")
