"""Section 04 charts: accuracy vs bit-width by model class; format range-vs-precision map."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chartstyle import apply_style, PALETTE, save
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
apply_style()

# ---------------------------------------------------------------------------
# Chart 1: accuracy retention vs bit-width across model classes
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(DATA, "04_accuracy_by_bitwidth.csv"))
fig, ax = plt.subplots(figsize=(9.5, 6))
for col, lbl, c in [
    ("cnn_vision", "CNN / vision (W+A quant)", PALETTE["blue"]),
    ("llm", "LLM (weight-only, per-group)", PALETTE["teal"]),
    ("diffusion", "Diffusion (image gen)", PALETTE["purple"]),
]:
    ax.plot(df["bits"], df[col], "-o", color=c, lw=2.3, markersize=6, label=lbl)
ax.invert_xaxis()
ax.set_xticks(df["bits"])
ax.set_xlabel("Bit-width")
ax.set_ylabel("Accuracy / quality retention (% of FP16)")
ax.set_title("Accuracy degradation by bit-width across model classes (illustrative)")
ax.axvspan(4.5, 8.5, color=PALETTE["green"], alpha=0.06)
ax.axvspan(3.5, 4.5, color=PALETTE["amber"], alpha=0.08)
ax.axvspan(0.5, 3.5, color=PALETTE["red"], alpha=0.06)
ax.text(6.5, 40, "INT8/6\nsafe zone", fontsize=8, color=PALETTE["green"], ha="center")
ax.text(4.0, 40, "4-bit\nfrontier", fontsize=8, color=PALETTE["amber"], ha="center")
ax.text(2.2, 40, "sub-4-bit\ncliff", fontsize=8, color=PALETTE["red"], ha="center")
ax.legend(loc="lower right")
ax.set_ylim(15, 103)
save(fig, "04_accuracy_by_bitwidth")

# ---------------------------------------------------------------------------
# Chart 2: numeric format map — dynamic range vs relative precision
# x = log10 dynamic range (orders of magnitude of representable magnitudes)
# y = effective mantissa/precision bits (uniform-equivalent)
# ---------------------------------------------------------------------------
fmts = [
    # name, log10 dynamic range, precision bits, total bits, family
    ("INT2",      0.5, 2.0, 2,  "int"),
    ("INT4",      1.0, 4.0, 4,  "int"),
    ("INT8",      2.4, 8.0, 8,  "int"),
    ("NF4",       1.3, 4.0, 4,  "nonuniform"),
    ("FP4 (E2M1)",1.8, 1.0, 4,  "float"),
    ("FP6 (E3M2)",3.0, 2.0, 6,  "float"),
    ("FP8 E4M3",  5.7, 3.0, 8,  "float"),
    ("FP8 E5M2",  9.5, 2.0, 8,  "float"),
    ("FP16",     12.0, 10.0, 16, "float"),
    ("BF16",     38.0, 7.0, 16, "float"),
]
fam_c = {"int": PALETTE["blue"], "float": PALETTE["red"], "nonuniform": PALETTE["green"]}
fig, ax = plt.subplots(figsize=(9.8, 6.2))
for name, dr, prec, tb, fam in fmts:
    ax.scatter(dr, prec, s=90 + tb*18, color=fam_c[fam], edgecolor="white",
               linewidth=1.2, alpha=0.9, zorder=3)
    ax.annotate(name, (dr, prec), xytext=(7, 4), textcoords="offset points",
                fontsize=8.4, color=PALETTE["ink"])
ax.set_xlabel("Dynamic range  (orders of magnitude, log10 of max/min representable)")
ax.set_ylabel("Precision  (mantissa / uniform-equivalent bits)")
ax.set_title("Numeric format map: dynamic range vs. precision (bubble size ∝ total bits)")
from matplotlib.lines import Line2D
leg = [Line2D([0],[0], marker="o", color="w", markerfacecolor=fam_c[k], markersize=11, label=v)
       for k, v in {"int":"Integer (uniform)", "float":"Floating-point", "nonuniform":"Non-uniform (NF4)"}.items()]
ax.legend(handles=leg, loc="upper left")
save(fig, "04_format_range_precision")
print("done")
