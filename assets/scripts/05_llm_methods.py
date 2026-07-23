"""Section 05 charts: accuracy-vs-compression scatter; effective-bits bar by method."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from chartstyle import apply_style, PALETTE, save
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
apply_style()

df = pd.read_csv(os.path.join(DATA, "05_methods_tradeoff.csv"))
cat_c = {
    "baseline": PALETTE["ink"],
    "weight+act": PALETTE["amber"],
    "weight-only": PALETTE["blue"],
    "codebook/rotation": PALETTE["purple"],
    "rotation-W4A4": PALETTE["red"],
}

# ---------------------------------------------------------------------------
# Chart 1: accuracy retention vs effective bits (compression) scatter
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6.4))
for _, r in df.iterrows():
    ax.scatter(r["effective_bits"], r["accuracy_retention"], s=150,
               color=cat_c[r["category"]], edgecolor="white", linewidth=1.1, zorder=3)
    ax.annotate(r["method"], (r["effective_bits"], r["accuracy_retention"]),
                xytext=(6, 3), textcoords="offset points", fontsize=8, color=PALETTE["ink"])
ax.set_xlabel("Effective bits per weight  (lower = more compression →)")
ax.set_ylabel("Accuracy retention (% of FP16)")
ax.set_title("LLM quantization methods: accuracy retention vs. compression")
ax.invert_xaxis()
ax.axhline(98, color=PALETTE["muted"], lw=0.8, ls=":")
ax.text(15.5, 98.15, "98% retention", fontsize=8, color=PALETTE["muted"])
ax.axvspan(3.5, 4.5, color=PALETTE["blue"], alpha=0.05)
ax.text(4.0, 90, "4-bit\nworkhorse", fontsize=8, color=PALETTE["blue"], ha="center")
leg = [Line2D([0],[0], marker="o", color="w", markerfacecolor=cat_c[k], markersize=10, label=k)
       for k in cat_c]
ax.legend(handles=leg, loc="lower left", title="Method family")
ax.set_ylim(86, 101)
save(fig, "05_accuracy_vs_compression")

# ---------------------------------------------------------------------------
# Chart 2: effective bits per method (horizontal bar), colored by family
# ---------------------------------------------------------------------------
d2 = df[df["category"] != "baseline"].sort_values("effective_bits")
fig, ax = plt.subplots(figsize=(9.5, 6.6))
colors = [cat_c[c] for c in d2["category"]]
ax.barh(d2["method"], d2["effective_bits"], color=colors, edgecolor="white")
for i, (b, a) in enumerate(zip(d2["effective_bits"], d2["accuracy_retention"])):
    ax.annotate(f"{b:.1f}b · {a:.0f}%", (b, i), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=7.8, color=PALETTE["ink"])
ax.set_xlabel("Effective bits per weight")
ax.set_title("Effective bit-width by LLM quantization method (label: bits · accuracy retention)")
ax.set_xlim(0, 10)
save(fig, "05_effective_bits_by_method")
print("done")
