"""Section 13 charts: startup funding by company; funding by category."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chartstyle import apply_style, PALETTE, save
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
apply_style()

df = pd.read_csv(os.path.join(DATA, "13_startups.csv"))
cat_c = {"compression-sw": PALETTE["blue"], "inference-chip": PALETTE["red"],
         "edge-chip": PALETTE["teal"]}

# ---------------------------------------------------------------------------
# Chart 1: funding by startup (horizontal bar, colored by category)
# ---------------------------------------------------------------------------
d = df.sort_values("funding_musd")
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(d["startup"], d["funding_musd"], color=[cat_c[c] for c in d["category"]],
        edgecolor="white")
for i, (v, s) in enumerate(zip(d["funding_musd"], d["stage"])):
    ax.annotate(f"${v:g}M · {s}", (v, i), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=7.8, color=PALETTE["ink"])
ax.set_xlabel("Disclosed funding raised (US$M, approximate)")
ax.set_title("Quantization / compression / edge-inference startups by funding (approximate)")
ax.set_xlim(0, 470)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=cat_c[k], label=v) for k, v in
                   {"compression-sw":"Compression software", "inference-chip":"In-memory / inference chip",
                    "edge-chip":"Edge AI accelerator"}.items()], loc="lower right")
save(fig, "13_startup_funding")

# ---------------------------------------------------------------------------
# Chart 2: total funding + count by category
# ---------------------------------------------------------------------------
grp = df.groupby("category").agg(total=("funding_musd","sum"), count=("startup","count")).reset_index()
labels = {"compression-sw":"Compression\nsoftware", "inference-chip":"In-memory /\ninference chip",
          "edge-chip":"Edge AI\naccelerator"}
fig, ax = plt.subplots(figsize=(8.5, 5.6))
x = np.arange(len(grp))
bars = ax.bar(x, grp["total"], color=[cat_c[c] for c in grp["category"]], width=0.6)
ax.set_xticks(x); ax.set_xticklabels([labels[c] for c in grp["category"]])
ax.set_ylabel("Total disclosed funding (US$M)")
ax.set_title("Startup funding by category (with company counts)")
for i, (t, c) in enumerate(zip(grp["total"], grp["count"])):
    ax.annotate(f"${t:g}M\n{c} cos", (i, t), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8.5, color=PALETTE["ink"])
save(fig, "13_funding_by_category")
print("done")
