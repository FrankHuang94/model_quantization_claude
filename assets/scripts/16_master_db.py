"""Section 16 charts: entity count by category; chipmaker precision-support heatmap."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from chartstyle import apply_style, PALETTE, save
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
apply_style()

df = pd.read_csv(os.path.join(DATA, "master_database.csv"))

# ---------------------------------------------------------------------------
# Chart 1: entity count by category
# ---------------------------------------------------------------------------
counts = df["category"].value_counts()
fig, ax = plt.subplots(figsize=(9.5, 5.6))
bars = ax.bar(counts.index, counts.values, color=PALETTE["blue"], width=0.62)
for b in bars:
    ax.annotate(f"{int(b.get_height())}", (b.get_x()+b.get_width()/2, b.get_height()),
                xytext=(0,3), textcoords="offset points", ha="center", fontsize=9)
ax.set_ylabel("Number of entities")
ax.set_title(f"Master database: entity count by category (n={len(df)})")
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
save(fig, "16_entity_by_category")

# ---------------------------------------------------------------------------
# Chart 2: chipmaker precision-support heatmap (curated for the major chipmakers)
# ---------------------------------------------------------------------------
chips = ["Qualcomm","Apple","MediaTek","Samsung","Google","NVIDIA","Intel","AMD","ARM","Huawei"]
formats = ["INT16","INT8","INT4","INT2","FP16","FP8","FP4"]
# 2 native, 1 emerging/partial, 0 none
M = {
 "Qualcomm":[2,2,2,2,2,2,1],
 "Apple":   [1,2,2,0,2,1,0],
 "MediaTek":[2,2,2,1,2,1,0],
 "Samsung": [1,2,1,0,2,1,0],
 "Google":  [0,2,1,0,1,0,0],
 "NVIDIA":  [1,2,2,0,2,2,2],
 "Intel":   [1,2,2,0,2,1,0],
 "AMD":     [1,2,2,0,2,1,0],
 "ARM":     [2,2,1,0,1,0,0],
 "Huawei":  [1,2,2,0,2,1,0],
}
mat = np.array([M[c] for c in chips])
cmap = ListedColormap([PALETTE["grid"], PALETTE["amber"], PALETTE["teal"]])
norm = BoundaryNorm([-0.5,0.5,1.5,2.5], cmap.N)
fig, ax = plt.subplots(figsize=(8.8, 6.4))
ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
ax.set_xticks(range(len(formats))); ax.set_xticklabels(formats)
ax.set_yticks(range(len(chips))); ax.set_yticklabels(chips, fontsize=9)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        t = {0:"—",1:"◐",2:"●"}[mat[i,j]]
        ax.text(j,i,t,ha="center",va="center",fontsize=11,
                color="white" if mat[i,j]==2 else PALETTE["ink"])
ax.set_title("Precision-support heatmap across chipmakers (● native · ◐ emerging · — none)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=PALETTE["teal"],label="native"),
                   Patch(color=PALETTE["amber"],label="emerging"),
                   Patch(color=PALETTE["grid"],label="none")],
          loc="upper left", bbox_to_anchor=(1.01,1.0))
save(fig, "16_precision_heatmap")
print("done; entities =", len(df))
