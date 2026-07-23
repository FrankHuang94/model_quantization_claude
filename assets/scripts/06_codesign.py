"""Section 06 charts: precision-support heatmap; memory roofline for quantization."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from chartstyle import apply_style, PALETTE, save
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
apply_style()

# ---------------------------------------------------------------------------
# Chart 1: hardware x precision support heatmap (0 none, 1 partial/emerging, 2 native)
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(DATA, "06_hw_precision.csv"), index_col="hardware")
mat = df.values
cmap = ListedColormap([PALETTE["grid"], PALETTE["amber"], PALETTE["teal"]])
norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
fig, ax = plt.subplots(figsize=(9.6, 6.6))
im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
ax.set_xticks(range(len(df.columns))); ax.set_xticklabels(df.columns)
ax.set_yticks(range(len(df.index))); ax.set_yticklabels(df.index, fontsize=8.5)
for i in range(mat.shape[0]):
    for j in range(mat.shape[1]):
        txt = {0: "—", 1: "◐", 2: "●"}[mat[i, j]]
        ax.text(j, i, txt, ha="center", va="center",
                color="white" if mat[i, j] == 2 else PALETTE["ink"], fontsize=11)
ax.set_title("Edge silicon × precision-format support (● native · ◐ emerging/partial · — none)")
ax.set_xlabel("Precision format")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=PALETTE["teal"], label="native"),
                   Patch(color=PALETTE["amber"], label="emerging/partial"),
                   Patch(color=PALETTE["grid"], label="none")],
          loc="upper left", bbox_to_anchor=(1.01, 1.0))
plt.setp(ax.get_xticklabels(), rotation=0)
save(fig, "06_hw_precision_heatmap")

# ---------------------------------------------------------------------------
# Chart 2: roofline — why quantization helps memory-bound workloads
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 6))
ai = np.logspace(-1, 3, 400)  # arithmetic intensity (FLOP/byte)
peak = 40e12   # 40 TOPS illustrative
bw = 100e9     # 100 GB/s illustrative
roof = np.minimum(peak, bw * ai)
ax.loglog(ai, roof/1e12, color=PALETTE["ink"], lw=2, label="Roofline (attainable TOPS)")
ax.axhline(peak/1e12, color=PALETTE["muted"], ls="--", lw=0.8)
ax.axvline(peak/bw, color=PALETTE["muted"], ls=":", lw=0.8)
ax.text(peak/bw*1.1, 0.2, "ridge point", rotation=90, fontsize=8, color=PALETTE["muted"])
# workload markers
ax.scatter([0.5], [bw*0.5/1e12], s=120, color=PALETTE["teal"], zorder=5)
ax.annotate("LLM decode (memory-bound)\nquantize weights → move right→ faster",
            (0.5, bw*0.5/1e12), xytext=(0.6, 1.2), fontsize=8.2, color=PALETTE["teal"])
ax.scatter([50], [peak/1e12], s=120, color=PALETTE["red"], zorder=5)
ax.annotate("Vision CNN / prefill (compute-bound)\nquantize compute → raise roof",
            (50, peak/1e12), xytext=(3, peak/1e12*1.15), fontsize=8.2, color=PALETTE["red"])
ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
ax.set_ylabel("Attainable throughput (TOPS)")
ax.set_title("Roofline: quantization helps memory-bound and compute-bound workloads differently")
ax.legend(loc="lower right")
save(fig, "06_roofline")
print("done")
