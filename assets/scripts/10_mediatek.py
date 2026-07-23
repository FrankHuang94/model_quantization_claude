"""Section 10 charts: Dimensity APU roadmap; MediaTek vs Qualcomm precision positioning."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chartstyle import apply_style, PALETTE, save
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
apply_style()

df = pd.read_csv(os.path.join(DATA, "10_mediatek.csv"))

# ---------------------------------------------------------------------------
# Chart 1: Dimensity APU roadmap (capability milestones over time)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5.6))
milestones = {
    2021: "APU 590\nINT8/FP16",
    2022: "APU 690\nhigher throughput",
    2023: "APU 790\nINT4 + transformer accel",
    2024: "APU 890\non-device LLM + spec. decode",
    2025: "next APU\nFP8 + LiteRT-NeuroPilot",
}
xs = list(milestones.keys())
ax.plot(xs, [1]*len(xs), color=PALETTE["teal"], lw=2, zorder=1)
for x, txt in milestones.items():
    ax.scatter(x, 1, s=180, color=PALETTE["blue"], edgecolor="white", zorder=3)
    ax.annotate(txt, (x, 1), xytext=(0, 18 if x % 2 == 1 else -34),
                textcoords="offset points", ha="center", fontsize=8.4, color=PALETTE["ink"])
    ax.annotate(df[df.year==x].chip.values[0], (x, 1), xytext=(0, 6),
                textcoords="offset points", ha="center", fontsize=7, color=PALETTE["muted"])
ax.set_ylim(0.4, 1.6); ax.set_yticks([])
ax.set_xlim(2020.5, 2025.7)
ax.set_xlabel("Year")
ax.set_title("MediaTek Dimensity APU quantization capability roadmap")
save(fig, "10_mediatek_roadmap")

# ---------------------------------------------------------------------------
# Chart 2: precision support MediaTek vs Qualcomm (side-by-side)
# ---------------------------------------------------------------------------
formats = ["FP16", "INT16", "INT8", "INT4", "FP8", "INT2"]
mtk = [2, 2, 2, 2, 1, 0]   # 2 native, 1 emerging, 0 none (2025 flagship)
qc =  [2, 2, 2, 2, 2, 2]
x = np.arange(len(formats)); w = 0.38
fig, ax = plt.subplots(figsize=(9.5, 5.6))
ax.bar(x - w/2, mtk, w, color=PALETTE["blue"], label="MediaTek Dimensity 9500")
ax.bar(x + w/2, qc, w, color=PALETTE["amber"], label="Qualcomm 8 Elite Gen 5")
ax.set_xticks(x); ax.set_xticklabels(formats)
ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["none", "emerging", "native"])
ax.set_title("Disclosed precision support: MediaTek vs. Qualcomm 2025 flagships")
ax.legend()
save(fig, "10_mediatek_vs_qualcomm")
print("done")
