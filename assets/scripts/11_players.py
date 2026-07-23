"""Section 11 charts: multi-player capability radar; AI-PC NPU TOPS bar."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chartstyle import apply_style, PALETTE, SERIES, save
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
apply_style()

df = pd.read_csv(os.path.join(DATA, "11_players_radar.csv"))
dims = ["silicon_breadth", "tooling_maturity", "ondevice_llm", "quant_research",
        "market_reach", "openness"]
dim_labels = ["Silicon\nprecision breadth", "Tooling\nmaturity", "On-device\nLLM support",
              "Quantization\nresearch", "Market\nreach", "Openness /\nportability"]

# ---------------------------------------------------------------------------
# Chart 1: radar of selected players
# ---------------------------------------------------------------------------
angles = np.linspace(0, 2*np.pi, len(dims), endpoint=False).tolist()
angles += angles[:1]
fig, ax = plt.subplots(figsize=(8.6, 8.6), subplot_kw=dict(polar=True))
show = ["NVIDIA", "Intel", "Google", "AMD", "ARM"]
for i, p in enumerate(show):
    row = df[df.player == p][dims].values.flatten().tolist()
    row += row[:1]
    ax.plot(angles, row, color=SERIES[i], lw=2, label=p)
    ax.fill(angles, row, color=SERIES[i], alpha=0.06)
ax.set_xticks(angles[:-1]); ax.set_xticklabels(dim_labels, fontsize=9)
ax.set_ylim(0, 10); ax.set_yticks([2,4,6,8,10])
ax.set_yticklabels(["2","4","6","8","10"], fontsize=8, color=PALETTE["muted"])
ax.set_title("Capability comparison across edge-AI players (illustrative 0–10 scoring)", pad=24)
ax.legend(loc="upper right", bbox_to_anchor=(1.22, 1.1))
save(fig, "11_players_radar")

# ---------------------------------------------------------------------------
# Chart 2: AI-PC / edge NPU peak TOPS (vendor-claimed) bar
# ---------------------------------------------------------------------------
npu = [
    ("Intel NPU5 (Core Ultra 300)", 50, PALETTE["blue"]),
    ("AMD XDNA 2 (Strix)", 50, PALETTE["red"]),
    ("Qualcomm Snapdragon X", 45, PALETTE["amber"]),
    ("Apple M4 (NE)", 38, PALETTE["slate"]),
    ("Intel NPU (Meteor Lake)", 11, PALETTE["cyan"]),
]
names = [n for n, _, _ in npu]; vals = [v for _, v, _ in npu]; cols = [c for _, _, c in npu]
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.barh(names, vals, color=cols, edgecolor="white")
for i, v in enumerate(vals):
    ax.annotate(f"{v} TOPS", (v, i), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=PALETTE["ink"])
ax.set_xlabel("Vendor-claimed NPU peak throughput (TOPS, mixed precision — not comparable across vendors)")
ax.set_title("AI-PC / edge NPU peak TOPS (vendor claims — read with Section 04 caveats)")
ax.set_xlim(0, 62)
ax.invert_yaxis()
save(fig, "11_aipc_npu_tops")
print("done")
