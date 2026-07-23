"""Section 08 charts: ANE TOPS growth timeline; Apple quantization capability roadmap."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chartstyle import apply_style, PALETTE, save
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
apply_style()

df = pd.read_csv(os.path.join(DATA, "08_apple_ane.csv"))

# ---------------------------------------------------------------------------
# Chart 1: ANE vendor-TOPS growth over time (A-series vs M-series)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
for line, c in [("iPhone", PALETTE["blue"]), ("Mac", PALETTE["amber"])]:
    d = df[df["line"] == line]
    ax.plot(d["year"], d["tops_vendor"], "-o", color=c, lw=2.2, markersize=6,
            label=f"{line} (A-series)" if line == "iPhone" else f"{line} (M-series)")
    for _, r in d.iterrows():
        ax.annotate(r["chip"], (r["year"], r["tops_vendor"]), xytext=(0, 7),
                    textcoords="offset points", fontsize=7, ha="center", color=PALETTE["ink"])
ax.set_xlabel("Year"); ax.set_ylabel("Neural Engine peak throughput (vendor-claimed TOPS)")
ax.set_title("Apple Neural Engine throughput growth, 2017–2025 (vendor TOPS — not independently verified)")
ax.legend(loc="upper left")
ax.annotate("A11: first ANE\n2 cores, 0.6 TOPS", (2017, 0.6), xytext=(2017.3, 8),
            fontsize=8, color=PALETTE["muted"], arrowprops=dict(arrowstyle="->", color=PALETTE["muted"]))
save(fig, "08_apple_ane_timeline")

# ---------------------------------------------------------------------------
# Chart 2: Apple quantization capability roadmap (Gantt-style bars)
# ---------------------------------------------------------------------------
caps = [
    ("INT8 Core ML quantization", 2017, 2025),
    ("Weight palettization (1-8 bit)", 2021, 2025),
    ("Per-grouped-channel palettization", 2024, 2025),
    ("INT4 block-wise weight quant", 2024, 2025),
    ("Stateful KV cache (Core ML)", 2024, 2025),
    ("MLX framework (on-device LLM)", 2023, 2025),
    ("Apple Intelligence (~3B on-device)", 2024, 2025),
    ("M5 GPU Neural Accelerators", 2025, 2025.6),
]
fig, ax = plt.subplots(figsize=(10.5, 5.6))
for i, (name, s, e) in enumerate(caps):
    ax.barh(i, e - s if e > s else 0.35, left=s, height=0.55,
            color=PALETTE["teal"], edgecolor="white")
    ax.annotate(f"{int(s)}", (s, i), xytext=(-6, 0), textcoords="offset points",
                va="center", ha="right", fontsize=8, color=PALETTE["ink"])
ax.set_yticks(range(len(caps))); ax.set_yticklabels([c[0] for c in caps], fontsize=8.8)
ax.invert_yaxis()
ax.set_xlim(2016.5, 2026)
ax.set_xlabel("Year introduced →")
ax.set_title("Apple on-device quantization capability roadmap")
save(fig, "08_apple_roadmap")
print("done")
