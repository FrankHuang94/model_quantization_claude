"""Section 02 charts: milestone timeline + era-progression of achievable bit-width."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from chartstyle import apply_style, PALETTE, SERIES, save
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
apply_style()

ERA_COLORS = {
    "binary/ternary": PALETTE["red"],
    "int8-standardization": PALETTE["blue"],
    "llm-era": PALETTE["teal"],
    "sub-4bit": PALETTE["purple"],
}
ERA_LABELS = {
    "binary/ternary": "Binary/ternary era (2015–16)",
    "int8-standardization": "INT8 standardization (2017–21)",
    "llm-era": "LLM-quantization era (2022–23)",
    "sub-4bit": "Sub-4-bit / FP-low era (2024–26)",
}

# ---------------------------------------------------------------------------
# Chart 1: milestone timeline (lollipop, lanes to avoid overlap)
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(DATA, "02_milestones.csv"))
fig, ax = plt.subplots(figsize=(13.5, 7.2))
# jitter x within year by lane for readability
for _, r in df.iterrows():
    x = r["year"] + (r["lane"] - 1.5) * 0.16
    y = r["lane"]
    c = ERA_COLORS[r["era"]]
    ax.plot([x, x], [0, y], color=c, lw=1.4, alpha=0.6, zorder=1)
    ax.scatter(x, y, s=70, color=c, edgecolor="white", linewidth=1, zorder=3)
    ax.annotate(r["milestone"], (x, y), xytext=(0, 8), textcoords="offset points",
                rotation=32, fontsize=7.6, ha="left", va="bottom", color=PALETTE["ink"])
ax.axhline(0, color=PALETTE["muted"], lw=1.2)
ax.set_ylim(-0.4, 4.6)
ax.set_xlim(2014.4, 2026.9)
ax.set_yticks([])
ax.set_xticks(range(2015, 2027))
ax.set_xlabel("Year")
ax.set_title("Landmark quantization milestones, 2015–2026")
ax.legend(handles=[Patch(color=ERA_COLORS[k], label=ERA_LABELS[k]) for k in ERA_LABELS],
          loc="upper left", ncol=2)
ax.spines["left"].set_visible(False)
save(fig, "02_milestone_timeline")

# ---------------------------------------------------------------------------
# Chart 2: "achievable production bit-width over time" (stepped)
# ---------------------------------------------------------------------------
years = [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
# practical production floor for *general* deployment (bits) — narrative, illustrative
vision_floor = [16, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 4]
llm_floor =    [None, None, None, None, None, None, 16, 8, 4, 4, 4, 4]
research_floor=[1, 1, 2, 2, 2, 2, 2, 3, 2, 1.58, 1.58, 1.58]
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.step(years, vision_floor, where="post", color=PALETTE["blue"], lw=2.4,
        label="Vision CNN — general production floor")
llm_years = [y for y, v in zip(years, llm_floor) if v is not None]
llm_vals = [v for v in llm_floor if v is not None]
ax.step(llm_years, llm_vals, where="post", color=PALETTE["teal"], lw=2.4,
        label="On-device LLM — general production floor")
ax.step(years, research_floor, where="post", color=PALETTE["red"], lw=2.0,
        ls="--", label="Research frontier (demonstrated)")
ax.set_yticks([1, 1.58, 2, 4, 8, 16])
ax.set_yticklabels(["1 (binary)", "1.58 (ternary)", "2", "4", "8", "16 (FP)"])
ax.invert_yaxis()
ax.set_xlabel("Year"); ax.set_ylabel("Bit-width (lower = more aggressive)")
ax.set_title("Descending bit-width: production floor vs. research frontier over time")
ax.legend(loc="lower left")
save(fig, "02_bitwidth_over_time")
print("done")
