"""Section 12 charts: landmark-method output over time; landmark methods by lab."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chartstyle import apply_style, PALETTE, save
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
apply_style()

# ---------------------------------------------------------------------------
# Chart 1: landmark methods per year + cumulative
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(DATA, "12_paper_output.csv"))
fig, ax = plt.subplots(figsize=(10, 5.8))
ax.bar(df["year"], df["landmark_methods"], color=PALETTE["blue"], width=0.6,
       label="Landmark quantization methods per year")
ax.set_xlabel("Year"); ax.set_ylabel("Landmark methods (count)", color=PALETTE["blue"])
ax.set_title("Quantization research output: landmark methods per year and cumulative (illustrative)")
ax2 = ax.twinx()
ax2.plot(df["year"], df["cumulative"], "-o", color=PALETTE["red"], lw=2.2,
         label="Cumulative")
ax2.set_ylabel("Cumulative landmark methods", color=PALETTE["red"])
ax2.grid(False)
ax2.tick_params(axis="y", colors=PALETTE["red"])
ax.axvspan(2021.5, 2024.5, color=PALETTE["amber"], alpha=0.08)
ax.text(2023, 5.5, "LLM-quant\nacceleration", ha="center", fontsize=8.5, color=PALETTE["amber"])
l1, la1 = ax.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
ax.legend(l1+l2, la1+la2, loc="upper left")
save(fig, "12_paper_output")

# ---------------------------------------------------------------------------
# Chart 2: landmark methods by lab
# ---------------------------------------------------------------------------
d2 = pd.read_csv(os.path.join(DATA, "12_lab_contributions.csv")).sort_values("landmark_methods")
fig, ax = plt.subplots(figsize=(9.5, 6))
ax.barh(d2["lab"], d2["landmark_methods"], color=PALETTE["teal"], edgecolor="white")
for i, (v, f) in enumerate(zip(d2["landmark_methods"], d2["focus"])):
    ax.annotate(f"{v}  · {f}", (v, i), xytext=(4, 0), textcoords="offset points",
                va="center", fontsize=8, color=PALETTE["ink"])
ax.set_xlabel("Landmark quantization methods (illustrative count)")
ax.set_title("Leading labs by landmark quantization-method contributions")
ax.set_xlim(0, 8)
save(fig, "12_lab_contributions")
print("done")
