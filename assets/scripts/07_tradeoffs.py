"""Section 07 charts: energy-efficiency gains by precision; multi-axis tradeoff."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from chartstyle import apply_style, PALETTE, save
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
apply_style()

df = pd.read_csv(os.path.join(DATA, "07_tradeoffs.csv"))

# ---------------------------------------------------------------------------
# Chart 1: relative energy per inference + accuracy overlay
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 6))
x = np.arange(len(df))
eff = (1 - df["rel_energy_per_inference"]) * 100  # % energy saved
bars = ax.bar(x, df["rel_energy_per_inference"], color=PALETTE["blue"], width=0.55,
              label="Relative energy / inference (vs FP16)")
ax.set_xticks(x); ax.set_xticklabels(df["precision"])
ax.set_ylabel("Relative energy per inference (FP16 = 1.0)")
ax.set_title("Energy per inference and accuracy by quantization level (illustrative)")
for i, (e, s) in enumerate(zip(df["rel_energy_per_inference"], eff)):
    ax.annotate(f"-{s:.0f}% energy", (i, e), xytext=(0, 4), textcoords="offset points",
                ha="center", fontsize=8, color=PALETTE["blue"])
ax2 = ax.twinx()
ax2.plot(x, df["accuracy_retention"], "-o", color=PALETTE["red"], lw=2.2,
         label="Accuracy retention (%)")
ax2.set_ylabel("Accuracy retention (% of FP16)", color=PALETTE["red"])
ax2.set_ylim(35, 103)
ax2.grid(False)
ax2.tick_params(axis="y", colors=PALETTE["red"])
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc="upper center")
save(fig, "07_energy_efficiency")

# ---------------------------------------------------------------------------
# Chart 2: normalized multi-axis (energy, memory, latency, accuracy-loss) bars
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
w = 0.2
metrics = [
    ("rel_energy_per_inference", "Energy/infer", PALETTE["blue"]),
    ("rel_memory", "Memory", PALETTE["teal"]),
    ("rel_latency_membound", "Latency (mem-bound)", PALETTE["amber"]),
]
for k, (col, lbl, c) in enumerate(metrics):
    ax.bar(x + (k-1)*w, df[col], w, color=c, label=lbl)
ax.set_xticks(x); ax.set_xticklabels(df["precision"])
ax.set_ylabel("Relative to FP16 (lower = better resource use)")
ax.set_title("Resource cost by precision: energy, memory, latency (all vs FP16 = 1.0)")
ax.legend(loc="upper right")
# accuracy annotation
for i, a in enumerate(df["accuracy_retention"]):
    ax.annotate(f"acc {a:.0f}%", (i, 1.02), ha="center", fontsize=7.6, color=PALETTE["red"])
ax.set_ylim(0, 1.15)
save(fig, "07_resource_tradeoff")
print("done")
