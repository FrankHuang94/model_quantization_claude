"""Section 03 charts: granularity vs accuracy, PTQ vs QAT recovery."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from chartstyle import apply_style, PALETTE, SERIES, save
apply_style()

# ---------------------------------------------------------------------------
# Chart 1: accuracy retention vs bit-width by granularity (illustrative curves)
# ---------------------------------------------------------------------------
bits = np.array([8, 6, 5, 4, 3, 2])
# accuracy retention (% of FP16) — illustrative, based on typical LLM PTQ behavior
per_tensor = [99.5, 98.6, 97.2, 92.0, 74.0, 45.0]
per_channel = [99.8, 99.4, 98.8, 96.0, 85.0, 60.0]
per_group128 = [99.9, 99.7, 99.4, 98.3, 92.0, 74.0]
per_group32 = [99.95, 99.85, 99.7, 99.1, 95.5, 82.0]
fig, ax = plt.subplots(figsize=(9.5, 6))
for y, lbl, c in [
    (per_tensor, "Per-tensor", PALETTE["red"]),
    (per_channel, "Per-channel", PALETTE["amber"]),
    (per_group128, "Per-group (g=128)", PALETTE["blue"]),
    (per_group32, "Per-group (g=32)", PALETTE["teal"]),
]:
    ax.plot(bits, y, "-o", color=c, lw=2.2, markersize=6, label=lbl)
ax.set_xticks(bits)
ax.invert_xaxis()
ax.set_ylim(40, 101)
ax.set_xlabel("Weight bit-width")
ax.set_ylabel("Accuracy retention (% of FP16 baseline)")
ax.set_title("Quantization granularity vs. accuracy retention (illustrative, LLM weight-only PTQ)")
ax.axhline(99, color=PALETTE["muted"], lw=0.8, ls=":")
ax.text(8, 99.2, "99% retention", fontsize=8, color=PALETTE["muted"])
ax.legend(title="Scale granularity", loc="lower left")
ax.annotate("finer granularity\n→ smaller error\n→ more scale overhead",
            xy=(3, 92), xytext=(5.2, 68), fontsize=8.5, color=PALETTE["muted"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["muted"]))
save(fig, "03_granularity_accuracy")

# ---------------------------------------------------------------------------
# Chart 2: PTQ vs QAT accuracy across bit-widths (grouped bars)
# ---------------------------------------------------------------------------
labels = ["INT8", "INT6", "INT4", "INT3", "INT2"]
ptq = [99.6, 98.5, 94.0, 80.0, 52.0]
qat = [99.9, 99.6, 98.2, 93.0, 78.0]
x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(9.5, 5.8))
b1 = ax.bar(x - w/2, ptq, w, label="PTQ (post-training)", color=PALETTE["amber"])
b2 = ax.bar(x + w/2, qat, w, label="QAT (quant-aware training)", color=PALETTE["teal"])
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(40, 102)
ax.set_ylabel("Accuracy retention (% of FP baseline)")
ax.set_title("PTQ vs. QAT accuracy by bit-width (illustrative; the gap widens as bits fall)")
ax.legend()
for b in list(b1) + list(b2):
    ax.annotate(f"{b.get_height():.0f}", (b.get_x()+b.get_width()/2, b.get_height()),
                xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
save(fig, "03_ptq_vs_qat")
print("done")
