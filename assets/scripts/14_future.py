"""Section 14 charts: projected production bit-width frontier; future-tech adoption timeline."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from chartstyle import apply_style, PALETTE, save
apply_style()

# ---------------------------------------------------------------------------
# Chart 1: projected production-viable bit-width (LLM weights) 2020-2030
# ---------------------------------------------------------------------------
years = [2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030]
hist = [16, 16, 8, 4, 4, 4, 4, None, None, None, None]
proj = [None]*6 + [4, 3.5, 3, 2.5, 2]
fig, ax = plt.subplots(figsize=(10, 5.8))
hy = [y for y, v in zip(years, hist) if v is not None]
hv = [v for v in hist if v is not None]
ax.plot(hy, hv, "-o", color=PALETTE["blue"], lw=2.4, label="Historical (production floor)")
py = [y for y, v in zip(years, proj) if v is not None]
pv = [v for v in proj if v is not None]
ax.plot(py, pv, "--o", color=PALETTE["red"], lw=2.2, label="Projected (⚠️ speculative)")
ax.fill_between(py, pv, [p-0.7 for p in pv], color=PALETTE["red"], alpha=0.08)
ax.axvline(2026, color=PALETTE["muted"], ls=":", lw=0.8)
ax.text(2026.05, 14, "now", fontsize=8, color=PALETTE["muted"])
ax.set_yticks([2, 2.5, 3, 3.5, 4, 8, 16]); ax.invert_yaxis()
ax.set_xlabel("Year"); ax.set_ylabel("Production-viable LLM weight bit-width")
ax.set_title("Projected descent of the production bit-width frontier (post-2026 speculative)")
ax.legend(loc="lower left")
save(fig, "14_bitwidth_projection")

# ---------------------------------------------------------------------------
# Chart 2: anticipated adoption timeline of future quantization tech (Gantt)
# ---------------------------------------------------------------------------
techs = [
    ("FP4 / MXFP4 mainstream (edge)", 2026, 2028, PALETTE["amber"]),
    ("Rotation W4A4 production", 2026, 2028, PALETTE["teal"]),
    ("2-bit codebook fast kernels", 2027, 2029, PALETTE["purple"]),
    ("Quantization-native models at scale", 2027, 2030, PALETTE["red"]),
    ("Diffusion/multimodal edge quant", 2026, 2029, PALETTE["blue"]),
    ("In-memory compute products", 2027, 2030, PALETTE["green"]),
    ("Tooling consolidation (MX/MLIR)", 2026, 2029, PALETTE["slate"]),
    ("Automated hardware-aware quant", 2027, 2030, PALETTE["cyan"]),
]
fig, ax = plt.subplots(figsize=(11, 5.8))
for i, (name, s, e, c) in enumerate(techs):
    ax.barh(i, e - s, left=s, height=0.55, color=c, edgecolor="white", alpha=0.85)
    ax.annotate(f"{s}–{e}", (s, i), xytext=(-4, 0), textcoords="offset points",
                va="center", ha="right", fontsize=7.5, color=PALETTE["ink"])
ax.set_yticks(range(len(techs))); ax.set_yticklabels([t[0] for t in techs], fontsize=8.6)
ax.invert_yaxis()
ax.set_xlim(2025.5, 2030.5)
ax.set_xlabel("Anticipated production-adoption window (⚠️ speculative)")
ax.set_title("Anticipated adoption timeline of emerging quantization technologies")
save(fig, "14_adoption_timeline")
print("done")
