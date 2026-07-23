"""Section 01 charts: technique maturity quadrant + players silicon-vs-tooling."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from chartstyle import apply_style, PALETTE, MATURITY_COLORS, save, CHART_DIR
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
apply_style()

# ---------------------------------------------------------------------------
# Chart 1: technique maturity quadrant (accuracy retention vs production adoption)
# ---------------------------------------------------------------------------
tech = pd.read_csv(os.path.join(DATA, "01_technique_maturity.csv"))
fig, ax = plt.subplots(figsize=(9.2, 6.4))
cmap = {"production": MATURITY_COLORS["production"],
        "sdk": MATURITY_COLORS["sdk"],
        "research": MATURITY_COLORS["research"]}
for _, r in tech.iterrows():
    ax.scatter(r["adoption"], r["accuracy_retention"], s=180,
               color=cmap[r["maturity"]], edgecolor="white", linewidth=1.2,
               zorder=3)
    ax.annotate(r["technique"], (r["adoption"], r["accuracy_retention"]),
                xytext=(6, 6), textcoords="offset points", fontsize=8.5,
                color=PALETTE["ink"])
ax.axvline(5, color=PALETTE["muted"], lw=0.8, ls="--", zorder=1)
ax.axhline(5, color=PALETTE["muted"], lw=0.8, ls="--", zorder=1)
ax.set_xlim(0, 10.5); ax.set_ylim(0, 10.5)
ax.set_xlabel("Production adoption  (0 = research-only  →  10 = ubiquitous)")
ax.set_ylabel("Accuracy retention vs FP16  (0 = poor  →  10 = near-lossless)")
ax.set_title("Quantization technique landscape: accuracy retention vs. production adoption")
legend = [Line2D([0],[0], marker="o", color="w", markerfacecolor=cmap[k],
                 markersize=11, label=v)
          for k, v in {"production":"Production-standard",
                       "sdk":"SDK / emerging",
                       "research":"Research-stage"}.items()]
ax.legend(handles=legend, loc="lower right", title="Maturity")
ax.text(2.5, 9.7, "High accuracy · low adoption", color=PALETTE["muted"], fontsize=8, ha="center")
ax.text(7.8, 0.4, "Aggressive · adoption-limited", color=PALETTE["muted"], fontsize=8, ha="center")
save(fig, "01_technique_maturity_quadrant")

# ---------------------------------------------------------------------------
# Chart 2: players — silicon precision support vs software tooling maturity
# ---------------------------------------------------------------------------
pl = pd.read_csv(os.path.join(DATA, "01_players_silicon_tooling.csv"))
fig, ax = plt.subplots(figsize=(9.2, 6.4))
for _, r in pl.iterrows():
    ax.scatter(r["silicon_score"], r["tooling_score"], s=210,
               color=PALETTE["blue"], edgecolor="white", linewidth=1.2, zorder=3)
    ax.annotate(r["player"], (r["silicon_score"], r["tooling_score"]),
                xytext=(7, -3), textcoords="offset points", fontsize=8.8,
                color=PALETTE["ink"])
ax.axvline(5, color=PALETTE["muted"], lw=0.8, ls="--")
ax.axhline(5, color=PALETTE["muted"], lw=0.8, ls="--")
ax.set_xlim(0, 10.5); ax.set_ylim(0, 10.5)
ax.set_xlabel("Silicon precision support  (breadth of INT/FP formats in shipping NPUs)")
ax.set_ylabel("Software / quantization tooling maturity")
ax.set_title("Edge-AI players: silicon precision support vs. quantization tooling")
ax.text(2.5, 9.7, "Tooling-led", color=PALETTE["muted"], fontsize=8, ha="center")
ax.text(8.2, 0.4, "Silicon-led", color=PALETTE["muted"], fontsize=8, ha="center")
save(fig, "01_players_silicon_vs_tooling")
print("done")
