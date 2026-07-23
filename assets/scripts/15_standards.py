"""Section 15 charts: benchmark coverage matrix; representative results across chips."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from chartstyle import apply_style, PALETTE, save
apply_style()

# ---------------------------------------------------------------------------
# Chart 1: benchmark suite x coverage-dimension matrix
# ---------------------------------------------------------------------------
suites = ["MLPerf Inference\n(DC/Edge)", "MLPerf Mobile", "MLPerf Tiny", "MLPerf Client",
          "AI-Benchmark\n(ETH)", "Geekbench AI", "Procyon AI", "lm-eval-harness\n(LLM)"]
dims = ["Vision", "LLM/genAI", "Mobile NPU", "tinyML", "Quant-specific", "Public/comparable"]
# 0 none, 1 partial, 2 strong
M = np.array([
    [2,1,1,0,1,2],  # MLPerf Inference
    [2,1,2,0,1,2],  # MLPerf Mobile
    [1,0,0,2,1,2],  # MLPerf Tiny
    [2,2,1,0,1,2],  # MLPerf Client
    [2,1,2,0,1,1],  # AI-Benchmark
    [2,1,2,0,0,1],  # Geekbench AI
    [2,1,1,0,0,1],  # Procyon AI
    [0,2,0,0,2,2],  # lm-eval-harness
])
cmap = ListedColormap([PALETTE["grid"], PALETTE["amber"], PALETTE["teal"]])
norm = BoundaryNorm([-0.5,0.5,1.5,2.5], cmap.N)
fig, ax = plt.subplots(figsize=(9.8, 6.2))
ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
ax.set_xticks(range(len(dims))); ax.set_xticklabels(dims, fontsize=9)
ax.set_yticks(range(len(suites))); ax.set_yticklabels(suites, fontsize=8.5)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        t = {0:"—",1:"◐",2:"●"}[M[i,j]]
        ax.text(j, i, t, ha="center", va="center", fontsize=11,
                color="white" if M[i,j]==2 else PALETTE["ink"])
ax.set_title("Benchmark suite coverage across dimensions (● strong · ◐ partial · — none)")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=PALETTE["teal"],label="strong"),
                   Patch(color=PALETTE["amber"],label="partial"),
                   Patch(color=PALETTE["grid"],label="none")],
          loc="upper left", bbox_to_anchor=(1.01,1.0))
save(fig, "15_benchmark_coverage")

# ---------------------------------------------------------------------------
# Chart 2: representative relative on-device LLM throughput (illustrative)
# ---------------------------------------------------------------------------
chips = ["Snapdragon\n8 Elite Gen5", "Dimensity\n9500", "Apple\nM5", "Intel\nCore Ultra 300", "AMD\nRyzen AI 300"]
tok_s = [55, 50, 60, 45, 45]  # illustrative relative on-device 7B-4bit decode tok/s
fig, ax = plt.subplots(figsize=(9.5, 5.2))
bars = ax.bar(chips, tok_s, color=[PALETTE["blue"],PALETTE["teal"],PALETTE["slate"],
                                   PALETTE["amber"],PALETTE["red"]], width=0.6)
for b in bars:
    ax.annotate(f"~{b.get_height():.0f}", (b.get_x()+b.get_width()/2, b.get_height()),
                xytext=(0,3), textcoords="offset points", ha="center", fontsize=9)
ax.set_ylabel("Illustrative 7B 4-bit decode throughput (tok/s)")
ax.set_title("Representative on-device LLM decode throughput (ILLUSTRATIVE — not measured/comparable)")
ax.set_ylim(0, 72)
ax.text(0.5, -0.22, "Values are illustrative placeholders, NOT independent benchmark results — see Section 04 on TOPS/throughput caveats.",
        transform=ax.transAxes, ha="center", fontsize=7.5, color=PALETTE["red"])
save(fig, "15_representative_results")
print("done")
