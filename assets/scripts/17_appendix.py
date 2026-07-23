"""Section 17 charts: confidence-level and maturity distribution across the database."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import pandas as pd
import matplotlib.pyplot as plt
from chartstyle import apply_style, PALETTE, save
DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
apply_style()

df = pd.read_csv(os.path.join(DATA, "master_database.csv"))
# primary confidence = first token
df["conf"] = df["source_confidence"].str.split(";").str[0]
df["mat"] = df["maturity"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4))

# confidence distribution
cc = df["conf"].value_counts()
conf_colors = {"official-spec":PALETTE["green"],"sdk-docs":PALETTE["teal"],
               "paper":PALETTE["blue"],"press":PALETTE["amber"],
               "inferred":PALETTE["red"],"speculative":PALETTE["purple"]}
bars = ax1.bar(cc.index, cc.values, color=[conf_colors.get(c,PALETTE["slate"]) for c in cc.index])
for b in bars:
    ax1.annotate(f"{int(b.get_height())}",(b.get_x()+b.get_width()/2,b.get_height()),
                 xytext=(0,3),textcoords="offset points",ha="center",fontsize=9)
ax1.set_title("Source-confidence distribution across the database")
ax1.set_ylabel("Entities")
plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=8.5)

# maturity distribution
mm = df["mat"].value_counts()
mat_colors={"production-shipped":PALETTE["green"],"sdk-limited":PALETTE["amber"],"research-only":PALETTE["red"]}
bars2 = ax2.bar(mm.index, mm.values, color=[mat_colors.get(m,PALETTE["slate"]) for m in mm.index])
for b in bars2:
    ax2.annotate(f"{int(b.get_height())}",(b.get_x()+b.get_width()/2,b.get_height()),
                 xytext=(0,3),textcoords="offset points",ha="center",fontsize=9)
ax2.set_title("Maturity distribution across the database")
ax2.set_ylabel("Entities")
plt.setp(ax2.get_xticklabels(), rotation=20, ha="right", fontsize=8.5)

fig.suptitle("Database methodology: how the reference's claims are graded", fontsize=13, fontweight="bold")
save(fig, "17_methodology_distribution")
print("done")
