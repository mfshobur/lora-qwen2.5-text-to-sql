import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE  = {"exec_accuracy": 0.434, "exec_match": 0.179}
FT    = {"exec_accuracy": 0.893, "exec_match": 0.523}

labels  = ["Execution accuracy", "Execution match"]
base_v  = [BASE["exec_accuracy"], BASE["exec_match"]]
ft_v    = [FT["exec_accuracy"],   FT["exec_match"]]

x     = np.arange(len(labels))
width = 0.32
gray  = "#94a3b8"
indigo = "#6366f1"

fig, ax = plt.subplots(figsize=(12, 6.75), dpi=100)
fig.patch.set_facecolor("white")

bars_base = ax.bar(x - width / 2, [v * 100 for v in base_v],  width, color=gray,   label="Base model",   zorder=3)
bars_ft   = ax.bar(x + width / 2, [v * 100 for v in ft_v],    width, color=indigo, label="Fine-tuned (LoRA)", zorder=3)

for bar in bars_base:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=12, color="#334155")

for bar in bars_ft:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=12, color=indigo, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=14)
ax.set_ylim(0, 105)
ax.set_ylabel("Score (%)", fontsize=12, color="#64748b")
ax.yaxis.set_tick_params(labelcolor="#64748b")
ax.set_axisbelow(True)
ax.yaxis.grid(True, color="#f1f5f9", linewidth=1.2)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.spines["bottom"].set_color("#e2e8f0")

ax.set_title(
    "Fine-tuning Qwen2.5-0.5B with LoRA",
    fontsize=16, fontweight="bold", color="#1e293b", pad=14,
)
ax.text(0.5, 1.01, "Evaluated on 419 held-out test questions",
        transform=ax.transAxes, ha="center", fontsize=11, color="#64748b")

ax.legend(handles=[
    mpatches.Patch(color=gray,   label="Base model"),
    mpatches.Patch(color=indigo, label="Fine-tuned (LoRA)"),
], fontsize=12, frameon=False, loc="upper left")

plt.tight_layout()
plt.savefig("chart.png", dpi=100, bbox_inches="tight", facecolor="white")
print("Wrote chart.png")
