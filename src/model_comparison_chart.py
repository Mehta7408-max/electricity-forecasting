"""
Model comparison bar charts — four metrics, four models (GAT excluded).
Output: src/artifacts/model_comparison.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT_PATH = Path(__file__).parent / "artifacts" / "model_comparison.png"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Data ──────────────────────────────────────────────────────────────────────
MODELS = [
    "XGBoost",
    "HomoGNN\n(GraphSAGE)",
    "HeteroSAGE",
    "ST-HeteroSAGE\n(ours)",
]

METRICS = {
    "MAE  (DKK)  ↓ lower is better": {
        "values": [179.49, 173.19, 162.78, 151.08],
        "lower_better": True,
    },
    "RMSE  (DKK)  ↓ lower is better": {
        "values": [243.43, 225.73, 213.32, 204.36],
        "lower_better": True,
    },
    "R²  ↑ higher is better": {
        "values": [0.568, 0.629, 0.668, 0.696],
        "lower_better": False,
    },
    "sMAPE  (%)  ↓ lower is better": {
        "values": [54.51, 57.44, 52.20, 51.64],
        "lower_better": True,
    },
}

# ── Colours ───────────────────────────────────────────────────────────────────
C_WIN    = "#27AE60"   # winner
C_BASE   = "#5B8DB8"   # regular bars
C_BG     = "#F7F9FC"
C_GRID   = "#DDDDDD"
C_TEXT   = "#1C2833"
C_ANNOT  = "#FFFFFF"   # value labels on bars

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor(C_BG)
fig.suptitle(
    "Model Comparison  —  Test Set Performance (DK1 + DK2)",
    fontsize=14, fontweight="bold", color=C_TEXT, y=1.005,
)

axes_flat = axes.flatten()
x = np.arange(len(MODELS))
bar_w = 0.52

for ax, (title, meta) in zip(axes_flat, METRICS.items()):
    values = meta["values"]
    lower_better = meta["lower_better"]

    # Identify winner index
    win_idx = int(np.argmin(values) if lower_better else np.argmax(values))

    colors = [C_WIN if i == win_idx else C_BASE for i in range(len(MODELS))]

    bars = ax.bar(x, values, width=bar_w, color=colors,
                  edgecolor="white", linewidth=0.8, zorder=3)

    # Value labels inside / above bars
    for bar, val, c in zip(bars, values, colors):
        bar_h = bar.get_height()
        # For R² use 3 decimals, others use 2
        fmt = f"{val:.3f}" if "R²" in title else f"{val:.1f}"
        # Place label inside bar if tall enough, else above
        if bar_h > max(values) * 0.12:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar_h * 0.5, fmt,
                    ha="center", va="center",
                    fontsize=9.5, fontweight="bold", color=C_ANNOT, zorder=4)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar_h + max(values) * 0.01, fmt,
                    ha="center", va="bottom",
                    fontsize=9.5, fontweight="bold", color=C_TEXT, zorder=4)

    ax.set_title(title, fontsize=10.5, fontweight="bold",
                 color=C_TEXT, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=9, color=C_TEXT)
    ax.set_facecolor(C_BG)
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(C_GRID)
    ax.spines["bottom"].set_color(C_GRID)
    ax.tick_params(colors=C_TEXT, labelsize=9)

    # Tight y-axis: start just below min value for better visual diff
    margin = (max(values) - min(values)) * 0.25
    if lower_better:
        ax.set_ylim(max(0, min(values) - margin), max(values) + margin * 1.5)
    else:
        ax.set_ylim(max(0, min(values) - margin), min(1.0, max(values) + margin))

# ── Legend ────────────────────────────────────────────────────────────────────
legend_handles = [
    mpatches.Patch(facecolor=C_WIN,  edgecolor="white", label="ST-HeteroSAGE (winner)"),
    mpatches.Patch(facecolor=C_BASE, edgecolor="white", label="Comparison models"),
]
fig.legend(handles=legend_handles, loc="lower center",
           bbox_to_anchor=(0.50, -0.04), ncol=2,
           fontsize=10, framealpha=0.9, edgecolor=C_GRID)

plt.tight_layout(rect=[0, 0.03, 1, 1])
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor=C_BG)
plt.close()
print(f"Saved -> {OUT_PATH}")
