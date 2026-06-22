"""Draw a clean ST-HeteroSAGE architecture diagram as a PNG."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

OUT = Path(__file__).parent / "artifacts" / "figures" / "08_architecture.png"

NAVY = "#1f3b66"; GREEN = "#2ca02c"; BLUE = "#1f77b4"; ORANGE = "#ff7f0e"
LIGHT = "#eef3fa"; LIGHTG = "#eaf6ea"; LIGHTO = "#fdf0e3"

fig, ax = plt.subplots(figsize=(11, 9))
ax.set_xlim(0, 10); ax.set_ylim(0, 12); ax.axis("off")


def box(x, y, w, h, text, fc, ec, fs=10, bold=False, tc="#111111"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                       linewidth=1.6, edgecolor=ec, facecolor=fc)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, wrap=True)


def arrow(x1, y1, x2, y2, label=None, color="#333333"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=18,
                        linewidth=1.8, color=color)
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2 + 0.25, (y1 + y2) / 2, label, ha="left", va="center",
                fontsize=8.5, color=color, style="italic")


# Inputs
box(0.6, 10.6, 4.0, 1.1, "hour nodes\n201,596 × 17 features\n(lags · weather · fundamentals · calendar)",
    LIGHT, NAVY, 9)
box(5.6, 10.6, 3.8, 1.1, "market nodes\n4 × one-hot\n(DK1, DK2, HYDRO, DE)", LIGHT, NAVY, 9)

# Projection
box(0.6, 9.0, 8.8, 0.9, "Input projection → shared 128-dim hidden space\n(hour: Linear+ReLU   |   market: 2-layer residual MLP)",
    "#ffffff", NAVY, 9, bold=True, tc=NAVY)
arrow(2.6, 10.6, 2.6, 9.9)
arrow(7.5, 10.6, 7.5, 9.9)

# ST block container
big = FancyBboxPatch((0.4, 2.7), 9.2, 5.9, boxstyle="round,pad=0.05,rounding_size=0.15",
                     linewidth=2.2, edgecolor=NAVY, facecolor="#f7f9fc", linestyle="--")
ax.add_patch(big)
ax.text(5.0, 8.35, "ST BLOCK   ( × 2 stacked )", ha="center", va="center",
        fontsize=12, fontweight="bold", color=NAVY)

# Spatial
box(1.0, 6.4, 8.0, 1.3,
    "① SPATIAL  —  HeteroConv (GraphSAGE)\nsame-hour message passing across zones\n"
    "co_occurs_with · belongs_to · interconnects  →  BatchNorm + ReLU + residual",
    LIGHTG, GREEN, 9.5, tc="#14491f")

# reshape note + arrow
arrow(5.0, 6.4, 5.0, 5.5, "reshape [4T,H] → [4,H,T]")

# Temporal
box(1.0, 4.0, 8.0, 1.4,
    "② TEMPORAL  —  Causal TCN  (per zone)\ndilated 1-D convs · kernel 7 · dilations (1,4,24)\n"
    "~174h (≈ 1 week) receptive field · CAUSAL (past only → no leakage)\n→ BatchNorm + ReLU + residual",
    LIGHTO, ORANGE, 9.5, tc="#6b3b00")

arrow(5.0, 9.0, 5.0, 7.75)            # projection → ST block
arrow(5.0, 4.0, 5.0, 3.05)            # temporal → head

# Head
box(2.6, 2.0, 4.8, 0.95, "Regression head\nLinear 128 → 64 → ReLU → 1", "#ffffff", NAVY, 10, bold=True, tc=NAVY)
arrow(5.0, 2.7, 5.0, 2.95, color=NAVY)  # block boundary into head (visual)
arrow(5.0, 2.0, 5.0, 1.2, color=GREEN)

# Output
box(2.6, 0.25, 4.8, 0.9, "Predicted price (DKK/MWh)\nloss masked to DK1 + DK2 only",
    LIGHTG, GREEN, 10, bold=True, tc="#14491f")

ax.text(5.0, 11.95, "ST-HeteroSAGE — Architecture Data Flow", ha="center", va="top",
        fontsize=14, fontweight="bold", color=NAVY)

fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print("Saved", OUT)
