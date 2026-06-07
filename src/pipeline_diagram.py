"""
Generates a compact landscape pipeline architecture diagram for the
ST-HeteroSAGE electricity price forecasting system.

Horizontal left-to-right flow across five phases — fits a single
report page (landscape or half-page portrait).

Output: src/artifacts/pipeline_diagram.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

OUT_DIR = Path(__file__).parent / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "pipeline_diagram.png"

# ── Colours ───────────────────────────────────────────────────────────────────
C_DATA   = "#2E86C1"
C_PIPE   = "#7D3C98"
C_HOMO   = "#D68910"
C_HETERO = "#1E8449"
C_XGB    = "#C0392B"
C_EVAL   = "#1A5276"
C_WIN    = "#27AE60"
C_BG     = "#F7F9FC"
C_ARROW  = "#666666"
C_TEXT   = "#1C2833"


def box(ax, cx, cy, w, h, lines, fc, ec, lw=1.5, sizes=None, bold_first=True):
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.010",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
    )
    ax.add_patch(patch)
    if sizes is None:
        sizes = [9] * len(lines)
    n = len(lines)
    offs = [0] if n == 1 else [h * 0.58 / (n - 1) * (i - (n - 1) / 2) for i in range(n)]
    for i, (t, s, dy) in enumerate(zip(lines, sizes, offs)):
        ax.text(cx, cy - dy, t, ha="center", va="center",
                fontsize=s, fontweight="bold" if (bold_first and i == 0) else "normal",
                color=C_TEXT if i == 0 else "#444444", zorder=4)


def arr(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.5, rad=0.0):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=13,
                                connectionstyle=f"arc3,rad={rad}"), zorder=2)


def header(ax, cx, label, color):
    ax.text(cx, 0.965, label, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="none"), zorder=5)


# ── Canvas (landscape) ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 9))
ax.set_xlim(0, 1)
ax.set_ylim(-0.05, 1)
ax.axis("off")
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

fig.suptitle("ST-HeteroSAGE  —  Forecasting Pipeline",
             fontsize=15, fontweight="bold", color=C_TEXT, y=0.995)

# Five phase column centres
CX = [0.10, 0.30, 0.52, 0.74, 0.92]
for cx, lbl, col in zip(
    CX,
    ["1  DATA", "2  PIPELINE", "3  GRAPHS", "4  MODELS", "5  EVALUATION"],
    [C_DATA, C_PIPE, C_HOMO, C_HETERO, C_EVAL],
):
    header(ax, cx, lbl, col)

# faint column separators
for x in (0.20, 0.41, 0.63, 0.83):
    ax.axvline(x, color="#E0E0E0", lw=1, zorder=0)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — DATA SOURCES (stacked vertically)
# ─────────────────────────────────────────────────────────────────────────────
bw1 = 0.165
data_boxes = [
    (0.78, ["Energinet /", "Nord Pool", "prices, load, renew."]),
    (0.55, ["Energy-Charts", "(Fraunhofer)", "gas, CO2, DE prices"]),
    (0.32, ["Open-Meteo", "API", "weather variables"]),
]
for cy, lines in data_boxes:
    box(ax, CX[0], cy, bw1, 0.135, lines,
        fc="#EAF4FB", ec=C_DATA, sizes=[9, 9, 7.5])

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — PIPELINE (vertical stack of steps)
# ─────────────────────────────────────────────────────────────────────────────
bw2 = 0.175
pipe_boxes = [
    (0.78, ["Ingestion", "SQLite database"]),
    (0.58, ["Feature Eng.", "lags, rolling, cyclical"]),
    (0.38, ["Chrono Split", "80 / 10 / 10"]),
    (0.20, ["Scaler", "fit on train only"]),
]
for cy, lines in pipe_boxes:
    box(ax, CX[1], cy, bw2, 0.105, lines,
        fc="#F5EEF8", ec=C_PIPE, sizes=[9.5, 7.5])

# data -> pipeline (converge to ingestion)
for cy, _ in data_boxes:
    arr(ax, CX[0] + bw1 / 2, cy, CX[1] - bw2 / 2, 0.78, color=C_DATA, lw=1.2)
# pipeline internal flow
for i in range(len(pipe_boxes) - 1):
    arr(ax, CX[1], pipe_boxes[i][0] - 0.052,
        CX[1], pipe_boxes[i + 1][0] + 0.052, color=C_PIPE, lw=1.2)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — GRAPHS  (two graph boxes + xgboost passthrough)
# ─────────────────────────────────────────────────────────────────────────────
bw3 = 0.185
box(ax, CX[2], 0.80, bw3, 0.135,
    ["Homogeneous Graph", "4 zones x T", "17 feat/node",
     "spatial + temporal edges"],
    fc="#FEF9E7", ec=C_HOMO, sizes=[9.5, 8, 8, 7.5])
box(ax, CX[2], 0.54, bw3, 0.150,
    ["Heterogeneous Graph", "hour + market nodes", "5 edge types",
     "co_occurs, belongs_to,", "interconnects, lag_to"],
    fc="#EAFAF1", ec=C_HETERO, sizes=[9.5, 8, 8, 7.5, 7.5])
box(ax, CX[2], 0.27, bw3, 0.090,
    ["XGBoost path", "18 tabular features", "(no graph)"],
    fc="#FDEDEC", ec=C_XGB, sizes=[9, 7.5, 7.5])

# pipeline -> graphs
arr(ax, CX[1] + bw2 / 2, 0.30, CX[2] - bw3 / 2, 0.80, color=C_PIPE, lw=1.2, rad=0.1)
arr(ax, CX[1] + bw2 / 2, 0.25, CX[2] - bw3 / 2, 0.54, color=C_PIPE, lw=1.2, rad=0.05)
arr(ax, CX[1] + bw2 / 2, 0.20, CX[2] - bw3 / 2, 0.27, color=C_XGB, lw=1.2)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — MODELS (vertical stack, winner highlighted)
# ─────────────────────────────────────────────────────────────────────────────
bw4 = 0.175
model_boxes = [
    (0.85, C_HOMO,   "#FDFEFE", ["HomoGNN (SAGE)", "MAE 173.19"]),
    (0.66, C_HETERO, "#FDFEFE", ["HeteroSAGE", "MAE 162.78"]),
    (0.49, C_HETERO, "#F4F6F6", ["GAT (indicative)", "MAE 179.20"]),
    (0.31, C_XGB,    "#FDFEFE", ["XGBoost", "MAE 179.49"]),
    (0.13, C_WIN,    "#D5F5E3", ["ST-HeteroSAGE *", "MAE 151.08"]),
]
for cy, ec, fc, lines in model_boxes:
    is_win = ec == C_WIN
    box(ax, CX[3], cy, bw4, 0.095, lines,
        fc=fc, ec=ec, lw=2.4 if is_win else 1.5, sizes=[9.5, 8.5])

# graphs -> models
arr(ax, CX[2] + bw3 / 2, 0.80, CX[3] - bw4 / 2, 0.85, color=C_HOMO, lw=1.3)
arr(ax, CX[2] + bw3 / 2, 0.56, CX[3] - bw4 / 2, 0.66, color=C_HETERO, lw=1.3)
arr(ax, CX[2] + bw3 / 2, 0.52, CX[3] - bw4 / 2, 0.49, color=C_HETERO, lw=1.3)
arr(ax, CX[2] + bw3 / 2, 0.13, CX[3] - bw4 / 2, 0.13, color=C_HETERO, lw=1.3, rad=-0.1)
arr(ax, CX[2] + bw3 / 2, 0.27, CX[3] - bw4 / 2, 0.31, color=C_XGB, lw=1.3)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — EVALUATION (single column of outputs + winner)
# ─────────────────────────────────────────────────────────────────────────────
bw5 = 0.135
box(ax, CX[4], 0.78, bw5, 0.115,
    ["Unified Eval", "DK1 + DK2", "Mar-Sep 2025",
     "MAE/RMSE/R2/sMAPE"],
    fc="#EAF0FB", ec=C_EVAL, sizes=[9.5, 8, 7.5, 7.5])
box(ax, CX[4], 0.55, bw5, 0.115,
    ["Analyses", "Robustness", "Ablation", "Interpretability"],
    fc="#EAF0FB", ec=C_EVAL, sizes=[9, 7.5, 7.5, 7.5])
box(ax, CX[4], 0.28, bw5, 0.130,
    ["WINNER", "ST-HeteroSAGE", "MAE 151.08", "R2 0.696",
     "sMAPE 51.64%"],
    fc="#D5F5E3", ec=C_WIN, lw=2.4, sizes=[9.5, 9, 8, 8, 8])

# models -> evaluation (all converge)
for cy, ec, _, _ in model_boxes:
    arr(ax, CX[3] + bw4 / 2, cy, CX[4] - bw5 / 2, 0.78, color=C_EVAL, lw=1.0, rad=0.05)
arr(ax, CX[4], 0.72, CX[4], 0.61, color=C_EVAL, lw=1.2)
arr(ax, CX[4], 0.49, CX[4], 0.345, color=C_WIN, lw=1.5)

# ── Footer note ───────────────────────────────────────────────────────────────
ax.text(0.5, 0.02,
        "Deterministic seeds (torch 42 / numpy 42)   |   MLflow tracking   |   "
        "Docker containerised   |   manual-dispatch CI/CD with regression guard",
        ha="center", va="center", fontsize=8.5, color="#777777", style="italic")

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(fc="#EAF4FB", ec=C_DATA,   label="Data"),
    mpatches.Patch(fc="#F5EEF8", ec=C_PIPE,   label="Pipeline"),
    mpatches.Patch(fc="#FEF9E7", ec=C_HOMO,   label="Homogeneous"),
    mpatches.Patch(fc="#EAFAF1", ec=C_HETERO, label="Heterogeneous"),
    mpatches.Patch(fc="#FDEDEC", ec=C_XGB,    label="XGBoost"),
    mpatches.Patch(fc="#D5F5E3", ec=C_WIN,    label="Winner"),
]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.005),
          ncol=6, fontsize=8.5, framealpha=0.9, edgecolor="#CCCCCC")

plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor=C_BG)
plt.close()
print(f"Saved -> {OUT_PATH}")
