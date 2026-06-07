"""
Generates a top-down pipeline architecture diagram for the ST-HeteroSAGE
electricity price forecasting system.

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
C_DATA    = "#2E86C1"
C_PIPE    = "#7D3C98"
C_HOMO    = "#D68910"
C_HETERO  = "#1E8449"
C_XGB     = "#C0392B"
C_EVAL    = "#1A5276"
C_WIN     = "#27AE60"
C_BG      = "#F4F6F7"
C_ARROW   = "#555555"
C_TEXT    = "#1C2833"


def rbox(ax, cx, cy, w, h, lines, fc, ec, lw=1.6, sizes=None, bold_first=True):
    """
    Draw a rounded rect centred at (cx,cy) with stacked text lines.
    lines  : list of strings
    sizes  : list of font sizes matching lines (defaults to 9 for all)
    """
    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.015",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
    )
    ax.add_patch(patch)

    if sizes is None:
        sizes = [9] * len(lines)

    n = len(lines)
    # distribute lines evenly inside the box
    if n == 1:
        offsets = [0]
    else:
        step = h * 0.55 / (n - 1)
        offsets = [step * (i - (n - 1) / 2) for i in range(n)]

    for i, (txt, sz, dy) in enumerate(zip(lines, sizes, offsets)):
        weight = "bold" if (bold_first and i == 0) else "normal"
        color  = C_TEXT if i == 0 else "#444444"
        ax.text(cx, cy + dy, txt, ha="center", va="center",
                fontsize=sz, fontweight=weight, color=color, zorder=4)


def arr(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.4, rad=0.0):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        mutation_scale=12,
                        connectionstyle=f"arc3,rad={rad}"),
        zorder=2,
    )


def band(ax, ytop, ybot, label, color):
    ax.axhspan(ybot, ytop, xmin=0.0, xmax=1.0,
               facecolor=color, alpha=0.06, zorder=0)
    ax.text(-0.01, (ytop + ybot) / 2, label,
            ha="right", va="center", fontsize=8.5,
            color=color, fontweight="bold", rotation=90,
            transform=ax.transData, zorder=1, clip_on=False)


# ── Canvas ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 22))
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

fig.suptitle("ST-HeteroSAGE  —  End-to-End Forecasting Pipeline",
             fontsize=14, fontweight="bold", color=C_TEXT, y=0.992)

# ── Phase bands ───────────────────────────────────────────────────────────────
band(ax, 0.990, 0.875, "1  Data Sources",      C_DATA)
band(ax, 0.870, 0.715, "2  Data Pipeline",     C_PIPE)
band(ax, 0.710, 0.530, "3  Graph Construction",C_HOMO)
band(ax, 0.525, 0.295, "4  Model Training",    C_HETERO)
band(ax, 0.290, 0.005, "5  Evaluation",        C_EVAL)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1  —  Data Sources
# ─────────────────────────────────────────────────────────────────────────────
y1 = 0.925
w1, h1 = 0.24, 0.070
src = [
    (0.18, ["Energinet / Nord Pool",  "Day-ahead spot prices", "(DK1, DK2, DE, HYDRO)"]),
    (0.50, ["Energy-Charts (Fraunhofer ISE)", "Load, renewables,", "gas & CO2 prices"]),
    (0.82, ["Open-Meteo API",         "Temperature, wind,", "cloud cover, humidity"]),
]
for cx, lines in src:
    rbox(ax, cx, y1, w1, h1, lines,
         fc="#EAF4FB", ec=C_DATA, sizes=[9.5, 8, 8])

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2  —  Data Pipeline
# ─────────────────────────────────────────────────────────────────────────────
y2a = 0.820
rbox(ax, 0.50, y2a, 0.60, 0.055,
     ["Incremental Ingestion", "SQLite database  |  incremental fetch  |  MLflow tracking"],
     fc="#F5EEF8", ec=C_PIPE, sizes=[10, 8])

y2b = 0.745
pipe = [
    (0.18, ["Feature Engineering",  "Lags 24h / 48h / 168h", "Rolling stats  |  Cyclical encoding"]),
    (0.50, ["Chronological Split",  "Train 80%  |  Val 10%  |  Test 10%",
            "Test window: Mar - Sep 2025"]),
    (0.82, ["Scaler",               "StandardScaler", "fit on train partition only"]),
]
w2, h2 = 0.27, 0.070
for cx, lines in pipe:
    rbox(ax, cx, y2b, w2, h2, lines,
         fc="#F5EEF8", ec=C_PIPE, sizes=[9.5, 8, 8])

# arrows
for cx, _ in src:
    arr(ax, cx, y1 - h1 / 2, cx, y2a + 0.028, color=C_DATA)
for cx, _ in pipe:
    arr(ax, 0.50, y2a - 0.028, cx, y2b + h2 / 2, color=C_PIPE)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3  —  Graph Construction
# ─────────────────────────────────────────────────────────────────────────────
y3 = 0.635
wh, hh = 0.38, 0.090

rbox(ax, 0.27, y3, wh, hh,
     ["Homogeneous Graph",
      "4 zones x T timesteps  |  17 features per node",
      "Spatial: DK1-DK2, DK1-DE, DK2-DE, DK2-HYDRO",
      "Temporal: 24h and 168h bidirectional lags"],
     fc="#FEF9E7", ec=C_HOMO, sizes=[10, 8, 8, 8])

rbox(ax, 0.73, y3, wh + 0.04, hh,
     ["Heterogeneous Graph",
      "Hour nodes: 201,596 x 17 feat  |  Market nodes: 4 x one-hot",
      "5 edge types: co_occurs_with, belongs_to,",
      "rev_belongs_to, interconnects, lag_to"],
     fc="#EAFAF1", ec=C_HETERO, sizes=[10, 8, 8, 8])

# XGBoost — no graph needed
y_xgb = 0.570
rbox(ax, 0.50, y_xgb, 0.26, 0.052,
     ["XGBoost Baseline", "18 tabular features  |  no graph structure"],
     fc="#FDEDEC", ec=C_XGB, sizes=[9.5, 8])

# arrows phase 2 -> graphs
arr(ax, 0.18, y2b - h2 / 2, 0.27, y3 + hh / 2, color=C_PIPE)
arr(ax, 0.50, y2b - h2 / 2, 0.50, y3 + hh / 2, color=C_PIPE)
arr(ax, 0.82, y2b - h2 / 2, 0.73, y3 + hh / 2, color=C_PIPE)
arr(ax, 0.50, y2b - h2 / 2, 0.50, y_xgb + 0.026, color=C_XGB, lw=1.2)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4  —  Model Training
# ─────────────────────────────────────────────────────────────────────────────
y4 = 0.450
wm, hm = 0.195, 0.068

models = [
    (0.13, C_HOMO,   ["HomoGNN", "(GraphSAGE)", "MAE  173.19 DKK"]),
    (0.40, C_HETERO, ["HeteroSAGE",  "",        "MAE  162.78 DKK"]),
    (0.62, C_HETERO, ["GAT",         "(indicative)", "MAE  179.20 DKK"]),
    (0.84, C_WIN,    ["ST-HeteroSAGE", "(winner)", "MAE  151.08 DKK"]),
]
for cx, ec, lines in models:
    is_win = ec == C_WIN
    rbox(ax, cx, y4, wm, hm, lines,
         fc="#D5F5E3" if is_win else "#F9F9F9",
         ec=ec, lw=2.4 if is_win else 1.6,
         sizes=[10, 8.5, 8.5])

# XGBoost model box
rbox(ax, 0.50, y4, wm, hm,
     ["XGBoost", "", "MAE  179.49 DKK"],
     fc="#FDEDEC", ec=C_XGB, lw=1.6, sizes=[10, 8.5, 8.5])

# graph -> model arrows
arr(ax, 0.27, y3 - hh / 2, 0.13, y4 + hm / 2, color=C_HOMO)
arr(ax, 0.73, y3 - hh / 2, 0.40, y4 + hm / 2, color=C_HETERO)
arr(ax, 0.73, y3 - hh / 2, 0.62, y4 + hm / 2, color=C_HETERO)
arr(ax, 0.73, y3 - hh / 2, 0.84, y4 + hm / 2, color=C_HETERO)
arr(ax, 0.50, y_xgb - 0.026, 0.50, y4 + hm / 2, color=C_XGB, lw=1.2)

# checkpoints
y_ck = 0.375
ck_items = [
    (0.13, "best_homo_model.pt",      C_HOMO),
    (0.40, "best_hetero_model.pt",    C_HETERO),
    (0.50, "xgboost_model.json",      C_XGB),
    (0.62, "best_gat_model.pt",       C_HETERO),
    (0.84, "best_st_hetero_model.pt", C_WIN),
]
for cx, lbl, ec in ck_items:
    rbox(ax, cx, y_ck, 0.175, 0.032, [lbl],
         fc="#FDFEFE", ec=ec, lw=1.1, sizes=[7.5], bold_first=False)
    arr(ax, cx, y4 - hm / 2, cx, y_ck + 0.016, color=ec, lw=1.0)

ax.text(0.50, 0.330,
        "Deterministic seeds (torch 42 / numpy 42)   |   "
        "MLflow experiment tracking   |   Docker containerised",
        ha="center", va="center", fontsize=7.8,
        color="#777777", style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5  —  Evaluation
# ─────────────────────────────────────────────────────────────────────────────
y5a = 0.285
rbox(ax, 0.50, y5a, 0.75, 0.052,
     ["Unified Test Evaluation",
      "DK1 + DK2  |  Test window: March - September 2025  |  MAE, RMSE, R2, sMAPE"],
     fc="#EAF0FB", ec=C_EVAL, lw=2.0, sizes=[10.5, 8.5])

for cx, _, ec in ck_items:
    arr(ax, cx, y_ck - 0.016, cx, y5a + 0.026, color=C_EVAL, lw=1.0)

# Evaluation outputs
y5b = 0.205
ev = [
    (0.10, ["Leaderboard",       "MAE / RMSE / R2 / sMAPE"]),
    (0.29, ["Robustness",        "3 families x 4 intensity levels"]),
    (0.50, ["Ablation Study",    "Component contribution"]),
    (0.71, ["Feature Importance","Permutation-based"]),
    (0.90, ["Horizon Error",     "Hour-of-day  |  day-of-week"]),
]
we, he = 0.165, 0.058
for cx, lines in ev:
    rbox(ax, cx, y5b, we, he, lines,
         fc="#EAF0FB", ec=C_EVAL, lw=1.2, sizes=[9, 8])
    arr(ax, 0.50, y5a - 0.026, cx, y5b + he / 2, color=C_EVAL, lw=1.0)

# Winner banner
y5c = 0.118
rbox(ax, 0.50, y5c, 0.72, 0.068,
     ["Winner:  ST-HeteroSAGE",
      "MAE 151.08 DKK     RMSE 204.36 DKK     R2 0.696     sMAPE 51.64%",
      "Spatial-temporal heterogeneous GNN with CausalTCN temporal encoder"],
     fc="#D5F5E3", ec=C_WIN, lw=2.5,
     sizes=[12, 9, 8], bold_first=True)
for cx, _ in ev:
    arr(ax, cx, y5b - he / 2, 0.50, y5c + 0.034, color=C_WIN, lw=1.0)

# Dashboard
y5d = 0.042
rbox(ax, 0.50, y5d, 0.60, 0.050,
     ["Streamlit Dashboard",
      "Feature importance  |  Ablation  |  Robustness  |  Per-zone error  |  Live forecasts"],
     fc="#FDFEFE", ec="#AAAAAA", lw=1.2, sizes=[9.5, 8])
arr(ax, 0.50, y5c - 0.034, 0.50, y5d + 0.025, color="#AAAAAA", lw=1.2)

# ── Legend ────────────────────────────────────────────────────────────────────
handles = [
    mpatches.Patch(fc="#EAF4FB", ec=C_DATA,    label="Data Sources"),
    mpatches.Patch(fc="#F5EEF8", ec=C_PIPE,    label="Data Pipeline"),
    mpatches.Patch(fc="#FEF9E7", ec=C_HOMO,    label="Homogeneous Track"),
    mpatches.Patch(fc="#EAFAF1", ec=C_HETERO,  label="Heterogeneous Track"),
    mpatches.Patch(fc="#FDEDEC", ec=C_XGB,     label="XGBoost Baseline"),
    mpatches.Patch(fc="#EAF0FB", ec=C_EVAL,    label="Evaluation"),
    mpatches.Patch(fc="#D5F5E3", ec=C_WIN,     label="Winning Model"),
]
ax.legend(handles=handles, loc="lower center",
          bbox_to_anchor=(0.50, -0.012), ncol=7,
          fontsize=8, framealpha=0.9, edgecolor="#CCCCCC")

plt.tight_layout(rect=[0.04, 0.01, 1.0, 0.995])
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight", facecolor=C_BG)
plt.close()
print(f"Saved -> {OUT_PATH}")
