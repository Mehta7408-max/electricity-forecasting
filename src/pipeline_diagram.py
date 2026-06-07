"""
Generates a top-down pipeline architecture diagram for the ST-HeteroSAGE
electricity price forecasting system.

Output: src/artifacts/pipeline_diagram.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# ── Output path ───────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent / "artifacts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "pipeline_diagram.png"

# ── Colour palette ────────────────────────────────────────────────────────────
C_DATA    = "#4A90D9"   # blue     — data sources
C_PIPE    = "#7B68EE"   # indigo   — data pipeline
C_HOMO    = "#E8A838"   # amber    — homogeneous track
C_HETERO  = "#3DAA6B"   # green    — heterogeneous track
C_XGBOOST = "#D45F5F"   # red      — xgboost baseline
C_EVAL    = "#5B5EA6"   # navy     — evaluation
C_WINNER  = "#2ECC71"   # bright   — winner highlight
C_BG      = "#F7F9FC"   # near-white background
C_ARROW   = "#555555"
C_TEXT    = "#1A1A2E"
C_PHASE   = "#EEEEEE"   # phase band fill

FIG_W, FIG_H = 18, 24


def box(ax, x, y, w, h, label, sublabel=None,
        fc="#FFFFFF", ec="#AAAAAA", lw=1.5,
        fontsize=10, subfontsize=8.5, bold=False,
        radius=0.02, text_color=C_TEXT):
    """Draw a rounded rectangle with optional sublabel."""
    fancy = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3,
    )
    ax.add_patch(fancy)
    weight = "bold" if bold else "normal"
    dy = 0.012 if sublabel else 0
    ax.text(x, y + dy, label, ha="center", va="center",
            fontsize=fontsize, fontweight=weight,
            color=text_color, zorder=4, wrap=True,
            multialignment="center")
    if sublabel:
        ax.text(x, y - 0.025, sublabel, ha="center", va="center",
                fontsize=subfontsize, color="#555555", zorder=4,
                style="italic", multialignment="center")


def arrow(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.5, style="->"):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle=style, color=color,
                                lw=lw, connectionstyle="arc3,rad=0.0"),
                zorder=2)


def arrow_curve(ax, x0, y0, x1, y1, color=C_ARROW, lw=1.5, rad=0.2):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color=color,
                                lw=lw, connectionstyle=f"arc3,rad={rad}"),
                zorder=2)


def phase_band(ax, y_top, y_bot, label, color):
    """Horizontal shaded band for a pipeline phase."""
    ax.axhspan(y_bot, y_top, xmin=0.01, xmax=0.99,
               facecolor=color, alpha=0.07, zorder=0)
    ax.text(0.013, (y_top + y_bot) / 2, label,
            ha="left", va="center", fontsize=9, color=color,
            fontweight="bold", rotation=90, zorder=1,
            transform=ax.get_yaxis_transform())


# ── Figure setup ──────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)

fig.suptitle(
    "ST-HeteroSAGE — End-to-End Forecasting Pipeline",
    fontsize=15, fontweight="bold", color=C_TEXT, y=0.985,
)

# ── Phase bands ───────────────────────────────────────────────────────────────
phase_band(ax, 0.985, 0.870, "① Data Sources",    C_DATA)
phase_band(ax, 0.865, 0.700, "② Data Pipeline",   C_PIPE)
phase_band(ax, 0.695, 0.520, "③ Graph Construction", C_HOMO)
phase_band(ax, 0.515, 0.280, "④ Model Training",  C_HETERO)
phase_band(ax, 0.275, 0.010, "⑤ Evaluation",      C_EVAL)

BW, BH = 0.20, 0.050   # standard box width / height
SW, SH = 0.16, 0.044   # small box
TW, TH = 0.26, 0.054   # tall/wide box

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — DATA SOURCES
# ─────────────────────────────────────────────────────────────────────────────
y1 = 0.930
src_xs = [0.22, 0.50, 0.78]
src_labels = [
    ("Energinet / Nord Pool", "Day-ahead spot prices\n(DK1, DK2, DE, HYDRO)"),
    ("Energy-Charts\n(Fraunhofer ISE)", "Load, renewables,\ngas & CO₂ prices"),
    ("Open-Meteo API", "Temperature, wind,\ncloud, humidity"),
]
for x, (lbl, sub) in zip(src_xs, src_labels):
    box(ax, x, y1, BW, BH + 0.01, lbl, sublabel=sub,
        fc="#EAF3FB", ec=C_DATA, lw=1.8, fontsize=9.5, bold=True)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — DATA PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
y2a = 0.820
box(ax, 0.50, y2a, 0.55, BH,
    "Incremental Ingestion  →  SQLite Database",
    fc="#F0EDF9", ec=C_PIPE, lw=1.8, fontsize=10, bold=True)

y2b = 0.760
pipe_steps = [
    (0.22, "Feature Engineering",
     "Lags (24 h, 48 h, 168 h)\nRolling stats · Cyclical encoding"),
    (0.50, "Chronological Split",
     "Train 80% · Val 10% · Test 10%\n(Mar 2025 – Sep 2025 test window)"),
    (0.78, "Scaler Fit",
     "StandardScaler fitted on\ntraining partition only"),
]
for x, lbl, sub in pipe_steps:
    box(ax, x, y2b, BW + 0.02, BH + 0.01, lbl, sublabel=sub,
        fc="#F0EDF9", ec=C_PIPE, lw=1.5, fontsize=9.5, bold=True)

# arrows: sources → ingestion
for x in src_xs:
    arrow(ax, x, y1 - (BH + 0.01) / 2,
          x, y2a + BH / 2 + 0.005, color=C_DATA)

# fan-in arrows to ingestion box edges, then fan-out to pipe steps
arrow(ax, 0.50, y2a - BH / 2, 0.22, y2b + (BH + 0.01) / 2, color=C_PIPE)
arrow(ax, 0.50, y2a - BH / 2, 0.50, y2b + (BH + 0.01) / 2, color=C_PIPE)
arrow(ax, 0.50, y2a - BH / 2, 0.78, y2b + (BH + 0.01) / 2, color=C_PIPE)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — GRAPH CONSTRUCTION  (two parallel branches)
# ─────────────────────────────────────────────────────────────────────────────
y3a = 0.665
box(ax, 0.28, y3a, 0.36, BH + 0.01,
    "Homogeneous Graph",
    sublabel="4 zones × T timesteps  |  17 features/node\n"
             "Spatial edges: DK1–DK2, DK1–DE, DK2–DE, DK2–HYDRO\n"
             "Temporal edges: 24 h & 168 h (bidirectional)",
    fc="#FFF8E7", ec=C_HOMO, lw=1.8, fontsize=9.5, bold=True,
    subfontsize=7.8)

box(ax, 0.72, y3a, 0.40, BH + 0.01,
    "Heterogeneous Graph",
    sublabel="Hour nodes (201,596 × 17 feat)  +  Market nodes (4 × one-hot)\n"
             "5 edge types: co_occurs_with · belongs_to · rev_belongs_to\n"
             "interconnects · lag_to (24 h, 48 h, 168 h)",
    fc="#E8F6EE", ec=C_HETERO, lw=1.8, fontsize=9.5, bold=True,
    subfontsize=7.8)

# pipe steps → graph boxes
for src_x, dest_x in [(0.22, 0.28), (0.50, 0.50), (0.78, 0.72)]:
    arrow(ax, src_x, y2b - (BH + 0.01) / 2,
          dest_x, y3a + (BH + 0.01) / 2 + 0.005,
          color=C_PIPE, lw=1.2)

# XGBoost branch — comes off the pipeline directly (no graph needed)
y_xgb = 0.600
box(ax, 0.50, y_xgb, SW + 0.02, SH,
    "XGBoost Baseline",
    sublabel="18 features · no graph structure",
    fc="#FDEAEA", ec=C_XGBOOST, lw=1.8, fontsize=9.5, bold=True,
    subfontsize=8)
arrow(ax, 0.50, y2b - (BH + 0.01) / 2,
      0.50, y_xgb + SH / 2, color=C_XGBOOST, lw=1.5)

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────
# Homogeneous track
y4a = 0.490
box(ax, 0.18, y4a, 0.26, BH,
    "HomoGNN (GraphSAGE)",
    sublabel="128 hidden · 3 layers\nMAE 173.19 DKK",
    fc="#FFF8E7", ec=C_HOMO, lw=1.8, fontsize=9.5, bold=True,
    subfontsize=8)

# Heterogeneous track — three models
hetero_models = [
    (0.50, "HeteroSAGE", "MAE 162.78 DKK"),
    (0.68, "GAT", "MAE 179.20 DKK"),
    (0.86, "ST-HeteroSAGE", "MAE 151.08 DKK  ★"),
]
for x, lbl, sub in hetero_models:
    is_winner = "★" in sub
    box(ax, x, y4a, 0.155, BH,
        lbl, sublabel=sub,
        fc="#D5F5E3" if is_winner else "#E8F6EE",
        ec=C_WINNER if is_winner else C_HETERO,
        lw=2.5 if is_winner else 1.8,
        fontsize=9.5, bold=True, subfontsize=8)

arrow(ax, 0.28, y3a - (BH + 0.01) / 2,
      0.18, y4a + BH / 2, color=C_HOMO)

for x in [0.50, 0.68, 0.86]:
    arrow(ax, 0.72, y3a - (BH + 0.01) / 2,
          x, y4a + BH / 2, color=C_HETERO)

# XGBoost flows down
arrow(ax, 0.50, y_xgb - SH / 2,
      0.50, y4a + BH / 2, color=C_XGBOOST, lw=1.5)

# ── Frozen checkpoint labels ──────────────────────────────────────────────────
y_ckpt = 0.415
ckpt_items = [
    (0.18,  "best_homo_model.pt",       C_HOMO),
    (0.50,  "best_hetero_model.pt",     C_HETERO),
    (0.68,  "best_gat_model.pt",        C_HETERO),
    (0.86,  "best_st_hetero_model.pt",  C_WINNER),
]
for x, lbl, ec in ckpt_items:
    box(ax, x, y_ckpt, 0.155, 0.030, f"[ckpt] {lbl}",
        fc="#F9F9F9", ec=ec, lw=1.2, fontsize=7.5)
    arrow(ax, x, y4a - BH / 2, x, y_ckpt + 0.015, color=ec, lw=1.0)

# MLflow tracking note
ax.text(0.50, 0.373,
        "All runs tracked in MLflow  ·  Deterministic seeds (torch 42 / numpy 42)  ·  Docker containerised",
        ha="center", va="center", fontsize=8, color="#777777", style="italic")

# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
y5a = 0.325
box(ax, 0.50, y5a, 0.72, BH,
    "Unified Test Evaluation  (DK1 + DK2 · test window Mar – Sep 2025)",
    fc="#ECEAF7", ec=C_EVAL, lw=2.0, fontsize=10, bold=True)

# all checkpoints → unified evaluation
for x in [0.18, 0.50, 0.68, 0.86]:
    arrow(ax, x, y_ckpt - 0.015,
          x, y5a + BH / 2, color=C_EVAL, lw=1.2)

# Evaluation outputs
y5b = 0.240
eval_outputs = [
    (0.14, "Leaderboard\nMAE · RMSE · R² · sMAPE"),
    (0.34, "Robustness Analysis\n3 perturbation families · 4 levels"),
    (0.54, "Ablation Study\nComponent contribution"),
    (0.74, "Feature Importance\nPermutation-based"),
    (0.91, "Horizon Error\nHour-of-day · day-of-week"),
]
for x, lbl in eval_outputs:
    box(ax, x, y5b, 0.155, BH + 0.005, lbl,
        fc="#ECEAF7", ec=C_EVAL, lw=1.3, fontsize=8.5)
    arrow(ax, 0.50, y5a - BH / 2, x, y5b + (BH + 0.005) / 2,
          color=C_EVAL, lw=1.0)

# Winner banner
y5c = 0.145
box(ax, 0.50, y5c, 0.70, BH + 0.01,
    "Winner: ST-HeteroSAGE",
    sublabel="MAE 151.08 DKK  ·  RMSE 204.36 DKK  ·  R² 0.696  ·  sMAPE 51.64%",
    fc="#D5F5E3", ec=C_WINNER, lw=2.5, fontsize=11, bold=True,
    subfontsize=9, text_color="#1A6B3C")

for x in [0.14, 0.34, 0.54, 0.74, 0.91]:
    arrow(ax, x, y5b - (BH + 0.005) / 2,
          0.50, y5c + (BH + 0.01) / 2, color=C_WINNER, lw=1.0)

# Streamlit dashboard note
y5d = 0.065
box(ax, 0.50, y5d, 0.60, BH,
    "Streamlit Dashboard",
    sublabel="Interactive: feature importance · ablation · robustness · per-zone error · forecasts",
    fc="#FDFCFF", ec="#AAAAAA", lw=1.3, fontsize=9.5, bold=True,
    subfontsize=8)
arrow(ax, 0.50, y5c - (BH + 0.01) / 2,
      0.50, y5d + BH / 2, color="#AAAAAA", lw=1.2)

# ─────────────────────────────────────────────────────────────────────────────
# Legend
# ─────────────────────────────────────────────────────────────────────────────
legend_items = [
    (C_DATA,    "Data Sources"),
    (C_PIPE,    "Data Pipeline"),
    (C_HOMO,    "Homogeneous Track"),
    (C_HETERO,  "Heterogeneous Track"),
    (C_XGBOOST, "XGBoost Baseline"),
    (C_EVAL,    "Evaluation"),
    (C_WINNER,  "Winning Model"),
]
handles = [mpatches.Patch(facecolor=c, edgecolor="#888888", label=l)
           for c, l in legend_items]
ax.legend(handles=handles, loc="lower right",
          bbox_to_anchor=(0.99, 0.005),
          ncol=4, fontsize=8, framealpha=0.85,
          edgecolor="#CCCCCC")

plt.tight_layout(rect=[0.03, 0, 1, 0.98])
plt.savefig(OUT_PATH, dpi=180, bbox_inches="tight",
            facecolor=C_BG)
plt.close()
print(f"Saved → {OUT_PATH}")
