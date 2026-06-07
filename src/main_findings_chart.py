"""
Main findings figure — two panels:
  (left)  model comparison by MAE
  (right) ST-HeteroSAGE actual vs predicted average daily price profile (DK1, DK2)

Output: src/artifacts/main_findings.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

SRC = Path(__file__).parent
OUT = SRC / "artifacts" / "main_findings.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── Load actual vs predicted daily profiles ───────────────────────────────────
with open(SRC / "artifacts_hetero" / "day_ahead_results.json") as f:
    da = json.load(f)

def profile(zone):
    d = da["price_profiles"][zone]["by_hour"]
    hours = sorted(int(h) for h in d)
    actual    = [d[str(h)]["actual"]    for h in hours]
    predicted = [d[str(h)]["predicted"] for h in hours]
    return hours, actual, predicted

h_dk1, a_dk1, p_dk1 = profile("DK1")
h_dk2, a_dk2, p_dk2 = profile("DK2")

# ── Model comparison data ─────────────────────────────────────────────────────
models = ["XGBoost", "HomoGNN", "HeteroSAGE", "ST-HeteroSAGE"]
mae    = [179.49, 173.19, 162.78, 151.08]

# ── Colours ───────────────────────────────────────────────────────────────────
C_WIN   = "#27AE60"
C_BASE  = "#5B8DB8"
C_BG    = "#F7F9FC"
C_GRID  = "#DDDDDD"
C_TEXT  = "#1C2833"
C_ACT   = "#1C2833"   # actual line
C_DK1   = "#27AE60"   # predicted DK1
C_DK2   = "#E67E22"   # predicted DK2

fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.2),
                               gridspec_kw={"width_ratios": [1, 1.35]})
fig.patch.set_facecolor(C_BG)
fig.suptitle(
    "ST-HeteroSAGE  —  Main Findings",
    fontsize=15, fontweight="bold", color=C_TEXT, y=1.00,
)

# ── PANEL LEFT — MAE bars ─────────────────────────────────────────────────────
x = np.arange(len(models))
colors = [C_WIN if m == "ST-HeteroSAGE" else C_BASE for m in models]
bars = axL.bar(x, mae, width=0.6, color=colors, edgecolor="white",
               linewidth=1, zorder=3)

for bar, v in zip(bars, mae):
    axL.text(bar.get_x() + bar.get_width() / 2, v - 9,
             f"{v:.1f}", ha="center", va="top",
             fontsize=10.5, fontweight="bold", color="white", zorder=4)

# improvement annotation
axL.annotate("", xy=(3, 151.08), xytext=(0, 179.49),
             arrowprops=dict(arrowstyle="->", color="#888888", lw=1.4,
                             connectionstyle="arc3,rad=-0.25"), zorder=2)
axL.text(1.5, 187, "-15.8%  MAE", ha="center", fontsize=10,
         fontweight="bold", color="#555555")

axL.set_title("(a)  Test-set MAE by model  (DK1 + DK2)",
              fontsize=11, fontweight="bold", color=C_TEXT, pad=10)
axL.set_ylabel("MAE  (DKK / MWh)", fontsize=10, color=C_TEXT)
axL.set_xticks(x)
axL.set_xticklabels(models, fontsize=9.5, color=C_TEXT)
axL.set_ylim(130, 195)
axL.set_facecolor(C_BG)
axL.yaxis.grid(True, color=C_GRID, linewidth=0.8, zorder=0)
axL.set_axisbelow(True)
for s in ("top", "right"):
    axL.spines[s].set_visible(False)
for s in ("left", "bottom"):
    axL.spines[s].set_color(C_GRID)
axL.tick_params(colors=C_TEXT)

# ── PANEL RIGHT — actual vs predicted daily profile ───────────────────────────
axR.plot(h_dk1, a_dk1, color=C_ACT, lw=2.6, label="Actual (DK1)", zorder=3)
axR.plot(h_dk1, p_dk1, color=C_DK1, lw=2.2, ls="--",
         label="Predicted (DK1)", zorder=4)
axR.plot(h_dk2, a_dk2, color="#7F8C8D", lw=2.0, label="Actual (DK2)", zorder=3)
axR.plot(h_dk2, p_dk2, color=C_DK2, lw=2.0, ls="--",
         label="Predicted (DK2)", zorder=4)

# highlight evening peak window
axR.axvspan(17, 19, color="#E74C3C", alpha=0.08, zorder=0)
axR.text(18, axR.get_ylim()[1] * 0.97, "evening peak",
         ha="center", va="top", fontsize=8.5, color="#C0392B", style="italic")

axR.set_title("(b)  Average daily price profile:  actual vs ST-HeteroSAGE",
              fontsize=11, fontweight="bold", color=C_TEXT, pad=10)
axR.set_xlabel("Hour of day", fontsize=10, color=C_TEXT)
axR.set_ylabel("Price  (DKK / MWh)", fontsize=10, color=C_TEXT)
axR.set_xticks(range(0, 24, 2))
axR.set_xlim(0, 23)
axR.set_facecolor(C_BG)
axR.grid(True, color=C_GRID, linewidth=0.7, zorder=0)
axR.set_axisbelow(True)
for s in ("top", "right"):
    axR.spines[s].set_visible(False)
for s in ("left", "bottom"):
    axR.spines[s].set_color(C_GRID)
axR.tick_params(colors=C_TEXT)
axR.legend(fontsize=9, framealpha=0.9, edgecolor=C_GRID, loc="upper left")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT, dpi=180, bbox_inches="tight", facecolor=C_BG)
plt.close()
print(f"Saved -> {OUT}")
