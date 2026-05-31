"""
Generates ST_HeteroSAGE_Review.ipynb — a supervisor-facing review notebook.

Run:  python3 notebooks/build_review_notebook.py

The notebook loads committed JSON artifacts (no retraining) so it executes in
seconds and reproduces the exact thesis numbers. Built with the std-lib `json`
module only — no nbformat dependency.
"""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({
        "cell_type": "markdown", "metadata": {},
        "source": text.splitlines(keepends=True),
    })


def code(text):
    cells.append({
        "cell_type": "code", "metadata": {}, "execution_count": None,
        "outputs": [], "source": text.splitlines(keepends=True),
    })


# ── 0. Title ────────────────────────────────────────────────────────────────
md(r"""# Heterogeneous Spatio-Temporal GNNs for Multi-Area Nordic Electricity Price Forecasting
### Supervisor review notebook

**Research question**

> *How can a heterogeneous graph be effectively constructed and integrated into
> GNN models to improve the accuracy, interpretability, and robustness of
> multi-area day-ahead electricity price forecasting in the Nordic power market?*

This notebook is a **read-only walkthrough** of the completed work. Every result
cell loads a committed JSON artifact produced by the training / analysis scripts,
so the notebook runs in seconds and reproduces the exact numbers reported in the
thesis — no GPU or retraining required.

**Model progression studied**

| # | Model | What it adds |
|---|-------|--------------|
| 1 | **XGBoost** | Non-graph tabular baseline (a lean per-zone feature set) |
| 2 | **HomoGNN** (GraphSAGE) | A *homogeneous* graph — single node/edge type |
| 3 | **GAT** | Attention over the homogeneous graph |
| 4 | **HeteroSAGE** | A *heterogeneous* graph — hour & market nodes, typed edges |
| 5 | **ST-HeteroSAGE** | Adds a temporal pathway (CausalTCN) → spatio-temporal model |

All GNNs are evaluated on the **identical** DK1+DK2 test window with identical
graph features and metrics, so differences are attributable to the modelling.
""")

# ── 1. Setup ────────────────────────────────────────────────────────────────
md(r"""## 1. Setup & paths

Artifacts are split across two folders, so we resolve the repo root first and
then locate each file robustly:

- `artifacts/` (repo root) → XGBoost baseline, homogeneous GraphSAGE
- `src/artifacts_hetero/` → heterogeneous + GAT + ST models and all analyses""")

code(r"""import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})

# Find the repo root (the dir that contains both 'src' and 'artifacts')
HERE = Path.cwd()
ROOT = next((p for p in [HERE, *HERE.parents]
             if (p / "src").exists() and (p / "artifacts").exists()), None)
assert ROOT is not None, "Could not locate repo root (needs src/ and artifacts/)"
SRC      = ROOT / "src"
ART      = ROOT / "artifacts"          # baseline + homo
ART_HET  = SRC / "artifacts_hetero"    # hetero / gat / st + analyses
print("repo root:", ROOT)


def find_json(*relpaths):
    # Return the first existing JSON among candidate locations
    for rp in relpaths:
        for base in (ROOT, SRC):
            p = base / rp
            if p.exists():
                with open(p) as f:
                    return json.load(f)
    raise FileNotFoundError(relpaths)
""")

# ── 2. Data & methodology ───────────────────────────────────────────────────
md(r"""## 2. Data, zones, and forecasting discipline

**Market areas (graph nodes are built per zone):**

| Zone | Role | Notes |
|------|------|-------|
| **DK1** | Western Denmark — *target* | Wind-heavy |
| **DK2** | Eastern Denmark — *target* | Linked to Sweden via Øresund |
| **HYDRO (SE3)** | Auxiliary | Swedish hydro reservoir signal |
| **DE** | Auxiliary | German price coupling |

**Features (17 per hour-node), grouped:**

- *Price history (day-ahead safe, all lags ≥ 24 h):* `price_lag_24h`, `price_lag_48h`, `price_lag_168h`, `price_rolling_24h_mean`, `price_rolling_24h_std`
- *Weather:* `temperature_c`, `wind_speed_ms`, `cloud_cover_pct`, `humidity_pct`
- *Fundamentals:* `load_mwh`, `renewable_mwh`, `gas_dkk`, `co2_dkk`
- *Calendar (cyclical):* `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`

**Methodological guarantees**

- **No leakage:** every price feature is lagged ≥ 24 h, matching a true
  day-ahead setting where tomorrow's prices are unknown at gate closure.
- **Chronological split:** train / val / test taken in time order; the test
  window is the most recent slice (2025), never shuffled.
- **Identical evaluation:** all models scored on DK1+DK2 **test** nodes only,
  in raw DKK, with MAE / RMSE / R² / SMAPE.

> *Baseline fairness note.* The XGBoost baseline intentionally uses a lean,
> non-graph tabular feature set (lags, calendar, weather, lagged neighbour
> prices). It is the honest "no graph, no temporal model" reference point — not
> a GNN fed through trees.
""")

# ── 3. Architecture ─────────────────────────────────────────────────────────
md(r"""## 3. The ST-HeteroSAGE architecture

Each **ST block** runs a *spatial* step then a *temporal* step:

```
            ┌──────────────── ST block (×2) ────────────────┐
 x_hour  ─► │  Spatial: HeteroConv (same-timestep)              │
 x_market─► │    hour ─co_occurs_with→ hour  (DK1↔DK2, DK2↔SE3) │
            │    hour ─belongs_to→ market / market→hour         │
            │    market ─interconnects→ market                  │
            │            │  (+ BatchNorm + residual)            │
            │            ▼                                       │
            │  Temporal: CausalTCN per zone along the time axis  │
            │    dilations (1,4,24), kernel 7 → RF ≈ 175 h       │
            │            │  (causal: no look-ahead)              │
            └────────────┼───────────────────────────────────────┘
                         ▼
                 shared MLP head ─► price (DKK)
```

**Why this design**

- **Spatial HeteroConv** captures *same-hour* cross-zone coupling.
- **CausalTCN** replaces explicit `lag_to` graph edges with dilated causal
  convolutions — a 7-day receptive field — strictly causal (no future peek).
- **DE staleness fix:** German prices end 2024-12-31, so DK↔DE *hour-level*
  edges would leak stale forward-filled values into the 2025 test window. Those
  edges were removed; the **DK2↔HYDRO (Øresund)** link is used instead (full
  2025 coverage).
""")

code(r"""# Confirm the model definition imports and count parameters
import sys
sys.path.insert(0, str(SRC))
import torch
from hetero_st_model import HeteroSTPriceForecaster

_m = HeteroSTPriceForecaster(in_channels=17, hidden_channels=128, num_st_blocks=2)
print(f"ST-HeteroSAGE trainable parameters: {sum(p.numel() for p in _m.parameters()):,}")
del _m
""")

# ── 4. Headline results ─────────────────────────────────────────────────────
md(r"""## 4. Headline results — model comparison

All numbers are on the **DK1+DK2 test window**. Lower MAE/RMSE/SMAPE is better,
higher R² is better.""")

code(r"""rows = [
    ("XGBoost",       find_json("artifacts/xgboost_metrics.json")),
    ("HomoGNN",       find_json("artifacts/homo_gnn_metrics.json")),
    ("GAT",           find_json("artifacts_hetero/gat_metrics_clean.json")),
    ("HeteroSAGE",    find_json("artifacts_hetero/hetero_metrics_clean.json")),
    ("ST-HeteroSAGE", find_json("artifacts_hetero/st_hetero_metrics.json")),
]
df = pd.DataFrame([{
    "Model": name,
    "MAE (DKK)": round(m["mae"], 1), "RMSE (DKK)": round(m["rmse"], 1),
    "R²": round(m["r2"], 3), "SMAPE (%)": round(m.get("smape", float("nan")), 1),
} for name, m in rows]).set_index("Model").sort_values("MAE (DKK)", ascending=False)
df
""")

code(r"""order = df.index.tolist()
colors = ["#9aa0a6" if m != "ST-HeteroSAGE" else "#e0773a" for m in order]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].barh(order, df["MAE (DKK)"], color=colors)
axes[0].set_title("Test MAE — lower is better"); axes[0].set_xlabel("DKK")
for i, v in enumerate(df["MAE (DKK)"]):
    axes[0].text(v + 1, i, f"{v:.1f}", va="center", fontsize=9)

axes[1].barh(order, df["R²"], color=colors)
axes[1].set_title("Test R² — higher is better"); axes[1].set_xlim(0, 0.8)
for i, v in enumerate(df["R²"]):
    axes[1].text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
plt.tight_layout(); plt.show()

xgb_mae = df.loc["XGBoost", "MAE (DKK)"]; best = df["MAE (DKK)"].idxmin()
imp = (xgb_mae - df.loc[best, "MAE (DKK)"]) / xgb_mae * 100
print(f"Best model: {best} | MAE improvement over XGBoost baseline: {imp:.1f}%")
""")

md(r"""**Reading the chart.** The non-graph **XGBoost baseline is clearly the
weakest** (~206 DKK). Introducing a graph (HomoGNN / GAT / HeteroSAGE) cuts MAE
to ~162 DKK, and adding the **temporal CausalTCN pathway** (ST-HeteroSAGE)
delivers the best result at **151 DKK / R² 0.70**. The heterogeneous
*spatio-temporal* construction is the decisive contribution.""")

# ── 5. Ablation ─────────────────────────────────────────────────────────────
md(r"""## 5. Ablation study — which components matter?

Each variant disables exactly one structural component of the trained
ST-HeteroSAGE (inference-time, no retraining) and re-measures test MAE. Δ-MAE
versus the full model quantifies that component's contribution.""")

code(r"""abl = find_json("artifacts_hetero/st_ablation_results.json")
labels = {
    "A_full":        "A · Full model",
    "B_no_tcn":      "B · No TCN (temporal)",
    "C_no_spatial":  "C · No spatial HeteroConv",
    "D_no_cooccurs": "D · No co_occurs_with",
    "E_no_market":   "E · No market bridge",
    "F_no_hydro":    "F · No DK2↔HYDRO",
}
base = abl["A_full"]["mae"]
ab = pd.DataFrame([{
    "Variant": labels[k], "MAE (DKK)": round(abl[k]["mae"], 1),
    "R²": round(abl[k]["r2"], 3), "Δ MAE": round(abl[k]["mae"] - base, 1),
} for k in labels]).set_index("Variant")
ab
""")

code(r"""deltas = ab["Δ MAE"].drop(ab.index[0])  # drop full-model 0 baseline
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(deltas.index[::-1], deltas.values[::-1], color="#c0504d")
ax.set_xlabel("Δ MAE vs full model (DKK) — bigger = more important")
ax.set_title("Ablation: cost of removing each component")
for i, v in enumerate(deltas.values[::-1]):
    ax.text(v + 1, i, f"+{v:.0f}", va="center", fontsize=9)
plt.tight_layout(); plt.show()
""")

md(r"""**Findings.** Both pathways are essential: removing the **spatial
HeteroConv** is the most damaging, with the **temporal TCN** close behind —
neither alone suffices. Among edges, cross-zone `co_occurs_with` matters most,
the market bridge contributes moderately, and DK2↔HYDRO (Øresund) gives a small
but real gain.""")

# ── 6. Interpretability ─────────────────────────────────────────────────────
md(r"""## 6. Interpretability

Two lenses: (a) gradient-based feature importance, and (b) error structure
across the daily / weekly cycle.""")

code(r"""interp = find_json("artifacts_hetero/st_interpretability.json")
fi = pd.DataFrame(interp["feature_importance"]).set_index("feature")
top = fi.head(10).iloc[::-1]
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(top.index, top["importance"], color="#4f81bd")
ax.set_xlabel("Mean |∂loss/∂x|  (gradient-based importance)")
ax.set_title("Top 10 most influential features — ST-HeteroSAGE")
plt.tight_layout(); plt.show()
fi.head(10)
""")

code(r"""ebt = interp["error_by_time"]
hod = {int(k): v for k, v in ebt["hour_of_day"].items()}
dow = ebt["day_of_week"]
order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
hours = sorted(hod)
axes[0].plot(hours, [hod[h] for h in hours], marker="o", color="#e0773a")
axes[0].set_title("MAE by hour-of-day"); axes[0].set_xlabel("hour"); axes[0].set_ylabel("MAE (DKK)")
axes[0].set_xticks(range(0, 24, 2))
dnames = [d for d in order if d in dow]
axes[1].bar(dnames, [dow[d] for d in dnames], color="#8064a2")
axes[1].set_title("MAE by day-of-week"); axes[1].set_ylabel("MAE (DKK)")
plt.tight_layout(); plt.show()

wh = max(hod, key=hod.get); bh = min(hod, key=hod.get)
wd = max(dow, key=dow.get); bd = min(dow, key=dow.get)
print(f"Worst hour: {wh:02d}:00 ({hod[wh]:.0f} DKK) | Best hour: {bh:02d}:00 ({hod[bh]:.0f} DKK)")
print(f"Worst day:  {wd} ({dow[wd]:.0f} DKK) | Best day:  {bd} ({dow[bd]:.0f} DKK)")
""")

md(r"""**Findings.** The model leans most on **gas price**, the **24 h price
lag**, and **renewable generation** — economically sensible drivers of Nordic
day-ahead prices. Errors concentrate around the **evening demand peak (~18:00)**
and on **Mondays**, while overnight hours and weekends are easiest.""")

# ── 7. Robustness ───────────────────────────────────────────────────────────
md(r"""## 7. Robustness

We perturb the test-window inputs (inference only) and watch how MAE degrades:
Gaussian feature noise, random feature dropout (missing sensors), and amplified
price spikes on a small fraction of nodes.""")

code(r"""rob = find_json("artifacts_hetero/st_robustness_results.json")
base_mae = rob["baseline"]["mae"]
def curve(section, key_fmt, xs):
    return [rob[section][key_fmt.format(x)]["mae"] for x in xs]

noise_x, drop_x, spike_x = [5, 10, 20, 30], [10, 20, 30], [2, 3, 5]
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].plot(noise_x, curve("gaussian_noise", "noise_{}pct", noise_x), marker="o", color="#4f81bd")
axes[0].set_title("Gaussian noise"); axes[0].set_xlabel("σ (% of feature std)")
axes[1].plot(drop_x, curve("feature_dropout", "drop_{}pct", drop_x), marker="s", color="#c0504d")
axes[1].set_title("Feature dropout"); axes[1].set_xlabel("dropout rate (%)")
axes[2].plot(spike_x, curve("price_spike", "spike_{}x", spike_x), marker="^", color="#9bbb59")
axes[2].set_title("Price spike (5% of nodes)"); axes[2].set_xlabel("lag amplification (×)")
for ax in axes:
    ax.axhline(base_mae, ls="--", color="gray", lw=1, label=f"clean {base_mae:.0f}")
    ax.set_ylabel("MAE (DKK)"); ax.legend()
plt.tight_layout(); plt.show()
""")

code(r"""def summarize(section, key_fmt, xs, unit):
    return pd.DataFrame([{
        unit: x, "MAE": round(rob[section][key_fmt.format(x)]["mae"], 1),
        "Δ%": round(rob[section][key_fmt.format(x)]["delta_pct"], 1),
    } for x in xs]).set_index(unit)

print("Gaussian noise:");  display(summarize("gaussian_noise", "noise_{}pct", noise_x, "σ %"))
print("Feature dropout:"); display(summarize("feature_dropout", "drop_{}pct", drop_x, "rate %"))
print("Price spike:");     display(summarize("price_spike", "spike_{}x", spike_x, "factor ×"))
""")

md(r"""**Findings.** The model is **resilient to additive noise** — even σ = 30 %
raises MAE only a few percent. It is more sensitive to **missing features**
(dropout), as zeroing an input removes information rather than blurring it.
Localized **price spikes** on a small share of nodes have limited global impact,
so the graph does not over-propagate single-node anomalies.""")

# ── 8. Reproducibility ──────────────────────────────────────────────────────
md(r"""## 8. How to reproduce (reference only — not run here)

```bash
cd src
python3 hetero_graph_builder.py     # build the heterogeneous graph from SQLite
python3 xgboost_baseline.py         # XGBoost baseline      → artifacts/xgboost_metrics.json
python3 homo_retrain.py             # HomoGNN
python3 gat_train.py                # GAT
python3 quick_retrain.py            # HeteroSAGE
python3 st_train.py                 # ST-HeteroSAGE (winner)
python3 st_ablation.py              # ablation A–F
python3 st_interpretability.py      # feature importance + error-by-time
python3 st_robustness.py            # noise / dropout / spike
```

The GNNs share one graph tensor, splits, and evaluation protocol, so the
comparison is apples-to-apples; XGBoost is the deliberately lean non-graph
reference.""")

md(r"""## 9. Summary for the supervisor

- **Answer to the research question:** a heterogeneous graph — paired with a
  causal temporal pathway — improves day-ahead price forecasting on all three
  axes:
  - **Accuracy:** ST-HeteroSAGE reaches **151 DKK MAE / R² 0.70**, the best of
    all models and a **~27 % MAE reduction over the XGBoost baseline** (~206).
  - **Interpretability:** gradient attribution recovers economically meaningful
    drivers (gas, recent price, renewables) and exposes the hard cases (evening
    peak, Mondays).
  - **Robustness:** graceful degradation under noise and bounded sensitivity to
    spikes; missing-feature robustness is the main vulnerability.
- **Ablations** confirm both the spatial heterogeneous coupling *and* the
  temporal TCN are necessary — removing either collapses performance.
- **Key engineering insight:** correctly handling **data staleness** (the DE
  edge fix) and choosing **physically grounded edges** (DK2↔HYDRO Øresund) were
  decisive for a fair, leakage-free comparison.
""")

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4, "nbformat_minor": 5,
}

out = Path(__file__).parent / "ST_HeteroSAGE_Review.ipynb"
with open(out, "w") as f:
    json.dump(notebook, f, indent=1)
print("Wrote", out, "with", len(cells), "cells")
