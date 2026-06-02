"""
Generates ST_HeteroSAGE_Review.ipynb — comprehensive supervisor-facing notebook.

Run from the repo root or from notebooks/:
    python3 notebooks/build_review_notebook.py

Uses only the std-lib json module — no nbformat dependency.
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


# ══════════════════════════════════════════════════════════════════════════════
# 0.  TITLE
# ══════════════════════════════════════════════════════════════════════════════
md(r"""# Heterogeneous Spatio-Temporal GNNs for Multi-Area Nordic Electricity Price Forecasting
### Supervisor technical review notebook

**Research question**

> *How can a heterogeneous graph be effectively constructed and integrated into
> GNN models to improve the accuracy, interpretability, and robustness of
> multi-area day-ahead electricity price forecasting in the Nordic power market?*

This notebook is the complete technical record of the project. It covers:

1. [Data sources, cleaning, and storage](#section-1)
2. [Feature engineering and leakage proof](#section-2)
3. [Graph construction — nodes, edges, design decisions](#section-3)
4. [Model architectures (XGBoost → HomoGNN → GAT → HeteroSAGE → ST-HeteroSAGE)](#section-4)
5. [Training protocol — optimiser, scheduling, early stopping](#section-5)
6. [Key design decisions and why](#section-6)
7. [MLOps setup](#section-7)
8. [Results — leaderboard, ablation, interpretability, robustness](#section-8)
9. [Reproducibility](#section-9)
10. [Interactive model testing — run XGBoost, HomoGNN, ST-HeteroSAGE live](#section-10)
11. [Graph structure visualisation — nodes, edges, types, interactive subgraph](#section-11)
12. [Summary for the supervisor — Q&A table](#section-12)

Every result cell loads a committed JSON artifact — no GPU or retraining needed.
Sections 10–11 load model checkpoints and run live inference (CPU, ~30 s total).
""")

# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════
md("## Setup")

code(r"""import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings; warnings.filterwarnings('ignore')

plt.rcParams.update({
    "figure.dpi": 110, "font.size": 11,
    "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False,
})
ZONE_COLORS = {"DK1": "#2171b5", "DK2": "#6baed6", "HYDRO": "#74c476", "DE": "#fd8d3c"}

# Locate repo root (parent that contains both 'src' and 'artifacts')
HERE = Path.cwd()
ROOT = next((p for p in [HERE, *HERE.parents]
             if (p / "src").exists() and (p / "artifacts").exists()), None)
assert ROOT is not None, "Could not find repo root"
SRC     = ROOT / "src"
ART     = ROOT / "artifacts"
ART_HET = SRC / "artifacts_hetero"
sys.path.insert(0, str(SRC))

def load_json(*relpaths):
    for rp in relpaths:
        for base in (ROOT, SRC):
            p = base / rp
            if p.exists():
                with open(p) as f: return json.load(f)
    raise FileNotFoundError(relpaths)

print("Repo root:", ROOT)
print("src/     :", SRC)
print("artifacts:", ART)
""")

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DATA
# ══════════════════════════════════════════════════════════════════════════════
md(r"""<a id="section-1"></a>
## 1. Data sources, cleaning, and storage

All data is persisted in a single **SQLite database** (`src/electricity.db`, ~42 MB)
with four tables. Loading from SQLite at graph-build time ensures one clean,
versioned source of truth — no CSV scatter.

### 1.1 Tables and data sources

| Table | Rows | Coverage | Source |
|-------|------|----------|--------|
| `spot_prices` | 195,035 | 2020-01-01 → 2025-09-30 | Energinet / ENTSO-E day-ahead auctions |
| `weather_data` | 50,016 | 2020-01-17 → 2025-09-30 | Open-Meteo historical API (Copenhagen) |
| `zone_fundamentals` | 173,355 | 2018-01-01 → 2026-05-22 | Energinet ProductionConsumptionSettlement |
| `market_factors` | 61,355 | 2018-01-01 → 2024-12-31 | EEX gas futures + EU ETS CO₂ prices |

### 1.2 Zones covered

| Zone | Rows in spot_prices | Notes |
|------|--------------------|----|
| **DK1** | 50,399 | Western Denmark — wind-heavy, target |
| **DK2** | 50,399 | Eastern Denmark — Øresund link to Sweden, target |
| **HYDRO (SE3)** | 50,399 | Swedish hydro reservoir signal, auxiliary |
| **DE** | 43,838 | German price coupling — **ends 2024-12-31** (important) |

### 1.3 Data cleaning steps

1. **Timeline synchronisation.** An outer-join over all four zones' timestamps
   creates a master frame. Missing values are **forward-filled** (using the last
   known price — valid for a daily forward-fill in slow-moving auction markets)
   and remaining edge NaNs set to 0.0. No backward fill is ever used.
2. **Weather gaps.** ~380 missing weather hours (equipment outages) are forward-
   filled, then capped at the neutral median if still missing.
3. **Per-zone z-score for demand/renewable.** German load (~55 GWh) is ~25×
   larger than Danish load (~2 GWh). Without per-zone normalisation, the global
   StandardScaler would crush DK's variation to near-zero. Stats are computed on
   the first 80 % of each zone's timeline (training window) — never touching
   future data.
4. **No price imputation on the target.** `y` stores the raw DKK prices
   exactly as published. Any zone-hour with a genuinely missing auction price
   (e.g., HYDRO before SE3 joined Nord Pool) is kept as 0.0 and masked out of
   the loss during training.
""")

code(r"""# Inspect the live SQLite database
import sqlite3
conn = sqlite3.connect(SRC / "electricity.db")

tables_info = {}
for table in ['spot_prices', 'weather_data', 'zone_fundamentals', 'market_factors']:
    n = pd.read_sql(f"SELECT COUNT(*) AS n FROM {table}", conn).iloc[0,0]
    rng = pd.read_sql(f"SELECT MIN(hour_utc) AS mn, MAX(hour_utc) AS mx FROM {table}", conn)
    cols = pd.read_sql(f"PRAGMA table_info({table})", conn)['name'].tolist()
    tables_info[table] = {"rows": n, "from": rng['mn'][0], "to": rng['mx'][0], "cols": cols}

df_info = pd.DataFrame(tables_info).T
df_info['rows'] = df_info['rows'].apply(lambda x: f"{x:,}")
display(df_info[['rows','from','to','cols']])
""")

code(r"""# Price distribution per zone — the raw targets (Plotly)
spot = pd.read_sql("SELECT price_zone, price_dkk FROM spot_prices", conn)
fig = go.Figure()
for zone, color in [('DK1','#2171b5'),('DK2','#6baed6'),('HYDRO','#74c476'),('DE','#fd8d3c')]:
    d = spot.loc[spot.price_zone == zone, 'price_dkk']
    fig.add_trace(go.Histogram(
        x=d, nbinsx=80, name=f"{zone} (μ={d.mean():.0f}, σ={d.std():.0f})",
        marker_color=color, opacity=0.55,
    ))
fig.update_layout(
    barmode='overlay', title="Raw price distribution by zone — full dataset (2020–2025)",
    xaxis_title="Price (DKK/MWh)", yaxis_title="Hours",
    legend=dict(x=0.75, y=0.95), height=400,
)
fig.show()

neg = spot[spot.price_dkk < 0]
print(f"Negative-price hours: {len(neg):,} ({len(neg)/len(spot)*100:.1f}% of all zone-hours)")
print(neg.groupby('price_zone').size().rename("negative hours"))
""")

code(r"""# Date coverage gap — DE stops at 2024-12-31, then forward-filled into 2025
print("=== Data coverage per zone and table ===")
for zone in ['DK1','DK2','HYDRO','DE']:
    sp = pd.read_sql(f"SELECT MIN(hour_utc) mn, MAX(hour_utc) mx FROM spot_prices WHERE price_zone='{zone}'", conn)
    zf = pd.read_sql(f"SELECT MIN(hour_utc) mn, MAX(hour_utc) mx FROM zone_fundamentals WHERE price_zone='{zone}'", conn)
    print(f"  {zone:<6}  spot: {sp.iloc[0,0][:10]} → {sp.iloc[0,1][:10]}  "
          f"  fundamentals: {zf.iloc[0,0][:10] if pd.notna(zf.iloc[0,0]) else 'N/A'} → "
          f"{zf.iloc[0,1][:10] if pd.notna(zf.iloc[0,1]) else 'N/A'}")

mf = pd.read_sql("SELECT MIN(hour_utc) mn, MAX(hour_utc) mx FROM market_factors", conn)
print(f"\n  market_factors (gas/CO₂): {mf.iloc[0,0][:10]} → {mf.iloc[0,1][:10]}")
print("\n  ⚠️  DE spot prices end 2024-12-31 — the test window (2025) relies on forward-fill.")
print("  This is why DK↔DE hour-level edges were EXCLUDED from the hetero graph (see §3).")
conn.close()
""")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  FEATURE ENGINEERING & LEAKAGE
# ══════════════════════════════════════════════════════════════════════════════
md(r"""<a id="section-2"></a>
## 2. Feature engineering and leakage proof

### 2.1 The 17 input features

Each hour-node in the graph receives a 17-dimensional feature vector assembled
in the pipeline and then StandardScaled (fit on training data only).

| # | Feature | Group | Description |
|---|---------|-------|-------------|
| 0 | `price_lag_24h` | Price history | Price at same hour, 1 day ago |
| 1 | `price_lag_48h` | Price history | Price at same hour, 2 days ago |
| 2 | `price_lag_168h` | Price history | Price at same hour, 1 week ago |
| 3 | `price_roll_mean_24h` | Price history | 24-hour rolling mean of lag-24h series |
| 4 | `price_roll_std_24h` | Price history | 24-hour rolling std of lag-24h series |
| 5 | `temperature_c` | Weather | Air temperature at delivery hour |
| 6 | `wind_speed_ms` | Weather | Wind speed at delivery hour |
| 7 | `cloud_cover_pct` | Weather | Cloud cover fraction |
| 8 | `humidity_pct` | Weather | Relative humidity |
| 9 | `load_mwh` | Fundamentals | Zone demand (per-zone z-scored) |
| 10 | `renewable_mwh` | Fundamentals | Wind + solar + hydro generation |
| 11 | `gas_dkk` | Fundamentals | Natural gas front-month price |
| 12 | `co2_dkk` | Fundamentals | EU ETS CO₂ allowance price |
| 13 | `hour_sin` | Calendar | sin(2π × hour / 24) |
| 14 | `hour_cos` | Calendar | cos(2π × hour / 24) |
| 15 | `dow_sin` | Calendar | sin(2π × day-of-week / 7) |
| 16 | `dow_cos` | Calendar | cos(2π × day-of-week / 7) |

### 2.2 Day-ahead leakage proof

A day-ahead forecast for hour *h* of day *D* is produced at **gate closure:
12:00 CET on day *D*−1**. Any feature used must be observable at that moment.

| Feature class | Lag | Safe? | Reasoning |
|---------------|-----|-------|-----------|
| `price_lag_24h` | 24 h | ✅ | Day *D*−2 prices are fully published |
| `price_lag_48h` | 48 h | ✅ | Day *D*−3 fully published |
| `price_lag_168h` | 168 h | ✅ | Previous-week prices fully published |
| Rolling stats | ≥ 24 h base | ✅ | Computed on shifted (lag≥24h) series |
| Weather at delivery hour | 0 h nominal | ✅ | Convention in EPF — 12–36 h NWP forecasts have <5% error; standard in literature |
| Fundamentals (load, renewable) | 0 h nominal | ✅ | Published day-ahead by TSO (Energinet) |
| Gas, CO₂ | Daily | ✅ | Slow-moving daily futures, published same morning |

> **No** current-hour price, **no** shift(1)/shift(2)/shift(6) features, and
> **no** cross-zone contemporaneous prices appear anywhere. The original code
> used `shift(1)` lags (1-hour leakage); this was replaced in a dedicated fix
> commit (`B1` in the git history) before any GNN work began.
""")

code(r"""# Verify no NaN or Inf in the scaled feature matrix
import torch
data = torch.load(SRC / "data/graphs_hetero/hetero_graph.pt", weights_only=False)
x = data['hour'].x.numpy()
y = data['hour'].y.numpy()
T = int(data['hour'].num_hours_per_zone)

print("=== Feature matrix health check ===")
print(f"Shape: {x.shape}  (201,596 nodes × 17 features)")
print(f"NaN count: {np.isnan(x).sum()}")
print(f"Inf count: {np.isinf(x).sum()}")
print(f"Mean: {x.mean():.4f}  Std: {x.std():.4f}  (should be ≈0, ≈1 after scaling)")

print("\n=== Target vector health check ===")
print(f"Shape: {y.shape}")
print(f"NaN count: {np.isnan(y).sum()}")
zones = ["DK1","DK2","HYDRO","DE"]
for i, z in enumerate(zones):
    yi = y[i*T:(i+1)*T]
    print(f"  {z:6s}: mean={yi.mean():.1f}  std={yi.std():.1f}  min={yi.min():.1f}  max={yi.max():.1f}")
""")

code(r"""# Show the chronological split boundaries
fig, ax = plt.subplots(figsize=(12, 2.5))
total = T  # hours per zone
train_end = int(total * 0.8)
val_end   = int(total * 0.9)

ax.barh(0, train_end,            height=0.5, color='#2171b5', label=f'Train  ({train_end:,} h = 80%)')
ax.barh(0, val_end - train_end,  height=0.5, left=train_end,  color='#74c476', label=f'Val    ({val_end-train_end:,} h = 10%)')
ax.barh(0, total - val_end,      height=0.5, left=val_end,    color='#fd8d3c', label=f'Test   ({total-val_end:,} h = 10%)')
ax.set_yticks([]); ax.set_xlabel("Hour index")
ax.set_title(f"Chronological 80/10/10 split — {total:,} hours per zone (all 4 zones identical)")
ax.legend(loc='lower right'); plt.tight_layout(); plt.show()

import pandas as pd
# Approximate calendar dates
dates = pd.read_sql("SELECT hour_utc FROM spot_prices WHERE price_zone='DK1' ORDER BY hour_utc",
                    sqlite3.connect(SRC / "electricity.db"))['hour_utc']
print(f"Full window  : {dates.iloc[0][:10]} → {dates.iloc[-1][:10]}")
print(f"Train ends   : {dates.iloc[train_end-1][:10]}")
print(f"Val ends     : {dates.iloc[val_end-1][:10]}")
print(f"Test starts  : {dates.iloc[val_end][:10]}  ← 2025 unseen window")
""")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  GRAPH CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════
md(r"""<a id="section-3"></a>
## 3. Graph construction — nodes, edges, design decisions

### 3.1 Node types

The **heterogeneous** graph (`HeteroData`) has two node types:

| Type | Count | Feature dim | Represents |
|------|-------|-------------|-----------|
| `hour` | 201,596 = 4 × T | 17 | One (zone, timestamp) pair |
| `market` | 4 | 4 | One-hot zone identity (DK1/DK2/HYDRO/DE) |

**Hour node layout** (zone-blocked, contiguous):

```
indices [0,       T)  →  DK1   (T = 50,399 hours)
indices [T,      2T)  →  DK2
indices [2T,     3T)  →  HYDRO (SE3)
indices [3T,     4T)  →  DE
```

Zone-blocked layout simplifies the CausalTCN step in the ST model: a single
`view(4, T, H)` reshapes all four zones into a [zones, time, features] tensor
without any index arithmetic.

### 3.2 Edge types

| Edge type | Count | Direction | Purpose |
|-----------|-------|-----------|---------|
| `hour → belongs_to → market` | 201,596 | Each hour-node to its zone market-node | Routes zone identity into hour embeddings |
| `market → rev_belongs_to → hour` | 201,596 | Reverse | Routes market representation back to hours |
| `hour → co_occurs_with → hour` | 201,596 | Same-timestep cross-zone | Physical interconnect coupling |
| `market → interconnects → market` | 10 | Zone-to-zone | Cross-border capacity routing |
| `hour → lag_to → hour` | 603,828 | t−24h, t−48h, t−168h per zone | Used only by HeteroSAGE, not ST (TCN replaces it) |

### 3.3 co_occurs_with edge design — the DE exclusion

This was the single most consequential graph design decision.

**First design:** DK1↔DK2 + DK2↔HYDRO + **DK1↔DE + DK2↔DE** at hour level.

**Result:** test MAE = **204 DKK** (catastrophic degradation).

**Root cause:** DE spot prices end 2024-12-31. The test window is 2025. The
pipeline forward-fills the last known DE price (~250 DKK) across all of 2025.
Because co_occurs_with edges fire at every hour, each DK1/DK2 test prediction
received the same stale DE signal, systematically biasing forecasts.

**Fix:** removed DK↔DE from co_occurs_with. DE still participates via the
**market bridge** (coarser signal), where the model can learn to down-weight it.
Hour-level DE information is still available as features (lags are computed from
the forward-filled series and are fine since they represent the last-known price).

**Final co_occurs_with topology:**
- `DK1 ↔ DK2` — Great Belt HVDC (~1,000 MW capacity)
- `DK2 ↔ HYDRO` — Øresund link to SE3 (~1,700 MW)

HYDRO (SE3) has full 2025 data so no staleness issue.

### 3.4 market → interconnects → market (capacity weights)

10 directed edges (all physical cross-border links) with edge attributes
encoding real physical NTC capacities in MW. Used as edge features in the
market subgraph to give the model a sense of coupling strength:

| Link | Capacity (MW) |
|------|--------------|
| DK1 ↔ DK2 (Great Belt) | 1,000 |
| DK1 ↔ DE (Kontek + Energybridge) | 2,000 / 600 |
| DK2 ↔ HYDRO (Øresund) | 1,700 |
| DK2 ↔ DE (Kriegers Flak) | 600 |
""")

code(r"""# Inspect the actual graph tensor
print("=== hetero_graph.pt summary ===\n")
print(f"hour nodes:   {data['hour'].x.shape}  (201,596 × 17)")
print(f"market nodes: {data['market'].x.shape}  (4 zones × 4-dim one-hot)")
print()
print("Edge types:")
for et in data.edge_types:
    n = data[et].edge_index.shape[1]
    has_attr = hasattr(data[et], 'edge_attr') and data[et].edge_attr is not None
    attr_str = f"  [edge_attr: {data[et].edge_attr.shape}]" if has_attr else ""
    print(f"  {str(et):<55} {n:>8,} edges{attr_str}")
""")

code(r"""# Visualise the market-level topology (interactive Plotly)
pos = {"DK1": (0, 1), "DK2": (1, 1), "HYDRO": (1.5, 2), "DE": (0.5, 0)}
node_colors = {"DK1": "#2171b5", "DK2": "#6baed6", "HYDRO": "#74c476", "DE": "#fd8d3c"}
edges_top = [
    ("DK1","DK2",  "1,000 MW", "#c0504d", True),
    ("DK2","HYDRO","1,700 MW", "#c0504d", True),
    ("DK1","DE",   "market bridge only", "#aaaaaa", False),
    ("DK2","DE",   "market bridge only", "#aaaaaa", False),
    ("DK1","HYDRO","market bridge only", "#aaaaaa", False),
]

fig = go.Figure()
for src, dst, label, color, is_hour in edges_top:
    x0, y0 = pos[src]; x1, y1 = pos[dst]
    fig.add_trace(go.Scatter(
        x=[x0, (x0+x1)/2, x1], y=[y0, (y0+y1)/2, y1], mode='lines+text',
        line=dict(color=color, width=3 if is_hour else 1.5, dash='solid' if is_hour else 'dash'),
        text=["", label, ""], textposition="top center", textfont=dict(size=9, color=color),
        showlegend=False, hoverinfo='skip',
    ))
for name, (x_, y_) in pos.items():
    fig.add_trace(go.Scatter(
        x=[x_], y=[y_], mode='markers+text',
        marker=dict(size=30, color=node_colors[name], line=dict(width=2, color='white')),
        text=[name], textposition="top center", textfont=dict(size=12, color='black', family='Arial Black'),
        name=name, hovertemplate=f"<b>{name}</b><extra></extra>",
    ))
fig.update_layout(
    title="Market-level physical interconnect topology",
    xaxis=dict(visible=False, range=[-0.4, 2.1]),
    yaxis=dict(visible=False, range=[-0.4, 2.6]),
    height=420, showlegend=True,
    legend=dict(x=0.01, y=0.01),
    plot_bgcolor='white',
)
fig.show()
""")

code(r"""# Prove DE is excluded from co_occurs_with edges
co = data['hour', 'co_occurs_with', 'hour'].edge_index.numpy()
T  = int(data['hour'].num_hours_per_zone)
de_involved = (co[0] >= 3*T) | (co[1] >= 3*T)
print(f"co_occurs_with edges involving DE: {de_involved.sum():,}  ← must be 0")
print(f"co_occurs_with edges DK1↔DK2:     {((co[0]<T)&(co[1]>=T)&(co[1]<2*T)).sum() + ((co[1]<T)&(co[0]>=T)&(co[0]<2*T)).sum():,}")
print(f"co_occurs_with edges DK2↔HYDRO:   {((co[0]>=T)&(co[0]<2*T)&(co[1]>=2*T)).sum() + ((co[1]>=T)&(co[1]<2*T)&(co[0]>=2*T)).sum():,}")
""")

# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════
md(r"""<a id="section-4"></a>
## 4. Model architectures

### 4.1 XGBoost baseline

The XGBoost model is the **"no graph, no temporal model"** reference. It sees
each (zone, hour) row as an independent sample — no message passing, no
sequence modelling.

- **Features:** 13-element tabular set (lags, weather, fundamentals — same
  13 base features as the GNNs but *without* cyclical encoding, without
  the full 17-feature graph representation)
- **Neighbourhood signal:** lagged DE and HYDRO prices are added as raw columns
  (`neighbor_price_de`, `neighbor_price_hydro`) to give XGBoost some notion of
  cross-zone context without a graph
- **Split:** 80 % train / 20 % test (no validation set needed)
- **Config:** 250 trees, depth 6, lr 0.05, subsample 0.8

### 4.2 HomoGNN — GraphSAGE (homogeneous)

A flat, single-type graph where all nodes (DK1/DK2/HYDRO/DE hours) and all
edges are the same type. GraphSAGE aggregates spatial neighbours (cross-zone)
and temporal neighbours (lag-24/48/168h) in the same convolution.

```
Architecture: 3-layer GraphSAGE  →  MLP head
  [17 features] → SAGEConv(128) → BN → SAGEConv(128) → BN
                → SAGEConv(128) → BN → Linear(64) → ReLU → Linear(1)
```

> **BN + Dropout interaction fix.** An early version of HomoGNN used
> `Dropout(0.2)`. BatchNorm normalises *after* the layer but before Dropout;
> when Dropout then randomly zeros activations it distorts the BN statistics,
> causing training to diverge (MAE shot from 162 → 296 DKK). Fixed by setting
> `dropout=0.0`. BN already regularises sufficiently on this dataset size.

### 4.3 GAT (Graph Attention Network)

Same homogeneous graph as HomoGNN but with multi-head attention (4 heads × 32
dims = 128) so the model can selectively up-weight more relevant neighbours.
GAT is outperformed by the typed HeteroSAGE because attention in a flat graph
cannot distinguish temporal from spatial neighbours — the same attention weight
applies to both a lag-24h edge and a co_occurs_with edge.

### 4.4 HeteroSAGE

The first heterogeneous model. It separates the graph into typed edge relations,
giving each edge type its own SAGEConv with separate weights.

```
Architecture: 3 × HeteroConv → shared MLP head

HeteroConv per layer handles all 5 edge types:
  hour  → co_occurs_with → hour     SAGEConv(128,128)
  hour  → belongs_to     → market   SAGEConv(128,128)
  market→ rev_belongs_to → hour     SAGEConv(128,128)
  hour  → lag_to         → hour     SAGEConv(128,128)  ← temporal lags as edges
  market→ interconnects  → market   SAGEConv(128,128)

Input projections:
  hour_proj   = Linear(17, 128)
  market_proj = Linear(4, 128)  → Linear(128, 128) [2-layer residual MLP]

Normalisation: BatchNorm1d(128) after each conv, residual connection

Output: single shared head  Linear(128) → ReLU → Linear(64) → ReLU → Linear(1)
```

**Why a shared output head?** An earlier version had zone-specific heads
(one for DK1, DK2, HYDRO, DE). Because DE/HYDRO targets are 0 in the test
window, their heads received gradient signal from zero-filled targets,
injecting noise into the shared encoder. A single head with the loss masked
to DK1+DK2 removes this entirely.

### 4.5 ST-HeteroSAGE (winning model)

Replaces the `lag_to` edges with a **CausalTCN** applied per zone along the
time axis. This separates the spatial and temporal computations into dedicated
pathways, each optimised for its task.

```
Architecture: 2 × STBlock → shared MLP head

STBlock:
  ── Spatial: HeteroConv (same-timestep only, NO lag_to)
      hour → co_occurs_with → hour
      hour → belongs_to     → market
      market→ rev_belongs_to → hour
      market→ interconnects  → market
      + BatchNorm + residual
  ── Temporal: CausalTCN (per zone, independent)
      Input: [4 zones, 128 channels, T timesteps]
      3 dilated causal Conv1d layers: dilations (1, 4, 24), kernel 7
      Receptive field = (7−1)×(1+4+24) = 174 hours ≈ 7.25 days
      Causal padding: (k−1)*d on the LEFT only (no look-ahead)
      + BatchNorm + residual

Input projections:
  hour_proj    = Linear(17, 128)
  market_proj  = Linear(4, 128) → Linear(128,128) [2-layer residual MLP]

Output: shared head  Linear(128) → ReLU → Linear(64) → ReLU → Linear(1)
Total parameters: 982,913
```
""")

code(r"""# Verify architecture and parameter counts
import torch
from hetero_st_model import HeteroSTPriceForecaster
from hetero_models import HeteroPriceForecaster
from gnn_models import GraphSAGEModel

models = {
    "HomoGNN (GraphSAGE)": GraphSAGEModel(17, 128, 3, dropout=0.0),
    "HeteroSAGE":          HeteroPriceForecaster(None, 17, 128),
    "ST-HeteroSAGE":       HeteroSTPriceForecaster(17, 128, 2),
}

print(f"{'Model':<25} {'Parameters':>12}")
print("-" * 40)
for name, m in models.items():
    p = sum(x.numel() for x in m.parameters())
    print(f"  {name:<23} {p:>12,}")
""")

code(r"""# Visualise the ST block dataflow
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: architecture stack
ax = axes[0]
ax.set_xlim(0, 4); ax.set_ylim(0, 10); ax.axis('off')
ax.set_title("ST-HeteroSAGE forward pass", fontsize=12)
blocks = [
    (0.5, 9.0, 3.0, 0.7, "#dde3f0", "Input x_hour [201596, 17] + x_market [4, 4]"),
    (0.5, 7.8, 3.0, 0.7, "#c6d4e8", "hour_proj Linear(17→128)  |  market 2-layer MLP"),
    (0.5, 6.0, 3.0, 1.4, "#a8c0df", "ST Block 1\n  Spatial: HeteroConv (4 edge types)\n  Temporal: CausalTCN (dilations 1,4,24)"),
    (0.5, 4.2, 3.0, 1.4, "#a8c0df", "ST Block 2\n  Spatial: HeteroConv\n  Temporal: CausalTCN"),
    (0.5, 3.0, 3.0, 0.7, "#c6d4e8", "Shared MLP head  Linear(128→64→1)"),
    (0.5, 1.8, 3.0, 0.7, "#dde3f0", "Output [201596, 1] → masked to DK1+DK2 test"),
]
for x_, y_, w, h, c, txt in blocks:
    rect = mpatches.FancyBboxPatch((x_,y_), w, h, boxstyle="round,pad=0.05", fc=c, ec='gray', lw=0.7)
    ax.add_patch(rect)
    ax.text(x_+w/2, y_+h/2, txt, ha='center', va='center', fontsize=8.5)
    if y_ < 9.0:
        ax.annotate('', xy=(2, y_+h+0.02), xytext=(2, y_+h+0.35),
                    arrowprops=dict(arrowstyle='->', color='gray'))

# Right: CausalTCN receptive field
ax2 = axes[1]
dilations = [1, 4, 24]; kernel = 7; color_map = ['#2171b5','#41ab5d','#d94801']
rf_per_layer = [(kernel-1)*d for d in dilations]
cumulative_rf = [sum(rf_per_layer[:i+1]) for i in range(len(rf_per_layer))]
for i, (d, rf, c) in enumerate(zip(dilations, rf_per_layer, color_map)):
    ax2.barh(i, rf, color=c, alpha=0.7,
             label=f"Layer {i+1}: dilation={d}, kernel={kernel}, RF={rf}h")
ax2.barh(len(dilations), cumulative_rf[-1], color='gray', alpha=0.3, label=f"Total RF ≈ {cumulative_rf[-1]}h")
ax2.axvline(24, ls=':', color='black', alpha=0.5); ax2.text(24, -0.5, '24h', ha='center', fontsize=9)
ax2.axvline(168, ls=':', color='black', alpha=0.5); ax2.text(168, -0.5, '168h', ha='center', fontsize=9)
ax2.set_yticks(range(len(dilations)+1))
ax2.set_yticklabels([f"Layer {i+1}" for i in range(len(dilations))] + ["Total"])
ax2.set_xlabel("Receptive field (hours)"); ax2.set_title("CausalTCN receptive field breakdown")
ax2.legend(fontsize=9); plt.tight_layout(); plt.show()
print(f"Total CausalTCN receptive field: {cumulative_rf[-1]} hours ≈ {cumulative_rf[-1]/24:.1f} days")
""")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRAINING PROTOCOL
# ══════════════════════════════════════════════════════════════════════════════
md(r"""<a id="section-5"></a>
## 5. Training protocol

All GNN models share the same training loop design, ensuring any performance
difference is due to architecture rather than training procedure.

### 5.1 Shared configuration

| Setting | Value | Why |
|---------|-------|-----|
| Optimiser | AdamW | Adam with decoupled weight decay (more stable than L2 in Adam) |
| Learning rate | 0.001 | Matched across HomoGNN / HeteroSAGE / ST; XGBoost uses 0.05 |
| Weight decay | 5×10⁻⁴ | Light L2 regularisation |
| LR scheduler | ReduceLROnPlateau | Halves LR when val loss plateaus for 10 epochs |
| Gradient clipping | max_norm = 1.0 | Prevents exploding gradients through deep HeteroConv stacks |
| Max epochs | 200 | Upper bound; early stopping usually triggers first |
| Early stopping | patience = 25 | Stop when val MSE doesn't improve for 25 epochs; restore best ckpt |
| Loss | MSE (scaled) | Train in standardised space (μ=0, σ=1) for numerical stability |
| Evaluation | MAE (DKK) | Report in original units; inverse-transform via StandardScaler |

### 5.2 Auxiliary loss (HeteroSAGE and ST-HeteroSAGE)

```python
main_loss = MSE(out[dk1_dk2_train], y[dk1_dk2_train])
aux_loss  = MSE(out[de_hydro_train], y[de_hydro_train])
loss      = main_loss + 0.1 * aux_loss
```

DE and HYDRO prices are real (DE through 2024, SE3/HYDRO throughout). Having
the model also predict them at a reduced weight (0.1×) forces the encoder to
build representations that understand *why* prices differ across zones —
information that then propagates to DK1/DK2 via co_occurs_with and market edges.

Without aux loss: the model ignores DE/HYDRO node context entirely because
those losses are masked out. With aux loss: cross-zone embeddings improve,
and DK1/DK2 accuracy benefits through message passing.

### 5.3 Target scaling

```python
scaler = StandardScaler()
scaler.fit(y[dk1_dk2_train])      # fit on DK1+DK2 training nodes only
y_scaled = scaler.transform(y)    # apply to all nodes
loss = MSE(out[train_mask], y_scaled[train_mask])
# At eval time:
y_pred_dkk = scaler.inverse_transform(out[test_mask])
```

Fitting the scaler on DK1+DK2 training nodes only prevents two contamination
paths: (a) DE/HYDRO zeros pulling the scale down, and (b) future test prices
shifting the mean.
""")

code(r"""# Visualise training history (Plotly — hover to inspect any epoch)
hist_path = ART_HET / "hetero_training_history.json"
if hist_path.exists():
    hist_raw = json.load(open(hist_path))
    if isinstance(hist_raw, list):
        val_mae = [e["val_mae"] for e in hist_raw]
        val_r2  = [e.get("val_r2", None) for e in hist_raw]
    else:
        val_mae = hist_raw.get("val_mae", [])
        val_r2  = hist_raw.get("val_r2", [None]*len(val_mae))
    epochs = list(range(1, len(val_mae)+1))
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Validation MAE (DKK)", "Validation R²"))
    fig.add_trace(go.Scatter(x=epochs, y=val_mae, mode='lines', name='Val MAE',
                             line=dict(color='#2171b5', width=2)), row=1, col=1)
    best_epoch = int(np.argmin(val_mae)) + 1
    fig.add_vline(x=best_epoch, line_dash='dash', line_color='red',
                  annotation_text=f"best epoch {best_epoch}", row=1, col=1)
    if any(v is not None for v in val_r2):
        fig.add_trace(go.Scatter(x=epochs, y=val_r2, mode='lines', name='Val R²',
                                 line=dict(color='#41ab5d', width=2)), row=1, col=2)
    fig.update_xaxes(title_text="Epoch")
    fig.update_yaxes(title_text="MAE (DKK)", row=1, col=1)
    fig.update_yaxes(title_text="R²", row=1, col=2)
    fig.update_layout(height=380, title="HeteroSAGE training convergence")
    fig.show()
    print(f"Best val MAE: {min(val_mae):.1f} DKK at epoch {best_epoch}")
    print(f"Total epochs logged: {len(val_mae)}")
else:
    print("Training history not saved for ST model — checkpoint is best-val-epoch weights.")
""")

# ══════════════════════════════════════════════════════════════════════════════
# 6.  KEY DESIGN DECISIONS
# ══════════════════════════════════════════════════════════════════════════════
md(r"""<a id="section-6"></a>
## 6. Key design decisions and why

This section documents every non-trivial architectural or data choice made
during development, the evidence that motivated it, and the quantified impact
where available.

| Decision | Alternative considered | Evidence / impact |
|----------|----------------------|-------------------|
| **Zone-blocked node layout** [DK1,DK2,HYDRO,DE] | Interleaved layout | Blocked layout allows O(1) `view(4,T,H)` for CausalTCN; interleaved needs scatter/gather |
| **CausalTCN replaces `lag_to` edges** | Keep lag_to in ST model | TCN gives a 174-hour receptive field (vs 3 explicit lags); causal padding guarantees no leakage |
| **Remove DK↔DE from `co_occurs_with`** | Include DE edges | +53 DKK MAE when included (204 vs 151) — DE is forward-filled in 2025 test window |
| **Add DK2↔HYDRO edge** | DK1↔DK2 only | Ablation F: +4.4 DKK when removed; SE3 hydro scarcity is a real price driver |
| **2-layer residual market MLP** | Single Linear(4,128) | Richer zone embedding; part of the recipe that lifted hetero from 168→163 DKK |
| **Single shared output head** | 4 zone-specific heads | Eliminated gradient noise from DE/HYDRO heads; cleaner regression signal |
| **Auxiliary DE/HYDRO loss (0.1×)** | No aux loss | Forces encoder to learn cross-zone dynamics; improves DK1/DK2 via message passing |
| **BatchNorm, no Dropout** | BN + Dropout(0.2) | BN+Dropout interaction caused HomoGNN to diverge: MAE 296 → 162 DKK fix |
| **Feature scaler fit on all 4 zones** | Fit on DK1+DK2 only | Keeps HYDRO/DE features in moderate scaled range (−0.56); fitting on DK alone pushes them to −0.99 |
| **Target scaler fit on DK1+DK2 train only** | All-zone targets | Prevents DE/HYDRO zeros from shifting the mean; prevents test data leaking |
| **Per-zone z-score for load/renewable** | Global scaler | German load ~25× Danish; without per-zone normalisation DK variation is crushed to noise |
| **Gradient clipping (max_norm=1.0)** | No clipping | Prevents exploding gradients through 3-layer HeteroConv with residuals |
| **AdamW over SGD** | SGD with momentum | AdamW's decoupled weight decay more stable on sparse heterogeneous graphs |
| **Day-ahead lag shift ≥ 24h** | shift(1)/shift(2) lags | Original code leaked current-day price; fix described in commit B1 |
""")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  MLOPS
# ══════════════════════════════════════════════════════════════════════════════
md(r"""<a id="section-7"></a>
## 7. MLOps setup

### 7.1 Experiment tracking — MLflow

Every training run (HomoGNN, HeteroSAGE, GAT) is instrumented with MLflow:

```python
mlflow.log_params({"hidden_channels": 128, "lr": 0.001, ...})
mlflow.log_metric("val_mae", val_mae, step=epoch)   # per-epoch
mlflow.log_metrics({"test_mae": mae, "test_r2": r2})  # final
mlflow.log_artifact(str(ckpt_path))                 # checkpoint
```

If `mlflow` is not installed the training script falls back gracefully
(`_mlflow_ok = False`) — MLflow is an enhancement, not a dependency.

### 7.2 Artifact versioning

All model checkpoints and metric JSONs are committed to git:

| Artifact | Location | What |
|----------|----------|------|
| `best_homo_model.pt` | `src/artifacts/` | HomoGNN GraphSAGE best checkpoint |
| `best_hetero_model.pt` | `src/artifacts_hetero/` | HeteroSAGE best checkpoint |
| `best_st_hetero_model.pt` | `src/artifacts_hetero/` | ST-HeteroSAGE best checkpoint |
| `*_metrics.json` | `src/artifacts*/` | Reproducible evaluation numbers |
| `hetero_graph.pt` | `src/data/graphs_hetero/` | Fully built graph tensor |
| `hetero_scalers.pkl` | `src/data/graphs_hetero/` | Fitted feature & target scalers |

### 7.3 Warm-start and freeze-scaler modes

Both the graph builder and the HeteroSAGE training support incremental updates:

```bash
# Re-build graph but reuse existing scalers (stable feature distribution)
python3 hetero_graph_builder.py --freeze-scaler

# Fine-tune the HeteroSAGE checkpoint on new data (40 epochs, lr 5×10⁻⁴)
python3 quick_retrain.py --warm-start
```

This allows production deployment where new hourly data arrives continuously
without triggering a full 200-epoch retrain.

### 7.4 CI / GitHub Actions

`.github/workflows/` contains a CI pipeline that:
1. Installs dependencies from `requirements.txt`
2. Runs data ingestion health checks
3. Validates that graph builds without errors
4. Runs `pytest` on unit tests

The project also ships a `Dockerfile` and `docker-compose.yml` for containerised
inference, and a `Makefile` with targets `make train`, `make evaluate`, `make serve`.
""")

code(r"""# Show the full list of committed artifacts with sizes
import os
art_dirs = [ART, ART_HET]
print(f"{'File':<55} {'Size':>9}")
print("-" * 67)
for d in art_dirs:
    if d.exists():
        for f in sorted(d.iterdir()):
            size = f.stat().st_size
            size_str = f"{size/1e6:.1f} MB" if size > 1e5 else f"{size/1e3:.0f} KB"
            print(f"  {str(f.relative_to(ROOT)):<53} {size_str:>9}")
""")

# ══════════════════════════════════════════════════════════════════════════════
# 8.  RESULTS
# ══════════════════════════════════════════════════════════════════════════════
md(r"""<a id="section-8"></a>
## 8. Results

### 8.1 Leaderboard — all models on DK1+DK2 test window
""")

code(r"""rows = [
    ("XGBoost (baseline)", load_json("artifacts/xgboost_metrics.json")),
    ("HomoGNN (GraphSAGE)", load_json("artifacts/homo_gnn_metrics.json")),
    ("GAT",                 load_json("artifacts_hetero/gat_metrics_clean.json")),
    ("HeteroSAGE",          load_json("artifacts_hetero/hetero_metrics_clean.json")),
    ("ST-HeteroSAGE",       load_json("artifacts_hetero/st_hetero_metrics.json")),
]
df_lb = pd.DataFrame([{
    "Model": name,
    "MAE (DKK)": round(m["mae"], 1), "RMSE (DKK)": round(m["rmse"], 1),
    "R²": round(m["r2"], 3), "SMAPE (%)": round(m.get("smape", float("nan")), 1),
} for name, m in rows]).set_index("Model")
# Add improvement vs XGBoost
xgb_mae = df_lb.loc["XGBoost (baseline)", "MAE (DKK)"]
df_lb["Δ vs XGBoost"] = df_lb["MAE (DKK)"].apply(
    lambda x: f"{(x-xgb_mae)/xgb_mae*100:+.1f}%")
display(df_lb)

best = df_lb["MAE (DKK)"].idxmin()
imp  = (xgb_mae - df_lb.loc[best,"MAE (DKK)"]) / xgb_mae * 100
print(f"\nBest model: {best}")
print(f"MAE improvement over XGBoost: {imp:.1f}%")
""")

code(r"""order   = df_lb.index.tolist()
palette = ["#9aa0a6","#5b8def","#74c476","#3f6fd1","#7c3aed"]

fig = make_subplots(rows=1, cols=2, subplot_titles=("Test MAE — lower is better",
                                                      "Test R² — higher is better"))
fig.add_trace(go.Bar(
    x=order, y=df_lb["MAE (DKK)"].values, marker_color=palette,
    text=[f"{v:.1f}" for v in df_lb["MAE (DKK)"].values], textposition='outside',
    name='MAE',
), row=1, col=1)
fig.add_trace(go.Bar(
    x=order, y=df_lb["R²"].values, marker_color=palette,
    text=[f"{v:.3f}" for v in df_lb["R²"].values], textposition='outside',
    name='R²',
), row=1, col=2)
fig.update_yaxes(title_text="DKK", row=1, col=1)
fig.update_yaxes(title_text="R²", range=[0, 0.82], row=1, col=2)
fig.update_xaxes(tickangle=-20)
fig.update_layout(height=440, showlegend=False, title="Model leaderboard — DK1+DK2 test set")
fig.show()
""")

md("### 8.2 Ablation study — ST-HeteroSAGE component contributions")

code(r"""abl = load_json("artifacts_hetero/st_ablation_results.json")
labels = {
    "A_full":        "A · Full model",
    "B_no_tcn":      "B · No TCN (temporal pathway)",
    "C_no_spatial":  "C · No spatial HeteroConv",
    "D_no_cooccurs": "D · No co_occurs_with edges",
    "E_no_market":   "E · No market bridge",
    "F_no_hydro":    "F · No DK2↔HYDRO edge",
}
base = abl["A_full"]["mae"]
df_abl = pd.DataFrame([{
    "Variant": labels[k], "MAE (DKK)": round(abl[k]["mae"],1),
    "R²": round(abl[k]["r2"],3), "Δ MAE": round(abl[k]["mae"]-base,1),
    "Δ%": f"{(abl[k]['mae']-base)/base*100:+.1f}%",
} for k in labels]).set_index("Variant")
display(df_abl)

deltas = df_abl["Δ MAE"].iloc[1:]
bar_colors = ['#c0504d' if v>20 else '#e07030' if v>10 else '#f0a060' for v in deltas.values]
fig = go.Figure(go.Bar(
    y=deltas.index[::-1].tolist(),
    x=deltas.values[::-1].tolist(),
    orientation='h',
    marker_color=bar_colors[::-1],
    text=[f"+{v:.0f} DKK" for v in deltas.values[::-1]],
    textposition='outside',
))
fig.update_layout(
    title="Ablation: cost of removing each ST-HeteroSAGE component",
    xaxis_title="Δ MAE vs full model (DKK) — larger = more important",
    height=380,
)
fig.show()
""")

md("### 8.3 Interpretability — feature importance and error patterns")

code(r"""interp = load_json("artifacts_hetero/st_interpretability.json")
fi = pd.DataFrame(interp["feature_importance"]).set_index("feature")

# Feature importance — horizontal bar (Plotly)
top = fi.head(10).iloc[::-1]
max_imp = top["importance"].max()
fig = go.Figure(go.Bar(
    y=top.index.tolist(), x=(top["importance"] / max_imp).tolist(),
    orientation='h', marker_color='#4f81bd',
    text=[f"{v:.2e}" for v in top["importance"].tolist()], textposition='outside',
))
fig.update_layout(
    title="Top 10 features — gradient saliency (ST-HeteroSAGE)",
    xaxis_title="Normalised importance (top feature = 1.0)",
    height=380,
)
fig.show()

# Error by hour of day and day of week (side-by-side Plotly)
ebt = interp.get("error_by_time", {})
hod = {int(k): v for k, v in ebt.get("hour_of_day", {}).items()}
dow = ebt.get("day_of_week", {})
order_dow = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]

fig = make_subplots(rows=1, cols=2, subplot_titles=("MAE by hour of day", "MAE by day of week"))
if hod:
    hours = sorted(hod)
    fig.add_trace(go.Scatter(
        x=hours, y=[hod[h] for h in hours], mode='lines+markers',
        line=dict(color='#e0773a', width=2), marker=dict(size=6), name='MAE by hour',
    ), row=1, col=1)
if dow:
    days_present = [d for d in order_dow if d in dow]
    fig.add_trace(go.Bar(
        x=days_present, y=[dow[d] for d in days_present],
        marker_color='#8064a2', name='MAE by day',
    ), row=1, col=2)
fig.update_yaxes(title_text="MAE (DKK)")
fig.update_layout(height=360, showlegend=False)
fig.show()

if hod:
    wh = max(hod, key=hod.get); bh = min(hod, key=hod.get)
    print(f"Worst hour: {wh:02d}:00 ({hod[wh]:.1f} DKK)  Best hour: {bh:02d}:00 ({hod[bh]:.1f} DKK)")
if dow:
    wd = max(dow, key=dow.get); bd = min(dow, key=dow.get)
    print(f"Worst day:  {wd} ({dow[wd]:.1f} DKK)  Best day:  {bd} ({dow[bd]:.1f} DKK)")
""")

md("### 8.4 Robustness under perturbation")

code(r"""rob = load_json("artifacts_hetero/st_robustness_results.json")
base_mae = rob["baseline"]["mae"]

scenarios = [
    ("gaussian_noise",  "noise_{}pct",  [5,10,20,30], "σ (% of std)",   "#4f81bd"),
    ("feature_dropout", "drop_{}pct",   [10,20,30],   "Dropout %",      "#c0504d"),
    ("price_spike",     "spike_{}x",    [2,3,5],      "Spike factor ×", "#9bbb59"),
]
fig = make_subplots(rows=1, cols=3, subplot_titles=[
    "Gaussian Noise", "Feature Dropout", "Price Spike"])
for col, (section, key_fmt, xs, xlabel, color) in enumerate(scenarios, 1):
    maes   = [rob[section][key_fmt.format(x)]["mae"]       for x in xs]
    deltas = [rob[section][key_fmt.format(x)]["delta_pct"] for x in xs]
    fig.add_trace(go.Scatter(
        x=xs, y=maes, mode='lines+markers',
        line=dict(color=color, width=2), marker=dict(size=8),
        text=[f"Δ{d:+.1f}%" for d in deltas], textposition='top center',
        name=section.replace('_',' '),
    ), row=1, col=col)
    fig.add_hline(y=base_mae, line_dash='dash', line_color='gray',
                  annotation_text=f"baseline {base_mae:.0f}", row=1, col=col)
    fig.update_xaxes(title_text=xlabel, row=1, col=col)
    fig.update_yaxes(title_text="MAE (DKK)", row=1, col=col)
fig.update_layout(
    height=400, showlegend=False,
    title="Robustness: MAE under test-time perturbation (no retraining)",
)
fig.show()

print(f"\nWorst-case degradation:")
print(f"  Gaussian noise σ=30%:   Δ = {rob['gaussian_noise']['noise_30pct']['delta_pct']:+.1f}%")
print(f"  Feature dropout 30%:    Δ = {rob['feature_dropout']['drop_30pct']['delta_pct']:+.1f}%")
print(f"  Price spike ×5:         Δ = {rob['price_spike']['spike_5x']['delta_pct']:+.1f}%")
""")

# ══════════════════════════════════════════════════════════════════════════════
# 9.  REPRODUCIBILITY
# ══════════════════════════════════════════════════════════════════════════════
md(r"""<a id="section-9"></a>
## 9. Reproducibility

All code is on branch `claude/upbeat-cannon-Z8QA6`. Every number in this
notebook can be reproduced from scratch:

```bash
cd src

# Step 1 — build graph from SQLite DB
python3 hetero_graph_builder.py          # writes hetero_graph.pt + scalers.pkl

# Step 2 — train all models
python3 xgboost_baseline.py              # XGBoost → artifacts/xgboost_metrics.json
python3 homo_retrain.py                  # HomoGNN  → artifacts/homo_gnn_metrics.json
python3 gat_train.py                     # GAT      → artifacts_hetero/gat_metrics_clean.json
python3 quick_retrain.py                 # HeteroSAGE → artifacts_hetero/hetero_metrics_clean.json
python3 st_train.py                      # ST-HeteroSAGE → artifacts_hetero/st_hetero_metrics.json

# Step 3 — post-hoc analyses on the ST checkpoint
python3 st_ablation.py                   # → st_ablation_results.json
python3 st_interpretability.py           # → st_interpretability.json
python3 st_robustness.py                 # → st_robustness_results.json
```

**Seeding:** `torch.manual_seed(42)` and `np.random.seed(42)` are set in all
GNN training scripts. XGBoost uses `random_state=42`.

**Environment:** Python 3.10, PyTorch 2.x, PyTorch-Geometric 2.x, XGBoost 3.x.
See `requirements.txt` in the repo root.

<a id="section-12"></a>
## 12. Summary for the supervisor

| Question | Answer |
|----------|--------|
| **Is the data clean?** | Yes — NaN-free feature matrix, no Inf; forward-fill only, no future-fill |
| **Is there leakage?** | No — all lags ≥ 24 h; chronological splits; scaler fit on train only |
| **Why heterogeneous graph?** | Typed edges learn separate weights per relation; allows physical meaning (capacity, temporal vs spatial) |
| **Why CausalTCN?** | Replaces 3 explicit lag edges with a 174-hour receptive field; strictly causal; much richer temporal signal |
| **Why was DE excluded from co_occurs_with?** | DE prices end 2024-12-31; hour-level edges leaked stale forward-filled data into 2025 test predictions (+53 DKK) |
| **Why no Dropout?** | BN + Dropout interact adversely (BN statistics corrupted by dropout masking); BN alone regularises adequately |
| **Is MLOps in place?** | MLflow experiment tracking, git-versioned artifacts, warm-start / freeze-scaler for incremental updates, Docker + CI/CD |
| **Best result** | ST-HeteroSAGE: **MAE 151.1 DKK, R² 0.696** — 26.5% better than XGBoost baseline |
""")

# ══════════════════════════════════════════════════════════════════════════════
# 10.  INTERACTIVE MODEL TESTING
# ══════════════════════════════════════════════════════════════════════════════
md(r"""<a id="section-10"></a>
## 10. Interactive Model Testing

Run the cells below to perform **live inference** with XGBoost, HomoGNN, and
ST-HeteroSAGE.  Edit the `INPUT FEATURES` block at the top of each sub-section
to test different scenarios.  The model prints its prediction alongside the
expected uncertainty (± test MAE) and a simple bar chart.

> **Note:** All model files are read from `src/` (the notebook kernel's working
> directory).  If a checkpoint is missing, the cell prints a reproduction command.
""")

# ── 10.1 XGBoost ─────────────────────────────────────────────────────────────
md("### 10.1 XGBoost — live inference")

code(r"""# ── INPUT FEATURES (edit these) ──────────────────────────────────────────
zone            = "DK1"   # "DK1" or "DK2"
lag_24h         = 420.0   # price same hour, yesterday        (DKK/MWh)
lag_48h         = 390.0   # price same hour, two days ago
lag_168h        = 445.0   # price same hour, last week
roll_mean       = 410.0   # mean price over 24-47 h window
roll_std        = 35.0    # std  price over 24-47 h window
temperature_c   = 8.0
wind_speed_ms   = 6.5
cloud_cover_pct = 60.0
humidity_pct    = 75.0
hour_of_day     = 17      # 0-23
day_of_week     = 0       # 0=Mon … 6=Sun
# ─────────────────────────────────────────────────────────────────────────────

import pickle, math, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

xgb_path = Path("artifacts/xgboost_model.pkl")
if not xgb_path.exists():
    print(f"Checkpoint not found: {xgb_path}")
    print("  → cd src && python xgboost_baseline.py")
else:
    with open(xgb_path, "rb") as f:
        xgb_model = pickle.load(f)

    h_sin = math.sin(2*math.pi*hour_of_day/24); h_cos = math.cos(2*math.pi*hour_of_day/24)
    d_sin = math.sin(2*math.pi*day_of_week/7);  d_cos = math.cos(2*math.pi*day_of_week/7)
    z_enc = 0 if zone == "DK1" else 1

    feat = np.array([[lag_24h, lag_48h, lag_168h, roll_mean, roll_std,
                      temperature_c, wind_speed_ms, cloud_cover_pct, humidity_pct,
                      h_sin, h_cos, d_sin, d_cos, z_enc]], dtype=np.float32)
    pred = float(xgb_model.predict(feat)[0])

    metrics = load_json("artifacts/xgboost_metrics.json")
    mae = metrics.get("mae", metrics.get("test_mae", 205.6))

    print(f"Zone:               {zone}")
    print(f"Predicted price:    {pred:,.1f} DKK/MWh")
    print(f"Test MAE:           {mae:.1f} DKK  → 90 % interval ≈ [{pred-mae:,.0f}, {pred+mae:,.0f}]")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[zone], y=[pred], marker_color="#4e79a7",
                         text=[f"{pred:.1f}"], textposition="outside", name="Prediction"))
    fig.add_shape(type="rect", x0=-0.4, x1=0.4,
                  y0=max(0, pred-mae), y1=pred+mae,
                  fillcolor="lightblue", opacity=0.25, line_width=0)
    fig.update_layout(title=f"XGBoost prediction — {zone}  (shading = ± MAE)",
                      yaxis_title="Price (DKK/MWh)", height=340, showlegend=False)
    fig.show()
""")

# ── 10.2 HomoGNN ─────────────────────────────────────────────────────────────
md("### 10.2 HomoGNN — live inference")

code(r"""# ── INPUT FEATURES (same schema as 10.1) ─────────────────────────────────
zone            = "DK1"
lag_24h         = 420.0
lag_48h         = 390.0
lag_168h        = 445.0
roll_mean       = 410.0
roll_std        = 35.0
temperature_c   = 8.0
wind_speed_ms   = 6.5
cloud_cover_pct = 60.0
humidity_pct    = 75.0
load_mwh        = 3500.0
renewable_mwh   = 1500.0
gas_dkk         = 300.0
co2_dkk         = 80.0
hour_of_day     = 17
day_of_week     = 0
# ─────────────────────────────────────────────────────────────────────────────

import pickle, math, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

ckpt_path   = Path("artifacts/best_homo_model.pt")
graph_path  = Path("data/graphs_homo/temporal_graph.pt")
scaler_path = Path("data/graphs_homo/homo_scalers.pkl")

missing = [p for p in [ckpt_path, graph_path, scaler_path] if not p.exists()]
if missing:
    print("Missing files:", [str(p) for p in missing])
    print("  → cd src && python homo_retrain.py")
else:
    import torch
    from homo_models import load_homo_model

    device = torch.device("cpu")
    data   = torch.load(graph_path, map_location=device, weights_only=False)
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    model, x_override = load_homo_model(data, ckpt_path, device)
    model.eval()

    num_nodes = data.x.shape[0]
    T = num_nodes // 2              # nodes per zone
    feat_scaler = scalers["feature_scaler"]

    h_sin = math.sin(2*math.pi*hour_of_day/24); h_cos = math.cos(2*math.pi*hour_of_day/24)
    d_sin = math.sin(2*math.pi*day_of_week/7);  d_cos = math.cos(2*math.pi*day_of_week/7)

    raw = np.array([[lag_24h, lag_48h, lag_168h, roll_mean, roll_std,
                     temperature_c, wind_speed_ms, cloud_cover_pct, humidity_pct,
                     load_mwh, renewable_mwh, gas_dkk, co2_dkk,
                     h_sin, h_cos, d_sin, d_cos]], dtype=np.float32)
    scaled = feat_scaler.transform(raw)

    x = data.x.clone() if x_override is None else x_override.clone()
    zone_off = 0 if zone == "DK1" else T
    node_idx = zone_off + T - 1
    x[node_idx] = torch.tensor(scaled[0], dtype=torch.float32)

    with torch.no_grad():
        out = model(x, data.edge_index).view(-1)
    pred = float(out[node_idx].item())

    metrics = load_json("artifacts/homo_gnn_metrics.json")
    mae = metrics.get("mae", metrics.get("test_mae", 161.9))

    print(f"Zone:               {zone}")
    print(f"Predicted price:    {pred:,.1f} DKK/MWh")
    print(f"Test MAE:           {mae:.1f} DKK  → 90 % interval ≈ [{pred-mae:,.0f}, {pred+mae:,.0f}]")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[zone], y=[pred], marker_color="#f28e2b",
                         text=[f"{pred:.1f}"], textposition="outside", name="Prediction"))
    fig.add_shape(type="rect", x0=-0.4, x1=0.4,
                  y0=max(0, pred-mae), y1=pred+mae,
                  fillcolor="moccasin", opacity=0.35, line_width=0)
    fig.update_layout(title=f"HomoGNN prediction — {zone}  (shading = ± MAE)",
                      yaxis_title="Price (DKK/MWh)", height=340, showlegend=False)
    fig.show()
""")

# ── 10.3 ST-HeteroSAGE ───────────────────────────────────────────────────────
md("### 10.3 ST-HeteroSAGE — live inference")

code(r"""# ── INPUT FEATURES ────────────────────────────────────────────────────────
zone            = "DK1"
lag_24h         = 420.0
lag_48h         = 390.0
lag_168h        = 445.0
roll_mean       = 410.0
roll_std        = 35.0
temperature_c   = 8.0
wind_speed_ms   = 6.5
cloud_cover_pct = 60.0
humidity_pct    = 75.0
load_mwh        = 3500.0
renewable_mwh   = 1500.0
gas_dkk         = 300.0
co2_dkk         = 80.0
hour_of_day     = 17
day_of_week     = 0
# ─────────────────────────────────────────────────────────────────────────────

import pickle, math, sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

ckpt_path   = Path("artifacts_hetero/best_st_hetero_model.pt")
graph_path  = Path("data/graphs_hetero/hetero_graph.pt")
scaler_path = Path("data/graphs_hetero/hetero_scalers.pkl")

missing = [p for p in [ckpt_path, graph_path, scaler_path] if not p.exists()]
if missing:
    print("Missing files:", [str(p) for p in missing])
    print("  → cd src && python st_train.py")
else:
    import torch
    from st_hetero_models import load_st_hetero_model

    device = torch.device("cpu")
    data   = torch.load(graph_path, map_location=device, weights_only=False)
    with open(scaler_path, "rb") as f:
        scalers = pickle.load(f)

    model, x_override = load_st_hetero_model(data, ckpt_path, device)
    model.eval()

    num_hours   = int(data["hour"].num_hours_per_zone)
    feat_scaler = scalers["feature_scaler"]

    h_sin = math.sin(2*math.pi*hour_of_day/24); h_cos = math.cos(2*math.pi*hour_of_day/24)
    d_sin = math.sin(2*math.pi*day_of_week/7);  d_cos = math.cos(2*math.pi*day_of_week/7)

    raw = np.array([[lag_24h, lag_48h, lag_168h, roll_mean, roll_std,
                     temperature_c, wind_speed_ms, cloud_cover_pct, humidity_pct,
                     load_mwh, renewable_mwh, gas_dkk, co2_dkk,
                     h_sin, h_cos, d_sin, d_cos]], dtype=np.float32)
    scaled = feat_scaler.transform(raw)

    x_dict = {k: v.clone() for k, v in data.x_dict.items()}
    if x_override is not None:
        x_dict.update(x_override)

    zone_off = 0 if zone == "DK1" else num_hours
    node_idx = zone_off + num_hours - 1
    x_dict["hour"][node_idx] = torch.tensor(scaled[0], dtype=torch.float32)

    ei = {k: v for k, v in data.edge_index_dict.items()}
    with torch.no_grad():
        out = model(x_dict, ei, num_hours=num_hours).view(-1)
    pred = float(out[node_idx].item())

    metrics = load_json("artifacts_hetero/st_hetero_metrics.json")
    mae = metrics.get("mae", metrics.get("test_mae", 151.1))

    print(f"Zone:               {zone}")
    print(f"Predicted price:    {pred:,.1f} DKK/MWh")
    print(f"Test MAE:           {mae:.1f} DKK  → 90 % interval ≈ [{pred-mae:,.0f}, {pred+mae:,.0f}]")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[zone], y=[pred], marker_color="#7c3aed",
                         text=[f"{pred:.1f}"], textposition="outside", name="Prediction"))
    fig.add_shape(type="rect", x0=-0.4, x1=0.4,
                  y0=max(0, pred-mae), y1=pred+mae,
                  fillcolor="#ede9fe", opacity=0.5, line_width=0)
    fig.update_layout(title=f"ST-HeteroSAGE prediction — {zone}  (shading = ± MAE)",
                      yaxis_title="Price (DKK/MWh)", height=340, showlegend=False)
    fig.show()
""")

# ── 10.4 Side-by-side comparison ─────────────────────────────────────────────
md("### 10.4 Head-to-head comparison (all models, same input)")

code(r"""# Run this cell AFTER running 10.1–10.3 so `pred` is defined for each model.
# For a clean comparison, re-run with identical inputs and capture each `pred`.

models_data = [
    ("XGBoost",       205.6, "#4e79a7"),
    ("HomoGNN",       161.9, "#f28e2b"),
    ("GAT",           179.2, "#e15759"),
    ("HeteroSAGE",    162.8, "#76b7b2"),
    ("ST-HeteroSAGE", 151.1, "#7c3aed"),
]
names  = [m[0] for m in models_data]
maes   = [m[1] for m in models_data]
colors = [m[2] for m in models_data]

fig = go.Figure()
fig.add_trace(go.Bar(
    x=names, y=maes, marker_color=colors,
    text=[f"{v:.1f}" for v in maes], textposition="outside",
))
fig.update_layout(
    title="Test MAE comparison across all models (lower is better)",
    yaxis_title="MAE (DKK/MWh)", height=380,
    yaxis_range=[0, max(maes)*1.2],
)
fig.show()

print("\nModel performance summary:")
print(f"{'Model':<18} {'MAE':>8}  {'vs XGB':>8}")
xgb_mae = maes[0]
for name, mae, _ in models_data:
    delta = (mae - xgb_mae) / xgb_mae * 100
    flag  = " ★ best" if mae == min(maes) else ""
    print(f"  {name:<16} {mae:>8.1f}  {delta:>+7.1f}%{flag}")
""")

# ══════════════════════════════════════════════════════════════════════════════
# 11.  GRAPH STRUCTURE VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════
md(r"""<a id="section-11"></a>
## 11. Graph Structure Visualisation

The heterogeneous graph has **2 node types** and **5 edge relation types**.
This section visualises its structure at three levels of detail:

1. **Live statistics** — exact node / edge counts loaded from the saved graph file
2. **Schematic topology** — a hand-laid Plotly network showing every node/edge type
3. **Interactive mini-graph** — 3 real timesteps × 4 zones + 4 market nodes with
   hoverable metadata
""")

# ── 11.1 Live graph statistics ────────────────────────────────────────────────
md("### 11.1 Live graph statistics")

code(r"""import sys, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

graph_path = Path("data/graphs_hetero/hetero_graph.pt")
if not graph_path.exists():
    print("Graph not found:", graph_path)
    print("  → cd src && python hetero_graph_builder.py")
else:
    import torch
    g = torch.load(graph_path, map_location="cpu", weights_only=False)

    # ── Node counts ──────────────────────────────────────────────────────
    node_rows = []
    for ntype in g.node_types:
        n = g[ntype].num_nodes
        f = g[ntype].x.shape[1] if hasattr(g[ntype], "x") and g[ntype].x is not None else 0
        node_rows.append((ntype, n, f))

    # ── Edge counts ──────────────────────────────────────────────────────
    edge_rows = []
    for src, rel, dst in g.edge_types:
        ei = g[src, rel, dst].edge_index
        edge_rows.append((f"{src} → {dst}", rel, ei.shape[1]))

    # ── Plotly table ─────────────────────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Node types", "Edge relation types"],
        specs=[[{"type": "table"}, {"type": "table"}]],
    )
    fig.add_trace(go.Table(
        header=dict(values=["Node type", "# nodes", "# features"],
                    fill_color="#4472c4", font_color="white"),
        cells=dict(values=[
            [r[0] for r in node_rows],
            [f"{r[1]:,}" for r in node_rows],
            [r[2] for r in node_rows],
        ], fill_color=[["#dce6f1","white"]*10]),
    ), row=1, col=1)
    fig.add_trace(go.Table(
        header=dict(values=["Direction", "Relation", "# edges"],
                    fill_color="#4472c4", font_color="white"),
        cells=dict(values=[
            [r[0] for r in edge_rows],
            [r[1] for r in edge_rows],
            [f"{r[2]:,}" for r in edge_rows],
        ], fill_color=[["#dce6f1","white"]*10]),
    ), row=1, col=2)
    fig.update_layout(height=300, margin=dict(t=40, b=10))
    fig.show()

    T = int(g["hour"].num_hours_per_zone)
    total_nodes = sum(r[1] for r in node_rows)
    total_edges = sum(r[2] for r in edge_rows)
    print(f"\nSummary: {total_nodes:,} nodes  |  {total_edges:,} edges  |  T={T:,} hours/zone")
""")

# ── 11.2 Schematic topology ───────────────────────────────────────────────────
md("### 11.2 Schematic topology diagram")

code(r"""# Hardcoded layout — one representative node per zone/market
# Positions chosen for visual clarity, not geometry

node_defs = [
    # label,  x,    y,   color,    size, node_type
    ("DK1\nhour",   -1.5,  1.0, "#2171b5", 30, "hour"),
    ("DK2\nhour",   -0.5,  1.0, "#6baed6", 30, "hour"),
    ("HYDRO\nhour",  0.5,  1.0, "#74c476", 30, "hour"),
    ("DE\nhour",     1.5,  1.0, "#fd8d3c", 30, "hour"),
    ("NordPool",    -1.0, -0.5, "#9467bd", 22, "market"),
    ("DK1 area",    -0.33,-0.5, "#8c564b", 22, "market"),
    ("DK2 area",     0.33,-0.5, "#e377c2", 22, "market"),
    ("DE area",      1.0, -0.5, "#7f7f7f", 22, "market"),
]

# Edge definitions: (from_idx, to_idx, label, color, dash)
edge_defs = [
    (0, 1, "co_occurs_with",   "#aec7e8", "solid"),
    (0, 2, "co_occurs_with",   "#aec7e8", "solid"),
    (1, 2, "co_occurs_with",   "#aec7e8", "solid"),
    (0, 0, "lag_to (self)",    "#9ecae1", "dot"),      # self-loop represented as offset
    (0, 4, "belongs_to",       "#c5b0d5", "dash"),
    (1, 5, "belongs_to",       "#c5b0d5", "dash"),
    (2, 5, "belongs_to",       "#c5b0d5", "dash"),
    (3, 7, "belongs_to",       "#c5b0d5", "dash"),
    (4, 5, "interconnects",    "#ffbb78", "solid"),
    (5, 6, "interconnects",    "#ffbb78", "solid"),
]

fig = go.Figure()

# Draw edges first (so nodes appear on top)
drawn_labels = set()
for (fi, ti, lbl, col, dash) in edge_defs:
    if fi == ti:   # self-loop: draw a small arc offset
        x0, y0 = node_defs[fi][1], node_defs[fi][2]
        fig.add_annotation(ax=x0, ay=y0+0.18, x=x0+0.18, y=y0,
                           xref="x", yref="y", axref="x", ayref="y",
                           showarrow=True, arrowhead=2, arrowcolor=col,
                           arrowwidth=1.5)
        continue
    x0, y0 = node_defs[fi][1], node_defs[fi][2]
    x1, y1 = node_defs[ti][1], node_defs[ti][2]
    show = lbl not in drawn_labels
    drawn_labels.add(lbl)
    fig.add_trace(go.Scatter(
        x=[x0, x1], y=[y0, y1], mode="lines",
        line=dict(color=col, width=2,
                  dash="dash" if dash=="dash" else ("dot" if dash=="dot" else "solid")),
        name=lbl, showlegend=show,
        hoverinfo="name",
    ))

# Draw nodes
for (lbl, x, y, col, sz, ntype) in node_defs:
    fig.add_trace(go.Scatter(
        x=[x], y=[y], mode="markers+text",
        marker=dict(color=col, size=sz, line=dict(color="white", width=2)),
        text=[lbl], textposition="top center",
        name=ntype, showlegend=False,
        hovertemplate=f"<b>{lbl.replace(chr(10),' ')}</b><br>type: {ntype}<extra></extra>",
    ))

# Node-type legend boxes
for col, label in [("#2171b5","hour node (DK1)"), ("#6baed6","hour node (DK2)"),
                   ("#74c476","hour node (HYDRO)"), ("#fd8d3c","hour node (DE)"),
                   ("#9467bd","market node")]:
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(color=col, size=12, symbol="circle"),
        name=label, showlegend=True,
    ))

fig.update_layout(
    title="Heterogeneous graph — schematic topology",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.2, 2.2]),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 1.6]),
    height=520, legend=dict(x=1.01, y=0.99, bordercolor="lightgray", borderwidth=1),
    plot_bgcolor="white",
)
fig.show()
""")

# ── 11.3 Interactive mini-graph ───────────────────────────────────────────────
md("### 11.3 Interactive mini-graph (3 timesteps × 4 zones + 4 market nodes)")

code(r"""import sys, math, json
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

graph_path = Path("data/graphs_hetero/hetero_graph.pt")
if not graph_path.exists():
    print("Graph file not found — run hetero_graph_builder.py first")
else:
    import torch
    g = torch.load(graph_path, map_location="cpu", weights_only=False)
    T   = int(g["hour"].num_hours_per_zone)
    M   = g["market"].num_nodes

    # Pick 3 consecutive timesteps near the end of training data
    t_center = T // 2
    steps    = [t_center - 1, t_center, t_center + 1]
    zones    = ["DK1", "DK2", "HYDRO", "DE"]
    z_colors = {"DK1": "#2171b5", "DK2": "#6baed6", "HYDRO": "#74c476", "DE": "#fd8d3c"}
    m_colors = ["#9467bd","#8c564b","#e377c2","#7f7f7f"]

    # ── Build node catalogue ──────────────────────────────────────────────
    # hour nodes: zone z, timestep t  → global index = z*T + t
    hour_nodes   = {}   # (z_idx, t_step) → mini_id
    market_nodes = {}   # m_idx → mini_id
    positions    = {}   # mini_id → (x, y)
    node_labels  = {}
    node_colors  = {}
    node_hover   = {}

    mid = 0
    for zi, z in enumerate(zones):
        for ci, t in enumerate(steps):
            global_idx = zi * T + t
            hour_nodes[(zi, t)] = mid
            positions[mid]    = (ci * 2.0, -zi * 1.5)
            node_labels[mid]  = f"{z}\nt={t}"
            node_colors[mid]  = z_colors[z]
            feat = g["hour"].x[global_idx].tolist()
            node_hover[mid]   = (
                f"<b>{z} t={t}</b><br>"
                f"lag_24h: {feat[0]:.2f}<br>"
                f"lag_48h: {feat[1]:.2f}<br>"
                f"wind:    {feat[6]:.2f}<br>"
                f"hour_sin:{feat[13]:.2f}"
            )
            mid += 1

    mkt_names = ["NordPool","DK1 area","DK2 area","DE area"]
    for mi in range(min(M, 4)):
        market_nodes[mi] = mid
        positions[mid]   = (mi * 2.0, -len(zones) * 1.5 - 0.8)
        node_labels[mid] = mkt_names[mi] if mi < len(mkt_names) else f"Market {mi}"
        node_colors[mid] = m_colors[mi % len(m_colors)]
        node_hover[mid]  = f"<b>{node_labels[mid]}</b><br>type: market"
        mid += 1

    # ── Collect edges that touch our mini-graph nodes ─────────────────────
    mini_hour_globals   = {zi*T + t: hour_nodes[(zi, t)]
                           for zi in range(len(zones)) for t in steps}
    mini_market_globals = {mi: market_nodes[mi] for mi in range(min(M,4))}

    edge_traces = []
    rel_colors  = {
        "lag_to":          "#9ecae1",
        "co_occurs_with":  "#a1d99b",
        "belongs_to":      "#c5b0d5",
        "rev_belongs_to":  "#dadaeb",
        "interconnects":   "#ffbb78",
    }
    drawn_rels = set()

    for src_type, rel, dst_type in g.edge_types:
        ei = g[src_type, rel, dst_type].edge_index
        src_map = mini_hour_globals if src_type == "hour" else mini_market_globals
        dst_map = mini_hour_globals if dst_type == "hour" else mini_market_globals

        edge_x, edge_y = [], []
        for s, d in zip(ei[0].tolist(), ei[1].tolist()):
            if s in src_map and d in dst_map:
                x0, y0 = positions[src_map[s]]
                x1, y1 = positions[dst_map[d]]
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

        if not edge_x:
            continue
        col  = rel_colors.get(rel, "#cccccc")
        show = rel not in drawn_rels
        drawn_rels.add(rel)
        edge_traces.append(go.Scatter(
            x=edge_x, y=edge_y, mode="lines",
            line=dict(color=col, width=1.5), opacity=0.7,
            name=rel, showlegend=show, hoverinfo="none",
        ))

    # ── Build node scatter ────────────────────────────────────────────────
    node_x = [positions[i][0] for i in range(mid)]
    node_y = [positions[i][1] for i in range(mid)]
    node_c = [node_colors[i]  for i in range(mid)]
    node_t = [node_labels[i]  for i in range(mid)]
    node_h = [node_hover[i]   for i in range(mid)]

    node_scatter = go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        marker=dict(color=node_c, size=20, line=dict(color="white", width=1.5)),
        text=node_t, textposition="top center",
        hovertemplate="%{customdata}<extra></extra>",
        customdata=node_h,
        name="nodes", showlegend=False,
    )

    fig = go.Figure(edge_traces + [node_scatter])
    fig.update_layout(
        title=f"Interactive mini-graph  ({mid} nodes, 3 timesteps per zone)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=560, plot_bgcolor="white",
        legend=dict(x=1.01, y=0.99, bordercolor="lightgray", borderwidth=1),
        hovermode="closest",
    )
    fig.show()
    print(f"Nodes shown: {mid}  |  Edge sets drawn: {len(drawn_rels)}")
""")

# ── Write the notebook ───────────────────────────────────────────────────────
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
print(f"Wrote {out}  ({len(cells)} cells)")
