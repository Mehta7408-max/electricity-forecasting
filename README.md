# Heterogeneous Spatio-Temporal GNNs for Nordic Electricity Price Forecasting

Multi-area day-ahead electricity price forecasting for the Nordic power market,
built around a **heterogeneous spatio-temporal graph neural network**
(ST-HeteroSAGE) and benchmarked against XGBoost and three GNN baselines.

> **Research question**
> *How can a heterogeneous graph be effectively constructed and integrated into
> GNN models to improve the accuracy, interpretability, and robustness of
> multi-area day-ahead electricity price forecasting in the Nordic power market?*

---

## 🏆 Results

Test set: chronological 80/10/10 split (test window ≈ Mar–Sep 2025), DK1 + DK2.

| Model | MAE (DKK) | RMSE (DKK) | R² | Notes |
|-------|-----------|-----------|-----|-------|
| XGBoost (tabular baseline) | 205.62 | 296.41 | 0.520 | 13 engineered features |
| Homogeneous GNN (GraphSAGE) | 161.91 | 212.53 | 0.671 | single node type |
| GAT | 179.20 | — | 0.591 | attention, heterogeneous |
| HeteroSAGE | 162.78 | — | 0.668 | typed edges, served by the API |
| **ST-HeteroSAGE (ours ★)** | **151.08** | **204.36** | **0.696** | CausalTCN + HeteroConv |

ST-HeteroSAGE is **26.5 % better than XGBoost** and **6.7 % better than the
homogeneous GraphSAGE baseline**.

---

## 📊 The heterogeneous graph

**2 node types, 5 edge relation types.** Four market zones (DK1, DK2, HYDRO, DE)
share the `hour` node type; zone identity is carried by the block layout and the
market edges rather than by separate node types.

### Nodes
| Type | Count | Meaning |
|------|-------|---------|
| `hour` | 201,596 | one node per hour per zone (50,399 h × 4 zones) |
| `market` | 4 | one node per area (NordPool, DK1 area, DK2 area, DE area) |

**17 features per `hour` node:** 5 price lag/rolling (`lag_24h`, `lag_48h`,
`lag_168h`, `roll24_mean`, `roll24_std`), 4 weather (`temp`, `wind`, `cloud`,
`humidity`), 4 fundamentals (`load_mwh`, `renewable_mwh`, `gas_dkk`, `co2_dkk`),
4 cyclical (`hour_sin/cos`, `week_sin/cos`).

### Edges
| Relation | Meaning |
|----------|---------|
| `lag_to` | temporal — links an hour to a later hour (24/48/168 h) |
| `co_occurs_with` | spatial — same-hour price co-movement across zones |
| `belongs_to` / `rev_belongs_to` | hour ↔ its market node |
| `interconnects` | market ↔ market (transmission capacity) |

**Zone-blocked layout** — DK1 = `[0, T)`, DK2 = `[T, 2T)`, HYDRO = `[2T, 3T)`,
DE = `[3T, 4T)` with `T = 50,399`. This enables the O(1) `view(4, T, H)` reshape
the CausalTCN uses for per-zone temporal convolution.

> **DE staleness fix:** German prices end 2024-12-31, so DE is deliberately
> excluded from hour-level `co_occurs_with` edges to stop stale forward-filled
> 2025 values from leaking into test predictions.

---

## 🧠 ST-HeteroSAGE architecture

Each spatio-temporal block applies **spatial** message passing then a
**temporal** convolution:

```
x_dict ─► HeteroConv (co_occurs_with, belongs_to, interconnects)   ← spatial
       ─► per-zone CausalTCN (dilations 1,4,24; kernel 7; RF ≈ 174h) ← temporal
       ─► BatchNorm + residual
   (×2 ST blocks) ─► regression head ─► price
```

- **CausalTCN** replaces explicit lag edges with a strictly causal 174-hour
  receptive field — a much richer temporal signal.
- **BatchNorm, no Dropout** — the two interact adversely (dropout masking
  corrupts BN statistics); BN alone regularises adequately here.

---

## 📦 Project layout

```
electricity-forecasting/
├── src/
│   ├── hetero_graph_builder.py   # build hetero_graph.pt + scalers from SQLite
│   ├── hetero_st_model.py        # ST-HeteroSAGE (CausalTCN + HeteroConv)
│   ├── hetero_models.py          # HeteroSAGE / GAT
│   ├── st_train.py               # train ST-HeteroSAGE
│   ├── homo_retrain.py           # homogeneous GraphSAGE baseline
│   ├── xgboost_baseline.py       # tabular baseline
│   ├── st_ablation.py            # ablation study
│   ├── st_interpretability.py    # feature importance + error analysis
│   ├── st_robustness.py          # perturbation robustness
│   ├── model_api.py              # FastAPI serving (HeteroSAGE)
│   ├── dashboard.py              # Streamlit dashboard (7 pages)
│   ├── pipeline.py               # end-to-end MLOps pipeline
│   ├── monitoring.py             # rolling MAE + drift
│   ├── data/graphs_hetero/       # hetero_graph.pt, hetero_scalers.pkl
│   └── artifacts_hetero/         # checkpoints + metrics JSON
├── notebooks/
│   └── ST_HeteroSAGE_Review.ipynb # supervisor technical review (Plotly, live inference)
├── artifacts/                    # xgboost_baseline.pkl + metrics
├── Dockerfile                    # CPU-only torch image (Python 3.11)
├── docker-compose.yml            # mlflow + api + dashboard + training profiles
└── requirements.txt
```

---

## 🚀 Quick start

### Option A — Docker (recommended)

```bash
docker compose up --build
```

Brings up three services:

| Service | URL | Purpose |
|---------|-----|---------|
| MLflow | http://localhost:5000 | experiment tracking |
| API | http://localhost:8000 | `/predict`, `/metrics`, `/health` (HeteroSAGE) |
| Dashboard | http://localhost:8501 | Streamlit UI |

### Option B — local

```bash
pip install torch==2.2.2+cpu torch-geometric>=2.3.0 \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
pip install -r requirements.txt

# API (terminal 1)
python src/model_api.py
# Dashboard (terminal 2) — works even without the API (local XGBoost fallback)
streamlit run src/dashboard.py --server.port=8501
```

---

## 🖥️ Dashboard pages

1. **Overview & Leaderboard** — 5-model MAE / R² comparison
2. **Graph Structure** — live node/edge stats, spatio-temporal schematic, and an
   interactive mini-graph (hover any node for its real price / weather)
3. **Live Prediction** — HeteroSAGE via the API, or a local XGBoost fallback when
   the API is offline
4. **Forecast Analysis** — per-zone day-ahead profiles
5. **Interpretability** — feature importance, error by hour/day
6. **Robustness** — MAE under noise / dropout / price-spike perturbations
7. **MLOps & Monitoring** — pipeline status, rolling MAE, drift

---

## 🔁 Reproducing the results

```bash
cd src

# 1 — build the graph from the SQLite DB
python hetero_graph_builder.py        # → hetero_graph.pt + hetero_scalers.pkl

# 2 — train all models
python xgboost_baseline.py            # → artifacts/xgboost_metrics.json
python homo_retrain.py                # → artifacts/homo_gnn_metrics.json
python gat_train.py                   # → artifacts_hetero/gat_metrics_clean.json
python hetero_train.py                # → artifacts_hetero/hetero_metrics_clean.json
python st_train.py                    # → artifacts_hetero/st_hetero_metrics.json

# 3 — post-hoc analyses on the ST checkpoint
python st_ablation.py                 # → st_ablation_results.json
python st_interpretability.py         # → st_interpretability.json
python st_robustness.py               # → st_robustness_results.json
```

**Seeding:** `torch.manual_seed(42)` and `np.random.seed(42)` in all GNN scripts;
XGBoost uses `random_state=42`.

**Environment:** Python 3.11, PyTorch 2.2 (CPU), PyTorch-Geometric ≥ 2.3,
XGBoost ≥ 2.0, scikit-learn ≥ 1.8.

---

## 🔄 Data sources

| Source | Data | Cost |
|--------|------|------|
| Energinet / Nord Pool | DK1, DK2 spot prices, load, renewables | free |
| Energy-Charts (Fraunhofer ISE) | DE spot prices | free |
| Open-Meteo | temperature, wind, cloud, humidity | free |

Coverage: hourly, **2019-12-31 → 2025-09-30** (50,399 hours per zone). All price
lags are ≥ 24 h (known at gate closure); scalers are fit on the training split
only; splits are strictly chronological — no leakage.

---

## ⚙️ MLOps

- **MLflow** experiment tracking (file-backed store)
- **Git-versioned artifacts** — checkpoints, metrics, scalers committed
- **Warm-start / freeze-scaler** for incremental retraining on new data
- **Docker + docker-compose** for reproducible serving and training
- **Monitoring** — rolling MAE and feature-drift checks via the API

---

## 📚 References

1. Kipf & Welling (2017). *Semi-Supervised Classification with Graph Convolutional Networks.*
2. Veličković et al. (2018). *Graph Attention Networks.*
3. Hamilton et al. (2017). *Inductive Representation Learning on Large Graphs* (GraphSAGE).
4. Bai et al. (2018). *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling* (TCN).
