"""
Streamlit MLOps dashboard for the electricity price forecasting project.

A single-file, multi-page app (st.sidebar.radio navigation) that works both:
  - online  — talking to the FastAPI server (live predictions, pipeline triggers)
  - offline — reading evaluation artifacts directly from disk

Run with:
    streamlit run src/dashboard.py --server.port=8501
"""
import os
import sys
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

try:
    import requests
except Exception:  # pragma: no cover - requests is a hard dependency, but be safe
    requests = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_SRC = _THIS.parent
PROJECT_ROOT = _SRC.parent if _SRC.name == "src" else _SRC

sys.path.insert(0, str(_SRC))

_ARTIFACTS = PROJECT_ROOT / "artifacts"
_SRC_ARTIFACTS = _SRC / "artifacts"
_HETERO = _SRC / "artifacts_hetero"

XGB_METRICS_PATH = _ARTIFACTS / "xgboost_metrics.json"
XGB_MODEL_PATH = _ARTIFACTS / "xgboost_baseline.pkl"
SCALER_PATH = _SRC / "data" / "graphs_hetero" / "hetero_scalers.pkl"
DATASET_START = "2019-12-31 23:00:00"  # first timestamp; node t = start + t hours
HOMO_METRICS_PATH = _SRC_ARTIFACTS / "homo_gnn_metrics.json"
GAT_METRICS_PATH = _HETERO / "gat_metrics_clean.json"
HETERO_METRICS_PATH = _HETERO / "hetero_metrics_clean.json"
ST_HETERO_METRICS_PATH = _HETERO / "st_hetero_metrics.json"
DAY_AHEAD_PATH = _HETERO / "day_ahead_results.json"
ABLATION_PATH = _HETERO / "st_ablation_results.json"
ROBUSTNESS_PATH = _HETERO / "st_robustness_results.json"
INTERP_PATH = _HETERO / "st_interpretability.json"
LAST_RUN_PATH = _HETERO / "last_pipeline_run.json"
GRAPH_PATH = _SRC / "data" / "graphs_hetero" / "hetero_graph.pt"

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

HETERO_COLOR = "#2563eb"
ST_COLOR = "#7c3aed"
NEUTRAL_COLORS = ["#9ca3af", "#6b7280", "#4b5563", HETERO_COLOR, ST_COLOR]
ZONE_COLORS = {"DK1": "#2171b5", "DK2": "#6baed6", "HYDRO": "#74c476", "DE": "#fd8d3c"}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_json_safe(path):
    """Load a JSON file, returning a dict (or list) or None on any error."""
    try:
        p = Path(path)
        if not p.exists():
            return None
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def api_get(endpoint):
    """GET against the FastAPI server. Returns parsed JSON or None on failure."""
    if requests is None:
        return None
    url = f"{API_BASE.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        resp = requests.get(url, timeout=3)
        if resp.status_code == 200:
            return resp.json()
        return None
    except Exception:
        return None


def api_post(endpoint, json_payload=None, params=None):
    """POST against the FastAPI server. Returns parsed JSON or None on failure."""
    if requests is None:
        return None
    url = f"{API_BASE.rstrip('/')}/{endpoint.lstrip('/')}"
    try:
        resp = requests.post(url, json=json_payload or {}, params=params, timeout=3)
        if resp.status_code == 200:
            return resp.json()
        # Surface useful error detail back to the caller
        try:
            return {"_error": resp.status_code, "detail": resp.json().get("detail")}
        except Exception:
            return {"_error": resp.status_code, "detail": resp.text}
    except Exception:
        return None


def api_is_up():
    """Return True if GET /health succeeds."""
    return api_get("/health") is not None


def api_banner():
    """Render a small connectivity banner in the sidebar."""
    if api_is_up():
        st.sidebar.success("🟢 API connected")
        return True
    st.sidebar.info("🔵 Offline mode (reading artifacts directly)")
    return False


def _fmt(x, nd=2):
    try:
        return round(float(x), nd)
    except Exception:
        return x


# ---------------------------------------------------------------------------
# Page 1 — Overview & Leaderboard
# ---------------------------------------------------------------------------
def page_overview():
    st.title("⚡ Electricity Price Forecasting — MLOps Dashboard")
    st.markdown(
        "**Research question:** *How can a heterogeneous graph be effectively constructed "
        "and integrated into GNN models to improve accuracy, interpretability, and "
        "robustness of multi-area day-ahead electricity price forecasting in the Nordic "
        "power market?*"
    )

    xgb = load_json_safe(XGB_METRICS_PATH)
    homo = load_json_safe(HOMO_METRICS_PATH)
    gat = load_json_safe(GAT_METRICS_PATH)
    hetero = load_json_safe(HETERO_METRICS_PATH)
    st_m = load_json_safe(ST_HETERO_METRICS_PATH)

    if not any([xgb, homo, gat, hetero, st_m]):
        st.warning("Run the training scripts (make train-all) to generate model metrics.")
        return

    models = [
        ("XGBoost (tabular baseline)", xgb),
        ("Homogeneous GNN (GraphSAGE)", homo),
        ("GAT", gat),
        ("HeteroSAGE", hetero),
        ("ST-HeteroSAGE (ours ★)", st_m),
    ]

    rows = []
    for name, m in models:
        if not m:
            rows.append({"Model": name, "MAE (DKK)": None, "RMSE": None, "R²": None, "SMAPE (%)": None})
            continue
        rows.append({
            "Model": name,
            "MAE (DKK)": _fmt(m.get("mae")),
            "RMSE": _fmt(m.get("rmse")),
            "R²": _fmt(m.get("r2"), 4),
            "SMAPE (%)": _fmt(m.get("smape")),
        })
    df = pd.DataFrame(rows)

    st.subheader("🏆 Model Leaderboard (DK1+DK2 test set, chronological 80/10/10 split)")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # ---- metric row -------------------------------------------------------
    best = st_m or hetero
    if best:
        c1, c2, c3 = st.columns(3)
        label = "ST-HeteroSAGE" if st_m else "HeteroSAGE"
        c1.metric(f"{label} MAE", f"{_fmt(best.get('mae'))} DKK")
        c2.metric(f"{label} R²", f"{_fmt(best.get('r2'), 4)}")
        if xgb and xgb.get("mae"):
            impr = (xgb["mae"] - best.get("mae", 0)) / xgb["mae"] * 100.0
            c3.metric("MAE improvement vs XGBoost", f"{impr:+.1f}%")
        else:
            c3.metric("MAE improvement vs XGBoost", "n/a")

    # ---- charts -----------------------------------------------------------
    chart_df = df.dropna(subset=["MAE (DKK)"])
    if not chart_df.empty:
        def _bar_color(name):
            if "ST-HeteroSAGE" in name:
                return ST_COLOR
            if "HeteroSAGE" in name:
                return HETERO_COLOR
            return "#9ca3af"

        colors = [_bar_color(n) for n in chart_df["Model"]]

        col_a, col_b = st.columns(2)
        with col_a:
            fig_mae = go.Figure(
                go.Bar(
                    x=chart_df["Model"],
                    y=chart_df["MAE (DKK)"],
                    marker_color=colors,
                    text=chart_df["MAE (DKK)"],
                    textposition="outside",
                )
            )
            fig_mae.update_layout(
                title="MAE across models (lower is better)",
                yaxis_title="MAE (DKK)",
                xaxis_tickangle=-20,
                height=440,
            )
            st.plotly_chart(fig_mae, use_container_width=True)

        with col_b:
            r2_df = chart_df.dropna(subset=["R²"])
            fig_r2 = go.Figure(
                go.Bar(
                    x=r2_df["Model"],
                    y=r2_df["R²"],
                    marker_color=[_bar_color(n) for n in r2_df["Model"]],
                    text=r2_df["R²"],
                    textposition="outside",
                )
            )
            fig_r2.update_layout(
                title="R² across models (higher is better)",
                yaxis_title="R²",
                xaxis_tickangle=-20,
                height=440,
            )
            st.plotly_chart(fig_r2, use_container_width=True)

    st.subheader("Verdict")
    st.markdown(
        "**ST-HeteroSAGE** (spatio-temporal heterogeneous GNN with CausalTCN) achieves the "
        "best results across all metrics: **MAE 151 DKK, R² 0.696** — a **26.5% MAE reduction** "
        "versus XGBoost and a 6.7% improvement over the homogeneous GraphSAGE baseline. "
        "The ablation study shows that CausalTCN accounts for the largest single gain, while "
        "the heterogeneous market bridge (market nodes + interconnect edges) provides a further "
        "meaningful improvement over a purely spatial or purely temporal approach."
    )


# ---------------------------------------------------------------------------
# Page 2 — Live Prediction
# Spatial edge types used for local ST-HeteroSAGE inference (lag_to handled by CausalTCN)
_ST_SPATIAL_EDGE_TYPES = [
    ('hour',   'co_occurs_with', 'hour'),
    ('hour',   'belongs_to',     'market'),
    ('market', 'rev_belongs_to', 'hour'),
    ('market', 'interconnects',  'market'),
]

_ST_GRAPH_PATH  = _SRC / "data" / "graphs_hetero" / "hetero_graph.pt"
_ST_CKPT_PATH   = _HETERO / "best_st_hetero_model.pt"
_ST_SCALER_PATH = _SRC / "data" / "graphs_hetero" / "hetero_scalers.pkl"


@st.cache_resource(show_spinner="Loading ST-HeteroSAGE…")
def _load_st_local():
    """Load ST-HeteroSAGE for offline inference. Returns (tuple) or raises on error."""
    import torch
    import pickle as _pkl
    from sklearn.preprocessing import StandardScaler
    from hetero_st_model import HeteroSTPriceForecaster

    missing = [str(p) for p in [_ST_GRAPH_PATH, _ST_CKPT_PATH, _ST_SCALER_PATH] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing model files: {missing}")

    device = torch.device("cpu")
    data = torch.load(_ST_GRAPH_PATH, map_location=device, weights_only=False)
    T = int(data['hour'].num_hours_per_zone)

    edge_index_dict = {et: data[et].edge_index.to(device) for et in _ST_SPATIAL_EDGE_TYPES}

    with open(_ST_SCALER_PATH, "rb") as f:
        scalers = _pkl.load(f)

    # Refit target_scaler on DK1+DK2 training nodes — matches st_train.py
    tr_mask = data['hour'].train_mask.cpu().numpy()
    dk12 = np.zeros(4 * T, dtype=bool)
    dk12[:2 * T] = True
    y_raw = data['hour'].y.cpu().numpy()
    target_scaler = StandardScaler()
    target_scaler.fit(y_raw[tr_mask & dk12].reshape(-1, 1))

    in_channels = data['hour'].x.shape[1]
    model = HeteroSTPriceForecaster(
        in_channels=in_channels, hidden_channels=128, num_st_blocks=2,
        temporal_dilations=(1, 4, 24), temporal_kernel=7,
    )
    model.load_state_dict(torch.load(_ST_CKPT_PATH, map_location=device, weights_only=False))
    model.eval()

    return model, data, scalers, target_scaler, edge_index_dict, T, device


def _predict_local_st(zone: str, hour_of_day: int):
    """
    Run ST-HeteroSAGE on the most-recent test node for (zone, hour_of_day).
    Returns (price_dkk, feature_dict, timestamp_str, error_str).
    Uses actual graph features — no manual feature override or broken scaler.
    """
    try:
        import torch
        import pandas as pd

        loaded = _load_st_local()
        model, data, scalers, target_scaler, edge_index_dict, T, device = loaded

        zone = zone.upper()
        zone_offset = 0 if zone == "DK1" else T

        # Find test nodes for this zone with matching hour_of_day
        test_mask = data['hour'].test_mask.numpy()
        start_ts = pd.Timestamp("2019-12-31 23:00:00")
        t_indices = np.arange(T)
        hours_arr = ((start_ts + pd.to_timedelta(t_indices, unit='h'))
                     .hour.to_numpy())
        hour_match = hours_arr == hour_of_day
        zone_test = test_mask[zone_offset: zone_offset + T]
        candidates = np.where(zone_test & hour_match)[0]
        if len(candidates) == 0:
            candidates = np.where(zone_test)[0]
        t_local = int(candidates[-1])
        node_idx = zone_offset + t_local

        # Inference — no feature override; graph features are already scaled
        x_dict = {'hour': data['hour'].x.to(device),
                  'market': data['market'].x.to(device)}
        with torch.no_grad():
            out = model(x_dict, edge_index_dict)
        predicted_dkk = float(
            target_scaler.inverse_transform([[out[node_idx].item()]])[0][0])

        # Inverse-transform first 13 features to physical units for display
        fs = scalers["feature_scaler"]
        raw_row = fs.inverse_transform(
            data['hour'].x[node_idx].numpy().reshape(1, -1))[0]
        feat_names = ['lag_24h', 'lag_48h', 'lag_168h', 'roll_mean', 'roll_std',
                      'temp_c', 'wind_ms', 'cloud_pct', 'humidity_pct',
                      'load_mwh', 'renewable_mwh', 'gas_dkk', 'co2_dkk']
        feat_dict = {n: float(raw_row[i]) for i, n in enumerate(feat_names)}
        ts_str = str(start_ts + pd.Timedelta(hours=int(t_local)))
        return predicted_dkk, feat_dict, ts_str, None

    except Exception as exc:
        return None, {}, "", str(exc)


def page_predict(api_up):
    st.title("🔮 Live Prediction")
    st.markdown(
        "Day-ahead price forecast using **ST-HeteroSAGE** (MAE ≈ 151 DKK). "
        "Select a zone and target hour — the model runs on the most recent "
        "matching node from the test set using its actual graph features."
    )

    if api_up:
        st.success("🟢 API connected — predictions served by ST-HeteroSAGE.")
    else:
        st.info(
            "🔵 API offline — running **ST-HeteroSAGE locally** inside the dashboard. "
            f"To use the API instead, start it with `make serve` (target: `{API_BASE}`)."
        )

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            zone = st.selectbox("Zone", ["DK1", "DK2"])
        with c2:
            hour_of_day = st.slider("Target hour of day", 0, 23, 18)
        submitted = st.form_submit_button("Predict price", type="primary")

    if submitted:
        payload = {"zone": zone, "hour_of_day": hour_of_day, "day_of_week": 0}
        result = api_post("/predict", payload) if api_up else None

        if result is not None and "_error" not in result:
            st.metric(
                f"Predicted price — {result.get('zone', zone)}  (ST-HeteroSAGE via API)",
                f"{result.get('predicted_price_dkk', float('nan')):.2f} DKK",
            )
        else:
            if result is not None and "_error" in result:
                st.warning(
                    f"API error {result['_error']}: {result.get('detail')} — "
                    "running ST-HeteroSAGE locally."
                )
            pred, feats, ts_str, err = _predict_local_st(zone, hour_of_day)
            if pred is None:
                st.error(f"ST-HeteroSAGE inference failed: {err}")
            else:
                st.metric(
                    f"Predicted price — {zone}  (ST-HeteroSAGE local)",
                    f"{pred:.2f} DKK",
                )
                st.caption(f"Test node: {ts_str}  |  MAE ≈ 151 DKK on DK1+DK2 test set.")
                if feats:
                    st.subheader("Features used (actual test data)")
                    fc1, fc2, fc3 = st.columns(3)
                    fc1.metric("lag_24h", f"{feats['lag_24h']:.1f} DKK")
                    fc1.metric("lag_48h", f"{feats['lag_48h']:.1f} DKK")
                    fc1.metric("lag_168h", f"{feats['lag_168h']:.1f} DKK")
                    fc2.metric("Temperature", f"{feats['temp_c']:.1f} °C")
                    fc2.metric("Wind", f"{feats['wind_ms']:.1f} m/s")
                    fc2.metric("Cloud cover", f"{feats['cloud_pct']:.0f} %")
                    fc3.metric("Gas price", f"{feats['gas_dkk']:.1f} DKK/MWh")
                    fc3.metric("CO₂ price", f"{feats['co2_dkk']:.1f} DKK/t")
                    fc3.metric("Renewable", f"{feats['renewable_mwh']:.0f} MWh")


# ---------------------------------------------------------------------------
# Page 3 — Forecast Analysis
# ---------------------------------------------------------------------------
def page_forecast():
    st.title("📈 Forecast Analysis")
    da = load_json_safe(DAY_AHEAD_PATH)
    if not da:
        st.warning("Run day_ahead_forecast.py to generate day_ahead_results.json.")
        return

    # ---- per-zone metrics -------------------------------------------------
    st.subheader("Per-zone day-ahead metrics — HeteroSAGE")
    pz = da.get("per_zone_metrics", {})
    rows = []
    for zone, m in pz.items():
        rows.append({
            "Zone": zone,
            "MAE (DKK)": _fmt(m.get("mae")),
            "RMSE": _fmt(m.get("rmse")),
            "R²": _fmt(m.get("r2"), 4),
            "SMAPE (%)": _fmt(m.get("smape")),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Note: **HYDRO** and **DE** are zero-filled market-context nodes (no real price "
        "target), so their near-zero MAE / R²=0 are expected and should be ignored. The "
        "real evaluation is on **DK1** and **DK2**."
    )

    zone = st.selectbox("Zone for profile charts", ["DK1", "DK2"])
    profiles = da.get("price_profiles", {}).get(zone, {})

    # ---- daily profile ----------------------------------------------------
    st.subheader(f"Daily price profile — {zone} (by hour of day)")
    by_hour = profiles.get("by_hour", {})
    if by_hour:
        hours = sorted(by_hour.keys(), key=lambda h: int(h))
        actual = [by_hour[h]["actual"] for h in hours]
        pred = [by_hour[h]["predicted"] for h in hours]
        xs = [int(h) for h in hours]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=actual, name="Actual", mode="lines+markers", line=dict(color="#111827")))
        fig.add_trace(go.Scatter(x=xs, y=pred, name="Predicted", mode="lines+markers", line=dict(color=HETERO_COLOR)))
        fig.update_layout(xaxis_title="Hour of day", yaxis_title="Price (DKK)", height=420)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No by-hour profile available for this zone.")

    # ---- weekly profile ---------------------------------------------------
    st.subheader(f"Weekly price profile — {zone} (by day of week)")
    by_dow = profiles.get("by_day_of_week", {})
    if by_dow:
        dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        days = sorted(by_dow.keys(), key=lambda d: int(d))
        labels = [dow_labels[int(d)] if int(d) < 7 else d for d in days]
        actual = [by_dow[d]["actual"] for d in days]
        pred = [by_dow[d]["predicted"] for d in days]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=actual, name="Actual", marker_color="#9ca3af"))
        fig.add_trace(go.Bar(x=labels, y=pred, name="Predicted", marker_color=HETERO_COLOR))
        fig.update_layout(barmode="group", xaxis_title="Day of week", yaxis_title="Price (DKK)", height=400)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No by-day-of-week profile available for this zone.")

    # ---- horizon degradation ---------------------------------------------
    st.subheader("Horizon degradation — DK1 (recursive 24-step forecast)")
    dk1_h = da.get("horizon_mae", {}).get("DK1", [])
    if dk1_h:
        steps = list(range(1, len(dk1_h) + 1))
        fig = go.Figure(
            go.Scatter(x=steps, y=dk1_h, mode="lines+markers", line=dict(color="#dc2626"))
        )
        fig.update_layout(
            xaxis_title="Forecast horizon (hours ahead)",
            yaxis_title="MAE (DKK, log scale)",
            yaxis_type="log",
            height=420,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "The y-axis is log-scaled: recursive forecasting feeds each prediction back "
            "in as the next step's lag feature, so errors **compound** and MAE blows up "
            "by orders of magnitude over a 24-hour horizon."
        )
    else:
        st.warning("No DK1 horizon_mae array available.")


# ---------------------------------------------------------------------------
# Page 4 — Interpretability
# ---------------------------------------------------------------------------
def page_interpretability():
    st.title("🔍 Interpretability")

    interp = load_json_safe(INTERP_PATH)
    abl = load_json_safe(ABLATION_PATH)

    # ---- feature importance (ST format: list of {feature, importance}) ---
    st.subheader("Feature importance (gradient saliency — ST-HeteroSAGE)")
    fi_raw = (interp or {}).get("feature_importance")
    if not fi_raw:
        st.warning("Run st_interpretability.py to generate st_interpretability.json.")
    else:
        # ST format: list[{feature, importance}] — already sorted descending
        if isinstance(fi_raw, list):
            items = sorted(fi_raw, key=lambda d: d["importance"])
            names = [d["feature"] for d in items]
            vals  = [d["importance"] for d in items]
            max_v = max(vals) if vals else 1.0
            vals_norm = [v / max_v for v in vals]
        else:
            # fallback: old zone-keyed dict format
            zone = st.selectbox("Zone", list(fi_raw.keys()))
            scores = fi_raw.get(zone, {})
            items_kv = sorted(scores.items(), key=lambda kv: kv[1])
            names = [k for k, _ in items_kv]
            vals_norm = [v for _, v in items_kv]
        fig = go.Figure(go.Bar(x=vals_norm, y=names, orientation="h", marker_color=ST_COLOR))
        fig.update_layout(
            title="ST-HeteroSAGE feature importance (normalised gradient saliency)",
            xaxis_title="Saliency (normalised, top feature = 1.0)",
            height=480,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Gas price and day-ahead lag (t-24h) dominate; renewable generation and "
            "weekly seasonality follow. No t-1 leakage — all lags ≥ 24 h."
        )

    # ---- temporal error profile ------------------------------------------
    err = (interp or {}).get("error_by_time", {})
    if err:
        col_h, col_d = st.columns(2)
        with col_h:
            st.subheader("MAE by hour of day")
            by_h = err.get("by_hour", {})
            if by_h:
                hours = sorted(by_h.keys(), key=int)
                fig = go.Figure(go.Bar(
                    x=[int(h) for h in hours],
                    y=[by_h[h] for h in hours],
                    marker_color=ST_COLOR,
                ))
                fig.update_layout(xaxis_title="Hour", yaxis_title="MAE (DKK)", height=340)
                st.plotly_chart(fig, use_container_width=True)
        with col_d:
            st.subheader("MAE by day of week")
            by_d = err.get("by_dow", {})
            if by_d:
                labels_dow = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
                days = sorted(by_d.keys(), key=int)
                fig = go.Figure(go.Bar(
                    x=[labels_dow[int(d)] if int(d) < 7 else d for d in days],
                    y=[by_d[d] for d in days],
                    marker_color=ST_COLOR,
                ))
                fig.update_layout(xaxis_title="Day", yaxis_title="MAE (DKK)", height=340)
                st.plotly_chart(fig, use_container_width=True)

    # ---- ablation (ST format: {A_full, B_no_tcn, C_no_spatial, ...}) ----
    st.subheader("Ablation study — overall MAE by ST-HeteroSAGE component removed")
    if not abl:
        st.warning("Run st_ablation.py to generate st_ablation_results.json.")
        return

    st_labels = {
        "A_full":      "A. Full ST-HeteroSAGE",
        "B_no_tcn":    "B. No CausalTCN",
        "C_no_spatial":"C. No spatial edges",
        "D_no_cooccurs":"D. No co-occurs edges",
        "E_no_market": "E. No market nodes",
        "F_no_hydro":  "F. No HYDRO zone",
    }
    full_mae = abl.get("A_full", {}).get("mae", 0)
    rows = []
    for key, label in st_labels.items():
        if key not in abl:
            continue
        m = abl[key]
        delta = _fmt(m.get("mae", 0) - full_mae)
        rows.append({
            "Variant": label,
            "key": key,
            "MAE (DKK)": _fmt(m.get("mae")),
            "Δ vs full": delta,
            "R²": _fmt(m.get("r2"), 4),
        })
    if not rows:
        st.warning("ST ablation file present but missing expected variants.")
        return

    abl_df = pd.DataFrame(rows)
    colors = [ST_COLOR if r == "A_full" else "#f59e0b" for r in abl_df["key"]]
    fig = go.Figure(
        go.Bar(
            x=abl_df["Variant"],
            y=abl_df["MAE (DKK)"],
            marker_color=colors,
            text=[f"Δ {d:+.1f}" if isinstance(d, (int, float)) else "" for d in abl_df["Δ vs full"]],
            textposition="outside",
        )
    )
    fig.update_layout(yaxis_title="MAE (DKK)", xaxis_tickangle=-20, height=460)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(abl_df.drop(columns=["key"]), use_container_width=True, hide_index=True)

    st.markdown(
        "**Which ST-HeteroSAGE components matter?**\n\n"
        "- **CausalTCN (B)** is the single biggest contributor: removing it increases MAE "
        "by **+106 DKK (+70%)** — dilated temporal convolutions capture the intra-day and "
        "weekly periodicity that GNN message-passing alone cannot.\n"
        "- **Spatial edges (C)** matter more than in the simpler HeteroSAGE: removing them "
        "adds **+158 DKK** because the TCN operates zone-independently and the GNN layer is "
        "the only path for cross-zone information.\n"
        "- **co_occurs_with edges (D)** and **market nodes (E)** each contribute ~10–25 DKK — "
        "meaningful but secondary to the temporal component."
    )


# ---------------------------------------------------------------------------
# Page 5 — Robustness
# ---------------------------------------------------------------------------
def page_robustness():
    st.title("🛡️ Robustness")
    rob = load_json_safe(ROBUSTNESS_PATH)
    if not rob:
        st.warning("Run st_robustness.py to generate st_robustness_results.json.")
        return

    # ST format: baseline = {mae, r2}; scenarios = {mae, r2, delta_mae, delta_pct}
    base_info = rob.get("baseline", {})
    # Accept both flat and DK1-nested formats
    base_mae = base_info.get("mae") or (base_info.get("DK1") or {}).get("mae")
    if base_mae is not None:
        st.metric("Baseline MAE (clean inputs)", f"{base_mae:.2f} DKK")

    families = [
        ("gaussian_noise", "Gaussian noise injection"),
        ("feature_dropout", "Feature dropout"),
        ("price_spike", "Price spike"),
    ]

    for key, label in families:
        group = rob.get(key, {})
        if not group:
            continue
        st.subheader(label)
        scenarios = list(group.keys())

        def _get_mae(s_dict):
            if "mae" in s_dict:
                return s_dict["mae"]
            return (s_dict.get("DK1") or {}).get("mae", 0)

        def _get_delta(s_dict):
            if "delta_pct" in s_dict:
                return s_dict["delta_pct"]
            return (s_dict.get("DK1") or {}).get("mae_delta_pct", 0)

        maes   = [_get_mae(group[s])   for s in scenarios]
        deltas = [_get_delta(group[s]) for s in scenarios]

        fig = go.Figure(
            go.Bar(
                x=scenarios,
                y=maes,
                marker_color=ST_COLOR,
                text=[f"{d:+.2f}%" for d in deltas],
                textposition="outside",
            )
        )
        if base_mae is not None:
            fig.add_hline(
                y=base_mae,
                line_dash="dash",
                line_color="#dc2626",
                annotation_text="baseline",
                annotation_position="top left",
            )
        fig.update_layout(yaxis_title="MAE (DKK)", height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**Takeaway:**\n\n"
        "- **Gaussian noise** causes only a modest MAE increase even at 30% noise (+6.5%), "
        "because the CausalTCN and graph structure distribute the signal across multiple "
        "lag paths — no single input is critical.\n"
        "- **Feature dropout** degrades performance more sharply (up to +30% at 30% dropout), "
        "since randomly zeroing features removes temporal context the TCN relies on.\n"
        "- **Price spike multipliers** have minimal impact: the spike is applied to ground-truth "
        "targets only, not to input lags, so the model's predictions are unchanged."
    )


# ---------------------------------------------------------------------------
# Page 6 — MLOps & Monitoring
# ---------------------------------------------------------------------------
def page_mlops(api_up):
    st.title("⚙️ MLOps & Monitoring")

    # ---- pipeline status --------------------------------------------------
    st.subheader("Pipeline status")
    status = api_get("/pipeline/status") if api_up else None
    source = "API"
    if status is None:
        status = load_json_safe(LAST_RUN_PATH)
        source = "last_pipeline_run.json"

    if not status:
        st.warning("No pipeline run found yet. Trigger one below or run `make pipeline`.")
    else:
        st.caption(f"Source: {source}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Status", str(status.get("status", "unknown")))
        dur = status.get("duration_seconds")
        c2.metric("Duration", f"{dur:.1f}s" if isinstance(dur, (int, float)) else "—")
        c3.metric("Improved", "✅" if status.get("improved") else "—")

        if status.get("timestamp"):
            st.write(f"**Timestamp:** {status['timestamp']}")

        stages = status.get("stages_completed", []) or []
        if stages:
            st.write("**Stages completed:**")
            st.write("  ".join(
                f"{'⏭️' if ':skipped' in str(s) else '✅'} {str(s).replace(':skipped', '')}"
                for s in stages
            ))

        errors = status.get("errors") or ([status["error"]] if status.get("error") else [])
        if errors:
            st.error("Errors:\n\n" + "\n".join(f"- {e}" for e in errors))

    # ---- trigger ----------------------------------------------------------
    st.subheader("Trigger a pipeline run")
    force_rebuild = st.checkbox("Force full rebuild (fresh scaler + from-scratch retrain)", value=False)
    if api_up:
        if st.button("🚀 Trigger Pipeline Run", type="primary"):
            resp = api_post("/pipeline/run", params={"force_rebuild": force_rebuild})
            if resp is None:
                st.error("Could not reach the API.")
            elif "_error" in resp:
                st.error(f"API returned {resp['_error']}: {resp.get('detail')}")
            else:
                st.success(f"{resp.get('status', 'ok')}: {resp.get('message', '')}")
    else:
        st.button("🚀 Trigger Pipeline Run", disabled=True)
        st.caption("Triggering the pipeline requires the API server (`make serve`).")

    # ---- monitoring report -----------------------------------------------
    st.subheader("Monitoring report")
    report = api_get("/monitoring/report") if api_up else None
    mon_source = "API"
    if report is None:
        try:
            from monitoring import get_monitoring_report
            report = get_monitoring_report()
            mon_source = "monitoring.get_monitoring_report() (direct import)"
        except Exception as exc:
            report = None
            st.warning(f"Monitoring report unavailable: {exc}")

    if report:
        st.caption(f"Source: {mon_source}")
        c1, c2, c3 = st.columns(3)
        dk1 = report.get("rolling_mae_dk1", {}) or {}
        dk2 = report.get("rolling_mae_dk2", {}) or {}
        c1.metric(
            "Rolling MAE — DK1",
            f"{dk1['mae']:.2f} DKK" if dk1.get("mae") is not None else "no data",
            help=f"{dk1.get('n_samples', 0)} samples / {dk1.get('window_hours', '?')}h window",
        )
        c2.metric(
            "Rolling MAE — DK2",
            f"{dk2['mae']:.2f} DKK" if dk2.get("mae") is not None else "no data",
            help=f"{dk2.get('n_samples', 0)} samples / {dk2.get('window_hours', '?')}h window",
        )
        c3.metric("Predictions (last 24h)", report.get("predictions_last_24h", 0))

        drift = report.get("drift_status", {}) or {}
        d_status = drift.get("status", "unknown")
        if d_status == "drift_detected":
            st.error(f"Drift detected: {drift.get('drifted_features', [])}")
        else:
            note = drift.get("note", "")
            st.success(f"Drift status: {d_status}" + (f" — {note}" if note else ""))


# ---------------------------------------------------------------------------
# Page — Graph Structure
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_hetero_graph():
    """Load the heterogeneous graph object (cached across reruns)."""
    try:
        import torch
        return torch.load(GRAPH_PATH, map_location="cpu", weights_only=False)
    except Exception as exc:
        return {"_error": str(exc)}


@st.cache_resource(show_spinner=False)
def _load_feature_scaler():
    """Load the fitted feature scaler (for inverse-transforming hover values)."""
    try:
        import pickle
        with open(SCALER_PATH, "rb") as f:
            return pickle.load(f).get("feature_scaler")
    except Exception:
        return None


def _node_datetime(t):
    """Map an in-zone timestep index to its real timestamp."""
    return (pd.Timestamp(DATASET_START) + pd.Timedelta(hours=int(t))).strftime("%Y-%m-%d %H:%M")


def page_graph():
    st.title("🕸️ Graph Structure")
    st.markdown(
        "How the heterogeneous graph fed to **ST-HeteroSAGE** is constructed — "
        "**2 node types** (`hour`, `market`) and **5 edge relation types**. "
        "The schematic below needs no model files; the statistics and the "
        "interactive mini-graph load the real `hetero_graph.pt`."
    )

    g = _load_hetero_graph()
    graph_ok = not (isinstance(g, dict) and "_error" in g)
    if not graph_ok:
        st.warning(
            f"Could not load `{GRAPH_PATH.name}` "
            f"({g.get('_error', 'missing')}). The schematic still renders below; "
            "run `python src/hetero_graph_builder.py` to enable the live sections."
        )

    # ---- 1. Live statistics ----------------------------------------------
    st.subheader("1. Graph statistics")
    if graph_ok:
        node_rows = []
        for ntype in g.node_types:
            n = g[ntype].num_nodes
            f = g[ntype].x.shape[1] if getattr(g[ntype], "x", None) is not None else 0
            node_rows.append({"Node type": ntype, "# nodes": f"{n:,}", "# features": f})

        edge_rows = []
        for src, rel, dst in g.edge_types:
            ei = g[src, rel, dst].edge_index
            edge_rows.append({
                "Direction": f"{src} → {dst}", "Relation": rel,
                "# edges": f"{ei.shape[1]:,}",
            })

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Node types**")
            st.dataframe(pd.DataFrame(node_rows), use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**Edge relation types**")
            st.dataframe(pd.DataFrame(edge_rows), use_container_width=True, hide_index=True)

        T = int(g["hour"].num_hours_per_zone)
        total_nodes = sum(g[nt].num_nodes for nt in g.node_types)
        total_edges = sum(g[et].edge_index.shape[1] for et in g.edge_types)
        m1, m2, m3 = st.columns(3)
        m1.metric("Total nodes", f"{total_nodes:,}")
        m2.metric("Total edges", f"{total_edges:,}")
        m3.metric("Hours per zone (T)", f"{T:,}")
    else:
        st.info("Live statistics unavailable — graph file not loaded.")

    # ---- 2. Schematic topology -------------------------------------------
    st.subheader("2. Schematic — spatio-temporal layout")
    st.caption(
        "Each **zone is a chain of hourly nodes** (here 3 timesteps shown; the real "
        "graph has ~50k per zone). Read left→right as time. Horizontal arrows are the "
        "temporal links; vertical lines are same-hour price co-movement across zones; "
        "dashed lines attach hours to their market node."
    )

    zones = ["DK1", "DK2", "HYDRO", "DE"]
    zone_y = {"DK1": 4.0, "DK2": 3.0, "HYDRO": 2.0, "DE": 1.0}
    cols_x = [0.0, 2.0, 4.0]               # t-2, t-1, t
    col_lbl = ["t−2", "t−1", "t (target)"]
    market_y = -0.4
    market_x = {"NordPool": 0.0, "DK1 area": 1.5, "DK2 area": 3.0, "DE area": 4.0}
    market_color = {"NordPool": "#9467bd", "DK1 area": "#8c564b",
                    "DK2 area": "#e377c2", "DE area": "#7f7f7f"}

    fig = go.Figure()
    legend_done = set()

    def _edge(x0, y0, x1, y1, color, name, dash="solid", width=2):
        show = name not in legend_done
        legend_done.add(name)
        fig.add_trace(go.Scatter(
            x=[x0, x1], y=[y0, y1], mode="lines",
            line=dict(color=color, width=width, dash=dash),
            name=name, showlegend=show, hoverinfo="name",
        ))

    # lag_to — temporal, horizontal within each zone row (arrows)
    for z in zones:
        y = zone_y[z]
        for ci in range(len(cols_x) - 1):
            _edge(cols_x[ci], y, cols_x[ci + 1], y, "#3182bd",
                  "lag_to (temporal: hour→next hour)", width=2.5)
            fig.add_annotation(ax=cols_x[ci], ay=y, x=cols_x[ci + 1], y=y,
                               xref="x", yref="y", axref="x", ayref="y",
                               showarrow=True, arrowhead=3, arrowsize=1.2,
                               arrowwidth=2, arrowcolor="#3182bd")

    # co_occurs_with — spatial, vertical at each timestep (DE excluded)
    co_zones = ["DK1", "DK2", "HYDRO"]
    for ci in range(len(cols_x)):
        for a, b in zip(co_zones[:-1], co_zones[1:]):
            _edge(cols_x[ci], zone_y[a], cols_x[ci], zone_y[b], "#31a354",
                  "co_occurs_with (spatial: same-hour price co-movement)", width=1.8)

    # belongs_to — each zone's target hour → its market node (dashed)
    belongs = {"DK1": "DK1 area", "DK2": "DK2 area", "HYDRO": "NordPool", "DE": "DE area"}
    for z, mk in belongs.items():
        _edge(cols_x[-1], zone_y[z], market_x[mk], market_y, "#9e9ac8",
              "belongs_to (hour → market)", dash="dash", width=1.5)

    # interconnects — market ↔ market (transmission capacity)
    for a, b in [("NordPool", "DK1 area"), ("DK1 area", "DK2 area"), ("DK2 area", "DE area")]:
        _edge(market_x[a], market_y, market_x[b], market_y, "#fd8d3c",
              "interconnects (market ↔ market: transmission)", width=2)

    # Hour nodes
    for z in zones:
        y = zone_y[z]
        fig.add_trace(go.Scatter(
            x=cols_x, y=[y] * len(cols_x), mode="markers+text",
            marker=dict(color=ZONE_COLORS[z], size=30, line=dict(color="white", width=2)),
            text=[z if ci == 0 else "" for ci in range(len(cols_x))],
            textposition="middle left", showlegend=False,
            hovertemplate=f"<b>{z}</b> hour node<extra></extra>",
        ))
    # Market nodes
    for mk, mx in market_x.items():
        fig.add_trace(go.Scatter(
            x=[mx], y=[market_y], mode="markers+text",
            marker=dict(color=market_color[mk], size=24, symbol="square",
                        line=dict(color="white", width=2)),
            text=[mk], textposition="bottom center", showlegend=False,
            hovertemplate=f"<b>{mk}</b> market node<extra></extra>",
        ))
    # Column time labels
    for ci, lbl in enumerate(col_lbl):
        fig.add_annotation(x=cols_x[ci], y=zone_y["DK1"] + 0.55, text=f"<b>{lbl}</b>",
                           showarrow=False, font=dict(size=12, color="#555"))

    fig.update_layout(
        title="Heterogeneous graph — spatio-temporal schematic (2 node types, 5 edge types)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.2, 5.2]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.4, 5.0]),
        height=560, plot_bgcolor="white",
        legend=dict(x=1.01, y=0.99, bordercolor="lightgray", borderwidth=1,
                    font=dict(size=10)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- 3. Interactive mini-graph ---------------------------------------
    st.subheader("3. Interactive mini-graph")
    if not graph_ok:
        st.info("Mini-graph unavailable — graph file not loaded.")
        return

    st.caption(
        "A real slice of the graph: consecutive hours per zone plus the 4 market "
        "nodes. Hover a node for its **actual** price and weather (inverse-transformed "
        "to real units). Edges are filtered from the full graph for exactly these nodes."
    )
    import numpy as np
    import torch

    T = int(g["hour"].num_hours_per_zone)
    M = g["market"].num_nodes
    zones = ["DK1", "DK2", "HYDRO", "DE"]
    m_colors = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    mkt_names = ["NordPool", "DK1 area", "DK2 area", "DE area"]
    scaler = _load_feature_scaler()

    c_a, c_b = st.columns(2)
    n_steps = c_a.slider("Hours per zone", min_value=2, max_value=6, value=3)
    # Anchor on the start of the test window so labels show recent, real dates
    default_anchor = int(T * 0.85)
    t_start = c_b.slider("Start hour index", min_value=0, max_value=max(0, T - n_steps),
                         value=min(default_anchor, T - n_steps), step=24)
    steps = list(range(t_start, t_start + n_steps))

    # ── Global indices of the selected hour nodes (zone z occupies [z*T, (z+1)*T)) ──
    hour_global = {}      # global idx -> mini id
    market_global = {}    # global idx (0..M-1) -> mini id
    positions, labels, colors_, hovers = {}, {}, {}, {}
    mid = 0

    # Pre-build an inverse-transformed view of just the selected hour rows
    sel_global = [zi * T + t for zi in range(len(zones)) for t in steps]
    feats_scaled = g["hour"].x[sel_global].numpy()
    feats_raw = scaler.inverse_transform(feats_scaled) if scaler is not None else feats_scaled
    raw_by_global = {gi: feats_raw[i] for i, gi in enumerate(sel_global)}

    for zi, z in enumerate(zones):
        for ci, t in enumerate(steps):
            gi = zi * T + t
            hour_global[gi] = mid
            positions[mid] = (ci * 2.0, -zi * 1.5)
            labels[mid] = z if ci == 0 else ""
            colors_[mid] = ZONE_COLORS[z]
            price = float(g["hour"].y[gi])
            r = raw_by_global[gi]
            hovers[mid] = (
                f"<b>{z}</b> &nbsp; {_node_datetime(t)}<br>"
                f"actual price: {price:,.0f} DKK<br>"
                f"lag_24h: {r[0]:,.0f} DKK<br>"
                f"temp: {r[5]:.1f} °C &nbsp; wind: {r[6]:.1f} m/s"
            )
            mid += 1

    for mi in range(min(M, 4)):
        market_global[mi] = mid
        positions[mid] = (mi * 2.0, -len(zones) * 1.5 - 0.8)
        labels[mid] = mkt_names[mi] if mi < len(mkt_names) else f"Market {mi}"
        colors_[mid] = m_colors[mi % len(m_colors)]
        hovers[mid] = f"<b>{labels[mid]}</b><br>market node"
        mid += 1

    # ── Vectorised edge filtering with torch.isin ──────────────────────────
    rel_colors = {
        "lag_to": "#3182bd", "co_occurs_with": "#31a354", "belongs_to": "#9e9ac8",
        "rev_belongs_to": "#bcbddc", "interconnects": "#fd8d3c",
    }
    hour_keep = torch.tensor(sorted(hour_global), dtype=torch.long)
    mkt_keep = torch.tensor(sorted(market_global), dtype=torch.long)
    edge_traces, drawn_rels = [], set()

    for src_type, rel, dst_type in g.edge_types:
        ei = g[src_type, rel, dst_type].edge_index
        src_keep = hour_keep if src_type == "hour" else mkt_keep
        dst_keep = hour_keep if dst_type == "hour" else mkt_keep
        mask = torch.isin(ei[0], src_keep) & torch.isin(ei[1], dst_keep)
        if not bool(mask.any()):
            continue
        kept = ei[:, mask]
        src_map = hour_global if src_type == "hour" else market_global
        dst_map = hour_global if dst_type == "hour" else market_global
        ex, ey = [], []
        for s, d in zip(kept[0].tolist(), kept[1].tolist()):
            x0, y0 = positions[src_map[s]]
            x1, y1 = positions[dst_map[d]]
            ex += [x0, x1, None]
            ey += [y0, y1, None]
        show = rel not in drawn_rels
        drawn_rels.add(rel)
        edge_traces.append(go.Scatter(
            x=ex, y=ey, mode="lines",
            line=dict(color=rel_colors.get(rel, "#cccccc"), width=1.5),
            opacity=0.75, name=rel, showlegend=show, hoverinfo="none",
        ))

    node_scatter = go.Scatter(
        x=[positions[i][0] for i in range(mid)],
        y=[positions[i][1] for i in range(mid)],
        mode="markers+text",
        marker=dict(color=[colors_[i] for i in range(mid)], size=22,
                    line=dict(color="white", width=1.5)),
        text=[labels[i] for i in range(mid)], textposition="middle left",
        customdata=[hovers[i] for i in range(mid)],
        hovertemplate="%{customdata}<extra></extra>", showlegend=False,
    )
    fig2 = go.Figure(edge_traces + [node_scatter])
    fig2.update_layout(
        title=f"Interactive mini-graph — {n_steps} hours/zone from {_node_datetime(t_start)}",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=560, plot_bgcolor="white", hovermode="closest",
        legend=dict(x=1.01, y=0.99, bordercolor="lightgray", borderwidth=1),
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption(f"Nodes shown: {mid}  |  Edge relation types present in this slice: {len(drawn_rels)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Electricity Forecasting MLOps",
        page_icon="⚡",
        layout="wide",
    )

    st.sidebar.title("⚡ Electricity Forecasting")
    api_up = api_banner()

    if st.sidebar.button("🔄 Clear cache"):
        st.cache_data.clear()
        st.rerun()

    page = st.sidebar.radio(
        "Navigate",
        [
            "📊 Overview & Leaderboard",
            "🕸️ Graph Structure",
            "🔮 Live Prediction",
            "📈 Forecast Analysis",
            "🔍 Interpretability",
            "🛡️ Robustness",
            "⚙️ MLOps & Monitoring",
        ],
    )

    st.sidebar.caption(f"API base: `{API_BASE}`")

    if page == "📊 Overview & Leaderboard":
        page_overview()
    elif page == "🕸️ Graph Structure":
        page_graph()
    elif page == "🔮 Live Prediction":
        page_predict(api_up)
    elif page == "📈 Forecast Analysis":
        page_forecast()
    elif page == "🔍 Interpretability":
        page_interpretability()
    elif page == "🛡️ Robustness":
        page_robustness()
    elif page == "⚙️ MLOps & Monitoring":
        page_mlops(api_up)


if __name__ == "__main__":
    main()
