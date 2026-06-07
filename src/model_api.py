"""
FastAPI serving for HeteroSTPriceForecaster (ST-HeteroSAGE).

GET  /health             -> model status
GET  /metrics            -> test metrics from st_hetero_metrics.json
POST /predict            -> day-ahead price forecast for DK1 or DK2
POST /pipeline/run       -> trigger the full MLOps pipeline as a background task
GET  /pipeline/status    -> return last pipeline run status
GET  /monitoring/report  -> return rolling MAE and drift status
POST /monitor/log-actual -> log the actual observed price for MAE tracking
"""
import sys
import json
import math
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parent
_ARTIFACTS_HETERO = _SRC / "artifacts_hetero"
_GRAPH_DIR = _SRC / "data" / "graphs_hetero"
_METRICS_FILE = _ARTIFACTS_HETERO / "st_hetero_metrics.json"
_SCALER_FILE = _GRAPH_DIR / "hetero_scalers.pkl"
_CKPT_FILE = _ARTIFACTS_HETERO / "best_st_hetero_model.pt"

# Spatial-only edge types — lag_to is handled by the CausalTCN inside the model
SPATIAL_EDGE_TYPES = [
    ('hour',   'co_occurs_with', 'hour'),
    ('hour',   'belongs_to',     'market'),
    ('market', 'rev_belongs_to', 'hour'),
    ('market', 'interconnects',  'market'),
]

# Per-zone training stats for load/renewable z-scoring (from hetero_pipeline.py, 80% train split)
_ZONE_LOAD_STATS = {
    "DK1": {"load_mwh": (2615.27, 505.81),  "renewable_mwh": (1760.94, 1181.19)},
    "DK2": {"load_mwh": (1566.48, 307.03),  "renewable_mwh": (619.19,  452.03)},
}

app = FastAPI(
    title="Electricity Price Forecasting API",
    description="ST-HeteroSAGE (CausalTCN + HeteroConv) — DK1 & DK2 zones",
    version="2.0.0",
)

# Track pipeline state between background runs
_pipeline_status: dict = {"status": "idle"}

# ---------------------------------------------------------------------------
# Startup: load model + scalers
# ---------------------------------------------------------------------------
_model = None
_scalers = None
_hetero_data = None
_num_hours = None
_device = None
_edge_index_dict = None
_target_scaler = None


@app.on_event("startup")
async def load_model():
    global _model, _scalers, _hetero_data, _num_hours, _device, _edge_index_dict, _target_scaler

    try:
        import torch
        from sklearn.preprocessing import StandardScaler
        from hetero_config import DEVICE, GRAPH_DIR, ARTIFACTS_DIR
        from hetero_st_model import HeteroSTPriceForecaster

        _device = DEVICE

        # Load graph
        data = torch.load(GRAPH_DIR / "hetero_graph.pt", map_location=DEVICE, weights_only=False)
        _num_hours = int(data["hour"].num_hours_per_zone)
        _hetero_data = data

        # Spatial-only edge dict (lag_to handled by CausalTCN)
        _edge_index_dict = {et: data[et].edge_index.to(DEVICE) for et in SPATIAL_EDGE_TYPES}

        # Load scalers
        with open(GRAPH_DIR / "hetero_scalers.pkl", "rb") as f:
            _scalers = pickle.load(f)

        # Refit target_scaler on DK1+DK2 training nodes (matches st_train.py)
        T = _num_hours
        tr_mask = data['hour'].train_mask.cpu().numpy()
        dk12_mask = np.zeros(4 * T, dtype=bool)
        dk12_mask[:2 * T] = True
        y_raw = data['hour'].y.cpu().numpy()
        _target_scaler = StandardScaler()
        _target_scaler.fit(y_raw[tr_mask & dk12_mask].reshape(-1, 1))

        # Build and load model
        in_channels = data['hour'].x.shape[1]
        _model = HeteroSTPriceForecaster(
            in_channels=in_channels,
            hidden_channels=128,
            num_st_blocks=2,
            temporal_dilations=(1, 4, 24),
            temporal_kernel=7,
        ).to(DEVICE)
        _model.load_state_dict(torch.load(ARTIFACTS_DIR / "best_st_hetero_model.pt",
                                           map_location=DEVICE, weights_only=False))
        _model.eval()

        print(f"[API] ST-HeteroSAGE loaded — {_num_hours} hours/zone, device={DEVICE}")

    except Exception as exc:
        print(f"[API] WARNING: model load failed — {exc}")
        print("[API] /predict will return 503 until model is available.")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
class PredictRequest(BaseModel):
    zone: str = Field(..., description="DK1 or DK2")
    hour_of_day: int = Field(..., ge=0, le=23, description="Target hour of day (0–23)")
    day_of_week: int = Field(default=1, ge=0, le=6, description="Day of week (Mon=0, Sun=6)")
    lag_24h: float = Field(default=800.0, description="Price lag 24 h (DKK)")
    lag_48h: float = Field(default=750.0, description="Price lag 48 h (DKK)")
    lag_168h: float = Field(default=720.0, description="Price lag 168 h / 1-week (DKK)")
    roll_mean: float = Field(default=750.0, description="24 h rolling mean price (DKK)")
    roll_std: float = Field(default=150.0, description="24 h rolling std price (DKK)")
    temp_c: float = Field(default=10.0, description="Temperature (°C)")
    wind_ms: float = Field(default=6.0, description="Wind speed (m/s)")
    cloud_pct: float = Field(default=50.0, description="Cloud cover (%)")
    humidity_pct: float = Field(default=75.0, description="Humidity (%)")
    load_mwh: float = Field(default=2600.0, description="Load (MWh)")
    renewable_mwh: float = Field(default=1700.0, description="Renewable generation (MWh)")
    gas_dkk: float = Field(default=400.0, description="Gas price (DKK/MWh)")
    co2_dkk: float = Field(default=200.0, description="CO2 price (DKK/t)")


class PredictResponse(BaseModel):
    zone: str
    predicted_price_dkk: float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "model": "ST-HeteroSAGE", "zones": ["DK1", "DK2"]}


@app.get("/metrics")
def get_metrics():
    if not _METRICS_FILE.exists():
        raise HTTPException(status_code=404, detail="Metrics file not found")
    with open(_METRICS_FILE) as f:
        return json.load(f)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _model is None or _hetero_data is None or _target_scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded — check startup logs")

    zone = req.zone.upper()
    if zone not in ("DK1", "DK2"):
        raise HTTPException(status_code=422, detail="zone must be DK1 or DK2")

    try:
        import torch

        T = _num_hours
        zone_offset = 0 if zone == "DK1" else T

        # Per-zone z-score for load/renewable (matches hetero_pipeline.py)
        zs = _ZONE_LOAD_STATS[zone]
        z_load = (req.load_mwh - zs["load_mwh"][0]) / zs["load_mwh"][1]
        z_ren  = (req.renewable_mwh - zs["renewable_mwh"][0]) / zs["renewable_mwh"][1]

        # Cyclical calendar features
        h_sin = math.sin(2 * math.pi * req.hour_of_day / 24.0)
        h_cos = math.cos(2 * math.pi * req.hour_of_day / 24.0)
        w_sin = math.sin(2 * math.pi * req.day_of_week / 7.0)
        w_cos = math.cos(2 * math.pi * req.day_of_week / 7.0)

        # 17-feature vector matching hetero_graph_builder.py column order
        raw = np.array([[req.lag_24h, req.lag_48h, req.lag_168h,
                         req.roll_mean, req.roll_std,
                         req.temp_c, req.wind_ms, req.cloud_pct, req.humidity_pct,
                         z_load, z_ren, req.gas_dkk, req.co2_dkk,
                         h_sin, h_cos, w_sin, w_cos]], dtype=np.float32)

        from sklearn.preprocessing import StandardScaler  # already imported at startup
        scaled = _scalers["feature_scaler"].transform(raw)

        # Override the last test node in this zone with the user feature vector
        test_mask = _hetero_data['hour'].test_mask.cpu().numpy()
        zone_test = test_mask[zone_offset: zone_offset + T]
        node_idx = zone_offset + int(np.where(zone_test)[0][-1])

        x_hour = _hetero_data['hour'].x.clone().to(_device)
        x_hour[node_idx] = torch.tensor(scaled[0], dtype=torch.float32).to(_device)
        x_dict = {'hour': x_hour, 'market': _hetero_data['market'].x.to(_device)}

        with torch.no_grad():
            out = _model(x_dict, _edge_index_dict)

        predicted_dkk = float(_target_scaler.inverse_transform([[out[node_idx].item()]])[0][0])
        return PredictResponse(zone=zone, predicted_price_dkk=predicted_dkk)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Pipeline & monitoring endpoints
# ---------------------------------------------------------------------------

def _run_pipeline_task(force_rebuild: bool):
    """Background task: run the full pipeline and update _pipeline_status."""
    global _pipeline_status
    _pipeline_status["status"] = "running"
    _pipeline_status["started_at"] = __import__("datetime").datetime.utcnow().isoformat()
    try:
        from pipeline import run_pipeline
        result = run_pipeline(force_rebuild_graph=force_rebuild)
        _pipeline_status = {
            "status": result.status,
            "finished_at": __import__("datetime").datetime.utcnow().isoformat(),
            "stages_completed": result.stages_completed,
            "metrics": result.metrics,
            "improved": result.improved,
            "errors": result.errors,
            "duration_seconds": result.duration_seconds,
        }
    except Exception as exc:
        _pipeline_status = {
            "status": "failed",
            "error": str(exc),
            "finished_at": __import__("datetime").datetime.utcnow().isoformat(),
        }


@app.post("/pipeline/run")
async def trigger_pipeline(
    background_tasks: BackgroundTasks,
    force_rebuild: bool = False,
):
    """Trigger the full MLOps pipeline as a background task. Returns immediately."""
    global _pipeline_status
    if _pipeline_status.get("status") == "running":
        return {"status": "already_running", "message": "Pipeline is already in progress"}
    _pipeline_status["status"] = "running"
    background_tasks.add_task(_run_pipeline_task, force_rebuild)
    return {"status": "started", "message": "Pipeline running in background"}


@app.get("/pipeline/status")
def pipeline_status():
    """Return last pipeline run status."""
    try:
        from pipeline import get_pipeline_status
        persisted = get_pipeline_status()
        # Merge with in-memory status (running state is only in memory)
        if _pipeline_status.get("status") == "running":
            return _pipeline_status
        return persisted
    except Exception as exc:
        return {"status": "unknown", "error": str(exc)}


@app.get("/monitoring/report")
def monitoring_report():
    """Return rolling MAE and drift status."""
    try:
        from monitoring import get_monitoring_report
        return get_monitoring_report()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/monitor/log-actual")
def log_actual_price(zone: str, timestamp: str, actual_price_dkk: float):
    """Log the actual observed price for a previous prediction (enables MAE tracking)."""
    try:
        from monitoring import log_prediction
        log_prediction(zone=zone, predicted_dkk=None, features={}, actual_dkk=actual_price_dkk)
        return {"status": "logged", "zone": zone, "timestamp": timestamp, "actual_price_dkk": actual_price_dkk}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
