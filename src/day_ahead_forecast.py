"""
Multi-area day-ahead forecasting analysis.

Addresses the gaps in the original evaluation:
  1. Per-zone metrics for all four areas  (DK1, DK2, HYDRO, DE)
  2. Hourly and weekly price profiles     (avg predicted vs actual by time slot)
  3. 24-step recursive horizon simulation (MAE vs forecast horizon)
     — at each step the lag_1h feature is replaced with the prior prediction,
       compounding autoregressive error across the test window.

Works with both legacy (9-feature) and current (13-feature) checkpoints.
Outputs saved to artifacts_hetero/day_ahead_results.json
"""
import sys
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from hetero_config import DEVICE, ARTIFACTS_DIR, GRAPH_DIR
from hetero_models import load_hetero_model
from hetero_pipeline import prepare_multi_area_data

ZONE_NAMES = ['DK1', 'DK2', 'HYDRO', 'DE']
LAG1_IDX = 0   # index of price_lag_1h in feature vector


def _load():
    data = torch.load(GRAPH_DIR / "hetero_graph.pt", map_location=DEVICE, weights_only=False)
    num_hours = int(data['hour'].num_hours_per_zone)
    model, x_override = load_hetero_model(data, ARTIFACTS_DIR / "best_hetero_model.pt", DEVICE)
    x_base = (x_override.cpu() if x_override is not None else data['hour'].x).clone()
    return model, data, num_hours, x_base


@torch.no_grad()
def _infer(model, data, x_base, num_hours):
    x_dict = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
    x_dict['hour'] = x_base.to(DEVICE)
    ei = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
    return model(x_dict, ei, num_hours=num_hours).view(-1).cpu().numpy()


def compute_per_zone_metrics(model, data, num_hours, x_base):
    out = _infer(model, data, x_base, num_hours)
    y   = data['hour'].y.cpu().numpy()
    tm  = data['hour'].test_mask.cpu().numpy()

    results = {}
    for z, name in enumerate(ZONE_NAMES):
        sl = slice(z * num_hours, (z + 1) * num_hours)
        zm = tm[sl]
        y_z, p_z = y[sl][zm], out[sl][zm]
        smape = float(np.mean(2 * np.abs(p_z - y_z) / (np.abs(y_z) + np.abs(p_z) + 1e-8)) * 100)
        mape  = float(np.mean(np.abs((p_z - y_z) / (np.abs(y_z) + 1e-8))) * 100)
        results[name] = {
            'mae':   round(float(mean_absolute_error(y_z, p_z)), 4),
            'rmse':  round(float(np.sqrt(np.mean((p_z - y_z)**2))), 4),
            'r2':    round(float(r2_score(y_z, p_z)), 4),
            'smape': round(smape, 4),
            'mape':  round(mape, 4),
        }
    return results


def compute_hourly_price_profiles(model, data, num_hours, x_base):
    """Average predicted vs actual price by hour-of-day and day-of-week."""
    import pandas as pd
    df_dk1, _, _, _, _ = prepare_multi_area_data()
    ts = pd.to_datetime(df_dk1['timestamp'].values)
    test_start  = int(num_hours * 0.9)
    hour_of_day = ts[test_start:].hour.values
    day_of_week = ts[test_start:].dayofweek.values

    out = _infer(model, data, x_base, num_hours)
    y   = data['hour'].y.cpu().numpy()
    tm  = data['hour'].test_mask.cpu().numpy()

    profiles = {}
    for z_idx, z_name in enumerate(['DK1', 'DK2']):
        sl = slice(z_idx * num_hours, (z_idx + 1) * num_hours)
        zm = tm[sl]
        y_z, p_z = y[sl][zm], out[sl][zm]

        by_hour = {
            str(h): {
                'actual':    round(float(y_z[hour_of_day == h].mean()), 2) if (hour_of_day == h).any() else 0.0,
                'predicted': round(float(p_z[hour_of_day == h].mean()), 2) if (hour_of_day == h).any() else 0.0,
            }
            for h in range(24)
        }
        by_dow = {
            str(d): {
                'actual':    round(float(y_z[day_of_week == d].mean()), 2) if (day_of_week == d).any() else 0.0,
                'predicted': round(float(p_z[day_of_week == d].mean()), 2) if (day_of_week == d).any() else 0.0,
            }
            for d in range(7)
        }
        profiles[z_name] = {'by_hour': by_hour, 'by_day_of_week': by_dow}
    return profiles


def simulate_recursive_horizon(model, data, num_hours, x_base, max_horizon=24):
    """
    Simulate autoregressive horizon degradation: at step h, the lag_1h feature
    of each test node is replaced with the step h-1 prediction (properly re-scaled).

    Predictions are in raw DKK; the feature vector expects StandardScaler-normalised
    values. We load the fitted scaler and apply the inverse/forward transform so
    inserted lag features stay in the correct range.

    Returns per-zone list of MAE for h = 1 … max_horizon.
    """
    import pickle
    scalers_path = GRAPH_DIR / "hetero_scalers.pkl"
    if scalers_path.exists():
        with open(scalers_path, "rb") as f:
            scalers = pickle.load(f)
        feat_scaler = scalers['feature_scaler']
        lag1_mean  = float(feat_scaler.mean_[LAG1_IDX])
        lag1_std   = float(feat_scaler.scale_[LAG1_IDX])
    else:
        # Fallback: estimate from x_base feature column
        col = x_base[:, LAG1_IDX].numpy()
        lag1_mean, lag1_std = float(col.mean()), float(col.std() + 1e-6)

    y          = data['hour'].y.cpu().numpy()
    tm         = data['hour'].test_mask.cpu().numpy()
    test_start = int(num_hours * 0.9)

    x_current  = x_base.numpy().copy()
    horizon_mae = {name: [] for name in ZONE_NAMES}

    for h in range(1, max_horizon + 1):
        out = _infer(model, data, torch.tensor(x_current, dtype=torch.float32), num_hours)

        for z, name in enumerate(ZONE_NAMES):
            sl = slice(z * num_hours, (z + 1) * num_hours)
            zm = tm[sl]
            horizon_mae[name].append(round(float(mean_absolute_error(y[sl][zm], out[sl][zm])), 4))

        # Scale prediction back to feature space before inserting as lag_1h
        for z in range(4):
            off = z * num_hours
            for t in range(test_start, num_hours - 1):
                raw_pred = out[off + t]
                scaled   = (raw_pred - lag1_mean) / lag1_std
                x_current[off + t + 1, LAG1_IDX] = scaled

    return horizon_mae


def run_day_ahead_analysis():
    print("\n📅 Running Day-Ahead Multi-Zone Forecast Analysis...")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model, data, num_hours, x_base = _load()
    n_feats = x_base.shape[1]
    print(f"   Model input features: {n_feats}")

    print("  [1/3] Per-zone metrics (DK1, DK2, HYDRO, DE)...")
    zone_metrics = compute_per_zone_metrics(model, data, num_hours, x_base)

    print("  [2/3] Hourly and weekly price profiles...")
    profiles = compute_hourly_price_profiles(model, data, num_hours, x_base)

    print("  [3/3] Recursive 24-step horizon simulation...")
    horizon = simulate_recursive_horizon(model, data, num_hours, x_base, max_horizon=24)

    summary = {
        'per_zone_metrics': zone_metrics,
        'price_profiles':   profiles,
        'horizon_mae':      horizon,
    }

    out_path = ARTIFACTS_DIR / "day_ahead_results.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)

    # ── readable output ───────────────────────────────────────────────────────
    print("\n📊 Per-Zone Test Metrics (note: HYDRO & DE have zero prices in current DB):")
    print(f"  {'Zone':<8} {'MAE':>8} {'RMSE':>8} {'R²':>7} {'SMAPE':>8} {'Data'}")
    print("  " + "-" * 54)
    data_status = {'DK1': 'real', 'DK2': 'real', 'HYDRO': 'zero-fill', 'DE': 'zero-fill'}
    for name, m in zone_metrics.items():
        flag = data_status.get(name, '?')
        print(f"  {name:<8} {m['mae']:>8.2f} {m['rmse']:>8.2f} {m['r2']:>7.4f} {m['smape']:>7.2f}%  {flag}")

    print("\n⏳ Recursive Horizon — DK1 MAE (DKK) [errors compound when lag≠actual]:")
    dk1_h = horizon.get('DK1', [])
    for step in [1, 2, 3, 6, 12, 18, 24]:
        if step <= len(dk1_h):
            v = dk1_h[step-1]
            note = " ← stable" if step <= 3 else (" ← degrading" if v < 1000 else " ← unstable")
            print(f"    h={step:2d}h → {v:.1f} DKK{note}")

    print("\n🕐 DK1 Daily Price Profile (avg by hour):")
    for h in range(0, 24, 3):
        slot = profiles.get('DK1', {}).get('by_hour', {}).get(str(h), {})
        a, p = slot.get('actual', 0), slot.get('predicted', 0)
        print(f"    {h:02d}h: actual={a:.0f}  predicted={p:.0f} DKK")

    print(f"\n✅ Saved → {out_path}")
    return summary


if __name__ == "__main__":
    run_day_ahead_analysis()
