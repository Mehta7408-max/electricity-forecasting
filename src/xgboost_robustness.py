"""
Robustness evaluation for the feature-parity XGBoost baseline.

Mirrors src/st_robustness.py exactly — same RNG seed, same perturbation
families and intensities — so the degradation curves are directly comparable
to ST-HeteroSAGE and HomoGNN:

  1. Gaussian noise   σ ∈ {5%, 10%, 20%, 30%} of each feature's std
  2. Feature dropout  randomly zero-out features at {10%, 20%, 30%}
  3. Price spike      multiply lag columns by {2×, 3×, 5×} for 5% of rows

All scenarios are inference-only — no retraining.
Results: artifacts/xgboost_robustness_results.json
"""
import sys, json, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from hetero_pipeline import prepare_multi_area_data

# Same feature order as xgboost_baseline.py. Lag/rolling columns are the
# price-derived features — the analogue of ST/Homo's LAG_COLS [0..4].
FEATURE_COLS = [
    'hour_of_day', 'minute', 'zone_id',
    'price_lag_24h', 'price_lag_48h', 'price_lag_168h',
    'price_rolling_24h_mean', 'price_rolling_24h_std',
    'neighbor_price_de', 'neighbor_price_hydro',
    'temperature_c', 'wind_speed_ms', 'cloud_cover_pct',
    'humidity_pct', 'load_mwh', 'renewable_mwh', 'gas_dkk', 'co2_dkk',
]
LAG_COLS = [3, 4, 5, 6, 7]   # price_lag_24h/48h/168h, rolling mean/std
RNG = np.random.default_rng(42)


def _setup():
    df_dk1, df_dk2, df_hydro, df_de, _ = prepare_multi_area_data()
    df_dk1 = df_dk1.copy(); df_dk2 = df_dk2.copy()
    df_dk1['zone_id'] = 0; df_dk2['zone_id'] = 1
    df_dk1['neighbor_price_de'] = df_de['price_lag_24h']
    df_dk1['neighbor_price_hydro'] = df_hydro['price_lag_24h']
    df_dk2['neighbor_price_de'] = df_de['price_lag_24h']
    df_dk2['neighbor_price_hydro'] = df_hydro['price_lag_24h']
    df = pd.concat([df_dk1, df_dk2], ignore_index=True)

    # Same 80/10/10 chronological split: test = last 10%.
    ts = sorted(df['timestamp'].unique())
    test_ts = ts[int(0.9 * len(ts))]
    test_mask = df['timestamp'] >= test_ts

    X_test = df.loc[test_mask, FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0).values
    y_test = df.loc[test_mask, 'price_dkk'].values

    with open(Path(__file__).parent.parent / "artifacts" / "xgboost_baseline.pkl", "rb") as f:
        model = pickle.load(f)
    return model, X_test, y_test


def _eval(model, X, y_true):
    y_pred = model.predict(X)
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "r2":  float(r2_score(y_true, y_pred))}


def run():
    print("\n🛡️  XGBoost (feature-parity) Robustness Evaluation")
    model, X_base, y_true = _setup()

    results = {}
    base = _eval(model, X_base, y_true)
    results['baseline'] = base
    print(f"  Baseline              MAE={base['mae']:.2f}  R²={base['r2']:.4f}")

    feat_std = X_base.std(axis=0)

    print("\n  [1] Gaussian noise:")
    gn = {}
    for frac in [0.05, 0.10, 0.20, 0.30]:
        X_noisy = X_base.copy() + RNG.normal(0, frac * feat_std, size=X_base.shape)
        m = _eval(model, X_noisy, y_true)
        delta = m['mae'] - base['mae']
        gn[f"noise_{int(frac*100)}pct"] = {**m, "delta_mae": delta,
                                           "delta_pct": delta / base['mae'] * 100}
        print(f"    σ={frac*100:.0f}%  MAE={m['mae']:.2f}  Δ={delta:+.2f} ({delta/base['mae']*100:+.1f}%)")
    results['gaussian_noise'] = gn

    print("\n  [2] Feature dropout:")
    fd = {}
    for rate in [0.10, 0.20, 0.30]:
        X_drop = X_base.copy()
        X_drop[RNG.random(X_drop.shape) < rate] = 0.0
        m = _eval(model, X_drop, y_true)
        delta = m['mae'] - base['mae']
        fd[f"drop_{int(rate*100)}pct"] = {**m, "delta_mae": delta,
                                          "delta_pct": delta / base['mae'] * 100}
        print(f"    rate={rate*100:.0f}%  MAE={m['mae']:.2f}  Δ={delta:+.2f} ({delta/base['mae']*100:+.1f}%)")
    results['feature_dropout'] = fd

    print("\n  [3] Price spike simulation (5% of rows):")
    ps = {}
    n_spike = max(1, int(0.05 * X_base.shape[0]))
    spike_idx = RNG.choice(X_base.shape[0], size=n_spike, replace=False)
    for factor in [2.0, 3.0, 5.0]:
        X_spike = X_base.copy()
        X_spike[np.ix_(spike_idx, LAG_COLS)] *= factor
        m = _eval(model, X_spike, y_true)
        delta = m['mae'] - base['mae']
        ps[f"spike_{int(factor)}x"] = {**m, "delta_mae": delta,
                                       "delta_pct": delta / base['mae'] * 100}
        print(f"    factor={factor:.0f}×  MAE={m['mae']:.2f}  Δ={delta:+.2f} ({delta/base['mae']*100:+.1f}%)")
    results['price_spike'] = ps

    out = Path(__file__).parent.parent / "artifacts" / "xgboost_robustness_results.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved → {out}")
    return results


if __name__ == "__main__":
    run()
