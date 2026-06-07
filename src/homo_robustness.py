"""
Robustness evaluation for the homogeneous GraphSAGE model.

Mirrors src/st_robustness.py exactly — same RNG seed, same perturbation
families and intensities — so the degradation curves are directly comparable
to ST-HeteroSAGE:

  1. Gaussian noise   σ ∈ {5%, 10%, 20%, 30%} of each feature's std
  2. Feature dropout  randomly zero-out features at {10%, 20%, 30%}
  3. Price spike      multiply lag columns (0–4) by {2×, 3×, 5×} for 5% of nodes

All scenarios are inference-only — no retraining.
Results: artifacts/homo_robustness_results.json
"""
import sys, json, pickle
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from gnn_config import DEVICE, GRAPH_DIR, ARTIFACTS_DIR
from gnn_graph_builder import load_graph
from gnn_models import GraphSAGEModel

LAG_COLS = [0, 1, 2, 3, 4]   # price_lag_24h/48h/168h, roll_mean/std
RNG = np.random.default_rng(42)


def _setup():
    data = load_graph(GRAPH_DIR / "temporal_graph.pt")
    test_mask = data.test_mask  # already DK1+DK2 only

    with open(GRAPH_DIR / "scaler.pkl", "rb") as f:
        target_scaler = pickle.load(f)['target_scaler']

    num_feats = data.x.shape[1]
    model = GraphSAGEModel(num_features=num_feats, hidden_channels=128,
                           num_layers=3, dropout=0.0).to(DEVICE)
    model.load_state_dict(torch.load(
        ARTIFACTS_DIR / "best_homo_model.pt", map_location=DEVICE, weights_only=False))
    model.eval()

    x_base = data.x.numpy().copy()             # [4T, 17] scaled space
    y_scaled = data.y.numpy()
    y_raw = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    edge_index = data.edge_index.to(DEVICE)
    return model, edge_index, x_base, y_raw, target_scaler, test_mask


@torch.no_grad()
def _eval(model, x_perturbed, edge_index, y_raw, target_scaler, test_mask):
    x = torch.tensor(x_perturbed, dtype=torch.float32, device=DEVICE)
    out_scaled = model(x, edge_index).cpu().numpy()
    tm = test_mask.numpy()
    y_pred = target_scaler.inverse_transform(out_scaled[tm].reshape(-1, 1)).flatten()
    y_true = y_raw[tm]
    return {"mae": float(mean_absolute_error(y_true, y_pred)),
            "r2":  float(r2_score(y_true, y_pred))}


def run():
    print("\n🛡️  HomoGNN (GraphSAGE) Robustness Evaluation")
    model, edge_index, x_base, y_raw, target_scaler, test_mask = _setup()

    results = {}
    base = _eval(model, x_base, edge_index, y_raw, target_scaler, test_mask)
    results['baseline'] = base
    print(f"  Baseline              MAE={base['mae']:.2f}  R²={base['r2']:.4f}")

    tm = test_mask.numpy()
    feat_std = x_base[tm].std(axis=0)

    print("\n  [1] Gaussian noise:")
    gn = {}
    for frac in [0.05, 0.10, 0.20, 0.30]:
        x_noisy = x_base.copy() + RNG.normal(0, frac * feat_std, size=x_base.shape)
        m = _eval(model, x_noisy, edge_index, y_raw, target_scaler, test_mask)
        delta = m['mae'] - base['mae']
        gn[f"noise_{int(frac*100)}pct"] = {**m, "delta_mae": delta,
                                           "delta_pct": delta / base['mae'] * 100}
        print(f"    σ={frac*100:.0f}%  MAE={m['mae']:.2f}  Δ={delta:+.2f} ({delta/base['mae']*100:+.1f}%)")
    results['gaussian_noise'] = gn

    print("\n  [2] Feature dropout:")
    fd = {}
    for rate in [0.10, 0.20, 0.30]:
        x_drop = x_base.copy()
        x_drop[RNG.random(x_drop.shape) < rate] = 0.0
        m = _eval(model, x_drop, edge_index, y_raw, target_scaler, test_mask)
        delta = m['mae'] - base['mae']
        fd[f"drop_{int(rate*100)}pct"] = {**m, "delta_mae": delta,
                                          "delta_pct": delta / base['mae'] * 100}
        print(f"    rate={rate*100:.0f}%  MAE={m['mae']:.2f}  Δ={delta:+.2f} ({delta/base['mae']*100:+.1f}%)")
    results['feature_dropout'] = fd

    print("\n  [3] Price spike simulation (5% of nodes):")
    ps = {}
    n_spike = max(1, int(0.05 * x_base.shape[0]))
    spike_idx = RNG.choice(x_base.shape[0], size=n_spike, replace=False)
    for factor in [2.0, 3.0, 5.0]:
        x_spike = x_base.copy()
        x_spike[np.ix_(spike_idx, LAG_COLS)] *= factor
        m = _eval(model, x_spike, edge_index, y_raw, target_scaler, test_mask)
        delta = m['mae'] - base['mae']
        ps[f"spike_{int(factor)}x"] = {**m, "delta_mae": delta,
                                       "delta_pct": delta / base['mae'] * 100}
        print(f"    factor={factor:.0f}×  MAE={m['mae']:.2f}  Δ={delta:+.2f} ({delta/base['mae']*100:+.1f}%)")
    results['price_spike'] = ps

    out = ARTIFACTS_DIR / "homo_robustness_results.json"
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Saved → {out}")
    return results


if __name__ == "__main__":
    run()
