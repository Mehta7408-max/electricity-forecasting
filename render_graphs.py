"""
Driver to render graphs from the repo's stored artifacts using gnn_visualization.
Saves PNGs to artifacts/figures/. Headless-safe (Agg backend).
"""
import json
import sys
import types
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).parent
OUT = ROOT / "artifacts" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# gnn_config imports torch (only needed for training/inference). Stub it so the
# pure-matplotlib plotting functions in gnn_visualization import cleanly.
_stub = types.ModuleType("gnn_config")
_stub.ARTIFACTS_DIR = OUT
sys.modules["gnn_config"] = _stub

import gnn_visualization as viz  # noqa: E402

sns.set_style("whitegrid")


def load(p):
    with open(p) as f:
        return json.load(f)


# 1) Training history. NOTE: the stored file has per-epoch lists for *_loss and
# train_mae, but val_mae/val_rmse/val_r2 are saved as final scalars only, so the
# library's plot_training_history (which assumes val_mae is a list) can't be used
# as-is. Plot the valid per-epoch curves directly.
hist = load(ROOT / "src" / "artifacts" / "training_history.json")
epochs = range(1, len(hist["train_loss"]) + 1)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(epochs, hist["train_loss"], label="Train Loss", linewidth=2)
axes[0].plot(epochs, hist["val_loss"], label="Val Loss", linewidth=2)
axes[0].set_title("Training & Validation Loss", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss (MSE)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(epochs, hist["train_mae"], label="Train MAE", linewidth=2, color="#1f77b4")
axes[1].axhline(hist["val_mae"], color="#ff7f0e", linestyle="--", linewidth=2,
                label=f"Final Val MAE = {hist['val_mae']:.1f}")
axes[1].set_title("Training MAE (val final R²={:.3f})".format(hist["val_r2"]),
                  fontsize=13, fontweight="bold")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE")
axes[1].legend(); axes[1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "01_training_history.png", dpi=200, bbox_inches="tight")
plt.close("all")

# 2) Model comparison across all trained models
models = {
    "ST-HeteroSAGE": load(ROOT / "src/artifacts_hetero/st_hetero_metrics.json"),
    "Hetero SAGE":   load(ROOT / "src/artifacts_hetero/hetero_metrics_clean.json"),
    "Homo SAGE":     load(ROOT / "src/artifacts/homo_gnn_metrics.json"),
    "GAT":           load(ROOT / "src/artifacts_hetero/gat_metrics_clean.json"),
    "XGBoost":       load(ROOT / "artifacts/xgboost_metrics.json"),
}
names = list(models)
specs = [("mae", "MAE (DKK/MWh)", False),
         ("rmse", "RMSE (DKK/MWh)", False),
         ("r2", "R²", True),
         ("smape", "sMAPE (%)", False)]
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
colors = ["#2ca02c", "#1f77b4", "#17becf", "#ff7f0e", "#d62728"]
for ax, (key, label, higher_better) in zip(axes.flatten(), specs):
    vals = [models[m].get(key, np.nan) for m in names]
    bars = ax.bar(names, vals, color=colors, alpha=0.85, edgecolor="black")
    ax.set_title(label + ("  (higher = better)" if higher_better else "  (lower = better)"),
                 fontsize=12, fontweight="bold")
    ax.set_ylabel(label)
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, alpha=0.3, axis="y")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}",
                ha="center", va="bottom", fontsize=9)
fig.suptitle("Model Comparison (DK1+DK2 test set)", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT / "02_model_comparison.png", dpi=200, bbox_inches="tight")
plt.close("all")

# 3) Day-ahead actual vs predicted hourly price profiles (DK1, DK2)
da = load(ROOT / "src/artifacts_hetero/day_ahead_results.json")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, zone in zip(axes, ["DK1", "DK2"]):
    by_hour = da["price_profiles"][zone]["by_hour"]
    hours = sorted(by_hour, key=int)
    actual = [by_hour[h]["actual"] for h in hours]
    pred = [by_hour[h]["predicted"] for h in hours]
    x = [int(h) for h in hours]
    ax.plot(x, actual, marker="o", label="Actual", linewidth=2)
    ax.plot(x, pred, marker="s", label="Predicted", linewidth=2, alpha=0.8)
    ax.set_title(f"{zone} – Avg Price Profile by Hour", fontsize=13, fontweight="bold")
    ax.set_xlabel("Hour of Day"); ax.set_ylabel("Price (DKK/MWh)")
    ax.set_xticks(range(0, 24, 2)); ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "03_dayahead_profiles.png", dpi=200, bbox_inches="tight")
plt.close("all")

# 4) Per-horizon MAE (24h ahead) for DK zones
fig, ax = plt.subplots(figsize=(13, 5))
for zone in ["DK1", "DK2"]:
    ax.plot(range(1, 25), da["horizon_mae"][zone], marker="o", label=zone, linewidth=2)
ax.set_title("Day-Ahead Forecast: MAE by Horizon Step", fontsize=13, fontweight="bold")
ax.set_xlabel("Forecast horizon (hours ahead)"); ax.set_ylabel("MAE (DKK/MWh)")
ax.set_xticks(range(1, 25)); ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "04_horizon_mae.png", dpi=200, bbox_inches="tight")
plt.close("all")

# 5) ST-HeteroSAGE ablation
abl = load(ROOT / "src/artifacts_hetero/st_ablation_results.json")
labels = {"A_full": "Full model", "B_no_tcn": "No TCN (temporal)",
          "C_no_spatial": "No spatial", "D_no_cooccurs": "No co-occurrence",
          "E_no_market": "No market", "F_no_hydro": "No hydro"}
keys = list(abl)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5))
mae_vals = [abl[k]["mae"] for k in keys]
r2_vals = [abl[k]["r2"] for k in keys]
disp = [labels.get(k, k) for k in keys]
a1.bar(disp, mae_vals, color="#1f77b4", alpha=0.85, edgecolor="black")
a1.set_title("Ablation – MAE (lower better)", fontsize=12, fontweight="bold")
a1.set_ylabel("MAE"); a1.tick_params(axis="x", rotation=25)
a1.grid(True, alpha=0.3, axis="y")
a2.bar(disp, r2_vals, color="#2ca02c", alpha=0.85, edgecolor="black")
a2.axhline(0, color="r", linestyle="--")
a2.set_title("Ablation – R² (higher better)", fontsize=12, fontweight="bold")
a2.set_ylabel("R²"); a2.tick_params(axis="x", rotation=25)
a2.grid(True, alpha=0.3, axis="y")
fig.suptitle("ST-HeteroSAGE Ablation Study", fontsize=15, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "05_ablation.png", dpi=200, bbox_inches="tight")
plt.close("all")

# 6) Feature importance (homogeneous interpretability, DK1 & DK2)
interp = load(ROOT / "src/artifacts_hetero/interpretability_summary.json")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, zone in zip(axes, ["DK1", "DK2"]):
    fi = interp["feature_importance"][zone]
    items = sorted(fi.items(), key=lambda kv: kv[1])
    feats = [k for k, _ in items]; vals = [v for _, v in items]
    ax.barh(feats, vals, color="#9467bd", alpha=0.85, edgecolor="black")
    ax.set_title(f"{zone} – Feature Importance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Relative importance")
    ax.grid(True, alpha=0.3, axis="x")
fig.tight_layout()
fig.savefig(OUT / "06_feature_importance.png", dpi=200, bbox_inches="tight")
plt.close("all")

# 7) Robustness: MAE degradation under perturbations (ST model)
rob = load(ROOT / "src/artifacts_hetero/st_robustness_results.json")
fig, ax = plt.subplots(figsize=(12, 5))
noise = rob["gaussian_noise"]; drop = rob["feature_dropout"]; spike = rob["price_spike"]
nx = [5, 10, 20, 30]
ax.plot(nx, [noise[f"noise_{p}pct"]["delta_pct"] for p in nx], marker="o", label="Gaussian noise")
ax.plot([10, 20, 30], [drop[f"drop_{p}pct"]["delta_pct"] for p in [10, 20, 30]], marker="s", label="Feature dropout")
ax.plot([2, 3, 5], [spike[f"spike_{p}x"]["delta_pct"] for p in [2, 3, 5]], marker="^", label="Price spike (x)")
ax.set_title("ST-HeteroSAGE Robustness: MAE Degradation", fontsize=13, fontweight="bold")
ax.set_xlabel("Perturbation magnitude"); ax.set_ylabel("MAE increase vs baseline (%)")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "07_robustness.png", dpi=200, bbox_inches="tight")
plt.close("all")

print("Saved figures:")
for p in sorted(OUT.glob("*.png")):
    print(" ", p.relative_to(ROOT))
