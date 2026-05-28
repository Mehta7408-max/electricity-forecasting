# OVERWRITE EXACTLY: src/gat_train.py
"""
Training Loop for Multi-Head Heterogeneous Graph Attention Network (HeteroGAT).
Uses HeteroGATPriceForecaster which supports per-edge-type GATConv and
optional attention weight extraction.
"""
import torch
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

from hetero_config import DEVICE, ARTIFACTS_DIR, GRAPH_DIR
from hetero_models import HeteroGATPriceForecaster


def train_gat_pipeline():
    print("\n⚡ Initializing Multi-Head HeteroGAT Training Pipeline...")

    data = torch.load(GRAPH_DIR / "hetero_graph.pt", map_location=DEVICE, weights_only=False)
    num_hours = int(data['hour'].num_hours_per_zone)

    model = HeteroGATPriceForecaster(
        metadata=data.metadata(),
        hour_in_features=data['hour'].x.shape[1],
        hidden_channels=128,
        heads=4,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Architecture: HeteroGAT (4 heads) | Parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=False
    )

    # data['hour'].y is stored as raw DKK — no inverse transform needed
    y_raw = data['hour'].y.to(DEVICE)
    y_raw_np = data['hour'].y.cpu().numpy()

    train_mask = data['hour'].train_mask.to(DEVICE)
    val_mask   = data['hour'].val_mask.to(DEVICE)
    test_mask  = data['hour'].test_mask.to(DEVICE)

    x_dict         = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}

    best_val_loss = float('inf')
    patience_counter = 0
    GAT_CHECKPOINT = ARTIFACTS_DIR / "best_gat_model.pt"

    for epoch in range(1, 201):
        model.train()
        optimizer.zero_grad()
        out = model(x_dict, edge_index_dict, num_hours=num_hours).view(-1)
        loss = F.mse_loss(out[train_mask], y_raw[train_mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_out  = model(x_dict, edge_index_dict, num_hours=num_hours).view(-1)
            val_loss = F.mse_loss(val_out[val_mask], y_raw[val_mask]).item()
            val_mae  = F.l1_loss(val_out[val_mask], y_raw[val_mask]).item()

        scheduler.step(val_loss)

        if epoch == 1 or epoch % 20 == 0:
            print(f"   Epoch {epoch:3d} | Train MSE: {loss.item():8.2f} | Val MSE: {val_loss:8.2f} | Val MAE: {val_mae:5.2f} DKK")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), GAT_CHECKPOINT)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 30:
                print(f"   Early stopping at epoch {epoch}.")
                break

    print("\n📊 Evaluating on test set (best checkpoint)...")
    model.load_state_dict(torch.load(GAT_CHECKPOINT, map_location=DEVICE, weights_only=False))
    model.eval()

    with torch.no_grad():
        final_out = model(x_dict, edge_index_dict, num_hours=num_hours).view(-1).cpu().numpy()

    test_mask_np = test_mask.cpu().numpy()
    y_test   = y_raw_np[test_mask_np]
    pred_test = final_out[test_mask_np]

    smape = float(np.mean(
        2.0 * np.abs(pred_test - y_test) / (np.abs(y_test) + np.abs(pred_test) + 1e-8)
    ) * 100)

    metrics = {
        "mae":   float(mean_absolute_error(y_test, pred_test)),
        "rmse":  float(np.sqrt(np.mean((pred_test - y_test) ** 2))),
        "r2":    float(r2_score(y_test, pred_test)),
        "smape": smape,
    }

    print(f"   MAE   : {metrics['mae']:.4f} DKK")
    print(f"   RMSE  : {metrics['rmse']:.4f} DKK")
    print(f"   R²    : {metrics['r2']:.4f}")
    print(f"   SMAPE : {metrics['smape']:.2f}%")

    with open(ARTIFACTS_DIR / "gat_metrics_clean.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ GAT training complete. Metrics saved to {ARTIFACTS_DIR}/gat_metrics_clean.json")


if __name__ == "__main__":
    train_gat_pipeline()
