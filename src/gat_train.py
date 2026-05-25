# OVERWRITE EXACTLY: src/gat_train.py
"""
Training Loop for Multi-Head Heterogeneous Graph Attention Network (HeteroGAT)
"""
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

from hetero_config import DEVICE, ARTIFACTS_DIR, GRAPH_DIR
from hetero_models import HeteroPriceForecaster

def train_gat_pipeline():
    print("\n⚡ Initializing Multi-Head HeteroGAT Training Pipeline...")
    
    # 1. Load Data Elements
    data = torch.load(GRAPH_DIR / "hetero_graph.pt", map_location=DEVICE)
    with open(GRAPH_DIR / "hetero_scalers.pkl", "rb") as f:
        scalers = pickle.load(f)
    target_scaler = scalers['target_scaler']

    # 2. Instantiate Model with 4 Attention Heads
    model = HeteroPriceForecaster(
        metadata=data.metadata(),
        hour_in_features=data['hour'].x.shape[1],
        hidden_channels=128,
        heads=4
    ).to(DEVICE)

    print(f"   Architecture: GATConv Multi-Head Mesh | Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Precise learning rate and weight decay for attention weight matrices
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)

    # Raw DKK Scaling Setup
    mean_val = target_scaler.mean_[0]
    scale_val = target_scaler.scale_[0]
    y_scaled = data['hour'].y.cpu().numpy().reshape(-1, 1)
    y_raw_np = target_scaler.inverse_transform(y_scaled).flatten()
    y_raw = torch.tensor(y_raw_np, dtype=torch.float32, device=DEVICE)

    train_mask = data['hour'].train_mask
    val_mask = data['hour'].val_mask
    test_mask = data['hour'].test_mask

    best_val_loss = float('inf')
    patience_counter = 0
    max_epochs = 200
    early_stopping_patience = 30
    
    GAT_CHECKPOINT = ARTIFACTS_DIR / "best_gat_model.pt"

    # 3. Training Loop
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        out_scaled = model(data.x_dict, data.edge_index_dict).view(-1)
        out_raw = out_scaled * scale_val + mean_val
        
        loss = F.mse_loss(out_raw[train_mask], y_raw[train_mask])
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Validation Loop
        model.eval()
        with torch.no_grad():
            val_out_scaled = model(data.x_dict, data.edge_index_dict).view(-1)
            val_out_raw = val_out_scaled * scale_val + mean_val
            val_loss = F.mse_loss(val_out_raw[val_mask], y_raw[val_mask]).item()
            val_mae = F.l1_loss(val_out_raw[val_mask], y_raw[val_mask]).item()

        scheduler.step(val_loss)

        if epoch == 1 or epoch % 10 == 0:
            print(f"   Epoch {epoch:3d} | Train MSE: {loss.item():8.2f} | Val MSE: {val_loss:8.2f} | Val MAE: {val_mae:5.2f} DKK")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), GAT_CHECKPOINT)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f"🛑 Early stopping triggered at epoch {epoch}.")
                break

    # 4. Final Out-of-Sample Test Evaluation
    print("\n📊 Executing Real-Scale Out-of-Sample Evaluation Matrix for GAT...")
    model.load_state_dict(torch.load(GAT_CHECKPOINT))
    model.eval()
    
    with torch.no_grad():
        final_out_scaled = model(data.x_dict, data.edge_index_dict).view(-1)
        final_out_raw = (final_out_scaled * scale_val + mean_val).cpu().numpy()
        
    y_test_raw = y_raw_np[test_mask.cpu().numpy()]
    predictions_test = final_out_raw[test_mask.cpu().numpy()]

    smape = np.mean(2.0 * np.abs(predictions_test - y_test_raw) / (np.abs(y_test_raw) + np.abs(predictions_test) + 1e-8)) * 100

    metrics = {
        "mae": float(mean_absolute_error(y_test_raw, predictions_test)),
        "rmse": float(np.sqrt(np.mean((predictions_test - y_test_raw) ** 2))),
        "r2": float(r2_score(y_test_raw, predictions_test)),
        "smape": float(smape)
    }

    print("\n📈 Multi-Head HeteroGAT Test Results:")
    print(f"   MAE   : {metrics['mae']:.4f} DKK")
    print(f"   RMSE  : {metrics['rmse']:.4f} DKK")
    print(f"   R²    : {metrics['r2']:.4f}")
    print(f"   SMAPE : {metrics['smape']:.2f}%")

    # Write output to its dedicated tracking path
    out_dir = Path("src/artifacts_hetero")
    with open(out_dir / "gat_metrics_clean.json", "w") as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    train_gat_pipeline()