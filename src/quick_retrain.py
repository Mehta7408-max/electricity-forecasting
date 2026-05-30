"""
Quick retrain script — trains HeteroPriceForecaster on the current graph
for up to 120 epochs with validation-based early stopping.
Overwrites best_hetero_model.pt when complete.
"""
import sys, json, time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from hetero_config import DEVICE, ARTIFACTS_DIR, GRAPH_DIR
from hetero_models import HeteroPriceForecaster
from mlflow_config import setup_mlflow


def quick_retrain(hidden_channels=64, max_epochs=120, patience=20):
    print("\n⚡ Quick retrain — HeteroPriceForecaster")
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── MLflow setup ───────────────────────────────────────────────────────────
    try:
        setup_mlflow()
        import mlflow
        _mlflow_run = mlflow.start_run(run_name="HeteroSAGE")
        _mlflow_ok = True
    except Exception as _e:
        print(f"   [MLflow] Disabled — {_e}")
        _mlflow_ok = False

    try:
        # ── Log hyperparameters ────────────────────────────────────────────────
        if _mlflow_ok:
            try:
                mlflow.log_params({
                    "hidden_channels": hidden_channels,
                    "max_epochs": max_epochs,
                    "patience": patience,
                    "lr": 0.002,
                    "weight_decay": 1e-4,
                })
            except Exception:
                pass

        data = torch.load(GRAPH_DIR / "hetero_graph.pt", map_location=DEVICE, weights_only=False)
        num_hours = int(data['hour'].num_hours_per_zone)

        model = HeteroPriceForecaster(
            metadata=data.metadata(),
            hour_in_features=data['hour'].x.shape[1],
            hidden_channels=hidden_channels,
        ).to(DEVICE)

        params = sum(p.numel() for p in model.parameters())
        print(f"   Params: {params:,} | hidden_channels={hidden_channels} | {num_hours} hours/zone")

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8
        )

        x_dict  = {k: v.to(DEVICE) for k, v in data.x_dict.items()}
        ei      = {k: v.to(DEVICE) for k, v in data.edge_index_dict.items()}
        y       = data['hour'].y.to(DEVICE)
        y_np    = data['hour'].y.cpu().numpy()
        tr_mask = data['hour'].train_mask.to(DEVICE)
        vl_mask = data['hour'].val_mask.to(DEVICE)
        te_mask = data['hour'].test_mask.to(DEVICE)

        ckpt_path = ARTIFACTS_DIR / "best_hetero_model.pt"
        best_val  = float('inf')
        patience_ctr = 0
        t0 = time.time()

        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            out  = model(x_dict, ei, num_hours=num_hours).view(-1)
            loss = F.mse_loss(out[tr_mask], y[tr_mask])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_out  = model(x_dict, ei, num_hours=num_hours).view(-1)
                val_loss = F.mse_loss(val_out[vl_mask], y[vl_mask]).item()
                val_mae  = F.l1_loss(val_out[vl_mask], y[vl_mask]).item()

            scheduler.step(val_loss)

            # Log per-epoch metrics to MLflow
            if _mlflow_ok:
                try:
                    mlflow.log_metric("train_mse", loss.item(), step=epoch)
                    mlflow.log_metric("val_mae", val_mae, step=epoch)
                except Exception:
                    pass

            if epoch == 1 or epoch % 10 == 0:
                elapsed = time.time() - t0
                print(f"   Epoch {epoch:3d} | Train MSE: {loss.item():10.1f} | Val MAE: {val_mae:6.1f} DKK | {elapsed:.0f}s")

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), ckpt_path)
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= patience:
                    print(f"   Early stopping at epoch {epoch}")
                    break

        # ── Final test evaluation ──────────────────────────────────────────────
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=False))
        model.eval()
        with torch.no_grad():
            final = model(x_dict, ei, num_hours=num_hours).view(-1).cpu().numpy()

        tm_np = te_mask.cpu().numpy()
        y_te  = y_np[tm_np]
        p_te  = final[tm_np]

        smape = float(np.mean(2 * np.abs(p_te - y_te) / (np.abs(y_te) + np.abs(p_te) + 1e-8)) * 100)
        metrics = {
            "mae":   float(mean_absolute_error(y_te, p_te)),
            "rmse":  float(np.sqrt(np.mean((p_te - y_te)**2))),
            "r2":    float(r2_score(y_te, p_te)),
            "smape": smape,
        }
        print(f"\n📊 Test Results:")
        print(f"   MAE:   {metrics['mae']:.2f} DKK")
        print(f"   RMSE:  {metrics['rmse']:.2f} DKK")
        print(f"   R²:    {metrics['r2']:.4f}")
        print(f"   SMAPE: {metrics['smape']:.2f}%")

        # Log final test metrics and checkpoint artifact
        if _mlflow_ok:
            try:
                mlflow.log_metrics({
                    "test_mae":   metrics["mae"],
                    "test_rmse":  metrics["rmse"],
                    "test_r2":    metrics["r2"],
                    "test_smape": metrics["smape"],
                })
                mlflow.log_artifact(str(ckpt_path))
            except Exception:
                pass

        with open(ARTIFACTS_DIR / "hetero_metrics_clean.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"\n✅ Checkpoint saved → {ckpt_path}")
        return metrics

    finally:
        if _mlflow_ok:
            try:
                mlflow.end_run()
            except Exception:
                pass


if __name__ == "__main__":
    quick_retrain()
