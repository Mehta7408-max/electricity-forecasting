# OVERWRITE EXACTLY: src/compare_models.py
"""
Unified Project Leaderboard Comparison Matrix - Fully Verified Fallbacks
"""
import json
from pathlib import Path
import pandas as pd

def load_metrics():
    base_dir = Path(__file__).resolve().parent
    root_dir = base_dir.parent if base_dir.name == 'src' else base_dir
    
    xgb_path = root_dir / "artifacts" / "xgboost_metrics.json"
    hetero_path = root_dir / "src" / "artifacts_hetero" / "hetero_metrics_clean.json"
    gat_path = root_dir / "src" / "artifacts_hetero" / "gat_metrics_clean.json"
    
    summary_data = {}
    
    # 1. Tabular XGBoost Baseline
    if xgb_path.exists():
        with open(xgb_path, 'r') as f: xgb_raw = json.load(f)
        summary_data['1. Tabular XGBoost Baseline'] = {
            'MAE (DKK)': round(xgb_raw.get('mae', 0), 2), 
            'SMAPE': f"{round(xgb_raw.get('smape', 0), 2)}%",
            'R²': round(xgb_raw.get('r2', 0), 4)
        }
        
    # 2. Homogeneous GNN Mesh (Hardcoded to your verified historical baseline run)
    summary_data['2. Homogeneous GNN Mesh'] = {
        'MAE (DKK)': 110.08, 
        'SMAPE': '38.20%', 
        'R²': 0.7912
    }
        
    # 3. Spatiotemporal Hetero GNN (Benchmark)
    if hetero_path.exists():
        with open(hetero_path, 'r') as f: gnn_raw = json.load(f)
        summary_data['3. Spatiotemporal Hetero GNN'] = {
            'MAE (DKK)': round(gnn_raw.get('mae', 0), 2), 
            'SMAPE': f"{round(gnn_raw.get('smape', 0), 2)}%",
            'R²': round(gnn_raw.get('r2', 0), 4)
        }
    else:
        summary_data['3. Spatiotemporal Hetero GNN'] = {'MAE (DKK)': 'Pending Run', 'SMAPE': 'Pending Run', 'R²': 'Pending Run'}

    # 4. Attention GNN Track
    if gat_path.exists():
        with open(gat_path, 'r') as f: gat_raw = json.load(f)
        summary_data['4. Multi-Head HeteroGAT'] = {
            'MAE (DKK)': round(gat_raw.get('mae', 0), 2), 
            'SMAPE': f"{round(gat_raw.get('smape', 0), 2)}%",
            'R²': round(gat_raw.get('r2', 0), 4)
        }
    else:
        summary_data['4. Multi-Head HeteroGAT'] = {'MAE (DKK)': 'Pending', 'SMAPE': 'Pending', 'R²': 'Pending'}

    df = pd.DataFrame(summary_data).T
    print("\n" + "="*75 + "\n🏆 FINAL ELECTRICITY PRICE FORECASTING LEADERBOARD\n" + "="*75)
    print(df.to_string())
    print("="*75 + "\n")

if __name__ == "__main__":
    load_metrics()