# OVERWRITE EXACTLY: src/hetero_graph_builder.py
"""
Advanced Graph Builder for Heterogeneous Multi-Area Electricity Forecasting.
Implements Cyclical Calendar Profiles and Weighted Spatial Interconnect Features.
"""
import sys
import torch
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import HeteroData

# Clean, working configuration imports
from hetero_config import GRAPH_DIR
from hetero_pipeline import prepare_multi_area_data 

def build_heterogeneous_spatiotemporal_graph(freeze_scaler=False):
    """
    Build the heterogeneous market graph.

    freeze_scaler:
        False (default) — fit fresh StandardScalers on the current training
            partition. Use for a full retrain from scratch.
        True — reuse the previously saved feature/target scalers from
            hetero_scalers.pkl instead of refitting. This keeps the input
            feature distribution fixed across incremental graph rebuilds so
            that warm-started model weights remain valid. Falls back to
            fitting fresh scalers if no saved pickle exists.
    """
    print("\n🏗️ Building Context-Aware Heterogeneous Market Graph...")
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Fetch Leakage-Proof Clean Data Frames (Bypasses manual DB mapping)
    df_dk1, df_dk2, df_hydro, df_de, _ = prepare_multi_area_data()
    
    # Sort to enforce chronological timeline validation integrity
    df_dk1 = df_dk1.sort_values('timestamp').reset_index(drop=True)
    df_dk2 = df_dk2.sort_values('timestamp').reset_index(drop=True)
    df_de = df_de.sort_values('timestamp').reset_index(drop=True)
    df_hydro = df_hydro.sort_values('timestamp').reset_index(drop=True)
    
    num_hours = len(df_dk1)
    
    # 2. Extract and Align Target Labels Matrix
    y_dk1 = df_dk1['price_dkk'].values.astype(np.float32)
    y_dk2 = df_dk2['price_dkk'].values.astype(np.float32)
    y_de = df_de['price_dkk'].values.astype(np.float32)
    y_hydro = df_hydro['price_dkk'].values.astype(np.float32)
    
    # 3. Feature Engineering: Cyclical Calendar Node Profiles
    timestamps = pd.to_datetime(df_dk1['timestamp'])
    hour_of_day = timestamps.dt.hour.values
    day_of_week = timestamps.dt.dayofweek.values
    
    hour_sin = np.sin(2 * np.pi * hour_of_day / 24.0)
    hour_cos = np.cos(2 * np.pi * hour_of_day / 24.0)
    week_sin = np.sin(2 * np.pi * day_of_week / 7.0)
    week_cos = np.cos(2 * np.pi * day_of_week / 7.0)
    
    def extract_features(df):
        cols = [
            'price_lag_1h', 'price_lag_2h', 'price_lag_6h',
            'price_rolling_6h_mean', 'price_rolling_6h_std',
            'temperature_c', 'wind_speed_ms', 'cloud_cover_pct', 'humidity_pct'
        ]
        # Return 0 defaults if weather variables aren't present in specific zone views
        for c in cols:
            if c not in df.columns:
                df[c] = 0.0
        return df[cols].values.astype(np.float32)

    x_dk1_base = extract_features(df_dk1)
    x_dk2_base = extract_features(df_dk2)
    x_de_base = extract_features(df_de)
    x_hydro_base = extract_features(df_hydro)
    
    # Append cyclical components directly to the feature arrays
    cyclical_stack = np.stack([hour_sin, hour_cos, week_sin, week_cos], axis=1).astype(np.float32)
    x_dk1 = np.hstack([x_dk1_base, cyclical_stack])
    x_dk2 = np.hstack([x_dk2_base, cyclical_stack])
    x_de = np.hstack([x_de_base, cyclical_stack])
    x_hydro = np.hstack([x_hydro_base, cyclical_stack])

    # 4. Fit Scalers sequentially to avoid cross-boundary distribution contamination
    train_idx_limit = int(num_hours * 0.8)

    scalers_path = GRAPH_DIR / "hetero_scalers.pkl"
    if freeze_scaler and scalers_path.exists():
        # Reuse previously fitted scalers so the feature distribution stays
        # fixed — required for stable warm-start fine-tuning on new data.
        with open(scalers_path, "rb") as f:
            _saved = pickle.load(f)
        feature_scaler = _saved['feature_scaler']
        target_scaler = _saved['target_scaler']
        print("   🔒 Frozen scaler mode — reusing saved feature/target scalers.")
    else:
        if freeze_scaler:
            print("   ⚠️  freeze_scaler requested but no saved scaler found — fitting fresh.")
        feature_scaler = StandardScaler()
        feature_scaler.fit(np.vstack([x_dk1[:train_idx_limit], x_dk2[:train_idx_limit], x_de[:train_idx_limit], x_hydro[:train_idx_limit]]))

        target_scaler = StandardScaler()
        target_scaler.fit(np.hstack([y_dk1[:train_idx_limit], y_dk2[:train_idx_limit]]).reshape(-1, 1))
    
    # 5. Build PyTorch Geometric HeteroData Object
    data = HeteroData()
    
    # Scale inputs and assign node representations
    x_all = np.vstack([
        feature_scaler.transform(x_dk1),
        feature_scaler.transform(x_dk2),
        feature_scaler.transform(x_hydro),
        feature_scaler.transform(x_de)
    ])
    y_all = np.hstack([y_dk1, y_dk2, y_hydro, y_de])
    
    data['hour'].x = torch.tensor(x_all, dtype=torch.float32)
    data['hour'].y = torch.tensor(y_all, dtype=torch.float32) # Stored raw for exact cash error evaluation

    # Set up localized Market Node static profiling matrix (4 Zones)
    # Mapping index sequence: 0: DK1, 1: DK2, 2: HYDRO, 3: DE
    data['market'].x = torch.eye(4, dtype=torch.float32)
    
    # 6. Establish Structural Graph Relationships (Edges)
    belongs_src = []
    belongs_dst = []
    for zone_idx in range(4):
        start_offset = zone_idx * num_hours
        belongs_src.extend(list(range(start_offset, start_offset + num_hours)))
        belongs_dst.extend([zone_idx] * num_hours)
        
    data['hour', 'belongs_to', 'market'].edge_index = torch.tensor([belongs_src, belongs_dst], dtype=torch.long)
    data['market', 'rev_belongs_to', 'hour'].edge_index = torch.tensor([belongs_dst, belongs_src], dtype=torch.long)
    
    # Chronological Autoregressive Lag Edges (Hour_t-1 connects to Hour_t)
    lag_src = []
    lag_dst = []
    for zone_idx in range(4):
        offset = zone_idx * num_hours
        for t in range(1, num_hours):
            lag_src.append(offset + (t - 1))
            lag_dst.append(offset + t)
            
    data['hour', 'lag_to', 'hour'].edge_index = torch.tensor([lag_src, lag_dst], dtype=torch.long)
    
    # Cross-Border Spatial Grid Interconnects (Market-to-Market Topology)
    # 0: DK1, 1: DK2, 2: HYDRO, 3: DE
    inter_src = [0, 1, 0, 3, 0, 2] # Two-way transmission paths
    inter_dst = [1, 0, 3, 0, 2, 0]
    
    # Physical Capacity Weight Matrix (Asymmetric size vectors in Megawatts)
    inter_weights = [1000.0, 1000.0, 2000.0, 1500.0, 600.0, 600.0]
    
    data['market', 'interconnects', 'market'].edge_index = torch.tensor([inter_src, inter_dst], dtype=torch.long)
    data['market', 'interconnects', 'market'].edge_attr = torch.tensor(inter_weights, dtype=torch.float32).view(-1, 1)
    
    # 7. Construct Non-Overlapping Spatiotemporal Validation Masks
    train_mask = torch.zeros(4 * num_hours, dtype=torch.bool)
    val_mask = torch.zeros(4 * num_hours, dtype=torch.bool)
    test_mask = torch.zeros(4 * num_hours, dtype=torch.bool)
    
    val_idx_limit = int(num_hours * 0.9)
    
    for zone_idx in range(4):
        offset = zone_idx * num_hours
        train_mask[offset : offset + train_idx_limit] = True
        val_mask[offset + train_idx_limit : offset + val_idx_limit] = True
        test_mask[offset + val_idx_limit : offset + num_hours] = True
        
    data['hour'].train_mask = train_mask
    data['hour'].val_mask = val_mask
    data['hour'].test_mask = test_mask
    
    # Keep track of individual zone sequence dimensions for our readout layer splits
    data['hour'].num_hours_per_zone = num_hours
    
    # Save elements out
    torch.save(data, GRAPH_DIR / "hetero_graph.pt")
    with open(GRAPH_DIR / "hetero_scalers.pkl", "wb") as f:
        pickle.dump({'feature_scaler': feature_scaler, 'target_scaler': target_scaler}, f)
        
    print(f"✅ Success! HeteroData artifact package compiled. Graph nodes: {data['hour'].x.shape[0]} entities.")

if __name__ == "__main__":
    freeze = "--freeze-scaler" in sys.argv
    build_heterogeneous_spatiotemporal_graph(freeze_scaler=freeze)