# OVERWRITE EXACTLY: src/gnn_graph_builder.py
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

from gnn_config import EDGE_TYPES, GRAPH_DIR
from hetero_pipeline import prepare_multi_area_data


class TemporalGraphBuilder:
    """Builds a flat, uniform spatial-temporal mesh network for fair benchmarking."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.zone_to_idx = {'DK1': 0, 'DK2': 1, 'DE': 2, 'HYDRO': 3}
        
    def _extract_features(self, row: pd.Series, feature_cols: list) -> np.ndarray:
        """Safely extracts features from a series, filling missing columns with 0.0."""
        vec = []
        for col in feature_cols:
            if col in row.index:
                val = row[col]
                # Guard against any random NaN values that sneaked through the outer join
                vec.append(0.0 if pd.isna(val) else float(val))
            else:
                vec.append(0.0) # Fallback for DE/HYDRO lacking synthetic weather attributes
        return np.array(vec, dtype=np.float32)

    def build_graph(self, train_mask_ratio: float = 0.8, val_mask_ratio: float = 0.1) -> Data:
        print("\n🔗 Constructing Multi-Zone Spatial-Temporal Mesh Graph...")
        
        df_dk1, df_dk2, df_hydro, df_de, _ = prepare_multi_area_data()
        num_hours = len(df_dk1)
        
        # 4 zones per timestep = total nodes in the system matrix
        num_nodes = num_hours * 4
        
        feature_cols = [
            'hour_of_day', 'price_lag_1h', 'price_lag_2h', 'price_lag_6h',
            'price_rolling_6h_mean', 'price_rolling_6h_std',
            'temperature_c', 'wind_speed_ms', 'cloud_cover_pct'
        ]
        
        # --- 1. Assemble the Flat Feature Map ---
        raw_x = np.zeros((num_nodes, len(feature_cols)), dtype=np.float32)
        raw_y = np.zeros((num_nodes, 1), dtype=np.float32)
        
        for t in range(num_hours):
            base_idx = t * 4
            raw_x[base_idx + 0] = self._extract_features(df_dk1.iloc[t], feature_cols)
            raw_x[base_idx + 1] = self._extract_features(df_dk2.iloc[t], feature_cols)
            raw_x[base_idx + 2] = self._extract_features(df_de.iloc[t], feature_cols)
            raw_x[base_idx + 3] = self._extract_features(df_hydro.iloc[t], feature_cols)
            
            raw_y[base_idx + 0] = float(df_dk1.iloc[t]['price_dkk']) if 'price_dkk' in df_dk1.columns else 0.0
            raw_y[base_idx + 1] = float(df_dk2.iloc[t]['price_dkk']) if 'price_dkk' in df_dk2.columns else 0.0
            raw_y[base_idx + 2] = float(df_de.iloc[t]['price_dkk']) if 'price_dkk' in df_de.columns else 0.0
            raw_y[base_idx + 3] = float(df_hydro.iloc[t]['price_dkk']) if 'price_dkk' in df_hydro.columns else 0.0
            
        # Scale and normalize features
        X_scaled = self.scaler.fit_transform(raw_x)
        
        # Fit target scale strictly on the training partition
        train_hours = int(num_hours * train_mask_ratio)
        self.target_scaler.fit(raw_y[:train_hours * 4])
        Y_scaled = self.target_scaler.transform(raw_y).flatten()
        
        # --- 2. Construct Spatial and Temporal Edges ---
        edge_list = []
        
        # Spatial interconnections (run for every hour step)
        for t in range(num_hours):
            b = t * 4
            # Bidirectional grid links: DK1<->DK2, DK1<->DE, DK2<->DE, DK2<->HYDRO
            spatial_pairs = [(0,1), (1,0), (0,2), (2,0), (1,2), (2,1), (1,3), (3,1)]
            for src, tgt in spatial_pairs:
                edge_list.append([b + src, b + tgt])
                
        # Temporal lags (connect across hours for the same zone)
        for t in range(num_hours):
            for edge_type, lag in EDGE_TYPES.items():
                if t >= lag:
                    old_b = (t - lag) * 4
                    new_b = t * 4
                    for z in range(4):  # Pass historical context to all 4 zones
                        edge_list.append([old_b + z, new_b + z])
                        edge_list.append([new_b + z, old_b + z])
                        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        # --- 3. Build Node Mask Allocations ---
        val_hours = int(num_hours * val_mask_ratio)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        # Train mask includes all 4 nodes during the training block
        train_mask[:train_hours * 4] = True
        
        # Validation and testing apply strictly to our core target areas (DK1 and DK2)
        for t in range(train_hours, train_hours + val_hours):
            val_mask[t * 4 + 0] = True
            val_mask[t * 4 + 1] = True
        for t in range(train_hours + val_hours, num_hours):
            test_mask[t * 4 + 0] = True
            test_mask[t * 4 + 1] = True
            
        data = Data(
            x=torch.FloatTensor(X_scaled),
            edge_index=edge_index,
            y=torch.FloatTensor(Y_scaled),
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
        data.num_nodes = num_nodes
        return data

    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump({'feature_scaler': self.scaler, 'target_scaler': self.target_scaler}, f)


def load_graph(graph_path: Path = None) -> Data:
    """Load a saved graph."""
    if graph_path is None:
        graph_path = GRAPH_DIR / "temporal_graph.pt"
    graph_data = torch.load(graph_path)
    return graph_data


def build_and_save_graph() -> Data:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    builder = TemporalGraphBuilder()
    graph_data = builder.build_graph()
    torch.save(graph_data, GRAPH_DIR / "temporal_graph.pt")
    builder.save(GRAPH_DIR / "scaler.pkl")
    print("✅ Stacked spatial-temporal mesh compiled successfully.")
    return graph_data

if __name__ == "__main__":
    build_and_save_graph()