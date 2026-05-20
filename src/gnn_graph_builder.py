"""
Graph Construction Module - Build homogeneous temporal graph for GNN

Graph Structure:
- Nodes: Each hour is a node with features (time, price lags, weather)
- Edges: Temporal connections (1h, 2h, 6h, 24h, 168h ahead)
- Homogeneous: All nodes are the same type (hourly timesteps)
"""

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

from gnn_config import EDGE_TYPES, NODE_FEATURES, GRAPH_DIR, DEVICE


class TemporalGraphBuilder:
    """Build temporal graph from time series data with standardized features and targets."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()  # Added target scaler
        self.feature_cols = NODE_FEATURES
    
    def build_graph(self, df: pd.DataFrame, train_mask_ratio: float = 0.7, 
                   val_mask_ratio: float = 0.15) -> Data:
        print("\n🔗 Building temporal graph...")
        num_nodes = len(df)
        
        # ── Extract and Normalize Node Features ────────────────────
        X = df[self.feature_cols].values
        X_scaled = self.scaler.fit_transform(X)
        node_features = torch.FloatTensor(X_scaled)
        
        # ── Extract and Normalize Labels ───────────────────────────
        y = df["price_dkk"].values.reshape(-1, 1)
        
        # Crucial: Fit the target scaler ONLY on training data to prevent data leakage
        train_size = int(num_nodes * train_mask_ratio)
        self.target_scaler.fit(y[:train_size])
        
        y_scaled = self.target_scaler.transform(y).flatten()
        labels = torch.FloatTensor(y_scaled)
        
        # ── Build Edges & Masks (Keep your original code here) ──────
        edge_index = self._build_temporal_edges(num_nodes)
        
        val_size = int(num_nodes * val_mask_ratio)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[:train_size] = True
        val_mask[train_size:train_size + val_size] = True
        test_mask[train_size + val_size:] = True
        
        timestamps = df["hour_utc"].values
        
        data = Data(
            x=node_features,
            edge_index=edge_index,
            y=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
        )
        
        data.num_nodes = num_nodes
        data.timestamps = timestamps
        data.feature_names = self.feature_cols
        
        return data
    
    def _build_temporal_edges(self, num_nodes: int) -> torch.LongTensor:
        edge_list = []
        for edge_type, lag in EDGE_TYPES.items():
            for i in range(num_nodes - lag):
                edge_list.append([i, i + lag])
            for i in range(num_nodes - lag):
                edge_list.append([i + lag, i])
        return torch.LongTensor(np.array(edge_list).T)
    
    def save(self, path: Path):
        """Save both scalers."""
        with open(path, 'wb') as f:
            pickle.dump({'feature_scaler': self.scaler, 'target_scaler': self.target_scaler}, f)
        print(f"💾 Saved scalers to {path}")
    
    def load(self, path: Path):
        """Load a saved scaler."""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"📂 Loaded scaler from {path}")


def build_and_save_graph(df: pd.DataFrame, save_path: Path = None) -> Data:
    """
    Build graph from DataFrame and save it.
    
    Args:
        df: Feature DataFrame.
        save_path: Path to save the graph.
    
    Returns:
        PyTorch Geometric Data object.
    """
    builder = TemporalGraphBuilder()
    graph_data = builder.build_graph(df)
    
    if save_path is None:
        save_path = GRAPH_DIR / "temporal_graph.pt"
    
    # Save graph
    torch.save(graph_data, save_path)
    print(f"💾 Saved graph to {save_path}")
    
    # Save scaler
    scaler_path = save_path.parent / "scaler.pkl"
    builder.save(scaler_path)
    
    return graph_data


def load_graph(graph_path: Path = None) -> Data:
    """Load a saved graph."""
    if graph_path is None:
        graph_path = GRAPH_DIR / "temporal_graph.pt"
    
    graph_data = torch.load(graph_path)
    print(f"📂 Loaded graph from {graph_path}")
    
    return graph_data


if __name__ == "__main__":
    from gnn_feature_engineering import prepare_data_for_gnn
    
    # Prepare data
    df = prepare_data_for_gnn()
    
    if df is not None:
        # Build and save graph
        graph = build_and_save_graph(df)
        
        print("\n📊 Graph Statistics:")
        print(f"   Number of nodes: {graph.num_nodes}")
        print(f"   Number of edges: {graph.edge_index.shape[1]}")
        print(f"   Number of features: {graph.x.shape[1]}")
        print(f"   Train nodes: {graph.train_mask.sum().item()}")
        print(f"   Val nodes: {graph.val_mask.sum().item()}")
        print(f"   Test nodes: {graph.test_mask.sum().item()}")