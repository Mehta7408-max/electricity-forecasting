"""
GNN Models - Graph Neural Network architectures for electricity price forecasting

Supported architectures:
- GCN (Graph Convolutional Network)
- GAT (Graph Attention Network)
- GraphSAGE
- GIN (Graph Isomorphism Network)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GINConv, global_mean_pool
from torch_geometric.nn import BatchNorm, LayerNorm


class GCNModel(nn.Module):
    """Graph Convolutional Network for regression."""
    
    def __init__(self, num_features, hidden_channels=128, num_layers=3, dropout=0.2):
        super(GCNModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input layer
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        self.convs.append(GCNConv(num_features, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))
        
        # Output projection
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))
        
        # Final regression layer
        self.fc = nn.Linear(hidden_channels, 1)
    
    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Regression output
        x = self.fc(x)
        return x.squeeze(-1)


class GATModel(nn.Module):
    """Graph Attention Network for regression."""
    
    def __init__(self, num_features, hidden_channels=128, num_layers=3, 
                 num_heads=4, dropout=0.2):
        super(GATModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(GATConv(num_features, hidden_channels // num_heads, 
                                  heads=num_heads, dropout=dropout))
        self.bns.append(BatchNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels // num_heads, 
                                      heads=num_heads, dropout=dropout))
            self.bns.append(BatchNorm(hidden_channels))
        
        # Output layer (single head for simplicity)
        self.convs.append(GATConv(hidden_channels, hidden_channels, 
                                  heads=1, dropout=dropout))
        self.bns.append(BatchNorm(hidden_channels))
        
        # Final regression layer
        self.fc = nn.Linear(hidden_channels, 1)
    
    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Regression output
        x = self.fc(x)
        return x.squeeze(-1)


class GraphSAGEModel(nn.Module):
    """GraphSAGE for regression."""
    
    def __init__(self, num_features, hidden_channels=128, num_layers=3, dropout=0.2):
        super(GraphSAGEModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        self.convs.append(SAGEConv(num_features, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.bns.append(BatchNorm(hidden_channels))
        
        # Output layer
        self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.bns.append(BatchNorm(hidden_channels))
        
        # Final regression layer
        self.fc = nn.Linear(hidden_channels, 1)
    
    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Regression output
        x = self.fc(x)
        return x.squeeze(-1)


class GINModel(nn.Module):
    """Graph Isomorphism Network for regression."""
    
    def __init__(self, num_features, hidden_channels=128, num_layers=3, dropout=0.2):
        super(GINModel, self).__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        # Input layer
        nn1 = nn.Sequential(
            nn.Linear(num_features, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(nn1))
        self.bns.append(BatchNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            nn_hidden = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Linear(hidden_channels, hidden_channels)
            )
            self.convs.append(GINConv(nn_hidden))
            self.bns.append(BatchNorm(hidden_channels))
        
        # Output layer
        nn_out = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.convs.append(GINConv(nn_out))
        self.bns.append(BatchNorm(hidden_channels))
        
        # Final regression layer
        self.fc = nn.Linear(hidden_channels, 1)
    
    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Regression output
        x = self.fc(x)
        return x.squeeze(-1)


def create_model(model_type: str, num_features: int, config: dict):
    """
    Factory function to create GNN models.
    
    Args:
        model_type: Type of model ('GCN', 'GAT', 'GraphSAGE', 'GIN').
        num_features: Number of input features.
        config: Model configuration dictionary.
    
    Returns:
        GNN model instance.
    """
    hidden_channels = config.get('hidden_channels', 128)
    num_layers = config.get('num_layers', 3)
    dropout = config.get('dropout', 0.2)
    
    if model_type == 'GCN':
        return GCNModel(num_features, hidden_channels, num_layers, dropout)
    
    elif model_type == 'GAT':
        num_heads = config.get('num_heads', 4)
        return GATModel(num_features, hidden_channels, num_layers, num_heads, dropout)
    
    elif model_type == 'GraphSAGE':
        return GraphSAGEModel(num_features, hidden_channels, num_layers, dropout)
    
    elif model_type == 'GIN':
        return GINModel(num_features, hidden_channels, num_layers, dropout)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation
    num_features = 19
    config = {
        'hidden_channels': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'num_heads': 4,
    }
    
    print("Testing model creation...")
    
    for model_type in ['GCN', 'GAT', 'GraphSAGE', 'GIN']:
        model = create_model(model_type, num_features, config)
        print(f"\n{model_type} Model:")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")