# OVERWRITE EXACTLY: src/simple_hetero_gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero

class FourAreaBaseModel(nn.Module):
    def __init__(self, hidden_channels, dropout=0.2):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Branching outputs tracking our four layout node keys
        self.out_dk1 = nn.Linear(hidden_channels, 1)
        self.out_dk2 = nn.Linear(hidden_channels, 1)
        self.out_hydro = nn.Linear(hidden_channels, 1)
        self.out_de = nn.Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {k: self.dropout(F.relu(x)) for k, x in x_dict.items()}
        
        x_dict = self.conv2(x_dict, edge_index_dict)
        x_dict = {k: F.relu(x) for k, x in x_dict.items()}
        
        return {
            'hour_dk1': self.out_dk1(x_dict['hour_dk1']),
            'hour_dk2': self.out_dk2(x_dict['hour_dk2']),
            'hour_hydro': self.out_hydro(x_dict['hour_hydro']),
            'hour_de': self.out_de(x_dict['hour_de'])
        }

def create_multi_area_hetero_gnn(metadata, hidden_channels=128, dropout=0.2):
    base_model = FourAreaBaseModel(hidden_channels, dropout)
    return to_hetero(base_model, metadata, aggr='sum')