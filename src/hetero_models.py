# OVERWRITE EXACTLY: src/hetero_models.py
"""
Production-Grade Heterogeneous Graph Neural Network with Weighted Spatial
Convolutions and Zone-Specific Readout Heads.
"""
import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear

class HeteroPriceForecaster(torch.nn.Module):
    def __init__(self, metadata, hour_in_features, hidden_channels=128):
        super().__init__()
        self.hidden_channels = hidden_channels
        
        # Feature Projections
        self.market_lin = Linear(4, hidden_channels)
        self.hour_lin = Linear(hour_in_features, hidden_channels)
        
        # Structural Layers
        self.conv1 = HeteroConv({
            ('market', 'interconnects', 'market'): SAGEConv(hidden_channels, hidden_channels),
            ('hour', 'belongs_to', 'market'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ('market', 'rev_belongs_to', 'hour'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ('hour', 'lag_to', 'hour'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')
        
        self.conv2 = HeteroConv({
            ('market', 'interconnects', 'market'): SAGEConv(hidden_channels, hidden_channels),
            ('market', 'rev_belongs_to', 'hour'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            ('hour', 'lag_to', 'hour'): SAGEConv(hidden_channels, hidden_channels),
        }, aggr='sum')
        
        # Zone-Specific Regression Output Heads
        self.head_dk1 = torch.nn.Sequential(Linear(hidden_channels, hidden_channels // 2), torch.nn.LeakyReLU(0.2), Linear(hidden_channels // 2, 1))
        self.head_dk2 = torch.nn.Sequential(Linear(hidden_channels, hidden_channels // 2), torch.nn.LeakyReLU(0.2), Linear(hidden_channels // 2, 1))
        self.head_de = torch.nn.Sequential(Linear(hidden_channels, hidden_channels // 2), torch.nn.LeakyReLU(0.2), Linear(hidden_channels // 2, 1))
        self.head_hydro = torch.nn.Sequential(Linear(hidden_channels, hidden_channels // 2), torch.nn.LeakyReLU(0.2), Linear(hidden_channels // 2, 1))

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None, num_hours=None):
        h_dict = {
            'market': F.leaky_relu(self.market_lin(x_dict['market']), 0.2),
            'hour': F.leaky_relu(self.hour_lin(x_dict['hour']), 0.2)
        }
        
        h_out1 = self.conv1(h_dict, edge_index_dict)
        h_dict['market'] = F.leaky_relu(h_out1['market'] + h_dict['market'], 0.2)
        h_dict['hour'] = F.leaky_relu(h_out1['hour'] + h_dict['hour'], 0.2)
        
        h_out2 = self.conv2(h_dict, edge_index_dict)
        h_dict['market'] = F.leaky_relu(h_out2['market'] + h_dict['market'], 0.2)
        h_dict['hour'] = F.leaky_relu(h_out2['hour'] + h_dict['hour'], 0.2)
        
        h_dk1 = h_dict['hour'][0 : num_hours]
        h_dk2 = h_dict['hour'][num_hours : 2 * num_hours]
        h_hydro = h_dict['hour'][2 * num_hours : 3 * num_hours]
        h_de = h_dict['hour'][3 * num_hours : 4 * num_hours]
        
        return torch.cat([self.head_dk1(h_dk1), self.head_dk2(h_dk2), self.head_hydro(h_hydro), self.head_de(h_de)], dim=0)