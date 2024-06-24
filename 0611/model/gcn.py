import torch
from torch_geometric.nn import GCNConv

import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_channels[0])
        self.conv2 = GCNConv(hidden_channels[0], 2)


    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index)
        x = F.softmax(x, dim=-1)
        return x
    
    def embed(self, data):
        x = self.conv1(data.x, data.edge_index)
        return x