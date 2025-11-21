import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from .base import BaseGCNModel


class BotRGCN(BaseGCNModel):
    def __init__(
        self,
        in_channels=128,  # node embeddings dim
        hidden_channels=128,
        num_relations=2,  # (0) followers, (1) following
        dropout=0.3,
    ):

        super().__init__(in_channels, hidden_channels, num_relations, dropout)

        self.rgcn = RGCNConv(in_channels, hidden_channels, num_relations=num_relations)

        self.linear_relu = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.LeakyReLU()
        )
        self.out_layer = nn.Linear(hidden_channels, 2)  # bot or not

    def forward(self, x, edge_index, edge_type):
        """
        x: (N, 128) node embeddings
        edge_index: (2, E) types of relations & number of edges
        edge_type:  (E,) long tensor [0 or 1]
        """

        # rgcn layer 1
        x = self.rgcn(x, edge_index, edge_type)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # rgcn layer 2
        x = self.rgcn(x, edge_index, edge_type)
        x = F.leaky_relu(x)

        # MLP head
        x = self.linear_relu(x)
        out = self.out_layer(x)

        return out
