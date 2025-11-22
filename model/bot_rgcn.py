import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from .base import BaseGCNModel


class BotRGCN(BaseGCNModel):
    """
    Homogeneous R-GCN model for user classification.

    Expects:
        x:          Tensor [N, F]          node features
        edge_index: LongTensor [2, E]      graph edges
        edge_type:  LongTensor [E]         relation IDs in [0, num_relations-1]
    """

    def __init__(
        self,
        in_channels: int = 128,  # dim of node embeddings
        hidden_channels: int = 128,
        num_relations: int = 2,  # (0) followers, (1) following
        dropout: float = 0.2,
    ):
        super().__init__(in_channels, hidden_channels, num_relations, dropout)

        self.dropout = dropout

        self.rgcn1 = RGCNConv(
            in_channels,
            hidden_channels,
            num_relations=num_relations,
        )

        # self.rgcn2 = RGCNConv(
        #     hidden_channels,
        #     hidden_channels,
        #     num_relations=num_relations,
        # )

        # MLP head
        self.linear_relu = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
        )
        self.out_layer = nn.Linear(hidden_channels, 2)  # bot / not-bot

    def forward(self, x, edge_index, edge_type):
        """
        x:          (N, F) node features
        edge_index: (2, E) edge list
        edge_type:  (E,)   long tensor with relation IDs [0, 1]
        """

        # R-GCN layer 1
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # R-GCN layer 2
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.leaky_relu(x)

        # MLP head
        x = self.linear_relu(x)
        out = self.out_layer(x)  # (N, 2)

        return out
