import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

from .base import BaseGCNModel


class MatGRUGate(nn.Module):
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows, rows))
        self.U = Parameter(torch.Tensor(rows, rows))
        self.bias = Parameter(torch.zeros(rows, cols))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-stdv, stdv)
        self.U.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        out = self.activation(self.W.matmul(x) + self.U.matmul(hidden) + self.bias)
        return out


class MatGRUCell(nn.Module):
    def __init__(self, rows, cols):
        super().__init__()
        self.rows = rows
        self.cols = cols

        self.update = MatGRUGate(rows, cols, nn.Sigmoid())
        self.reset = MatGRUGate(rows, cols, nn.Sigmoid())
        self.htilda = MatGRUGate(rows, cols, nn.Tanh())

    def forward(self, prev_Q):
        z_topk = prev_Q

        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class EvolvingRGCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_relations = num_relations

        # Initialize weight matrix for GCN
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()

        # GRU cell for weight evolution
        self.evolve_weights = MatGRUCell(in_channels, out_channels)

        # Relation-specific transformation (simplified RGCN approach)
        self.relation_weights = nn.ParameterList(
            [
                Parameter(torch.Tensor(in_channels, out_channels))
                for _ in range(num_relations)
            ]
        )
        for w in self.relation_weights:
            nn.init.xavier_uniform_(w)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, edge_index, edge_type):
        # Evolve the base weight matrix
        self.weight.data = self.evolve_weights(self.weight)

        # Aggregate messages by relation type
        out = torch.zeros(x.size(0), self.out_channels, device=x.device)

        for r in range(self.num_relations):
            # Get edges of this relation type
            mask = edge_type == r
            if mask.sum() == 0:
                continue

            edge_index_r = edge_index[:, mask]

            # Relation-specific transformation
            x_transformed = x.matmul(self.relation_weights[r])

            # Simple message passing (aggregation)
            row, col = edge_index_r
            deg = torch.zeros(x.size(0), device=x.device)
            deg.scatter_add_(0, row, torch.ones(row.size(0), device=x.device))
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0

            # Normalize
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            # Aggregate
            out.index_add_(0, row, x_transformed[col] * norm.unsqueeze(-1))

        # Apply base weight
        out = out.matmul(self.weight)

        return out


class BotEvolvingRGCN(BaseGCNModel):
    def __init__(
        self,
        in_channels=128,
        hidden_channels=128,
        num_relations=2,
        out_channels=2,
        dropout=0.3,
        num_layers=2,
    ):
        super().__init__(
            in_channels,
            hidden_channels,
            num_relations,
            out_channels,
            dropout,
            num_layers,
        )
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_relations = num_relations
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_layers = num_layers

        # Evolving RGCN layers
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(
            EvolvingRGCNLayer(in_channels, hidden_channels, num_relations)
        )

        # Additional layers
        for _ in range(num_layers - 1):
            self.layers.append(
                EvolvingRGCNLayer(hidden_channels, hidden_channels, num_relations)
            )

        # MLP head for classification
        self.linear_relu = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels), nn.LeakyReLU()
        )
        self.out_layer = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type):
        # Pass through evolving RGCN layers
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_type)
            if i < len(self.layers) - 1:
                x = F.leaky_relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        # Final activation
        x = F.leaky_relu(x)

        # MLP head
        x = self.linear_relu(x)
        out = self.out_layer(x)

        return out

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for module in [self.linear_relu, self.out_layer]:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
