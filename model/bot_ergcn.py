"""
Some functions and methods in this file are adapted from the EvolveGCN implementation.

Original paper:
    Pareja, A., Domeniconi, G., Chen, J., Ma, T., Suzumura, T., Kanezashi, H., 
    Kaler, T., Schardl, T. B., & Leiserson, C. E. (2020).
    EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs.
    In Proceedings of the AAAI Conference on Artificial Intelligence.
    arXiv:1902.10191

Original code repository:
    https://github.com/IBM/EvolveGCN/blob/master/egcn_h.py

License: Apache-2.0
"""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math


class mat_GRU_gate(torch.nn.Module):
    def __init__(self, rows, cols, activation):
        super().__init__()
        self.activation = activation
        self.W = Parameter(torch.Tensor(rows, rows))
        self.U = Parameter(torch.Tensor(rows, rows))
        self.bias = Parameter(torch.zeros(rows, cols))
        self.reset_param(self.W)
        self.reset_param(self.U)

    def reset_param(self, t):
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        out = self.activation(self.W.matmul(x) + self.U.matmul(hidden) + self.bias)
        return out


class TopK(torch.nn.Module):
    def __init__(self, feats, k):
        super().__init__()
        self.scorer = Parameter(torch.Tensor(feats, 1))
        self.reset_param(self.scorer)
        self.k = k

    def reset_param(self, t):
        stdv = 1. / math.sqrt(t.size(0))
        t.data.uniform_(-stdv, stdv)

    def forward(self, node_embs, mask):
        num_nodes = node_embs.size(0)

        scores = node_embs.matmul(self.scorer)  # (num_nodes, 1)
        scores = scores.squeeze(-1)  # (num_nodes,)
        scores = scores / (self.scorer.norm() + 1e-8) 
        scores = scores + mask

        valid_nodes = (mask > -float('inf')).sum().item()
        k_actual = min(self.k, valid_nodes, num_nodes)
        
        if k_actual == 0:
            return torch.zeros(node_embs.size(1), self.k, device=node_embs.device)
        
        vals, topk_indices = scores.topk(k_actual)

        valid_mask = vals > -float("Inf")
        topk_indices = topk_indices[valid_mask]
        
        if topk_indices.size(0) == 0:
            return torch.zeros(node_embs.size(1), self.k, device=node_embs.device)
        
        if topk_indices.size(0) < self.k:
            pad_size = self.k - topk_indices.size(0)
            # Pad with last valid index
            last_valid_idx = topk_indices[-1]
            padding = torch.full((pad_size,), last_valid_idx, dtype=topk_indices.dtype, device=topk_indices.device)
            topk_indices = torch.cat([topk_indices, padding])
        
        topk_indices = torch.clamp(topk_indices, 0, num_nodes - 1)
        
        tanh = torch.nn.Tanh()

        if isinstance(node_embs, torch.sparse.FloatTensor) or \
           isinstance(node_embs, torch.cuda.sparse.FloatTensor):
            node_embs = node_embs.to_dense()

        selected_embs = node_embs[topk_indices]  # (k, features)
        selected_scores = scores[topk_indices]   # (k,)
        
        weights = tanh(selected_scores).unsqueeze(-1)  # (k, 1)
        out = selected_embs * weights  # (k, features)

        return out.t()


class mat_GRU_cell(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.update = mat_GRU_gate(args.rows, args.cols, torch.nn.Sigmoid())
        self.reset = mat_GRU_gate(args.rows, args.cols, torch.nn.Sigmoid())
        self.htilda = mat_GRU_gate(args.rows, args.cols, torch.nn.Tanh())
        self.choose_topk = TopK(feats=args.rows, k=args.cols)

    def forward(self, prev_Q, prev_Z, mask):
        z_topk = self.choose_topk(prev_Z, mask)
        
        update = self.update(z_topk, prev_Q)
        reset = self.reset(z_topk, prev_Q)

        h_cap = reset * prev_Q
        h_cap = self.htilda(z_topk, h_cap)

        new_Q = (1 - update) * prev_Q + update * h_cap

        return new_Q


class GRCU_RGCN(torch.nn.Module):

    def __init__(self, in_feats, out_feats, num_relations, activation):
        super().__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_relations = num_relations
        self.activation = activation
        
        # Namespace for mat_GRU_cell
        cell_args = type('Args', (), {})()
        cell_args.rows = in_feats
        cell_args.cols = out_feats
        
        # For self-loop
        self.evolve_weights_self = mat_GRU_cell(cell_args)
        self.GCN_init_weights_self = Parameter(torch.Tensor(in_feats, out_feats))
        
        # For each relation
        self.evolve_weights_relations = nn.ModuleList([
            mat_GRU_cell(cell_args) for _ in range(num_relations)
        ])
        self.GCN_init_weights_relations = nn.ParameterList([
            Parameter(torch.Tensor(in_feats, out_feats))
            for _ in range(num_relations)
        ])
        
        # Initialize all weights
        self.reset_param(self.GCN_init_weights_self)
        for w in self.GCN_init_weights_relations:
            self.reset_param(w)

    def reset_param(self, t):
        stdv = 1. / math.sqrt(t.size(1))
        t.data.uniform_(-stdv, stdv)

    def forward(self, A_list, edge_type_list, node_embs_list, mask_list):
        GCN_weights_self = self.GCN_init_weights_self
        GCN_weights_relations = [w.clone() for w in self.GCN_init_weights_relations]
        
        out_seq = []
        
        for t in range(len(A_list)):
            edge_index = A_list[t]
            edge_type = edge_type_list[t]
            node_embs = node_embs_list[t]
            mask = mask_list[t]
            
            # Self-loop weight
            GCN_weights_self = self.evolve_weights_self(
                GCN_weights_self, node_embs, mask
            )
            
            # Relation weights
            for r in range(self.num_relations):
                GCN_weights_relations[r] = self.evolve_weights_relations[r](
                    GCN_weights_relations[r], node_embs, mask
                )
            
            # RGCN
            node_embs = self.apply_rgcn(
                node_embs, edge_index, edge_type,
                GCN_weights_self, GCN_weights_relations
            )
            
            out_seq.append(node_embs)
        
        return out_seq
    
    def apply_rgcn(self, x, edge_index, edge_type, W_self, W_relations):
        num_nodes = x.size(0)
        device = x.device
        
        # Self-loop
        out = x.matmul(W_self)
        
        # Relation-specific aggregation
        for r in range(self.num_relations):
            mask = edge_type == r
            if mask.sum() == 0:
                continue

            edge_index_r = edge_index[:, mask]
            row, col = edge_index_r
            
            # Normalization
            deg = torch.zeros(num_nodes, device=device)
            deg.scatter_add_(0, row, torch.ones(row.size(0), device=device))
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

            # Aggregate
            x_transformed = x.matmul(W_relations[r])
            messages = x_transformed[col] * norm.unsqueeze(-1)
            out.index_add_(0, row, messages)
        
        return self.activation(out)


class BotEvolveRGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        num_relations,
        out_channels,
        dropout=0.3,
        num_layers=2,
        device='cpu'
    ):
        super().__init__()
        
        feats = [in_channels] + [hidden_channels] * num_layers
        self.device = device
        self.num_relations = num_relations
        
        self.GRCU_layers = nn.ModuleList()
        for i in range(1, len(feats)):
            layer = GRCU_RGCN(
                in_feats=feats[i-1],
                out_feats=feats[i],
                num_relations=num_relations,
                activation=nn.LeakyReLU()
            )
            self.GRCU_layers.append(layer)
        
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, A_list, edge_type_list, Nodes_list, nodes_mask_list):
        for layer in self.GRCU_layers:
            Nodes_list = layer(A_list, edge_type_list, Nodes_list, nodes_mask_list)
        
        out_seq = []
        for node_embs in Nodes_list:
            out = self.classifier(node_embs)
            out_seq.append(out)
        
        return out_seq
