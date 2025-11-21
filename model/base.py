import torch.nn as nn
from abc import ABC, abstractmethod


class BaseGCNModel(nn.Module, ABC):
    def __init__(self, in_channels, hidden_channels, num_relations=2, out_channels=2, dropout=0.3, num_layers=2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.dropout = dropout
        self.num_layers = num_layers

    @abstractmethod
    def forward(self, x, edge_index, edge_type):
        pass

    def get_name(self):
        return self.__class__.__name__
