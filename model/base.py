import torch.nn as nn
from abc import ABC, abstractmethod


class BaseGCNModel(nn.Module, ABC):
    def __init__(self, in_channels, hidden_channels, num_relations=9, dropout=0.3):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_relations = num_relations
        self.dropout = dropout

    @abstractmethod
    def forward(self):
        pass

    def get_name(self):
        return self.__class__.__name__
