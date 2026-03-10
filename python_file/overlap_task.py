import torch
import torch.nn as nn
import torch.nn.functional as F


class model(nn.Module):
    def __init__(self, device, hidden_dim, embedding_dim=300) -> None:
        super(model, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim


    def forward(self, input):
        pass