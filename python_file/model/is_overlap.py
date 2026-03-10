import torch
import torch.nn as nn
import torch.nn.functional as F


class is_overlap_model(nn.Module):
    def __init__(self, input, device, hidden_dim, embedding_dim=300) -> None:
        super(is_overlap_model, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
