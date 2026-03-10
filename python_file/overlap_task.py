import torch
import torch.nn as nn
import torch.nn.functional as F


class model(nn.Module):
    def __init__(self, device, vocab_size, hidden_dim=256, embedding_dim=300) -> None:
        super(model, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.encoder = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=8, batch_first=True, norm_first=True)
        self.text_encoder = nn.TransformerEncoder(self.encoder, num_layers=6)

        self.txt_proj = nn.Linear(self.embedding_dim, self.hidden_dim)

        # 0 = No
        # 1 = Yes
        self.output = nn.Linear(self.hidden_dim, 1)


    def forward(self, input):
        pass