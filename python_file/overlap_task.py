import torch
import torch.nn as nn
import torch.nn.functional as F


class model(nn.Module):
    def __init__(self, device, vocab_size, max_len, hidden_dim=256, embedding_dim=300) -> None:
        super(model, self).__init__()

        # Embeddings for tokens, position, and segment
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_len, embedding_dim)
        self.seg_embedding = nn.Embedding(2, embedding_dim)

        # transformer encoder layer
        self.encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            batch_first=True,
            norm_first=True
        )
        # actual transformer block
        self.text_encoder = nn.TransformerEncoder(self.encoder, num_layers=6)

        # makes fully connected go from 300 to 256
        self.txt_proj = nn.Linear(embedding_dim, hidden_dim)

        # makes it into a binary classification
        # 0 = No
        # 1 = Yes
        self.output = nn.Linear(hidden_dim, 1)


    def forward(self, input_ids, segment_ids):
        # embeddings
        x = self.embedding(input_ids) + self.pos_embedding() + self.seg_embedding(segment_ids)

        # transformer sequence
        x = self.text_encoder(x)

        # fully connected into 256
        x = self.txt_proj(x)

        # outputs to 0/1
        return self.output(x[:, 0, :])