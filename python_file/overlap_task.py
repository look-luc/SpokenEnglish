import torch
import torch.nn as nn

class model(nn.Module):
    def __init__(self, vocab_size, max_len=512, hidden_dim_1=256, hidden_dim_2=128, embedding_dim=300) -> None:
        super(model, self).__init__()

        possible_labels = ['recognitional', 'other', 'transitional', 'progressional']

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_len, embedding_dim)
        self.seg_embedding = nn.Embedding(2, embedding_dim)

        # transformer encoder layer
        self.encoder = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=15,
            batch_first=True,
            norm_first=True
        )
        # actual transformer block
        self.text_encoder = nn.TransformerEncoder(self.encoder, num_layers=12)

        self.output = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_2, len(possible_labels))
        )


    def forward(self, input_ids, segment_ids):
        # get position
        positions = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        # embeddings
        x = self.embedding(input_ids) + self.pos_embedding(positions) + self.seg_embedding(segment_ids)

        # transformer sequence
        x = self.text_encoder(x)

        # outputs to 0/1
        return self.output(x[:, 0, :])
