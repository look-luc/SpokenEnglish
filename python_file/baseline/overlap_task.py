import torch
import torch.nn as nn


class model(nn.Module):
    def __init__(self, vocab_size, pad_token_id=0, max_len=512, hidden_dim_1=256, hidden_dim_2=128,
                 embedding_dim=300) -> None:
        super(model, self).__init__()

        possible_labels = ['recognitional', 'other', 'transitional', 'progressional', 'restatement']
        self.pad_token_id = pad_token_id

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(max_len, embedding_dim)
        self.seg_embedding = nn.Embedding(2, embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=4,
            batch_first=True,
            norm_first=True
        )
        self.text_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.pre_classifier_norm = nn.LayerNorm(embedding_dim)

        self.output = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_1),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim_2, len(possible_labels))
        )

    def forward(self, input_ids, segment_ids):
        src_key_padding_mask = (input_ids == self.pad_token_id)

        positions = torch.arange(input_ids.size(1), dtype=torch.long, device=input_ids.device)
        x = self.embedding(input_ids) + self.pos_embedding(positions) + self.seg_embedding(segment_ids)

        x = self.text_encoder(x, src_key_padding_mask=src_key_padding_mask)

        cls_output = x[:, 0, :]
        cls_output = self.pre_classifier_norm(cls_output)

        return self.output(cls_output)