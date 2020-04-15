import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from torchstruct import GeometricUnit

# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class PsiFold(nn.Module):
    """
    PsiFold implementation
    """
    def __init__(self, embed_dim=20, hidden_size=50, linear_units=20, n_layers=1, nhead = 5, dropout=0.5):
        super(PsiFold, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embed = nn.Embedding(20, embed_dim) # embedding for primary sequence
        # TODO replace by larger embedding layer?
        self.fc = nn.Linear(embed_dim + 21, hidden_size)

        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        # TODO add parameter for dim_feedforward
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = n_layers)

        self.geometry = GeometricUnit(hidden_size, linear_units)

    def forward(self, seq, pssm, length):
        """
        seq: (L x B)
        pssm: (L x B x 21)
        length: (L,)
        """

        L, B = seq.size()

        # (L x B x embed_dim)
        seq_embedding = self.embed(seq)

        encoder_in = torch.cat((seq_embedding, pssm), dim=2)
        encoder_in = self.fc(encoder_in)
        encoder_in = self.pos_encoder(encoder_in)

        mask = torch.arange(L).expand(B, L) >= length.unsqueeze(1)

        # (L x B x hidden_size)
        encoder_out = self.encoder(encoder_in, src_key_padding_mask=mask)

        # (L x B x 3)
        coords = self.geometry(encoder_out)

        return coords
