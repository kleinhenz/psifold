import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from psifold.geometry import pnerf

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
    def __init__(self, seq_embed_dim=16, kmer_embed_dim=256, hidden_size=512, n_layers=2, nhead=8, dim_feedforward=1024, dropout=0.1):
        super(PsiFold, self).__init__()

        # save info needed to recreate model from checkpoint
        self.model_name = "psifold"
        self.model_args = {"seq_embed_dim" : seq_embed_dim,
                           "kmer_embed_dim" : kmer_embed_dim,
                           "hidden_size" : hidden_size,
                           "n_layers": n_layers,
                           "nhead" : nhead,
                           "dim_feedforward" : dim_feedforward,
                           "dropout": dropout}

        self.seq_embed = nn.Embedding(20, seq_embed_dim)
        self.kmer_embed = nn.Embedding(22**3, kmer_embed_dim)

        input_dim = seq_embed_dim + kmer_embed_dim + 21
        self.fc0 = nn.Linear(input_dim, hidden_size)

        self.pos_encoder = PositionalEncoding(hidden_size, dropout)

        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc1 = nn.Linear(hidden_size, 9)

    def forward(self, seq, kmer, pssm, length):
        """
        seq: (L x B)
        pssm: (L x B x 21)
        length: (L,)
        """

        L, B = seq.size()

        # (L x B x seq_embed_dim)
        seq = self.seq_embed(seq)

        # (L x B x kmer_embed_dim)
        kmer = self.kmer_embed(kmer)

        # (L x B x input_dim)
        encoder_in = torch.cat((seq, kmer, pssm), dim=2)

        # (L x B x hidden_size)
        encoder_in = self.fc0(encoder_in)

        # (L x B x hidden_size)
        encoder_in = self.pos_encoder(encoder_in)

        # (L x B)
        mask = torch.arange(L, device=seq.device).expand(B, L) >= length.unsqueeze(1)

        # (L x B x hidden_size)
        encoder_out = self.encoder(encoder_in, src_key_padding_mask=mask)

        # (L x B x 9)
        srf = self.fc1(encoder_out)

        # (3L x B x 3)
        srf = srf.view(L, B, 3, 3).permute(0, 2, 1, 3).contiguous().view(3*L, B, 3)

        # (3L x B x 3)
        coords = pnerf(srf, nfrag=int(math.sqrt(L)))

        return coords
