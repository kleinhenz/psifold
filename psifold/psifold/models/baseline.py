import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from psifold import GeometricUnit

class Baseline(nn.Module):
    """
    PsiFold implementation
    """
    def __init__(self, seq_embed_dim=16, kmer_embed_dim=256, hidden_size=64, linear_units=32, n_layers=2, dropout=0.1):
        super(Baseline, self).__init__()

        # save info needed to recreate model from checkpoint
        self.model_name = "baseline"
        self.model_args = {"seq_embed_dim" : seq_embed_dim,
                           "kmer_embed_dim" : kmer_embed_dim,
                           "hidden_size" : hidden_size,
                           "linear_units" : linear_units,
                           "n_layers": n_layers,
                           "dropout": dropout}

        self.seq_embed = nn.Embedding(20, seq_embed_dim)
        self.kmer_embed = nn.Embedding(22**3, kmer_embed_dim)

        input_dim = seq_embed_dim + kmer_embed_dim + 21
        self.fc = nn.Linear(input_dim, hidden_size)

        layers = [nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]*n_layers
        self.encoder = nn.Sequential(*layers)

        self.geometry = GeometricUnit(hidden_size, linear_units)

    def forward(self, seq, kmer, pssm, length):
        """
        seq: (L x B)
        pssm: (L x B x 21)
        length: (L,)
        """

        L, B = seq.size()

        seq = self.seq_embed(seq)
        kmer = self.kmer_embed(kmer)

        # (L x B x input_dim)
        encoder_in = torch.cat((seq, kmer, pssm), dim=2)

        # (L x B x hidden_size)
        encoder_in = self.fc(encoder_in)

        # (L x B x hidden_size)
        encoder_out = self.encoder(encoder_in)

        # (L x B x 3)
        coords = self.geometry(encoder_out)

        return coords
