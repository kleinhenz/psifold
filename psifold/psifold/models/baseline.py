import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from psifold import GeometricUnit

class Baseline(nn.Module):
    """
    PsiFold implementation
    """
    def __init__(self, hidden_size=64, linear_units=32, n_layers=2, dropout=0.1):
        super(Baseline, self).__init__()

        # save info needed to recreate model from checkpoint
        self.model_name = "psifold"
        self.model_args = {"hidden_size" : hidden_size,
                           "linear_units" : linear_units,
                           "n_layers": n_layers,
                           "dropout": dropout}

        self.fc = nn.Linear(41, hidden_size)

        layers = [nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]*n_layers
        self.encoder = nn.Sequential(*layers)

        self.geometry = GeometricUnit(hidden_size, linear_units)

    def forward(self, seq, pssm, length):
        """
        seq: (L x B)
        pssm: (L x B x 21)
        length: (L,)
        """

        L, B = seq.size()

        # (L x B x 20)
        seq = F.one_hot(seq, 20).type(pssm.dtype)

        # (L x B x 41)
        encoder_in = torch.cat((seq, pssm), dim=2)

        # (L x B x hidden_size)
        encoder_in = self.fc(encoder_in)

        # (L x B x hidden_size)
        encoder_out = self.encoder(encoder_in)

        # (L x B x 3)
        coords = self.geometry(encoder_out)

        return coords
