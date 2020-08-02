import torch
import torch.nn as nn
import torch.nn.functional as F

import math

import psifold

class Baseline(nn.Module):
    def __init__(self, hidden_size=512, n_layers=2, dropout=0.1):
        super(Baseline, self).__init__()

        # save info needed to recreate model from checkpoint
        self.model_name = "baseline"
        self.model_args = {"hidden_size" : hidden_size,
                           "n_layers" : n_layers,
                           "dropout" : dropout}

        self.fc0 = nn.Linear(41, hidden_size)

        layers = [nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(dropout)]*n_layers
        self.encoder = nn.Sequential(*layers)

        self.fc1 = nn.Linear(hidden_size, 3)

    def forward(self, batch):
        seq = batch["seq"] # (L x B)
        pssm = batch["pssm"] # (L x B x 21)
        length = batch["length"] # (L,)

        L, B = seq.size()

        # (L x B x 20)
        seq = F.one_hot(seq, 20).type(pssm.dtype)

        # (L x B x (20 + 21))
        encoder_in = torch.cat((seq, pssm), dim=2)

        # (L x B x hidden_size)
        encoder_in = self.fc0(encoder_in)

        # (L x B x hidden_size)
        encoder_out = self.encoder(encoder_in)

        # (L x B x 3)
        srf = self.fc1(encoder_out)

        return srf
