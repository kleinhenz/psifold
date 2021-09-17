import math
import re

import numpy as np
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence

import psifold
from psifold.geometry import pnerf, torsion_to_srf

class RGN(nn.Module):
    """
    Recurrent Geometric Network implementation

    References:
    * https://doi.org/10.1016/j.cels.2019.03.006
    * https://github.com/aqlaboratory/rgn
    * https://github.com/conradry/pytorch-rgn/
    """
    def __init__(self, hidden_size=800, alphabet_size=60, n_layers=2, dropout=0.5):
        super(RGN, self).__init__()

        # save info needed to recreate model from checkpoint
        self.model_name = "rgn"
        self.model_args = {"hidden_size" : hidden_size,
                           "alphabet_size" : alphabet_size,
                           "n_layers": n_layers,
                           "dropout": dropout}

        self.lstm = nn.LSTM(input_size=41,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=False,
                            dropout=dropout,
                            bidirectional=True)

        self.fc = nn.Linear(2*hidden_size, alphabet_size)

        #initialize alphabet to random values between -pi and pi
        #each entry in the alphabet represents a triplet of angles
        u = torch.distributions.Uniform(-math.pi, math.pi)
        self.alphabet = nn.Parameter(u.rsample(torch.Size([alphabet_size, 3])))

        # (C-N, N-CA, CA-C)
        self.bond_lengths = nn.Parameter(100.0 * torch.tensor([1.3297, 1.4597, 1.5249]))
        self.bond_lengths.requires_grad = False

        # (CA-C-N, C-N-CA, N-CA-C)
        # NOTE values from original rgn code given by pi - self.bond_angles
        self.bond_angles = nn.Parameter(torch.tensor([1.1075, 1.0196, 1.2035]))
        self.bond_angles.requires_grad = False

    def forward(self, batch):
        seq = batch["seq"] # (L x B)
        pssm = batch["pssm"] # (L x B x 21)
        length = batch["length"].cpu() # (L,)

        L, B = seq.size()

        # (L x B x 20)
        seq = F.one_hot(seq, 20).type(pssm.dtype)

        # (L x B x (20 + 21))
        lstm_in = torch.cat((seq, pssm), dim=2)
        lstm_in = pack_padded_sequence(lstm_in, length)

        # (L x B x (2*hidden_size))
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out, _ = pad_packed_sequence(lstm_out)

        ## angularization ##

        # (L, B, alphabet_size)
        x = F.softmax(self.fc(lstm_out), dim=2)

        # (L, B, 3)
        sin = torch.matmul(x, torch.sin(self.alphabet))
        cos = torch.matmul(x, torch.cos(self.alphabet))
        phi = torch.atan2(sin, cos)

        # (3L, B, 3)
        srf = torsion_to_srf(self.bond_lengths, self.bond_angles, phi)

        # (3L, B, 3)
        coords = pnerf(srf, nfrag=int(math.sqrt(L)))

        return coords
