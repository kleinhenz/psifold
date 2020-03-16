import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import math

from torchstruct.util import geometric_unit

# adapted from https://github.com/conradry/pytorch-rgn/blob/master/rgn.ipynb
class RGN(nn.Module):
    def __init__(self, embed_dim=100, hidden_size=50, linear_units=20, n_layers=1):
        super(RGN, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        #initialize alphabet to random values between -pi and pi
        u = torch.distributions.Uniform(-math.pi, math.pi)
        self.alphabet = nn.Parameter(u.rsample(torch.Size([linear_units, 3])))

        # TODO make bond lengths/angles parameters instead of constant?

        # [C-N, N-CA, CA-C]
        self.bond_lengths = torch.tensor([132.868, 145.801, 152.326])
        # [CA-C-N, C-N-CA, N-CA-C]
        self.bond_angles = torch.tensor([2.028, 2.124, 1.941])

        self.embed = nn.Embedding(20, embed_dim) # embedding for primary sequence
        self.lstm = nn.LSTM(input_size=embed_dim + 21,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=False,
                            bidirectional=True)

        self.linear = nn.Linear(hidden_size*2, linear_units)

    def forward(self, inp):
        # (L x B)
        seq = inp["seq"]
        length = inp["length"]
        L, B = seq.size()

        # (L x B x 21)
        pssm = inp["pssm"]

        # (L x B x embed_dim)
        seq_embedding = self.embed(seq)

        # (L x B x (embed_dim + 21))
        lstm_in = torch.cat((seq_embedding, pssm), dim=2)
        lstm_in = pack_padded_sequence(lstm_in, length)

        # (L x B x (2*hidden_size))
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out, _ = pad_packed_sequence(lstm_out)

        # (L x B x linear_units)
        linear_out = self.linear(lstm_out)

        ### angularization ###

        # (L x B x linear_units)
        softmax_out = F.softmax(linear_out, dim=2)

        # (L x B x 3)
        sin = torch.matmul(softmax_out, torch.sin(self.alphabet))
        cos = torch.matmul(softmax_out, torch.cos(self.alphabet))

        # (L x B x 3)
        phi = torch.atan2(sin, cos)

        ### geometric units ###

        # initial coords
        # (3 x B x 3)
        coords = torch.eye(3).unsqueeze(1).repeat(1, B, 1)

        # (3 x B)
        r = self.bond_lengths.unsqueeze(1).repeat(1, B)
        theta = self.bond_angles.unsqueeze(1).repeat(1, B)

        for i in range(L):
            coords = geometric_unit(coords, r, theta, phi[i].transpose(0, 1))

        return coords
