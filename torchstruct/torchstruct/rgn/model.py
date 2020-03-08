import torch
import torch.nn as nn
import torch.nn.functional as F

import math

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

        #bond length vectors C-N, N-CA, CA-C
        self.avg_bond_lens = torch.tensor([1.329, 1.459, 1.525])
        #bond angle vector, in radians, CA-C-N, C-N-CA, N-CA-C
        self.avg_bond_angles = torch.tensor([2.034, 2.119, 1.937])

        self.embed = nn.Embedding(20, embed_dim) # embedding for primary sequence
        self.lstm = nn.LSTM(input_size=embed_dim + 21,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=True,
                            bidirectional=True)

        self.linear = nn.Linear(hidden_size*2, linear_units)

    def forward(self, inp):
        # (batch x N)
        seq = inp["seq"]

        # (batch x N x 21)
        pssm = inp["pssm"]

        # (batch x N x embed_dim)
        seq_embedding = self.embed(seq)

        # (batch x N x (embed_dim + 21))
        lstm_in = torch.cat((seq_embedding, pssm), dim=2)

        # (batch x N x (2*hidden_size))
        lstm_out, _ = self.lstm(lstm_in)

        # (batch x N x linear_units)
        linear_out = self.linear(lstm_out)

        # angularization

        # (batch x N x linear_units)
        softmax_out = F.softmax(linear_out, dim=2)

        # (batch x N x 3)
        sin = torch.matmul(softmax_out, torch.sin(self.alphabet))
        cos = torch.matmul(softmax_out, torch.cos(self.alphabet))

        # (batch x N x 3)
        phi = torch.atan2(sin, cos)

        return phi
