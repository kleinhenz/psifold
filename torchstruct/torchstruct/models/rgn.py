import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import math

from torchstruct import GeometricUnit

class RGN(nn.Module):
    """
    Recurrent Geometric Network implementation

    References:
    * https://doi.org/10.1016/j.cels.2019.03.006
    * https://github.com/aqlaboratory/rgn
    * https://github.com/conradry/pytorch-rgn/
    """
    def __init__(self, embed_dim=20, hidden_size=50, linear_units=20, n_layers=1, dropout=0.5):
        super(RGN, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # TODO make this one-hot to match original implementation?
        self.embed = nn.Embedding(20, embed_dim) # embedding for primary sequence
        self.lstm = nn.LSTM(input_size=embed_dim + 21,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=False,
                            dropout=dropout,
                            bidirectional=True)

        self.geometry = GeometricUnit(hidden_size*2, linear_units)

    def forward(self, seq, pssm, length):
        """
        seq: (L x B)
        pssm: (L x B x 21)
        length: (L,)
        """

        # (L x B x embed_dim)
        seq_embedding = self.embed(seq)

        # (L x B x (embed_dim + 21))
        lstm_in = torch.cat((seq_embedding, pssm), dim=2)
        lstm_in = pack_padded_sequence(lstm_in, length)

        # (L x B x (2*hidden_size))
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out, _ = pad_packed_sequence(lstm_out)

        # (L x B x 3)
        coords = self.geometry(lstm_out)

        return coords
