import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import math

from psifold import GeometricUnit

class RGN(nn.Module):
    """
    Recurrent Geometric Network implementation

    References:
    * https://doi.org/10.1016/j.cels.2019.03.006
    * https://github.com/aqlaboratory/rgn
    * https://github.com/conradry/pytorch-rgn/
    """
    def __init__(self, hidden_size=64, linear_units=32, n_layers=2, dropout=0.1):
        super(RGN, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size=41,
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

        # (L x B x 20)
        seq = F.one_hot(seq, 20).type(pssm.dtype)

        # (L x B x (embed_dim + 21))
        lstm_in = torch.cat((seq, pssm), dim=2)
        lstm_in = pack_padded_sequence(lstm_in, length)

        # (L x B x (2*hidden_size))
        lstm_out, _ = self.lstm(lstm_in)
        lstm_out, _ = pad_packed_sequence(lstm_out)

        # (L x B x 3)
        coords = self.geometry(lstm_out)

        return coords
