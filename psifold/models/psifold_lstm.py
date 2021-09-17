import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence, pack_sequence, pack_padded_sequence, pad_packed_sequence

import psifold

class PsiFoldLSTM(nn.Module):
    """
    PsiFold implementation
    """
    def __init__(self, hidden_size=800, n_layers=2, dropout=0.5):
        super(PsiFoldLSTM, self).__init__()

        # save info needed to recreate model from checkpoint
        self.model_name = "psifold_lstm"
        self.model_args = {"hidden_size" : hidden_size,
                           "n_layers": n_layers,
                           "dropout": dropout}

        self.lstm = nn.LSTM(input_size=41,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            batch_first=False,
                            dropout=dropout,
                            bidirectional=True)

        self.fc = nn.Linear(2*hidden_size, 3)

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

        # (L x B x 3)
        srf = self.fc(lstm_out)

        return srf
