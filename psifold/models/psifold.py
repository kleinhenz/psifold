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

class PsiFoldDataset(Dataset):
    def __init__(self, fname, section, verbose=False):
        self.class_re = re.compile("(.*)#")

        with h5py.File(fname, "r") as h5f:
            h5dset = h5f[section][:]

        N = len(h5dset)
        r = tqdm(range(N)) if verbose else range(N)
        self.dset = [self.read_record(h5dset[i]) for i in r]

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        return self.dset[idx]

    def read_record(self, record):
        # identifier
        ID = record["id"]

        # primary amino acid sequence (N)
        seq = torch.from_numpy(record["primary"])
        N = seq.size(0)

        # PSSM + information content (N x 21)
        pssm = torch.from_numpy(np.reshape(record["evolutionary"], (-1, 21), "C"))

        # coordinate available (N)
        mask = torch.from_numpy(record["mask"]).bool()

        # tertiary structure (N x 3)
        coords = torch.from_numpy(np.reshape(record["tertiary"], (-1, 3), "C")) / 100.0
        coords = coords[1::3] # c-alpha only

        # fill masked coords with nan to keep track
        coords = coords.masked_fill(mask.logical_not().unsqueeze(1), float("nan"))

        # srf coords (N x 3)
        r, theta, phi = psifold.geometry.internal_coords(coords, pad=True)
        srf = psifold.geometry.internal_to_srf(r, theta, phi)

        mask = (srf == srf).all(dim=-1)
        # always mask out first residue since we don't have full srf coords
        mask[0] = False

        out = {"id" : ID, "seq" : seq, "pssm" : pssm, "mask" : mask, "coords" : coords, "srf" : srf}

        # extract class from id if present
        class_match = self.class_re.match(ID)
        if class_match: out["class"] = class_match[1]

        return out

def collate_fn(batch):
    length = torch.tensor([x["seq"].size(0) for x in batch])
    sorted_length, sorted_indices = length.sort(0, True)

    ID = [batch[i]["id"] for i in sorted_indices]
    seq = pad_sequence([batch[i]["seq"] for i in sorted_indices])
    pssm = pad_sequence([batch[i]["pssm"] for i in sorted_indices])
    mask = pad_sequence([batch[i]["mask"] for i in sorted_indices])
    coords = pad_sequence([batch[i]["coords"] for i in sorted_indices], batch_first=False, padding_value=float("nan"))
    srf = pad_sequence([batch[i]["srf"] for i in sorted_indices], batch_first=False, padding_value=float("nan"))

    return {"id" : ID, "seq" : seq, "pssm" : pssm, "mask" : mask, "coords" : coords, "srf" : srf, "length" : sorted_length}

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

    def forward(self, seq, pssm, length):
        """
        seq: (L x B)
        pssm: (L x B x 21)
        length: (L,)
        """

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

class PsiFoldTransformerEncoder(nn.Module):
    """
    PsiFold implementation
    """
    def __init__(self, hidden_size=256, ff_dim=512, nhead=8, n_layers=3, dropout=0.1):
        super(PsiFoldTransformerEncoder, self).__init__()

        # save info needed to recreate model from checkpoint
        self.model_name = "psifold_transformer_encoder"
        self.model_args = {"hidden_size" : hidden_size,
                           "ff_dim" : ff_dim,
                           "nhead" : nhead,
                           "n_layers": n_layers,
                           "dropout": dropout}

        self.fc0 = nn.Linear(41, hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(hidden_size, nhead=nhead, dim_feedforward=ff_dim, dropout=dropout)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.fc1 = nn.Linear(hidden_size, 3)

    def forward(self, seq, pssm, length):
        """
        seq: (L x B)
        pssm: (L x B x 21)
        length: (L,)
        """

        L, B = seq.size()

        # (L x B x 20)
        seq = F.one_hot(seq, 20).type(pssm.dtype)

        # (L x B x (20 + 21))
        encoder_in = torch.cat((seq, pssm), dim=2)

        # (L x B x hidden_size)
        encoder_in = self.fc0(encoder_in)

        # (L x B x hidden_size)
        encoder_in = self.pos_encoder(encoder_in)

        # (L x B)
        mask = torch.arange(L, device=seq.device).expand(B, L) >= length.unsqueeze(1)

        # (L x B x hidden_size)
        encoder_out = self.encoder(encoder_in, src_key_padding_mask=mask)

        # (L x B x 3)
        srf = self.fc1(encoder_out)

        return srf
