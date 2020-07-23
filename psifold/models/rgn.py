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

class RGNDataset(Dataset):
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

        # coordinate available (3N)
        mask = torch.from_numpy(record["mask"]).bool().repeat_interleave(3)

        # tertiary structure (3N x 3)
        coords = torch.from_numpy(np.reshape(record["tertiary"], (-1, 3), "C"))

        # alpha carbon only
        if coords[0::3].eq(0).all() and coords[2::3].eq(0).all():
            mask = torch.logical_and(mask, torch.tensor([False, True, False]).repeat(N))

        out = {"id" : ID, "seq" : seq, "pssm" : pssm, "mask" : mask, "coords" : coords}

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
    coords = pad_sequence([batch[i]["coords"] for i in sorted_indices])

    return {"id" : ID, "seq" : seq, "pssm" : pssm, "mask" : mask, "coords" : coords, "length" : sorted_length}

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

        # (N-CA, CA-C, C-N)
        self.bond_lengths = nn.Parameter(torch.tensor([145.867432,152.534744,132.935516]))
        self.bond_lengths.requires_grad = False

        # (C-N-CA, N-CA-C, CA-C-N)
        # NOTE values from original rgn code given by pi - self.bond_angles
        self.bond_angles = nn.Parameter(torch.tensor([1.019982,1.204710,1.109421]))
        self.bond_angles.requires_grad = False

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
