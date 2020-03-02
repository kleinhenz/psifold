#!/usr/bin/env python

import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import h5py
import matplotlib.pyplot as plt

class ProteinNetDataset(Dataset):
    def __init__(self, input_file, input_section):
        self.h5f = h5py.File(input_file, "r")
        self.dset = self.h5f[input_section]

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        record = self.dset[idx]

        # primary amino acid sequence (N x 1)
        primary = torch.from_numpy(record["primary"][:, np.newaxis])

        # PSSM + information content (N x 21)
        evolutionary = torch.from_numpy(np.reshape(record["evolutionary"], (-1, 21), "C"))

        # coordinate available (N x 1)
        mask = torch.from_numpy(record["mask"][:, np.newaxis])

        # tertiary structure (3N x 3)
        tertiary = torch.from_numpy(np.reshape(record["tertiary"], (-1, 3), "C"))

        return {"primary" : primary, "evolutionary" : evolutionary, "mask" : mask, "tertiary" : tertiary}

    def close(self):
        self.h5f.close()

class RGN(nn.Module):
    def __init__(self, embed_dim = 20, hidden_size = 50, num_layers = 1, dropout = 0.1):
        super(RGN, self).__init__()
        self.embed = nn.Embedding(20, embed_dim) # embedding for primary sequence
        self.LSTM = nn.LSTM(input_size = embed_dim + 21,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            dropout = 0.1)

def main():
    parser = argparse.ArgumentParser(description="train RGN model")
    parser.add_argument("--input.file", default="input.h5", dest="input_file", help="hdf5 file containing proteinnet records")
    parser.add_argument("--input.section", default="/training/90", dest="input_section", help="hdf5 section containing proteinnet records")
    args = parser.parse_args()

    dset = ProteinNetDataset(args.input_file, args.input_section)
    print(dset[0]["primary"])

    dset.close()

if __name__ == "__main__":
    main()
