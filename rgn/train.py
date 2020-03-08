#!/usr/bin/env python

import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import h5py
import matplotlib.pyplot as plt

from model import RGN

class ProteinNetDataset(Dataset):
    def __init__(self, input_file, input_section):
        self.h5f = h5py.File(input_file, "r")
        self.dset = self.h5f[input_section]

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        record = self.dset[idx]

        # primary amino acid sequence (N)
        seq = torch.from_numpy(record["primary"])

        # PSSM + information content (N x 21)
        pssm = torch.from_numpy(np.reshape(record["evolutionary"], (-1, 21), "C"))

        # coordinate available (N)
        mask = torch.from_numpy(record["mask"])

        # tertiary structure (3N x 3)
        coords = torch.from_numpy(np.reshape(record["tertiary"], (-1, 3), "C"))

        return {"seq" : seq, "pssm" : pssm, "mask" : mask, "coords" : coords}

    def close(self):
        self.h5f.close()

def main():
    parser = argparse.ArgumentParser(description="train RGN model")
    parser.add_argument("--input.file", default="input.h5", dest="input_file", help="hdf5 file containing proteinnet records")
    parser.add_argument("--input.section", default="/training/90", dest="input_section", help="hdf5 section containing proteinnet records")
    args = parser.parse_args()

    dset = ProteinNetDataset(args.input_file, args.input_section)

    batch_size = 1
    dloader = torch.utils.data.DataLoader(dset, batch_size = batch_size, shuffle=True)

    rgn = RGN()

    example = next(iter(dloader))
    print(rgn(example))

    dset.close()

if __name__ == "__main__":
    main()
