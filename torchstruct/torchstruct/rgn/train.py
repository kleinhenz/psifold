#!/usr/bin/env python

import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import h5py
import matplotlib.pyplot as plt

from torchstruct.util import ProteinNetDataset
from torchstruct.rgn import RGN

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

    out = rgn(example)
    print(example["seq"].size())
    print(out.size())

    dset.close()

if __name__ == "__main__":
    main()
