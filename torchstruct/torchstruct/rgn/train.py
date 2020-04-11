#!/usr/bin/env python

import argparse

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
import h5py
import matplotlib.pyplot as plt

from torchstruct.util import ProteinNetDataset, collate_fn, dRMSD
from torchstruct.rgn import RGN

def main():
    parser = argparse.ArgumentParser(description="train RGN model")
    parser.add_argument("--input.file", default="input.h5", dest="input_file", help="hdf5 file containing proteinnet records")
    parser.add_argument("--train.section", default="/training/90", dest="train_section", help="hdf5 section containing training set")
    parser.add_argument("--val.section", default="/validation", dest="val_section", help="hdf5 section containing validation set")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_size", type=int, default=-1)
    args = parser.parse_args()

    h5f = h5py.File(args.input_file, "r")

    train = ProteinNetDataset(h5f[args.train_section])
    if args.train_size > 0:
        train = Subset(train, range(args.train_size))

    dloader = torch.utils.data.DataLoader(train, batch_size = args.batch_size, shuffle=True, collate_fn=collate_fn)

    val = ProteinNetDataset(h5f[args.val_section])
    val = collate_fn([val[i] for i in range(len(val))])

    model = RGN(hidden_size=100, linear_units=20, n_layers=2)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    N = len(train)
    N_batches = N // args.batch_size

    for epoch in range(args.epochs):
        train_loss = 0.0
        for batch_idx, batch in enumerate(dloader):
            out = model(batch)
            optimizer.zero_grad()
            loss = dRMSD(out, batch["coords"], batch["mask"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
            train_loss += loss.data
            optimizer.step()

        train_loss /= len(dloader)
        out = model(val)
        val_loss = dRMSD(out, val["coords"], val["mask"])

        print(f"epoch {epoch:d}: train_loss = {train_loss:0.4e}, val_loss = {val_loss:0.4e}")

    h5f.close()

if __name__ == "__main__":
    main()
