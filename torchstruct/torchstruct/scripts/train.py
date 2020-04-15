#!/usr/bin/env python

import argparse
import datetime

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
import h5py
import matplotlib.pyplot as plt

from torchstruct import ProteinNetDataset, collate_fn, train, validate, RGN, GTN

def main():
    parser = argparse.ArgumentParser(description="train RGN model")
    parser.add_argument("--input.file", default="input.h5", dest="input_file", help="hdf5 file containing proteinnet records")
    parser.add_argument("--train.section", default="/training/90", dest="train_section", help="hdf5 section containing training set")
    parser.add_argument("--val.section", default="/validation", dest="val_section", help="hdf5 section containing validation set")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--max_len", type=int, default=-1)
    parser.add_argument("--model", choices=["RGN", "GTN"], default="RGN")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dset = ProteinNetDataset(args.input_file, args.train_section)
    if args.max_len > 0:
        indices = [i for i, x in enumerate(train_dset) if x["seq"].numel() <= args.max_len]
        train_dset = Subset(train_dset, indices)

    if args.train_size > 0: train_dset = Subset(train_dset, range(args.train_size))
    train_dloader = torch.utils.data.DataLoader(train_dset, batch_size = args.batch_size, shuffle=True, collate_fn=collate_fn)

    val_dset = ProteinNetDataset(args.input_file, args.val_section)
    val_dloader = torch.utils.data.DataLoader(val_dset, batch_size = args.batch_size, shuffle=False, collate_fn=collate_fn)

    if args.model == "RGN":
        model = RGN(embed_dim=20, hidden_size=100, linear_units=20, n_layers=2)
    elif args.model == "GTN":
        model = GTN(embed_dim=20, hidden_size=100, linear_units=20, n_layers=2, nhead=4)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        start = datetime.datetime.now()
        train_loss = train(model, optimizer, train_dloader, device, output_frequency = 10)
        val_loss = validate(model, val_dloader, device)
        elapsed = datetime.datetime.now() - start
        print(f"epoch {epoch:d}: elapsed = {elapsed}, train dRMSD (A) = {train_loss/100:0.3f}, val dRMSD (A) = {val_loss/100:0.3f}")

if __name__ == "__main__":
    main()
