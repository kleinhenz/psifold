#!/usr/bin/env python

import argparse
import datetime

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
import h5py
import matplotlib.pyplot as plt

from torchstruct.util import ProteinNetDataset, collate_fn, dRMSD
from torchstruct.rgn import RGN

def validate(model, val_dloader, device):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_dloader:
            for k in ["seq", "pssm", "length", "coords", "mask"]:
                batch[k] = batch[k].to(device)

            out = model(batch["seq"], batch["pssm"], batch["length"])
            loss = dRMSD(out, batch["coords"], batch["mask"])
            val_loss += loss.data

    return val_loss

def train(model, optimizer, train_dloader, device, output_frequency = 60):
    model.train()
    train_loss = 0.0

    last_output = datetime.datetime.now()
    for batch_idx, batch in enumerate(train_dloader):
        for k in ["seq", "pssm", "length", "coords", "mask"]:
            batch[k] = batch[k].to(device)

        out = model(batch["seq"], batch["pssm"], batch["length"])
        optimizer.zero_grad()
        loss = dRMSD(out, batch["coords"], batch["mask"])
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=50.0)
        train_loss += loss.data
        optimizer.step()

        if ((datetime.datetime.now() - last_output).seconds > output_frequency):
            last_output = datetime.datetime.now()
            print(f"batch {batch_idx}/{len(train_dloader)}")

    train_loss /= len(train_dloader)

    return train_loss

def main():
    parser = argparse.ArgumentParser(description="train RGN model")
    parser.add_argument("--input.file", default="input.h5", dest="input_file", help="hdf5 file containing proteinnet records")
    parser.add_argument("--train.section", default="/training/90", dest="train_section", help="hdf5 section containing training set")
    parser.add_argument("--val.section", default="/validation", dest="val_section", help="hdf5 section containing validation set")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_size", type=int, default=-1)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dset = ProteinNetDataset(args.input_file, args.train_section)
    if args.train_size > 0: train_dset = Subset(train_dset, range(args.train_size))
    train_dloader = torch.utils.data.DataLoader(train_dset, batch_size = args.batch_size, shuffle=True, collate_fn=collate_fn)

    val_dset = ProteinNetDataset(args.input_file, args.val_section)
    val_dloader = torch.utils.data.DataLoader(val_dset, batch_size = args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = RGN(hidden_size=100, linear_units=20, n_layers=2)
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
