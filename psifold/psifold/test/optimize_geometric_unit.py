#!/usr/bin/env python

import argparse

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch

import psifold
from psifold import dRMSD, dRMSD_masked, ProteinNetDataset, make_data_loader, GeometricUnit

def run(batch, hidden_size, linear_units, epochs=1000, lr=1e-3):
    geo = GeometricUnit(hidden_size, linear_units)

    coords = batch["coords"]
    mask = batch["mask"]

    L, B  = batch["seq"].size()

    inp = torch.rand(L, B, hidden_size, requires_grad=True)
    optimizer = torch.optim.Adam([inp], lr=lr)

    loss_history = []
    epochs = 1000
    for epoch in tqdm(range(epochs)):
        out = geo(inp)
        loss = dRMSD_masked(coords, out, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item()/100.0)

    print(f"loss = {loss_history[-1]:0.2e}")
    loss_history = np.array(loss_history)

    fig, ax = plt.subplots()
    ax.plot(loss_history)
    ax.set_xlabel("iter")
    ax.set_ylabel("dRMSD")
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="train geometric unit on single batch")
    parser.add_argument("--input.file", default="input.h5", dest="input_file", help="hdf5 file containing proteinnet records")
    parser.add_argument("--train.section", default="/training/90", dest="train_section", help="hdf5 section containing training set")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--linear_units", type=int, default=32)

    args = parser.parse_args()

    dset = ProteinNetDataset(args.input_file, args.train_section)
    dloader = make_data_loader(dset, batch_size=args.batch_size, max_len=args.max_len)
    batch = next(iter(dloader))

    print(batch["id"])
    print(batch["length"])

    run(batch, args.hidden_size, args.linear_units, lr=args.learning_rate, epochs=args.epochs)

if __name__ == "__main__":
    main()
