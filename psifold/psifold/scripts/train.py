#!/usr/bin/env python

import argparse
import datetime

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
import h5py
import matplotlib.pyplot as plt

from psifold import ProteinNetDataset, make_data_loader, train, validate, make_model

def restore_from_checkpoint(checkpoint):
    model = make_model(checkpoint["model"], checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer

def main():
    parser = argparse.ArgumentParser(description="train RGN model")
    parser.add_argument("--input.file", default="input.h5", dest="input_file", help="hdf5 file containing proteinnet records")
    parser.add_argument("--train.section", default="/training/90", dest="train_section", help="hdf5 section containing training set")
    parser.add_argument("--val.section", default="/validation", dest="val_section", help="hdf5 section containing validation set")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train_size", type=int, default=-1)
    parser.add_argument("--max_len", type=int, default=-1)
    parser.add_argument("--model", choices=["rgn", "psifold"], default="psifold")
    parser.add_argument("--save_checkpoint", type=str, default="checkpoint.pt")
    parser.add_argument("--load_checkpoint", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("loading data...")
    train_dset = ProteinNetDataset(args.input_file, args.train_section)
    train_dloader = make_data_loader(train_dset,
                                     max_len=args.max_len,
                                     max_size=args.train_size,
                                     batch_size=args.batch_size,
                                     bucket_size=32*args.batch_size)

    val_dset = ProteinNetDataset(args.input_file, args.val_section)
    val_dloader = make_data_loader(val_dset)

    # TODO get these from command line
    if args.model == "rgn":
        model_args = {"hidden_size" : 64, "linear_units" : 32, "n_layers" : 2, "dropout" : 0.1}
    elif args.model == "psifold":
        model_args = {"hidden_size" : 64, "linear_units" : 32, "n_layers" : 2, "nhead" : 4, "dim_feedforward" : 256, "dropout" : 0.1}

    if args.load_checkpoint:
        print(f"restoring state from {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint)
        model, optimizer = restore_from_checkpoint(checkpoint)
    else:
        model = make_model(args.model, model_args)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    model.to(device)

    train_loss_history = []
    val_loss_history = []
    print("entering training loop...")
    for epoch in range(args.epochs):
        start = datetime.datetime.now()
        train_loss = train(model, optimizer, train_dloader, device, output_frequency = 10)
        val_loss = validate(model, val_dloader, device)
        elapsed = datetime.datetime.now() - start
        print(f"epoch {epoch:d}: elapsed = {elapsed}, train dRMSD (A) = {train_loss/100:0.3f}, val dRMSD (A) = {val_loss/100:0.3f}")

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        checkpoint = {
            "model" : args.model,
            "model_args" : model_args,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss_history": train_loss_history,
            "val_loss_history" : val_loss_history
            }
        torch.save(checkpoint, args.save_checkpoint)

if __name__ == "__main__":
    main()
