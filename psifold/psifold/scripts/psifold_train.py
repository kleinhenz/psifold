#!/usr/bin/env python

import argparse
import datetime
import math

import torch
from torch import nn, optim

import numpy as np
import h5py
import matplotlib.pyplot as plt

from psifold import ProteinNetDataset, make_data_loader, group_by_class, make_model, restore_from_checkpoint, run_train_loop

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    parser = argparse.ArgumentParser(description="train RGN model")
    parser.add_argument("--input.file", default="input.h5", dest="input_file", help="hdf5 file containing proteinnet records")
    parser.add_argument("--train.section", default="/training/90", dest="train_section", help="hdf5 section containing training set")
    parser.add_argument("--val.section", default="/validation", dest="val_section", help="hdf5 section containing validation set")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--max_len", type=int, default=None)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=None)

    parser.add_argument("--save_checkpoint", type=str, default="checkpoint.pt")
    parser.add_argument("--load_checkpoint", type=str, default="")

    parser.add_argument("--model", choices=["rgn", "psifold"], default="psifold")

    # rgn parameters
    parser.add_argument("--rgn_hidden_size", type=int, default=64)
    parser.add_argument("--rgn_linear_units", type=int, default=32)
    parser.add_argument("--rgn_n_layers", type=int, default=2)
    parser.add_argument("--rgn_dropout", type=float, default=0.5)

    # psifold argument
    parser.add_argument("--psifold_hidden_size", type=int, default=64)
    parser.add_argument("--psifold_linear_units", type=int, default=32)
    parser.add_argument("--psifold_n_layers", type=int, default=2)
    parser.add_argument("--psifold_nhead", type=int, default=4)
    parser.add_argument("--psifold_dim_feedforward", type=int, default=256)
    parser.add_argument("--psifold_dropout", type=float, default=0.5)

    args = parser.parse_args()
    print("running psifold_train...")
    print("args:", vars(args))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if torch.cuda.is_available(): print(torch.cuda.get_device_name())

    print("loading data...")
    train_dset = ProteinNetDataset(args.input_file, args.train_section)
    train_dloader = make_data_loader(train_dset,
                                     max_len=args.max_len,
                                     max_size=args.train_size,
                                     batch_size=args.batch_size,
                                     bucket_size=32*args.batch_size)
    val_dset = ProteinNetDataset(args.input_file, args.val_section)
    val_dset_groups = group_by_class(val_dset)
    val_dloader_dict = {k : make_data_loader(v, batch_size=args.batch_size) for k, v in val_dset_groups.items()}

    if args.model == "rgn":
        model_args = {"hidden_size" : args.rgn_hidden_size, "linear_units" : args.rgn_linear_units, "n_layers" : args.rgn_n_layers, "dropout" : args.rgn_dropout}
    elif args.model == "psifold":
        model_args = {"hidden_size" : args.psifold_hidden_size, "linear_units" : args.psifold_linear_units, "n_layers" : args.psifold_n_layers, "nhead" : args.psifold_nhead, "dim_feedforward" : args.psifold_dim_feedforward, "dropout" : args.psifold_dropout}

    if args.load_checkpoint:
        print(f"restoring state from {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint)
        model, optimizer, best_val_loss = restore_from_checkpoint(checkpoint, device)
    else:
        model = make_model(args.model, model_args)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        best_val_loss = math.inf

    n_params = count_parameters(model)
    print(f"n_params = {n_params}")

    print("entering training loop...")
    model = run_train_loop(model,
                           optimizer,
                           train_dloader,
                           val_dloader_dict,
                           device,
                           max_grad_norm=args.max_grad_norm,
                           epochs=args.epochs,
                           output_frequency=60,
                           checkpoint_file=args.save_checkpoint,
                           best_val_loss=best_val_loss)

if __name__ == "__main__":
    main()
