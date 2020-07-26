#!/usr/bin/env python

import argparse
import copy
import datetime
import math
import functools

import numpy as np
import h5py

from tqdm import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F

import psifold
from psifold import make_data_loader, count_parameters, group_by_class, pnerf
from psifold.models.psifold_trans_enc import PsiFoldTransformerEncoder
from psifold.data import PsiFoldDataset
from psifold.util import validate, train, run_train_loop

tmscore_path = "TMscore"

def restore_from_checkpoint(checkpoint, device):
    assert checkpoint["model_name"] == "psifold_transformer_encoder"
    model = PsiFoldLSTM(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    val_loss = checkpoint["val_loss"]
    train_loss_history = checkpoint["train_loss_history"]
    val_loss_history = checkpoint["val_loss_history"]
    return model, optimizer, val_loss, train_loss_history, val_loss_history

def criterion(srf_predict, batch):
    mask = batch["mask"]
    srf = batch["srf"]
    loss = (srf_predict[mask] - srf[mask]).pow(2).sum(-1).sqrt().mean()
    return loss

def compute_tm(srf_predict, batch):
    coords = pnerf(srf_predict, nfrag=int(math.sqrt(batch["seq"].size(0))))

    tm_scores = {}

    N = len(batch["id"])
    for i in range(N):
        ID = batch["id"][i]
        l = batch["length"][i]
        mask = batch["mask"][:,i]

        seq = batch["seq"][mask,i]
        ca_coords = coords[mask,i,:]
        ca_coords_ref = batch["coords"][mask,i,:]
        out = psifold.data.run_tm_score(seq, ca_coords, ca_coords_ref, tmscore_path=tmscore_path)

        tm_scores[ID] = out["tm"]

    return tm_scores

def main():
    parser = argparse.ArgumentParser(description="train RGN model")
    parser.add_argument("--input.file", default="input.h5", dest="input_file", help="hdf5 file containing proteinnet records")
    parser.add_argument("--train.section", default="/training/90", dest="train_section", help="hdf5 section containing training dataset")
    parser.add_argument("--val.section", default="/validation", dest="val_section", help="hdf5 section containing validation dataset")
    parser.add_argument("--test.section", default="/testing", dest="test_section", help="hdf5 section containing test dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_size", type=int, default=None)
    parser.add_argument("--complete_only", action="store_true")
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--tmscore_path", type=str, default="TMscore")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=None)

    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--latest_checkpoint_path", type=str, default="checkpoint_latest.pt")
    parser.add_argument("--best_checkpoint_path", type=str, default="checkpoint_best.pt")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    # model parameters
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--ff_dim", type=int, default=2048)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--dropout", type=float, default=0.1)

    args = parser.parse_args()
    print("running run_psifold...")
    print("args:", vars(args))

    global tmscore_path
    tmscore_path = args.tmscore_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if torch.cuda.is_available(): print(torch.cuda.get_device_name())

    print("loading data...")
    collate_fn = PsiFoldDataset.collate
    if args.train:
        train_dset = PsiFoldDataset(args.input_file, args.train_section, verbose=True)
        train_dloader = make_data_loader(train_dset,
                                         collate_fn,
                                         max_len=args.max_len,
                                         max_size=args.train_size,
                                         batch_size=args.batch_size,
                                         bucket_size=32*args.batch_size,
                                         complete_only=args.complete_only)
        val_dset = PsiFoldDataset(args.input_file, args.val_section, verbose=True)
        val_dset_groups = group_by_class(val_dset)
        val_dloader_dict = {k : make_data_loader(v, collate_fn, batch_size=args.batch_size) for k, v in val_dset_groups.items()}

    if args.test:
        test_dset = PsiFoldDataset(args.input_file, args.test_section, verbose=True)
        test_dset_groups = group_by_class(test_dset)
        test_dloader_dict = {k : make_data_loader(v, collate_fn, batch_size=args.batch_size) for k, v in test_dset_groups.items()}

    if args.load_checkpoint:
        print(f"restoring state from {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model, optimizer, best_val_loss, train_loss_history, val_loss_history = restore_from_checkpoint(checkpoint, device)
    else:
        model_args = {"hidden_size" : args.hidden_size, "ff_dim" : args.ff_dim, "nhead" : args.nhead, "n_layers" : args.n_layers, "dropout" : args.dropout}
        model = PsiFoldTransformerEncoder(**model_args)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        best_val_loss = math.inf
        train_loss_history = []
        val_loss_history = []

    print(f"{model.model_name}: {model.model_args}")
    n_params = count_parameters(model)
    print(f"n_params = {n_params}")

    if args.train:
        print("entering training loop...")
        model = run_train_loop(model,
                               criterion,
                               optimizer,
                               train_dloader,
                               val_dloader_dict,
                               device,
                               compute_tm,
                               max_grad_norm=args.max_grad_norm,
                               epochs=args.epochs,
                               output_frequency=60,
                               best_checkpoint_path=args.best_checkpoint_path,
                               latest_checkpoint_path=args.latest_checkpoint_path,
                               best_val_loss=best_val_loss,
                               train_loss_history=train_loss_history,
                               val_loss_history=val_loss_history)

    if args.test:
        test_loss, test_loss_by_group, tm_scores_by_group = validate(model, criterion, compute_tm, test_dloader_dict, device)
        print("test loss (A) by subgroup:\n" + "\n".join(f"{k} : {v:0.3f}" for k,v in test_loss_by_group.items()))

        print("test tm-scores by subgroup:")
        for group, tm_scores in tm_scores_by_group.items():
            scores = np.array(list(tm_scores.values()))
            q = np.quantile(scores, [0.25, 0.5, 0.75])
            print(f"{group}: {q[0]:0.2f}-{q[1]:0.2f}-{q[2]:0.2f}")

if __name__ == "__main__":
    main()
