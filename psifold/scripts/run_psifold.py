#!/usr/bin/env python

import argparse
import copy
import datetime
import math

import numpy as np
import h5py

from tqdm import tqdm

import torch
from torch import nn, optim
import torch.nn.functional as F

import psifold
from psifold.models.psifold import PsiFoldLSTM, PsiFoldTransformerEncoder, PsiFoldDataset, collate_fn
from psifold import make_data_loader, count_parameters, to_device, group_by_class, pnerf

tmscore_path = "TMscore"

def make_model(model_name, model_args):
    if model_name == "psifold_lstm":
        model = PsiFoldLSTM(**model_args)
    elif model_name == "psifold_transformer_encoder":
        model = PsiFoldTransformerEncoder(**model_args)
    else:
        raise Exception(f"model: {model_name} not recognized")
    return model

def restore_from_checkpoint(checkpoint, device):
    model = make_model(checkpoint["model_name"], checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    val_loss = checkpoint["val_loss"]
    train_loss_history = checkpoint["train_loss_history"]
    val_loss_history = checkpoint["val_loss_history"]
    return model, optimizer, val_loss, train_loss_history, val_loss_history

def tm_score_batch(batch, coords):
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

def validate(model, val_dloader_dict, device):
    model.eval()
    val_loss = 0.0
    val_loss_by_group = {}
    tm_scores_by_group = {}

    with torch.no_grad():
        for group, dloader in val_dloader_dict.items():
            val_loss_group = 0.0
            tm_scores_group = {}
            for batch in dloader:
                to_device(batch, device)
                seq, pssm, length, srf, mask = [batch[k] for k in ["seq", "pssm", "length", "srf", "mask"]]
                srf_predict = model(seq, pssm, length)
                loss = (srf_predict[mask] - srf[mask]).pow(2).sum(-1).sqrt().mean()

                val_loss += loss.item()
                val_loss_group += loss.item()

                coords_predict = pnerf(srf_predict, nfrag=int(math.sqrt(seq.size(0))))
                tm_scores_group.update(tm_score_batch(batch, coords_predict))

            val_loss_by_group[group] = val_loss_group / len(dloader)
            tm_scores_by_group[group] = tm_scores_group

    N = sum(len(dloader) for group, dloader in val_dloader_dict.items())
    val_loss /= N

    return val_loss, val_loss_by_group, tm_scores_by_group

def train(model, optimizer, train_dloader, device, max_grad_norm=None, output_frequency = 60):
    model.train()
    train_loss = 0.0

    last_output = datetime.datetime.now()
    for batch_idx, batch in enumerate(train_dloader):
        to_device(batch, device)
        seq, pssm, length, srf, mask = [batch[k] for k in ["seq", "pssm", "length", "srf", "mask"]]
        srf_predict = model(seq, pssm, length)
        optimizer.zero_grad()
        loss = (srf_predict[mask] - srf[mask]).pow(2).sum(-1).sqrt().mean()
        loss.backward()

        if max_grad_norm: nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        train_loss += loss.item()
        optimizer.step()

        if ((datetime.datetime.now() - last_output).seconds > output_frequency):
            last_output = datetime.datetime.now()
            print(f"batch {batch_idx}/{len(train_dloader)}")

    train_loss /= len(train_dloader)

    return train_loss

def run_train_loop(model, optimizer, train_dloader, val_dloader_dict, device,
        max_grad_norm=None, epochs=10, output_frequency=60,
        best_checkpoint_path="checkpoint_best.pt", latest_checkpoint_path="checkpoint_latest.pt",
        best_val_loss=math.inf, train_loss_history = [], val_loss_history = []):

    best_model_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        start = datetime.datetime.now()
        train_loss = train(model, optimizer, train_dloader, device, max_grad_norm=max_grad_norm, output_frequency=output_frequency)
        val_loss, val_loss_by_group, tm_scores_by_group = validate(model, val_dloader_dict, device)
        elapsed = datetime.datetime.now() - start
        print(f"epoch {epoch:d}: elapsed = {elapsed}, train loss = {train_loss:0.3f}, val loss = {val_loss:0.3f}")

        print("val loss (A) by subgroup:\n" + "\n".join(f"{k} : {v:0.3f}" for k,v in val_loss_by_group.items()))

        print("test tm-scores by subgroup:")
        for group, tm_scores in tm_scores_by_group.items():
            scores = np.array(list(tm_scores.values()))
            q = np.quantile(scores, [0.25, 0.5, 0.75])
            print(f"{group}: {q[0]:0.2f}-{q[1]:0.2f}-{q[2]:0.2f}")

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        checkpoint = {
            "model_name" : model.model_name,
            "model_args" : model.model_args,
            "epoch": epoch,
            "val_loss" : val_loss,
            "train_loss_history" : train_loss_history,
            "val_loss_history" : val_loss_history,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            }

        if latest_checkpoint_path:
            torch.save(checkpoint, latest_checkpoint_path)

        # checkpoint if we improve validation loss
        if val_loss < best_val_loss:
            print(f"val loss improved ({best_val_loss:0.3f} -> {val_loss:0.3f})\nsaving checkpoint")
            best_val_loss = val_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())

            if best_checkpoint_path:
                torch.save(checkpoint, best_checkpoint_path)

    model.load_state_dict(best_model_state_dict)

    return model

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
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--max_grad_norm", type=float, default=None)

    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--latest_checkpoint_path", type=str, default="checkpoint_latest.pt")
    parser.add_argument("--best_checkpoint_path", type=str, default="checkpoint_best.pt")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    # model parameters
    parser.add_argument("--model", choices=["psifold_lstm", "psifold_transformer_encoder"], default="psifold_lstm")
    parser.add_argument("--lstm_hidden_size", type=int, default=800)
    parser.add_argument("--lstm_n_layers", type=int, default=2)
    parser.add_argument("--lstm_dropout", type=float, default=0.5)

    parser.add_argument("--trenc_hidden_size", type=int, default=256)
    parser.add_argument("--trenc_ff_dim", type=int, default=512)
    parser.add_argument("--trenc_nhead", type=int, default=8)
    parser.add_argument("--trenc_n_layers", type=int, default=3)
    parser.add_argument("--trenc_dropout", type=float, default=0.1)

    args = parser.parse_args()
    print("running run_psifold...")
    print("args:", vars(args))

    global tmscore_path
    tmscore_path = args.tmscore_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if torch.cuda.is_available(): print(torch.cuda.get_device_name())

    print("loading data...")
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

    if args.model == "psifold_lstm":
        model_args = {"hidden_size" : args.lstm_hidden_size,
                      "n_layers" : args.lstm_n_layers,
                      "dropout" : args.lstm_dropout}

    elif args.model == "psifold_transformer_encoder":
        model_args = {"hidden_size" : args.trenc_hidden_size,
                      "ff_dim" : args.trenc_ff_dim,
                      "nhead" : args.trenc_nhead,
                      "n_layers" : args.trenc_n_layers,
                      "dropout" : args.trenc_dropout}

    if args.load_checkpoint:
        print(f"restoring state from {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model, optimizer, best_val_loss, train_loss_history, val_loss_history = restore_from_checkpoint(checkpoint, device)
    else:
        model = make_model(args.model, model_args)
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
                               optimizer,
                               train_dloader,
                               val_dloader_dict,
                               device,
                               max_grad_norm=args.max_grad_norm,
                               epochs=args.epochs,
                               output_frequency=60,
                               best_checkpoint_path=args.best_checkpoint_path,
                               latest_checkpoint_path=args.latest_checkpoint_path,
                               best_val_loss=best_val_loss,
                               train_loss_history=train_loss_history,
                               val_loss_history=val_loss_history)

    if args.test:
        test_loss, test_loss_by_group, tm_scores_by_group = validate(model, test_dloader_dict, device)
        print("test loss (A) by subgroup:\n" + "\n".join(f"{k} : {v:0.3f}" for k,v in test_loss_by_group.items()))

        print("test tm-scores by subgroup:")
        for group, tm_scores in tm_scores_by_group.items():
            scores = np.array(list(tm_scores.values()))
            q = np.quantile(scores, [0.25, 0.5, 0.75])
            print(f"{group}: {q[0]:0.2f}-{q[1]:0.2f}-{q[2]:0.2f}")

if __name__ == "__main__":
    main()
