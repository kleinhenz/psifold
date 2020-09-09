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
from psifold.models.psifold_lstm import PsiFoldLSTM
from psifold.models.baseline import Baseline
from psifold.data import PsiFoldDataset
from psifold.util import validate, train, run_train_loop
from psifold.optimization import poly_schedule, Lamb

tmscore_path = "TMscore"

cosine_similarity = nn.CosineSimilarity(dim=-1)

def make_transformer(args):
    model_args = {"hidden_size" : args.hidden_size, "n_layers" : args.n_layers, "dropout" : args.dropout}
    model = PsiFoldTransformerEncoder(**model_args)
    return model

def make_lstm(args):
    model_args = {"hidden_size" : args.hidden_size, "n_layers" : args.n_layers, "dropout" : args.dropout}
    model = PsiFoldLSTM(**model_args)
    return model

def make_baseline(args):
    model_args = {"hidden_size" : args.hidden_size, "n_layers" : args.n_layers, "dropout" : args.dropout}
    model = Baseline(**model_args)
    return model

def restore_from_checkpoint(checkpoint, device):
    model_name = checkpoint["model_name"]
    model_args = checkpoint["model_args"]

    if model_name == "psifold_transformer_encoder":
        model = PsiFoldTransformerEncoder(**model_args)
    elif model_name == "psifold_lstm":
        model = PsiFoldLSTM(**model_args)
    elif model_name == "baseline":
        model = Baseline(**model_args)
    else:
        assert False

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer = Lamb(model.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=1.0)
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    val_loss = checkpoint["val_loss"]
    train_loss_history = checkpoint["train_loss_history"]
    val_loss_history = checkpoint["val_loss_history"]
    return model, optimizer, scheduler, val_loss, train_loss_history, val_loss_history

def criterion(srf_predict, batch):
    mask = batch["mask"]
    srf = batch["srf"]
    loss = 1.0 - cosine_similarity(srf_predict[mask], srf[mask]).mean()
#    loss = (srf_predict[mask] - srf[mask]).pow(2).sum(-1).sqrt().mean()
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
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--tmscore_path", type=str, default="TMscore")

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--accumulate_steps", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=1024)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--enable_amp", action="store_true")

    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--reset_optim", action="store_true")

    parser.add_argument("--latest_checkpoint_path", type=str, default="checkpoint_latest.pt")
    parser.add_argument("--best_checkpoint_path", type=str, default="checkpoint_best.pt")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    subparsers = parser.add_subparsers()

    transformer_parser = subparsers.add_parser("transformer")
    lstm_parser = subparsers.add_parser("lstm")
    baseline_parser = subparsers.add_parser("baseline")

    # model parameters
    transformer_parser.add_argument("--hidden_size", type=int, default=512)
    transformer_parser.add_argument("--n_layers", type=int, default=8)
    transformer_parser.add_argument("--dropout", type=float, default=0.1)
    transformer_parser.set_defaults(make_model=make_transformer)

    lstm_parser.add_argument("--hidden_size", type=int, default=800)
    lstm_parser.add_argument("--n_layers", type=int, default=2)
    lstm_parser.add_argument("--dropout", type=float, default=0.5)
    lstm_parser.set_defaults(make_model=make_lstm)

    baseline_parser.add_argument("--hidden_size", type=int, default=512)
    baseline_parser.add_argument("--n_layers", type=int, default=2)
    baseline_parser.add_argument("--dropout", type=float, default=0.1)
    baseline_parser.set_defaults(make_model=make_baseline)

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
        val_dset_groups["train"] = torch.utils.data.Subset(train_dset, torch.randperm(len(train_dset))[:32])
        val_dloader_dict = {k : make_data_loader(v, collate_fn, batch_size=args.batch_size) for k, v in val_dset_groups.items()}

    if args.test:
        test_dset = PsiFoldDataset(args.input_file, args.test_section, verbose=True)
        test_dset_groups = group_by_class(test_dset)
        test_dloader_dict = {k : make_data_loader(v, collate_fn, batch_size=args.batch_size) for k, v in test_dset_groups.items()}

    if args.load_checkpoint:
        print(f"restoring state from {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model, optimizer, scheduler, best_val_loss, train_loss_history, val_loss_history = restore_from_checkpoint(checkpoint, device)
    else:
        model = args.make_model(args)
        model.to(device)
        best_val_loss = math.inf
        train_loss_history = []
        val_loss_history = []

    if (not args.load_checkpoint) or (args.load_checkpoint and args.reset_optim):
        optimizer = Lamb(model.parameters(), lr=args.learning_rate)
        n_steps = args.epochs * len(train_dloader)
        warmup_frac = args.warmup_steps / n_steps
        schedule_f = poly_schedule(warmup=warmup_frac, degree=1.0)
        f = lambda step : schedule_f(step/n_steps)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, f)

    print(f"{model.model_name}: {model.model_args}")
    n_params = count_parameters(model)
    print(f"n_params = {n_params}")

    if args.train:
        print("entering training loop...")
        checkpoint_extra_data = {"args" : vars(args)}
        model = run_train_loop(model,
                               criterion,
                               optimizer,
                               scheduler,
                               train_dloader,
                               val_dloader_dict,
                               device,
                               compute_tm,
                               accumulate_steps=args.accumulate_steps,
                               max_grad_norm=args.max_grad_norm,
                               enable_amp=args.enable_amp,
                               epochs=args.epochs,
                               output_frequency=60,
                               best_checkpoint_path=args.best_checkpoint_path,
                               latest_checkpoint_path=args.latest_checkpoint_path,
                               checkpoint_extra_data=checkpoint_extra_data,
                               best_val_loss=best_val_loss,
                               train_loss_history=train_loss_history,
                               val_loss_history=val_loss_history)

    if args.test:
        test_loss, test_loss_by_group, tm_scores_by_group = validate(model, criterion, compute_tm, test_dloader_dict, device)
        print("test loss (A) by subgroup:\n" + "\n".join(f"{k} : {v:0.3f}" for k,v in test_loss_by_group.items()))

        print("test tm-scores by subgroup:")
        for group, tm_scores in tm_scores_by_group.items():
            scores = np.array(list(tm_scores.values()))
            q = np.quantile(scores, [0.0, 0.25, 0.5, 0.75, 1.0])
            print(f"{group}: {q[0]:0.2f}-{q[1]:0.2f}-{q[2]:0.2f}-{q[3]:0.2f}-{q[4]:0.2f}")

if __name__ == "__main__":
    main()
