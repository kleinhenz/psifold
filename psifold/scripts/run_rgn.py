#!/usr/bin/env python

import argparse
import copy
import datetime
import math
import re

import numpy as np
import h5py

from tqdm import tqdm

import torch
from torch import nn, optim
from torch.utils.data import Dataset, Subset
from torch.nn.utils.rnn import pad_sequence, pack_sequence

import psifold
from psifold import RGN, dRMSD_masked, make_data_loader, count_parameters

tmscore_path = "TMscore"

def restore_from_checkpoint(checkpoint, device):
    model = RGN(**checkpoint["model_args"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    val_loss = checkpoint["val_loss"]
    train_loss_history = checkpoint["train_loss_history"]
    val_loss_history = checkpoint["val_loss_history"]
    return model, optimizer, val_loss, train_loss_history, val_loss_history

class RGNDataset(Dataset):
    def __init__(self, fname, section, verbose=False):
        self.class_re = re.compile("(.*)#")

        with h5py.File(fname, "r") as h5f:
            h5dset = h5f[section][:]

        N = len(h5dset)
        r = tqdm(range(N)) if verbose else range(N)
        self.dset = [self.read_record(h5dset[i]) for i in r]

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        return self.dset[idx]

    def read_record(self, record):
        # identifier
        ID = record["id"]

        # primary amino acid sequence (N)
        seq = torch.from_numpy(record["primary"])
        N = seq.size(0)

        # PSSM + information content (N x 21)
        pssm = torch.from_numpy(np.reshape(record["evolutionary"], (-1, 21), "C"))

        # coordinate available (3N)
        mask = torch.from_numpy(record["mask"]).bool().repeat_interleave(3)

        # tertiary structure (3N x 3)
        coords = torch.from_numpy(np.reshape(record["tertiary"], (-1, 3), "C"))

        # alpha carbon only
        if coords[0::3].eq(0).all() and coords[2::3].eq(0).all():
            mask = torch.logical_and(mask, torch.tensor([False, True, False]).repeat(N))

        out = {"id" : ID, "seq" : seq, "pssm" : pssm, "mask" : mask, "coords" : coords}

        # extract class from id if present
        class_match = self.class_re.match(ID)
        if class_match: out["class"] = class_match[1]

        return out

def collate_fn(batch):
    length = torch.tensor([x["seq"].size(0) for x in batch])
    sorted_length, sorted_indices = length.sort(0, True)

    ID = [batch[i]["id"] for i in sorted_indices]
    seq = pad_sequence([batch[i]["seq"] for i in sorted_indices])
    pssm = pad_sequence([batch[i]["pssm"] for i in sorted_indices])
    mask = pad_sequence([batch[i]["mask"] for i in sorted_indices])
    coords = pad_sequence([batch[i]["coords"] for i in sorted_indices])

    return {"id" : ID, "seq" : seq, "pssm" : pssm, "mask" : mask, "coords" : coords, "length" : sorted_length}

def group_by_class(dset):
    classes = np.array([x["class"] for x in dset])
    out = {c : Subset(dset, np.nonzero(classes == c)[0]) for c in np.unique(classes)}
    return out

def to_device(batch, device):
    for k in ["seq", "pssm", "length", "coords", "mask"]:
        batch[k] = batch[k].to(device)

def tm_score_batch(batch, coords):
    tm_scores = {}

    N = len(batch["id"])
    for i in range(N):
        ID = batch["id"][i]
        l = batch["length"][i]
        seq = batch["seq"][:l,i]
        ca_coords = coords[1:(3*l):3,i,:] / 100.0
        ca_coords_ref = batch["coords"][1:(3*l):3,i,:] / 100.0
        out = psifold.data.run_tm_score(seq, ca_coords, ca_coords_ref, tmscore_path=tmscore_path)

        tm_scores["ID"] = out["tm"]

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
                out = model(batch["seq"], batch["pssm"], batch["length"])
                loss = dRMSD_masked(out, batch["coords"], batch["mask"])

                val_loss += loss.item()
                val_loss_group += loss.item()

                tm_scores_group.update(tm_score_batch(batch, out))

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
        out = model(batch["seq"], batch["pssm"], batch["length"])
        optimizer.zero_grad()
        loss = dRMSD_masked(out, batch["coords"], batch["mask"])
        loss.backward()

        if max_grad_norm: nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        train_loss += loss.item()
        optimizer.step()

        if ((datetime.datetime.now() - last_output).seconds > output_frequency):
            last_output = datetime.datetime.now()
            print(f"batch {batch_idx}/{len(train_dloader)}")

    train_loss /= len(train_dloader)

    return train_loss

def run_train_loop(model, optimizer, train_dloader, val_dloader_dict, device, max_grad_norm=None, epochs=10, output_frequency=60, checkpoint_file="checkpoint.pt", best_val_loss=math.inf, train_loss_history = [], val_loss_history = []):
    best_model_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        start = datetime.datetime.now()
        train_loss = train(model, optimizer, train_dloader, device, max_grad_norm=max_grad_norm, output_frequency=output_frequency)
        val_loss, val_loss_by_group, tm_scores_by_group = validate(model, val_dloader_dict, device)
        elapsed = datetime.datetime.now() - start
        print(f"epoch {epoch:d}: elapsed = {elapsed}, train dRMSD (A) = {train_loss/100:0.3f}, val dRMSD (A) = {val_loss/100:0.3f}")

        print("val dRMSD (A) by subgroup:\n" + "\n".join(f"{k} : {v/100:0.3f}" for k,v in val_loss_by_group.items()))

        print("test tm-scores by subgroup:")
        for group, tm_scores in tm_scores_by_group.items():
            scores = np.array(list(tm_scores.values()))
            q = np.quantile(scores, [0.25, 0.5, 0.75])
            print(f"{group}: {q[0]:0.2f}-{q[1]:0.2f}-{q[2]:0.2f}")

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        # checkpoint if we improve validation loss
        if val_loss < best_val_loss:
            print("saving checkpoint")
            best_val_loss = val_loss
            best_model_state_dict = copy.deepcopy(model.state_dict())
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
            torch.save(checkpoint, checkpoint_file)

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

    parser.add_argument("--save_checkpoint", type=str, default="checkpoint.pt")
    parser.add_argument("--load_checkpoint", type=str, default="")

    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")

    # rgn parameters
    parser.add_argument("--hidden_size", type=int, default=800)
    parser.add_argument("--alphabet_size", type=int, default=60)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)

    args = parser.parse_args()
    print("running run_rgn...")
    print("args:", vars(args))

    global tmscore_path
    tmscore_path = args.tmscore_path

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if torch.cuda.is_available(): print(torch.cuda.get_device_name())

    print("loading data...")
    if args.train:
        train_dset = RGNDataset(args.input_file, args.train_section, verbose=True)
        train_dloader = make_data_loader(train_dset,
                                         collate_fn,
                                         max_len=args.max_len,
                                         max_size=args.train_size,
                                         batch_size=args.batch_size,
                                         bucket_size=32*args.batch_size,
                                         complete_only=args.complete_only)
        val_dset = RGNDataset(args.input_file, args.val_section, verbose=True)
        val_dset_groups = group_by_class(val_dset)
        val_dloader_dict = {k : make_data_loader(v, collate_fn, batch_size=args.batch_size) for k, v in val_dset_groups.items()}

    if args.test:
        test_dset = RGNDataset(args.input_file, args.test_section, verbose=True)
        test_dset_groups = group_by_class(test_dset)
        test_dloader_dict = {k : make_data_loader(v, collate_fn, batch_size=args.batch_size) for k, v in test_dset_groups.items()}

    model_args = {"hidden_size" : args.hidden_size, "alphabet_size" : args.alphabet_size, "n_layers" : args.n_layers, "dropout" : args.dropout}

    if args.load_checkpoint:
        print(f"restoring state from {args.load_checkpoint}")
        checkpoint = torch.load(args.load_checkpoint, map_location=device)
        model, optimizer, best_val_loss, train_loss_history, val_loss_history = restore_from_checkpoint(checkpoint, device)
    else:
        model = RGN(**model_args)
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
        best_val_loss = math.inf
        train_loss_history = []
        val_loss_history = []

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
                               checkpoint_file=args.save_checkpoint,
                               best_val_loss=best_val_loss,
                               train_loss_history=train_loss_history,
                               val_loss_history=val_loss_history)

    if args.test:
        test_loss, test_loss_by_group, tm_scores_by_group = validate(model, test_dloader_dict, device)
        print("test dRMSD (A) by subgroup:\n" + "\n".join(f"{k} : {v/100:0.3f}" for k,v in test_loss_by_group.items()))

        print("test tm-scores by subgroup:")
        for group, tm_scores in tm_scores_by_group.items():
            scores = np.array(list(tm_scores.values()))
            q = np.quantile(scores, [0.25, 0.5, 0.75])
            print(f"{group}: {q[0]:0.2f}-{q[1]:0.2f}-{q[2]:0.2f}")

if __name__ == "__main__":
    main()
