import copy
import datetime
import math

import torch
from torch import nn, optim

from psifold import dRMSD_masked, RGN, PsiFold, Baseline

def to_device(batch, device):
    for k in ["seq", "kmer", "pssm", "length", "coords", "mask"]:
        batch[k] = batch[k].to(device)

def make_model(model_name, model_args):
    if model_name == "rgn":
        model = RGN(**model_args)
    elif model_name == "psifold":
        model = PsiFold(**model_args)
    elif model_name == "baseline":
        model = Baseline(**model_args)
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
    return model, optimizer, val_loss

def validate(model, val_dloader_dict, device):
    model.eval()
    val_loss = 0.0
    val_loss_by_group = {}

    with torch.no_grad():
        for group, dloader in val_dloader_dict.items():
            val_loss_group = 0.0
            for batch in dloader:
                to_device(batch, device)
                out = model(batch["seq"], batch["kmer"], batch["pssm"], batch["length"])
                loss = dRMSD_masked(out, batch["coords"], batch["mask"])

                val_loss += loss.item()
                val_loss_group += loss.item()

            val_loss_by_group[group] = val_loss_group / len(dloader)

    N = sum(len(dloader) for group, dloader in val_dloader_dict.items())
    val_loss /= N

    return val_loss, val_loss_by_group

def train(model, optimizer, train_dloader, device, max_grad_norm=None, output_frequency = 60):
    model.train()
    train_loss = 0.0

    last_output = datetime.datetime.now()
    for batch_idx, batch in enumerate(train_dloader):
        to_device(batch, device)
        out = model(batch["seq"], batch["kmer"], batch["pssm"], batch["length"])
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

def run_train_loop(model, optimizer, train_dloader, val_dloader_dict, device, max_grad_norm=None, epochs=10, output_frequency=60, checkpoint_file="checkpoint.pt", best_val_loss=math.inf):
    best_model_state_dict = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        start = datetime.datetime.now()
        train_loss = train(model, optimizer, train_dloader, device, max_grad_norm=max_grad_norm, output_frequency=output_frequency)
        val_loss, val_loss_by_group = validate(model, val_dloader_dict, device)
        elapsed = datetime.datetime.now() - start
        print(f"epoch {epoch:d}: elapsed = {elapsed}, train dRMSD (A) = {train_loss/100:0.3f}, val dRMSD (A) = {val_loss/100:0.3f}")
        print("val dRMSD (A) by subgroup:\n" + "\n".join(f"{k} : {v/100:0.3f}" for k,v in val_loss_by_group.items()))

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
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                }
            torch.save(checkpoint, checkpoint_file)

    model.load_state_dict(best_model_state_dict)

    return model
