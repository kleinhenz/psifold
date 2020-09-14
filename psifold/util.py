import copy
import datetime
import math
import pathlib
import uuid

import numpy as np

import torch

from torch import nn, optim
from torch.cuda import amp
from torch.utils.data import Dataset, Subset

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def to_device(batch, device):
    for k,v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

def group_by_class(dset):
    classes = np.array([x["class"] for x in dset])
    out = {c : Subset(dset, np.nonzero(classes == c)[0]) for c in np.unique(classes)}
    return out

def validate(model, criterion, compute_tm, val_dloader_dict, device):
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
                out = model(batch)
                loss = criterion(out, batch)

                val_loss += loss.item()
                val_loss_group += loss.item()

                tm_scores_group.update(compute_tm(out, batch))

            val_loss_by_group[group] = val_loss_group / len(dloader)
            tm_scores_by_group[group] = tm_scores_group

    N = sum(len(dloader) for group, dloader in val_dloader_dict.items())
    val_loss /= N

    return val_loss, val_loss_by_group, tm_scores_by_group

def train(model, criterion, optimizer, scheduler, scaler, dloader, device, accumulate_steps=1, max_grad_norm=None, enable_amp=False, output_frequency = 60):
    model.train()
    train_loss = 0.0

    last_output = datetime.datetime.now()

    dloader_iter = iter(dloader)
    super_batches = torch.arange(len(dloader)).split(accumulate_steps)

    for super_batch_idx, super_batch in enumerate(super_batches):
        N = len(super_batch)
        optimizer.zero_grad()
        for _, batch in zip(range(N), dloader_iter):
            to_device(batch, device)

            with amp.autocast(enabled=enable_amp):
                out = model(batch)
                loss = criterion(out, batch) / N

            scaler.scale(loss).backward()
            train_loss += loss.item()

        if max_grad_norm: nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if ((datetime.datetime.now() - last_output).seconds > output_frequency):
            last_output = datetime.datetime.now()
            print(f"batch {super_batch_idx}/{len(super_batches)}")

    train_loss /= len(super_batches)
    return train_loss

def run_train_loop(model,
        criterion,
        optimizer,
        scheduler,
        train_dloader,
        val_dloader_dict,
        device,
        compute_tm,
        accumulate_steps=1,
        max_grad_norm=None,
        enable_amp=False,
        epochs=10,
        output_frequency=60,
        checkpoint_path="checkpoints",
        checkpoint_extra_data=None,
        best_val_loss=math.inf,
        train_loss_history = [],
        val_loss_history = []):

    best_model_state_dict = copy.deepcopy(model.state_dict())
    scaler = amp.GradScaler()

    workdir = pathlib.Path(checkpoint_path) / str(uuid.uuid4())
    workdir.mkdir(parents=True)
    latest_checkpoint_path = workdir / "latest.pt"
    best_checkpoint_path = workdir / "best.pt"

    for epoch in range(epochs):
        start = datetime.datetime.now()

        train_loss = train(model,
                criterion,
                optimizer,
                scheduler,
                scaler,
                train_dloader,
                device,
                accumulate_steps=accumulate_steps,
                max_grad_norm=max_grad_norm,
                enable_amp=enable_amp,
                output_frequency=output_frequency)

        val_loss, val_loss_by_group, tm_scores_by_group = validate(model, criterion, compute_tm, val_dloader_dict, device)

        elapsed = datetime.datetime.now() - start
        print(f"epoch {epoch:d}: elapsed = {elapsed}, train loss = {train_loss:0.3f}, val loss = {val_loss:0.3f}")

        print("val loss (A) by subgroup:\n" + "\n".join(f"{k} : {v:0.3f}" for k,v in val_loss_by_group.items()))

        print("test tm-scores by subgroup:")
        for group, tm_scores in tm_scores_by_group.items():
            scores = np.array(list(tm_scores.values()))
            q = np.quantile(scores, [0.0, 0.25, 0.5, 0.75, 1.0])
            print(f"{group}: {q[0]:0.2f}-{q[1]:0.2f}-{q[2]:0.2f}-{q[3]:0.2f}-{q[4]:0.2f}")

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        checkpoint = {
            "model_name" : model.model_name,
            "model_args" : model.model_args,
            "epoch": epoch,
            "val_loss" : val_loss,
            "train_loss_history" : train_loss_history,
            "val_loss_history" : val_loss_history,
            "tm_scores" : tm_scores_by_group,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict" : scheduler.state_dict(),
            "extra" : checkpoint_extra_data
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
