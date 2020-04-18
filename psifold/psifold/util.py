import datetime

import torch
import torch.nn as nn

from psifold import dRMSD, RGN, PsiFold

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

    val_loss /= len(val_dloader)

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

def run_train_loop(model, optimizer, train_dloader, val_dloader, device, epochs=10, output_frequency=60, checkpoint_file="checkpoint.pt"):
    train_loss_history = []
    val_loss_history = []
    for epoch in range(epochs):
        start = datetime.datetime.now()
        train_loss = train(model, optimizer, train_dloader, device, output_frequency=output_frequency)
        val_loss = validate(model, val_dloader, device)
        elapsed = datetime.datetime.now() - start
        print(f"epoch {epoch:d}: elapsed = {elapsed}, train dRMSD (A) = {train_loss/100:0.3f}, val dRMSD (A) = {val_loss/100:0.3f}")

        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

        checkpoint = {
            "model_name" : model.model_name,
            "model_args" : model.model_args,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss_history": train_loss_history,
            "val_loss_history" : val_loss_history
            }
        torch.save(checkpoint, checkpoint_file)

def make_model(model_name, model_args):
    if model_name == "rgn":
        model = RGN(**model_args)
    elif model_name == "psifold":
        model = PsiFold(**model_args)
    else:
        raise Exception(f"model: {model_name} not recognized")
    return model
