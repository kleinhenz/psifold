import datetime

import torch
import torch.nn as nn

from torchstruct import dRMSD

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
