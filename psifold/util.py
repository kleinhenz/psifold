import numpy as np

import torch
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
