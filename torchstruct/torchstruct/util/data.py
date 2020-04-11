import h5py
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_sequence

def collate_fn(batch):
    length = torch.tensor([x["seq"].size(0) for x in batch])
    sorted_length, sorted_indices = length.sort(0, True)

    ID = [batch[i]["id"] for i in sorted_indices]
    seq = pad_sequence([batch[i]["seq"] for i in sorted_indices])
    pssm = pad_sequence([batch[i]["pssm"] for i in sorted_indices])
    mask = pad_sequence([batch[i]["mask"] for i in sorted_indices])
    coords = pad_sequence([batch[i]["coords"] for i in sorted_indices])

    return {"id" : ID, "seq" : seq, "pssm" : pssm, "mask" : mask, "coords" : coords, "length" : sorted_length}

class ProteinNetDataset(Dataset):
    def __init__(self, h5dset):
        self.dset = h5dset

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        record = self.dset[idx]

        # identifier
        ID = record["id"]

        # primary amino acid sequence (N)
        seq = torch.from_numpy(record["primary"])

        # PSSM + information content (N x 21)
        pssm = torch.from_numpy(np.reshape(record["evolutionary"], (-1, 21), "C"))

        # coordinate available (3N)
        mask = torch.from_numpy(record["mask"]).bool().repeat_interleave(3)

        # tertiary structure (3N x 3)
        coords = torch.from_numpy(np.reshape(record["tertiary"], (-1, 3), "C"))

        return {"id" : ID, "seq" : seq, "pssm" : pssm, "mask" : mask, "coords" : coords}
