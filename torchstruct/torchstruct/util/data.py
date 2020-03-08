import h5py
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

class ProteinNetDataset(Dataset):
    def __init__(self, input_file, input_section):
        self.h5f = h5py.File(input_file, "r")
        self.dset = self.h5f[input_section]

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        record = self.dset[idx]

        # primary amino acid sequence (N)
        seq = torch.from_numpy(record["primary"])

        # PSSM + information content (N x 21)
        pssm = torch.from_numpy(np.reshape(record["evolutionary"], (-1, 21), "C"))

        # coordinate available (N)
        mask = torch.from_numpy(record["mask"])

        # tertiary structure (3N x 3)
        coords = torch.from_numpy(np.reshape(record["tertiary"], (-1, 3), "C"))

        return {"seq" : seq, "pssm" : pssm, "mask" : mask, "coords" : coords}

    def close(self):
        self.h5f.close()
