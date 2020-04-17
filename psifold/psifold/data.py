import h5py
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, Subset
from torch.nn.utils.rnn import pad_sequence, pack_sequence

class BucketByLenRandomBatchSampler(torch.utils.data.Sampler):
    """
    Bucket example by length and randomly sample from buckets
    """
    def __init__(self, lengths, batch_size=32, bucket_size=1024):
        self.batch_size = batch_size
        self.buckets = torch.argsort(lengths).split(bucket_size)

        self.nbatches = 0
        for bucket in self.buckets:
            self.nbatches += (len(bucket) + batch_size - 1) // batch_size

    def __iter__(self):
        batches = []
        for bucket in self.buckets:
            for indices in torch.randperm(len(bucket)).split(self.batch_size):
                batches.append(bucket[indices])
        assert len(batches) == self.nbatches

        for i in torch.randperm(len(batches)):
            yield batches[i]

    def __len__(self):
        return self.nbatches

def collate_fn(batch):
    length = torch.tensor([x["seq"].size(0) for x in batch])
    sorted_length, sorted_indices = length.sort(0, True)

    ID = [batch[i]["id"] for i in sorted_indices]
    seq = pad_sequence([batch[i]["seq"] for i in sorted_indices])
    pssm = pad_sequence([batch[i]["pssm"] for i in sorted_indices])
    mask = pad_sequence([batch[i]["mask"] for i in sorted_indices])
    coords = pad_sequence([batch[i]["coords"] for i in sorted_indices])

    return {"id" : ID, "seq" : seq, "pssm" : pssm, "mask" : mask, "coords" : coords, "length" : sorted_length}

# TODO extract class info (FM/TBM for test set, % seq id for validation set) from id field
class ProteinNetDataset(Dataset):
    def __init__(self, fname, section):
        with h5py.File(fname, "r") as h5f:
            self.dset = h5f[section][:]

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

def make_data_loader(dset, batch_size=32, max_len=None, max_size=None, bucket_size=None):
    """
    create a DataLoader for ProteinNet datasets

    Args:
        batch_size: approximate size of each batch (the last batch in the dset/bucket may be smaller)
        max_len: only include proteins with sequence length <= max_len
        max_size: only include first max_size elements of dataset
        bucket_size: size of buckets used by BucketByLenRandomBatchSampler
    """
    if max_len:
        assert max_len > 0
        indices = [i for i, x in enumerate(dset) if x["seq"].numel() <= max_len]
        dset = Subset(dset, indices)

    if max_size:
        assert max_size > 0
        dset = Subset(dset, range(max_size))

    if bucket_size:
        lengths = torch.tensor([x["seq"].shape[0] for x in dset])
        sampler = BucketByLenRandomBatchSampler(lengths, batch_size=batch_size, bucket_size=bucket_size)
    else:
        sampler = BatchSampler(RandomSampler(dset), batch_size=batch_size, drop_last=False)

    data_loader = DataLoader(dset, batch_sampler=sampler, collate_fn=collate_fn)

    return data_loader
