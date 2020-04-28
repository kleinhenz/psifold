import re

import h5py
import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, Subset
from torch.nn.utils.rnn import pad_sequence, pack_sequence

from psifold import internal_coords, internal_to_srf

def make_srf_dset_from_protein(coords, seq, kmer, mask):
    # fill masked coords with nan to keep track
    coords = coords.masked_fill(mask.logical_not().unsqueeze(1), float("nan"))

    # compute srf coords
    r, theta, phi = internal_coords(coords.unsqueeze(1), pad=True)
    srf = internal_to_srf(r, theta, phi).squeeze().view(-1, 3, 3)

    # update mask after propagating nan through calculation
    mask = (srf == srf).view(-1, 9).all(dim=-1)
    # always mask out first residue since we don't have full srf coords
    mask[0] = False

    # select valid regions
    srf = srf.masked_select(mask.view(-1,1,1)).view(-1, 3, 3)
    seq = seq.masked_select(mask)
    kmer = kmer.masked_select(mask)

    return {"srf" : srf, "seq" : seq, "kmer" : kmer}

def make_srf_dset(dset):
    srf_dset = {"srf" : [], "seq" : [], "kmer" : []}

    for x in tqdm(dset):
        out = make_srf_dset_from_protein(x["coords"], x["seq"], x["kmer"], x["mask"])
        for k in srf_dset: srf_dset[k].append(out[k])

    for k in srf_dset:
        srf_dset[k] = torch.cat(srf_dset[k])

    return srf_dset

def seq2kmer(seq):
    k, c = 3, 22 # only 3-mers for now
    basis = torch.tensor([c]).repeat(k).pow(torch.arange(k))
    seq_pad = torch.cat([torch.tensor([20]), seq, torch.tensor([21])]) # (add padding tokens)
    kmer = torch.tensor([torch.dot(seq_pad[i:i+3], basis) for i in range(seq.size(0))])
    return kmer

def kmer2aa(kmer, k=3, c=22):
    basis = torch.tensor([c]).repeat(k).pow(torch.arange(k))
    aa = torch.zeros_like(basis)
    for i in reversed(range(k)):
        q, r = divmod(kmer, basis[i].item())
        aa[i] = q
        kmer = r
    return aa

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
    kmer = pad_sequence([batch[i]["kmer"] for i in sorted_indices])
    pssm = pad_sequence([batch[i]["pssm"] for i in sorted_indices])
    mask = pad_sequence([batch[i]["mask"] for i in sorted_indices])
    coords = pad_sequence([batch[i]["coords"] for i in sorted_indices])

    return {"id" : ID, "seq" : seq, "kmer" : kmer, "pssm" : pssm, "mask" : mask, "coords" : coords, "length" : sorted_length}

class ProteinNetDataset(Dataset):
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

        # construct 3-mers
        kmer = seq2kmer(seq)

        # PSSM + information content (N x 21)
        pssm = torch.from_numpy(np.reshape(record["evolutionary"], (-1, 21), "C"))

        # coordinate available (3N)
        mask = torch.from_numpy(record["mask"]).bool().repeat_interleave(3)

        # tertiary structure (3N x 3)
        coords = torch.from_numpy(np.reshape(record["tertiary"], (-1, 3), "C"))

        # alpha carbon only
        if coords[0::3].eq(0).all() and coords[2::3].eq(0).all():
            mask = torch.logical_and(mask, torch.tensor([False, True, False]).repeat(N))

        out = {"id" : ID, "seq" : seq, "kmer" : kmer, "pssm" : pssm, "mask" : mask, "coords" : coords}

        # extract class from id if present
        class_match = self.class_re.match(ID)
        if class_match: out["class"] = class_match[1]

        return out

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

def group_by_class(dset):
    classes = np.array([x["class"] for x in dset])
    out = {c : Subset(dset, np.nonzero(classes == c)[0]) for c in np.unique(classes)}
    return out
