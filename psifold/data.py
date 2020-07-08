import re

import h5py
import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, Subset
from torch.nn.utils.rnn import pad_sequence, pack_sequence

from psifold import internal_coords, internal_to_srf

def collect_geometry(dset):
    bond_lengths = {"n_ca" : [], "ca_c" : [], "c_n" : []}
    bond_angles = {"n_ca_c" : [], "ca_c_n" : [], "c_n_ca" : []}
    # n_ca_c_n, ca_c_n_ca, c_n_ca_c
    bond_torsions = {"psi" : [], "omega": [], "phi" : []}

    for example in dset:
        assert example["mask"].all()
        coords = example["coords"].view(-1, 1, 3)
        r, theta, phi = internal_coords(coords)
        for i, (x, y, z) in enumerate(zip(bond_lengths, bond_angles, bond_torsions)):
            bond_lengths[x].append(r[i::3].squeeze())
            bond_angles[y].append(theta[i::3].squeeze())
            bond_torsions[z].append(phi[i::3].squeeze())

    for x in [bond_lengths, bond_angles, bond_torsions]:
        for k,v in x.items():
            x[k] = torch.cat(v)

    return bond_lengths, bond_angles, bond_torsions

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

def make_data_loader(dset, collate_fn, batch_size=32, max_len=None, max_size=None, bucket_size=None, complete_only=False):
    """
    create a DataLoader for ProteinNet datasets

    Args:
        batch_size: approximate size of each batch (the last batch in the dset/bucket may be smaller)
        max_len: only include proteins with sequence length <= max_len
        max_size: only include first max_size elements of dataset
        bucket_size: size of buckets used by BucketByLenRandomBatchSampler
    """

    if complete_only:
        indices = torch.tensor([i for i, x in enumerate(dset) if x["mask"].all()])
        dset = Subset(dset, indices)

    if max_len:
        assert max_len > 0
        indices = [i for i, x in enumerate(dset) if x["seq"].numel() <= max_len]
        dset = Subset(dset, indices)

    if max_size:
        assert max_size > 0
        dset = Subset(dset, torch.randperm(len(dset))[:max_size])

    if bucket_size:
        lengths = torch.tensor([x["seq"].shape[0] for x in dset])
        sampler = BucketByLenRandomBatchSampler(lengths, batch_size=batch_size, bucket_size=bucket_size)
    else:
        sampler = BatchSampler(RandomSampler(dset), batch_size=batch_size, drop_last=False)

    data_loader = DataLoader(dset, batch_sampler=sampler, collate_fn=collate_fn)

    return data_loader

def make_pdb_record(seq, ca_coords):
    """
    create a pdb record
    """

    aa_list = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TRY"]

    seq = record["seq"]
    coords = record["coords"] / 100.0
    ca_coords = coords[1::3]

    lines = []
    for i in range(len(seq)):

        aa = aa_list[seq[i]]
        x = ca_coords[i,0]
        y = ca_coords[i,1]
        z = ca_coords[i,2]
        occ = 1.0
        T = 10.0

        line = f"ATOM  {i:5d}  CA  {aa} A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{T:6.2f}           C  \n"
        lines.append(line)

    return "".join(lines)
