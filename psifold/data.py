import pathlib
import re
import subprocess
import tempfile

import h5py
import numpy as np

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, Subset
from torch.nn.utils.rnn import pad_sequence, pack_sequence

import psifold
from psifold import internal_coords, internal_to_srf

class PsiFoldDataset(Dataset):
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
        ID = record["id"].decode("utf-8")

        # primary amino acid sequence (N)
        seq = torch.from_numpy(record["primary"])
        N = seq.size(0)

        # PSSM + information content (N x 21)
        pssm = torch.from_numpy(np.reshape(record["evolutionary"], (-1, 21), "C"))

        # coordinate available (N)
        mask = torch.from_numpy(record["mask"]).bool()

        # tertiary structure (N x 3)
        coords = torch.from_numpy(np.reshape(record["tertiary"], (-1, 3), "C")) / 100.0
        coords = coords[1::3] # c-alpha only

        # fill masked coords with nan to keep track
        coords = coords.masked_fill(mask.logical_not().unsqueeze(1), float("nan"))

        # srf coords (N x 3)
        # note first srf coord is always fixed by padding
        r, theta, phi = psifold.geometry.internal_coords(coords, pad=True)
        srf = psifold.geometry.internal_to_srf(r, theta, phi)

        mask = (srf == srf).all(dim=-1)

        out = {"id" : ID, "seq" : seq, "pssm" : pssm, "mask" : mask, "coords" : coords, "srf" : srf}

        # extract class from id if present
        class_match = self.class_re.match(ID)
        if class_match: out["class"] = class_match[1]

        return out

    @staticmethod
    def collate(batch):
        length = torch.tensor([x["seq"].size(0) for x in batch])
        sorted_length, sorted_indices = length.sort(0, True)

        ID = [batch[i]["id"] for i in sorted_indices]
        seq = pad_sequence([batch[i]["seq"] for i in sorted_indices])
        pssm = pad_sequence([batch[i]["pssm"] for i in sorted_indices])
        mask = pad_sequence([batch[i]["mask"] for i in sorted_indices])
        coords = pad_sequence([batch[i]["coords"] for i in sorted_indices], batch_first=False, padding_value=float("nan"))
        srf = pad_sequence([batch[i]["srf"] for i in sorted_indices], batch_first=False, padding_value=float("nan"))

        return {"id" : ID, "seq" : seq, "pssm" : pssm, "mask" : mask, "coords" : coords, "srf" : srf, "length" : sorted_length}

class RGNDataset(Dataset):
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
        ID = record["id"].decode("utf-8")

        # primary amino acid sequence (N)
        seq = torch.from_numpy(record["primary"])
        N = seq.size(0)

        # PSSM + information content (N x 21)
        pssm = torch.from_numpy(np.reshape(record["evolutionary"], (-1, 21), "C"))

        # coordinate available (3N)
        mask = torch.from_numpy(record["mask"]).bool().repeat_interleave(3)

        # tertiary structure (3N x 3)
        coords = torch.from_numpy(np.reshape(record["tertiary"], (-1, 3), "C"))

        # alpha carbon only
        if coords[0::3].eq(0).all() and coords[2::3].eq(0).all():
            mask = torch.logical_and(mask, torch.tensor([False, True, False]).repeat(N))

        out = {"id" : ID, "seq" : seq, "pssm" : pssm, "mask" : mask, "coords" : coords}

        # extract class from id if present
        class_match = self.class_re.match(ID)
        if class_match: out["class"] = class_match[1]

        return out

    @staticmethod
    def collate(batch):
        length = torch.tensor([x["seq"].size(0) for x in batch])
        sorted_length, sorted_indices = length.sort(0, True)

        ID = [batch[i]["id"] for i in sorted_indices]
        seq = pad_sequence([batch[i]["seq"] for i in sorted_indices])
        pssm = pad_sequence([batch[i]["pssm"] for i in sorted_indices])
        mask = pad_sequence([batch[i]["mask"] for i in sorted_indices])
        coords = pad_sequence([batch[i]["coords"] for i in sorted_indices])

        return {"id" : ID, "seq" : seq, "pssm" : pssm, "mask" : mask, "coords" : coords, "length" : sorted_length}

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

def make_data_loader(dset, collate_fn, batch_size=32, max_len=None, max_size=None, bucket_size=None, complete_only=False, remove_empty=True):
    """
    create a DataLoader for ProteinNet datasets

    Args:
        batch_size: approximate size of each batch (the last batch in the dset/bucket may be smaller)
        max_len: only include proteins with sequence length <= max_len
        max_size: only include first max_size elements of dataset
        bucket_size: size of buckets used by BucketByLenRandomBatchSampler
    """

    if remove_empty:
        indices = torch.tensor([i for i, x in enumerate(dset) if not (~x["mask"]).all()])
        dset = Subset(dset, indices)

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

def run_tm_score(seq, ca_coords, ca_coords_ref, tmscore_path="TMscore", timeout=1.0):
    pdb = make_pdb_record(seq, ca_coords)
    pdb_ref = make_pdb_record(seq, ca_coords_ref)

    with tempfile.TemporaryDirectory() as tmpdirname:
        path_a = pathlib.Path(tmpdirname) / "model.pdb"
        with open(path_a, "w") as f:
            f.write(pdb)

        path_b = pathlib.Path(tmpdirname) / "native.pdb"
        with open(path_b, "w") as f:
            f.write(pdb_ref)

        proc = subprocess.run([tmscore_path, "model.pdb", "native.pdb"], cwd=tmpdirname, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)

    s = proc.stdout.decode()
    out = {"stdout" : s}

    tm_score = re.search("TM-score\s*=\s*(\d+\.\d*)", s)
    if tm_score: out["tm"] = float(tm_score[1])

    gdt_ts_score = re.search("GDT-TS-score\s*=\s*(\d+\.\d*)", s)
    if gdt_ts_score: out["gdt_ts"] = float(gdt_ts_score[1])

    rmsd = re.search("RMSD of  the common residues\s*=\s*(\d+\.\d*)", s)
    if rmsd: out["rmsd"] = float(rmsd[1])

    return out
