#!/usr/bin/env python

import argparse
import math

import torch
from torch import nn, optim

from psifold import ProteinNetDataset, make_data_loader, group_by_class, make_model, restore_from_checkpoint
from psifold.util import validate

def main():
    parser = argparse.ArgumentParser(description="test model")
    parser.add_argument("--input.file", default="input.h5", dest="input_file", help="hdf5 file containing proteinnet records")
    parser.add_argument("--input.section", default="/testing", dest="input_section", help="hdf5 section containing proteinnet records")
    parser.add_argument("--load_checkpoint", type=str, default="checkpoint.pt", help="path to checkpoint to load model from")

    args = parser.parse_args()
    print("running psifold_test...")
    print("args:", vars(args))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    if torch.cuda.is_available(): print(torch.cuda.get_device_name())

    test_dset = ProteinNetDataset(args.input_file, args.input_section)
    test_dset_groups = group_by_class(test_dset)
    test_dloader_dict = {k : make_data_loader(v, batch_size=16) for k, v in test_dset_groups.items()}

    print(f"restoring state from {args.load_checkpoint}")
    checkpoint = torch.load(args.load_checkpoint, map_location=device)
    model, _, _, _, _ = restore_from_checkpoint(checkpoint, device)
    model.to(device)

    test_loss, test_loss_by_group = validate(model, test_dloader_dict, device)
    print("test dRMSD (A) by subgroup:\n" + "\n".join(f"{k} : {v/100:0.3f}" for k,v in test_loss_by_group.items()))

if __name__ == "__main__":
    main()
