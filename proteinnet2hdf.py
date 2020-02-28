#!/usr/bin/env python3

import pathlib
import argparse

import tensorflow as tf
import numpy as np
import h5py

def create_hdf5_dtype():
    str_dt = h5py.string_dtype(encoding='utf-8', length=None)
    vlen_int64_dt = h5py.special_dtype(vlen=np.int64)
    vlen_float32_dt = h5py.special_dtype(vlen=np.float32)

    dtype = np.dtype({"names" : ["id", "primary", "evolutionary", "tertiary", "mask"],
                      "formats" : [str_dt, vlen_int64_dt, vlen_float32_dt, vlen_float32_dt, vlen_float32_dt]})

    return dtype

def parse_record(proto):
    context_features={'id': tf.io.FixedLenFeature((1,), tf.string)}
    sequence_features={
                        'primary':      tf.io.FixedLenSequenceFeature((1,),               tf.int64),
                        'evolutionary': tf.io.FixedLenSequenceFeature((21,), tf.float32, allow_missing=True),
                        'secondary':    tf.io.FixedLenSequenceFeature((1,),               tf.int64,   allow_missing=True),
                        'tertiary':     tf.io.FixedLenSequenceFeature((3,),  tf.float32, allow_missing=True),
                        'mask':         tf.io.FixedLenSequenceFeature((1,),               tf.float32, allow_missing=True)}

    context, features = tf.io.parse_single_sequence_example(proto,
                                                            context_features=context_features,
                                                            sequence_features=sequence_features)
    return context, features

def read_tf(files):
    dset = tf.data.TFRecordDataset(files)
    dset_parsed = dset.map(parse_record)

    dtype = create_hdf5_dtype()
    data = []
    for context, features in dset_parsed:
        protein_id = context["id"].numpy()[0].decode("utf-8")
        primary = features["primary"].numpy()
        evolutionary = features["evolutionary"].numpy()
        tertiary = features["tertiary"].numpy()
        mask = features["mask"].numpy()

        record = np.array((protein_id, primary, evolutionary, tertiary.flatten(), mask), dtype=dtype)
        data.append(record)

    data = np.array(data)

    return data

def main():
    parser = argparse.ArgumentParser(description="convert proteinnet tensorflow records to hdf5")
    parser.add_argument("--output", default="output.h5", help="name of output hdf5 archive")
    parser.add_argument("path", help="path to folder containing proteinnet records")
    args = parser.parse_args()

    root = pathlib.Path(args.path).resolve()

    dirs = ["testing", "validation"] + [f"training/{i}" for i in [30, 50, 70, 90, 95, 100]]
    dirs = [root / d for d in dirs]
    for d in dirs: assert d.exists()

    dtype = create_hdf5_dtype()
    with h5py.File(args.output, "w") as h5f:
        h5f.attrs["origin"] = str(root)
        for d in dirs:
            h5path = str(d.relative_to(root))
            print(f"processing {h5path}")

            files = sorted(d.glob("*"), key = lambda x : int(x.name))
            files = [str(f) for f in files]
            data = read_tf(files)
            dset = h5f.create_dataset(h5path, data.shape, dtype=dtype)
            dset[:] = data

if __name__ == "__main__":
    main()
