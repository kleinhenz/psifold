# PsiFold

PsiFold is python module for protein structure prediction.

# Installation
```
conda create -n psifold python=3.7
conda activate psifold
pip install -e .
```

# Usage
The following commands download the [ProteinNet](https://github.com/aqlaboratory/proteinnet) casp7 dataset and trains a lstm psifold model for 5 epochs.
```
# download tensorflow records
curl -LO "https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/tfrecords/casp7.tar.gz"
tar -xvzf casp7.tar.gz

# convert tensorflow records to hdf5 file
proteinnet2hdf --output="casp7.h5" casp7

# tensorflow records are no longer needed
rm -r casp7 casp7.tar.gz

# train lstm model
run_psifold --train \
            --input.file=casp7.h5 \
            --train.section=/training/90 \
            --batch_size=32 \
            --epochs=5 \
            --learning_rate=1e-3 \
            lstm

# evaluate the model
# choose --test.section="/testing" to evaluate the test set
run_psifold --test \
            --input.file=casp7.h5 \
            --load_checkpoint=checkpoint_best.pt \
            --test.section=/validation
```
