# PsiFold

PsiFold is python module for protein structure prediction.

# Installation
```
conda create -n psifold python=3.7
conda activate psifold
pip install -e .
```

# Usage
The following commands download the [ProteinNet](https://github.com/aqlaboratory/proteinnet) casp7 dataset and trains the psifold model for 5 epochs.
```
curl -LO "https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/tfrecords/casp7.tar.gz"
tar -xvzf casp7.tar.gz

# convert tensorflow records to hdf5 file
proteinnet2hdf --output="casp7.h5" casp7

# tensorflow records are no longer needed
rm -r casp7 casp7.tar.gz

# train model
# model with best validation loss will be saved in checkpoint.pt
# run psifold_train --help to get list of options
psifold_train --input.file="casp7.h5" \
              --save_checkpoint="checkpoint.pt" \
              --train.section="/training/90" \
              --val.section="/validation" \
              --max_len=400 \
              --batch_size=32 \
              --epochs=5 \
              --learning_rate=1e-3 \
              --model=psifold

# evaluate the model
# choose --input.section="/testing" to evaluate the test set
psifold_test --input.file="casp7.h5" --input.section="/validation" --load_checkpoint="checkpoint.pt"
```
