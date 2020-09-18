# PsiFold

PsiFold is a pytorch implementation of the [RGN](https://doi.org/10.1016/j.cels.2019.03.006) method for protein structure prediction as well as several variants.

# Installation
```
conda create -n psifold python=3.7
conda activate psifold
pip install -e .
```

# Usage
The following commands download the [ProteinNet](https://github.com/aqlaboratory/proteinnet) casp7 dataset and trains a rgn model for 5 epochs.
```
# download tensorflow records
curl -LO "https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/tfrecords/casp7.tar.gz"
tar -xvzf casp7.tar.gz

# convert tensorflow records to hdf5 file
proteinnet2hdf --output="casp7.h5" casp7

# tensorflow records are no longer needed
rm -r casp7 casp7.tar.gz

# train rgn model
run_rgn --train \
        --input.file=casp7.h5 \
        --train.section=/training/90 \
        --batch_size=32 \
        --epochs=5 \
        --learning_rate=1e-3

# evaluate the model
# choose --test.section="/testing" to evaluate the test set
run_rgn --test \
        --input.file=casp7.h5 \
        --load_checkpoint=checkpoint_best.pt \
        --test.section=/validation
```

# Variants
PsiFold also implements a variant of the RGN method which uses a transformer encoder stack (like BERT) to replace the LSTM and replaces the dRMSD loss with a local cosine similarity loss on the alpha carbon SRF coordinates.
Given the recent advances in NLP from transformers this seems like an interesting idea, but so far I have been unable to train a good model.
