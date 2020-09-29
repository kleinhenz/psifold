# PsiFold

PsiFold is a pytorch library for protein structure prediction which implements the [RGN](https://doi.org/10.1016/j.cels.2019.03.006) method as well a novel variant based on the transformer architecture.

# Installation
```
pip install git+https://git@github.com/kleinhenz/psifold.git
```

# Usage
The following script downloads the [ProteinNet](https://doi.org/10.1186/s12859-019-2932-0) casp7 dataset, trains a RGN model for 5 epochs, and evaluates the trained model.
In addition to computing the dRMSD loss, the evaluation step computes the [TMscore](https://doi.org/10.1002/prot.20264) for each predicted structure (the `TMscore` program must be in the `PATH` or else provided via the `tmscore_path` argument).
```
# download tensorflow records
curl -LO "https://sharehost.hms.harvard.edu/sysbio/alquraishi/proteinnet/tfrecords/casp7.tar.gz"
tar -xvzf casp7.tar.gz

# convert tensorflow records to hdf5 file
proteinnet2hdf --output="casp7.h5" casp7

# tensorflow records are no longer needed
rm -r casp7 casp7.tar.gz

# train rgn model
run_rgn \
  --train \
  --input.file=casp7.h5 \
  --train.section=/training/90 \
  --batch_size=32 \
  --epochs=5 \
  --learning_rate=1e-3

# evaluate the model
# choose --test.section="/testing" to evaluate the test set
run_rgn \
  --test \
  --compute_tm \
  --input.file=casp7.h5 \
  --load_checkpoint=checkpoints/{uuid}/latest.pt \
  --test.section=/validation
```

# Variants
Given the recent [advances](https://arxiv.org/abs/1706.03762) in NLP from the introduction of transformers, it is interesting to see if they may be beneficial in other problem domains.
PsiFold implements a variant of the RGN method which replaces the LSTM with a transformer encoder stack similar to [BERT](https://arxiv.org/abs/1810.04805).
Additionally this variant replaces the global dRMSD loss with a local cosine similarity loss on the alpha carbon SRF coordinates.
The reason for the local loss function is to try to ameliorate the vanishing/exploding gradient problems that come from backpropagation through the sequential [NERF](https://doi.org/10.1002/jcc.25772) reconstruction algorithm that is necessary for computing global loss functions such as dRMSD.
Additionally, the experience of NLP shows that transformers can show very good performance on problems with long range correlations even when the loss is completely local.

After the input data has been downloaded, a psifold model can be trained similarly to an RGN model using the `run_psifold` command as shown below
```
run_psifold \
  --train \
  --input.file=casp7.h5 \
  --train.section=/training/90 \
  --enable_amp \
  --epochs=10 \
  --accumulate_steps=8 \
  --batch_size=16 \
  --learning_rate=1e-3 \
  --max_len=512 \
  transformer \
  --n_layers=12 \
  --hidden_size=768
```
Unfortunately, these models appear to still be difficult to train and I have not yet obtained a good set of weights.
I had hoped that the use of the local loss function would fix the training difficulties of the RGN models but this seems to not be the case.
