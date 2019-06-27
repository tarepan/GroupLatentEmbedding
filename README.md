# Group Latent Embedding for Vector Quantized Variational Autoencoder in Non-Parallel Voice Conversion (Accepted by Interspeech 2019)

This is a Pytorch implementation of [Group Latent Embedding for Vector Quantized Variational Autoencoder in Non-Parallel Voice Conversion](https://psi.engr.tamu.edu/wp-content/uploads/2019/06/ding2019interspeech.pdf). This implementation is based on the VQ-VAE-WaveRNN implementation at [https://github.com/mkotha/WaveRNN](https://github.com/mkotha/WaveRNN).

## Dataset:

* [VCTK](https://datashare.is.ed.ac.uk/handle/10283/2651)
  * [Audio samples](https://shaojinding.github.io/samples/gle/gle_demo).

## Preparation

The preparation is similar to that at [https://github.com/mkotha/WaveRNN](https://github.com/mkotha/WaveRNN). We repeat it here for convenience.


### Requirements

* Python 3.6 or newer
* PyTorch with CUDA enabled
* [librosa](https://github.com/librosa/librosa)
* [apex](https://github.com/NVIDIA/apex) if you want to use FP16 (it probably
  doesn't work that well).


### Create config.py

```
cp config.py.example config.py
```

### Preparing VCTK

You can skip this section if you don't need a multi-speaker dataset.

1. Download and uncompress [the VCTK dataset](
  https://datashare.is.ed.ac.uk/handle/10283/2651).
2. `python preprocess_multispeaker.py /path/to/dataset/VCTK-Corpus/wav48
  /path/to/output/directory`
3. In `config.py`, set `multi_speaker_data_path` to point to the output
  directory.


## Usage

To run Group Latent Embedding:

```
$ python wavernn.py -m vqvae_group --num-group 41 --num-sample 10
```

The `-m` option can be used to tell the the script what model to train. By default, it trains a vanilla VQ-VAE model. 

Trained models are saved under the `model_checkpoints` directory.

By default, the script will take the latest snapshot and continues training
from there. To train a new model freshly, use the `--scratch` option.

Every 50k steps, the model is run to generate test audio outputs. The output
goes under the `model_outputs` directory.

When the `-g` option is given, the script produces the output using the saved
model, rather than training it.

`--num-group` specifies the number of groups. `--num-sample` specifies the number of atoms in each group. Note that num-group times num-sample should be equal to the total number of atoms in the embedding dictionary (`n_classes` in class `VectorQuantGroup` in `vector_quant.py`)

# Acknowledgement

The code is based on [mkotha/WaveRNN](https://github.com/mkotha/WaveRNN).

# Cite the work
S. Ding, and R. Gutierrez-Osuna. "Group Latent Embedding for Vector Quantized Variational Autoencoder in Non-Parallel Voice Conversion." Accepted by Interspeech 2019.
