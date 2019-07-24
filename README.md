# GLE-VQ-VAE VC
This repository is a Pytorch implementation of [Group Latent Embedding for Vector Quantized Variational Autoencoder in Non-Parallel Voice Conversion][originPaper] and fork of [official repo][].  
This implementation is based on the VQ-VAE-WaveRNN implementation at [https://github.com/mkotha/WaveRNN](https://github.com/mkotha/WaveRNN).  

The purpose of this fork is out-of-box reproduction of original results.  
Currently, original repo has missing internal dependencies [#1][i1]  [#2][i2] (needs fixes), but repo do not enable GitHub issues.  
If you can contact with authors and notify the bugs, I am happy.  

[originPaper]: https://psi.engr.tamu.edu/wp-content/uploads/2019/06/ding2019interspeech.pdf
[official repo]: https://github.com/shaojinding/GroupLatentEmbedding
[i1]: https://github.com/tarepan/GroupLatentEmbedding/issues/1
[i2]: https://github.com/tarepan/GroupLatentEmbedding/issues/2

## Dataset:

* [VCTK](https://datashare.is.ed.ac.uk/handle/10283/2651)
  <!-- * [Audio samples](https://shaojinding.github.io/samples/gle/gle_demo). -->

## Preparation
### Requirements
(move into pipenv rock file)

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

The `-m` option specify a model to train (default: GLE-VQ-VAE model, proposed in the paper.)

Below is model list (model_type: description).  

- vqvae_group: GLE-VQ-VAE with multi-speaker
- vqvae: VQ-VAE with multi-speaker
- wavernn: WaveRNN with single-speaker
- nc: ?? with single-speaker

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
