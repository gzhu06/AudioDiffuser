<div align="center">

# Audio Diffuser

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## Description

Diffuser designed for audio using denoising score matching formulation. We have included a collection or (re)implementations of diffusion models, together with VAE, VQ-GAN for various audio applications. 
This repository uses hydra-lightning config management which provides an easy-to-use interface for running diffusion-based models. 
In the application side, the diffusion process can be shared across different tasks, we will try to cover various audio based applications.
Additionally, this repo can be used as a learning resource for diffusion models. It includes detailed docstrings linked to the paper, comments, and notebooks for introducing diffusion models both theoretically and practically.

## Diffusion components
In this repo, we mainly use the desnoising score matching formulation discussed in [EDM](https://github.com/NVlabs/edm) for consistency based on [Audio Diffusion by Flavio](https://github.com/archinetai/audio-diffusion-pytorch). In EDM, a common framework is proposed for many diffusion models and decouples the design of sampling schedule, noise level parameterization etc.

### 1.Diffusion

### 2.Sampler

### 3.Scheduler

### 4.Backbones

### 5.Applications
Speech enhancement
Super resolution
source separation
vocoder
super-resolution


## How to run

### Install dependencies

```bash
# clone project
git clone https://github.com/gzhu06/DiffAudioSynthesizer
cd DiffAudioSynthesizer

# [OPTIONAL] create conda environment
conda create -n diffaudio python=3.8
conda activate diffaudio

# install pytorch (>=1.12.0), e.g. with cuda=10.2, we have:
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch

# install requirements
pip install -r requirements.txt
```
### Hydra-lightning

This is a great config management tool and it decouples dataloaders, training, network backbones.

### Run
Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash ddp mixed precision
CUDA_VISIBLE_DEVICES=0,3 python src/train.py trainer=ddp trainer.devices=2 experiment=example.yaml +trainer.precision=16
```

Or train model with  single GPU resume from a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=3 python src/train.py experiment=example1.yaml +trainer.precision=16 ckpt_path="/path/to/ckpt/name.ckpt"
```

Or evaluation:

```bash
CUDA_VISIBLE_DEVICES=3 python src/eval.py ckpt_path='dummy.ckpt' +trainer.precision=16 experiment=example2.yaml
```

# Code TODO
- [ ] v-objective, x-objective and epsilon (vp, ve) with edm framework
- [ ] DDIM
- [ ] consistency models
- [ ] applications: vocoder, super-resolution, speech enhancement and source separation
- [ ] rvq-vae (code is done, need to be verified with larger machines)
- [ ] ViT transformer backbone (?)
- [ ] discrete diffusion

# Notebooks TODO
- [ ] VDM formulation
- [ ] Sampler: ADM sampling
- [ ] Diffusion
- [ ] Scheduler

# Check TODO
- Diffusion: ADM sampling

### Generation Evaulation
We compare different frameworks by testing on sc09 dataset using [unconditional audio generation benchmark repo](https://github.com/gzhu06/Unconditional-Audio-Generation-Benchmark).

## Notebooks

We listed our note on diffusion models.

## Code References
- [EDM by Nvdia](https://github.com/NVlabs/edm)
- [Unconditional diffwave by philsyn](https://github.com/philsyn/DiffWave-unconditional)
- [Audio Diffusion by Flavio](https://github.com/archinetai/audio-diffusion-pytorch)
- [Imagen codebase by lucidrians](https://github.com/lucidrains/imagen-pytorch)

## Resources
This repo is generated with [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).
