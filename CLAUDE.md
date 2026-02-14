# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unofficial PyTorch implementation of ["Generative Modeling via Drifting" (Deng et al., 2026)](https://arxiv.org/abs/2602.04770). One-step generative model (1-NFE) that trains a DiT-style generator by computing a drifting field V via soft assignment matrices. Supports MNIST (pixel space) and CIFAR-10 (CNN feature space). ImageNet support is planned but not yet implemented.

## Environment Setup

Uses `uv` with Python 3.10. Dependencies: PyTorch 2.9.1 (CUDA 13.0), torchvision, einops, matplotlib, numpy.

```bash
uv venv --python 3.10 --seed
uv pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu130 --link-mode=copy
uv pip install einops --link-mode=copy
```

## Commands

```bash
# Training
python train.py --dataset mnist      # MNIST, pixel space (~20 min on GPU)
python train.py --dataset cifar10    # CIFAR-10, with feature encoder

# Sampling from trained checkpoint
python sample.py --checkpoint outputs/mnist/checkpoint_final.pt --dataset mnist
```

There is no test suite, linter, or build system. No pyproject.toml or requirements.txt — deps are in `k.oku`.

## Architecture

Six modules with clear separation of concerns:

- **model.py** — `DriftDiT` generator: patch-based DiT with adaLN-Zero, RoPE, QK-Norm, SwiGLU, classifier-free guidance. Two variants: `DriftDiT-Tiny` (MNIST) and `DriftDiT-Small` (CIFAR-10). Factory dict `DriftDiT_models` maps string names to constructors.
- **drifting.py** — Core algorithm: `compute_V()` implements Algorithm 2 (pairwise L2 distances → soft assignment via geometric mean of row/col softmax → drift field). `compute_V_multi_temperature()` runs at temperatures [0.02, 0.05, 0.2] and sums normalized fields. `normalize_features()` and `normalize_drift()` handle feature standardization.
- **feature_encoder.py** — `MultiScaleFeatureEncoder` (custom CNN) and `PretrainedResNetEncoder` (ImageNet ResNet18) for CIFAR-10 feature extraction. MNIST uses raw pixels. Factory: `create_feature_encoder()`.
- **train.py** — Training loop (Algorithm 1). Configs `MNIST_CONFIG` and `CIFAR10_CONFIG` contain all hyperparameters. Per-class batch sampling, queue-based positive sample management, EMA updates, warmup LR schedule.
- **sample.py** — One-step inference with `forward_with_cfg()`, grid visualization, alpha sweep.
- **utils.py** — `EMA`, `WarmupLRScheduler`, `SampleQueue`, checkpoint save/load, image grid utilities.

### Key data flow

1. Generator takes noise + class label + CFG alpha → image (one forward pass)
2. Feature encoder maps generated and real images to feature space (skip for MNIST)
3. `compute_V` computes drift field pointing generated features toward real data
4. Loss = MSE between features and stopgrad(features + V)
5. When ||V||=0, generated distribution matches real data

### Conditioning mechanism

Three embeddings summed: label (class + null for CFG dropout), alpha (Fourier features for CFG scale), style (random codebook tokens for diversity). Fed to each DiT block via adaLN-Zero modulation (6 params per block: shift/scale for attention and MLP, plus gates).

### Checkpoints

Saved to `outputs/{dataset}/` with model, EMA, optimizer, scheduler, epoch, step, and config. Training supports resumption via `--resume`.
