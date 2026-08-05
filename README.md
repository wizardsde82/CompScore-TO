# Diffusion-model-driven topology optimization generates mechanically superior patient-matched orthopedic implants

CompScore-TO is a conditional latent diffusion system for three-dimensional patient-matched orthopedic implant design. It combines anatomy cross-attention with timestep-adaptive compliance, stress-shielding, and additive-manufacturing objectives. The repository contains training, preprocessing, physics-surrogate, evaluation, configuration, and distributed-runtime components for the 64³-voxel experiments.

## Installation

Python 3.11 and a CUDA 12.4-capable PyTorch environment are required for GPU training.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Conda users can create the pinned environment directly.

```bash
conda env create -f environment.yml
conda activate compscore-to
pip install -e .
```

The container image uses PyTorch 2.6.0, CUDA 12.4, and cuDNN 9.

```bash
docker build -t compscore-to .
```

## Data

The verified official access points are collected in `datasets.txt`. NMDID requires a research request and institutional account. OAI requires acceptance of the NIH Data Sharing Agreement. The two training archives named in the study are omitted from that file because their cited landing pages could not be verified from the current network environment.

Input anatomy masks are NIfTI volumes. Preprocessing resamples masks to 1 mm isotropic spacing, retains the largest connected component, crops the anatomical region, and pads or crops to 64³.

```bash
compscore-prepare --input raw_masks --output data/processed/anatomy --site nmdid
```

Training manifests are JSON lists. Each record supplies relative paths for anatomy, topology, load, boundary-condition, and stress arrays, with scalar compliance, manufacturability, and volume-fraction targets. Arrays are stored in NumPy format.

## Training

The reported schedule requires an NVIDIA A100 80 GB GPU. The VAE uses batch size 64 for 120 epochs, AdamW at 2×10⁻⁴, weight decay 10⁻², cosine warm restarts, KL weight 10⁻⁴, and gradient clipping at 1.0.

```bash
compscore-train --manifest data/processed/manifest.json --stage vae
```

The anatomy encoder is pre-trained for 40 epochs on bone-mask reconstruction.

```bash
compscore-train --manifest data/processed/manifest.json --stage anatomy
```

The five-layer multi-physics graph surrogate is trained for 80 epochs on paired topology and direct-FEA samples.

```bash
compscore-train --manifest data/processed/manifest.json --stage surrogate
```

The EDM denoiser uses batch size 48 for 200,000 iterations, AdamW at 3×10⁻⁴, weight decay 10⁻³, 2,000 warmup steps, cosine decay, and EMA 0.9999.

```bash
compscore-train --manifest data/processed/manifest.json --stage diffusion
```

Configuration fields can be overridden after the command arguments.

```bash
compscore-train --manifest data/processed/manifest.json --stage diffusion diffusion.batch_size=48 seed=2026
```

The full training cost reported for the 64³ configuration is 72 A100 GPU-hours, excluding approximately 36,000 direct FEA solves needed to build the surrogate dataset. The prepared training pairs occupy storage according to the chosen stress-field representation; dense float32 volumes require approximately 75 GB before metadata and optimizer artifacts.

## Evaluation

Evaluation consumes an NPZ archive containing `generated_compliance`, `reference_compliance`, `stress_ratio`, `manufacturing`, `fatigue`, `generated_geometry`, and `anatomy_roi`. It reports compliance error, stress-shielding ratio, manufacturability, fatigue-life proxy, Dice, and BCa 95% confidence intervals with 10,000 resamples.

```bash
compscore-evaluate --results results/direct_fea_arrays.npz --output results/summary.json
```

The main retrospective cohort contains 1,280 anatomies with five random seeds for stochastic methods. Reference aggregate targets are compliance error 5.108 ± 1.31%, stress-shielding ratio 0.742 ± 0.038, manufacturability 91.2 ± 2.8%, fatigue-life proxy 1.423 ± 0.071, anatomy-fit Dice 0.893 ± 0.013, and generation time 5.2 ± 0.3 seconds on one A100. Final values must be computed from direct FEA outputs rather than surrogate predictions.

## Method configuration

The latent representation has four channels and 8× spatial compression per axis. Anatomy conditioning consists of 512 tokens with width 256. EDM noise follows log-normal parameters Pmean −1.2 and Pstd 1.2 with σmin 0.002 and σmax 80. The primary sampler setting uses 50 reverse steps. Base physics strengths are 5.0 for compliance, 3.0 for stress shielding, and 2.0 for manufacturability. SIMP material interpolation uses penalty 3 and Ti-6Al-4V modulus 110 GPa.

The manufacturability objective combines a 45-degree maximum overhang, 0.3 mm minimum feature size, and connected topology. Stress shielding is evaluated in a 2 mm peri-implant shell. Ground-truth generation uses eight physiological load cases and volume fractions 0.2, 0.3, 0.4, and 0.5.

## Scope

This software supports computational retrospective analysis. Mechanical values require direct finite-element verification. They are not clinical performance claims, and fabricated-device validation is outside the included workflow.

