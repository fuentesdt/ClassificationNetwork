# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Does

A 3D medical image classification neural network in MATLAB. It trains a volumetric CNN to perform binary classification on NIfTI (`.nii`) medical images using K-fold cross-validation, with Grad-CAM visualization for interpretability.

## Running the Code

This is a pure MATLAB project with no build system. Run the main script from the MATLAB command window or CLI:

```matlab
classifier3d
```

Or from the terminal:

```bash
matlab -batch "classifier3d"
```

## Data Setup

Input data must be organized in `adcnetwork/` (relative to the working directory) with one subdirectory per class — MATLAB's `imageDatastore` derives class labels from folder names. Images must be NIfTI format, 256×256×64 voxels (4D arrays: H×W×D×channels). The script writes Grad-CAM output NIfTI files to `newmap/`.

## Architecture

The entire pipeline lives in `classifier3d.m`, with two helper functions at the bottom of the file:

- **`mycustomreader(filename)`** — custom NIfTI reader using `niftiread`; squeezes the 4D volume to pass to the datastore
- **`stopTraining(info)`** — early stopping callback; halts training if val accuracy hasn't improved after 40 epochs or stays below 90%

**Network structure** (3D CNN, input 256×256×64):
1. Conv3D (3×3×3, 8 filters) → ReLU → MaxPool3D (4×4×4)
2. Conv3D (3×3×3, 8 filters) → ReLU → MaxPool3D (4×4×4)
3. Conv3D (3×3×3, 2 filters) → ReLU → GlobalAvgPool → Scale(0.25) → Softmax → Classification

**Training loop** runs a grid search over class weights `[1, 2, 10]` × epoch counts `[4, 8, 16, 32, 64, 128]` inside a 5-fold cross-validation. Optimizer: Adam, LR 0.001, L2 reg 0.1, batch size 4.

**Evaluation** computes accuracy and confusion matrix per fold, then generates Grad-CAM maps (saved as `.nii` files in `newmap/`) for each test sample.

## Key Parameters to Tune

| Variable | Location | Default | Purpose |
|---|---|---|---|
| `numFolds` | line 61 | 5 | K-fold count |
| `classWeights` | training loop | `[1, 2, 10]` | Grid search values |
| `maxEpochs` | training loop | `[4…128]` | Grid search values |
| `InitialLearnRate` | `trainingOptions` | 0.001 | Adam LR |
| `L2Regularization` | `trainingOptions` | 0.1 | Weight decay |
| `OutputNetwork` | `trainingOptions` | `'last-iteration'` | Change to `'best-validation'` to keep best checkpoint |

## MATLAB Toolbox Requirements

- Deep Learning Toolbox (for `trainNetwork`, `imageInputLayer`, `convolution3dLayer`, `gradCAM`, etc.)
- Image Processing Toolbox (for `niftiread`, `niftiwrite`)
- Statistics and Machine Learning Toolbox (for `cvpartition`)
- GPU support is optional; `trainNetwork` auto-detects available hardware
