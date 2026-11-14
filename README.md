# Deep Learning Experiments: Optimization, Symmetry, and Image Super-Resolution

This repository contains practical Deep Learning coursework experiments implemented primarily with PyTorch. The exercises explore neural-network training behavior and a small image super-resolution pipeline.

## Topics covered

- **Gradient accumulation** — accumulating gradients over multiple mini-batches before updating model parameters.
- **Symmetry breaking** — demonstrating why identical/zero initialization can prevent hidden units from learning distinct representations, and experimenting with randomized weight initialization.
- **Image super-resolution** — reducing image resolution, constructing neighborhood-based pixel features, and training a multilayer perceptron (MLP) to predict RGB values at the original resolution.
- **Image-quality evaluation** — comparing reconstructed images with reference images using SSIM and PSNR.

## Repository structure

```text
.
├── HW2.pdf
├── code/
│   ├── P2/       # Gradient accumulation
│   ├── P4/       # Symmetry and neural-network initialization
│   └── P5/       # Image super-resolution implementation and outputs
└── report/       # Assignment report
```

## Main technologies

- Python
- PyTorch
- NumPy
- Pillow (PIL)
- scikit-image
- Jupyter Notebook

## Getting started

Install the main dependencies:

```bash
pip install torch numpy pillow scikit-image jupyter
```

Open the notebooks under `code/` with Jupyter. Some image experiments expect the image assets and model checkpoints stored alongside the relevant code.

## Notes

The repository preserves the original assignment files, notebooks, model checkpoints, and generated image results. Exact runtime requirements may depend on the PyTorch and scikit-image versions used.
