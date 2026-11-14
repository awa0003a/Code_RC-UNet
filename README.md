# Code_RC-UNet
RC-UNet: Geometry-Aware Optic Disc Segmentation for Retinopathy of Prematurity
# RC-UNet: Geometry-Aware Optic Disc Segmentation for Retinopathy of Prematurity

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Status-Research-lightgrey" />
</p>

> **RC-UNet** is a geometry-constrained U-Net architecture for optic disc segmentation in neonatal retinal fundus images acquired during Retinopathy of Prematurity (ROP) screening. It introduces two novel components: a **Radial Depthwise Convolution Block** with exact ring-weight sharing, and a **Circular Ring Spatial Attention Gate** with geometrically constrained kernel support.

---

## Highlights

- **Exact geometric constraints** — radial kernels enforce ring weight sharing and disk-only support via `ring_basis()`; a built-in `geometry_audit()` verifies constraints at runtime
- **Three operating modes** — `radial` (constrained convolution + standard attention), `circular` (standard convolution + constrained attention), `combined` (both constraints active)
- **Uncertainty-aware inference** — MC-Dropout (Dropout2d) at the bottleneck enables predictive entropy and variance maps
- **Multi-camera support** — trained and evaluated across Clarity RetCam 3, RetCam Envision, and Phoenix ICON
- **Patient-level data splitting** — no patient leakage across train/validation/test splits
- **Single-file design** — entire pipeline in one `.py` file; no argparse, edit the config block at the top

---

## Architecture Overview

```
Input (3 × H × W)
    │
    ├── Encoder 1: ConvBlock          (3 → β)
    ├── Encoder 2: ConvBlock          (β → 2β)
    ├── Encoder 3: RadialConvBlock    (2β → 4β)   ← geometric constraint
    └── Bottleneck: RadialConvBlock   (4β → 8β)   ← geometric constraint
                 + CircularAttentionGate           ← ring-constrained spatial gate
                 + MC-Dropout2d
    ┌── Decoder 3: ConvBlock + skip   (8β → 4β)
    ├── Decoder 2: ConvBlock + skip   (4β → 2β)
    ├── Decoder 1: ConvBlock + skip   (2β → β)
    └── Output: 1×1 Conv → logits
```

Base channel width `β = 64` (Standard). Small (`β = 32`) and Large (`β = 128`) variants available.

---

## Dataset

This code was developed and evaluated on the **Retinal Image Dataset of Infants and Retinopathy of Prematurity** (Timkovič et al., *Scientific Data*, 2024).

| Split | Patients | Images |
|-------|----------|--------|
| Train | — | 907 |
| Validation | — | 195 |
| Test | — | 195 |

- **Version 1** (6,004 images): publicly available at [figshare](https://doi.org/10.6084/m9.figshare.c.6626162.v1)
- **Version 2** (1,297 annotated images): representative subset of 149 images from 7 patients available at [github.com/awa0003a/RC-UNET](https://github.com/awa0003a/RC-UNET_); complete dataset available from the corresponding author upon reasonable request

---

## Requirements

```bash
Python >= 3.9
torch >= 2.0
torchvision
numpy
Pillow
```

Install dependencies:

```bash
pip install torch torchvision numpy Pillow
```

---

## Quick Start

**1. Prepare your dataset**

Organise image–mask pairs in a single folder:

```
data/
├── img001.png
├── img001_gt.png
├── img002.png
├── img002_gt.png
...
```

Masks must follow the `<stem>_gt.<ext>` naming convention.

**2. Edit the configuration block**

Open `RC_UNet_Geometric_Improved_Single_Cell.py` and edit the top section:

```python
DATA_DIR   = '/path/to/your/dataset'
OUTPUT_DIR = '/path/to/output'
MODE       = 'combined'   # 'radial' | 'circular' | 'combined'
IMAGE_SIZE = 384
BATCH_SIZE = 4
EPOCHS     = 50
```

**3. Run training**

```bash
python RC_UNet_Geometric_Improved_Single_Cell.py
```

Outputs saved to `OUTPUT_DIR`:
- `best_model.pth` — best validation Dice checkpoint
- `metrics.csv` — per-epoch train/val metrics
- `test_results.json` — final test set evaluation
- `worst_predictions/` — worst-N failure cases for inspection

---

## Key Hyperparameters

| Parameter | Value |
|-----------|-------|
| Input resolution | 384 × 384 |
| Batch size | 4 |
| Optimizer | AdamW |
| Learning rate | 2 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁵ |
| Dropout rate | 0.20 |
| Early stopping patience | 20 epochs |
| Scheduler patience | 7 epochs |
| Loss | BCE + Soft Dice + Tversky |
| Normalisation | ImageNet (mean/std) |
| Seed | 42 |

---

## Geometric Constraint Verification

After training, verify that all geometric constraints hold:

```python
report = geometry_audit(model)
# Raises AssertionError if any constraint is violated
# Returns per-layer report of ring structure, disk support, and symmetry
```

---

## Modes

| Mode | Convolution | Attention |
|------|-------------|-----------|
| `radial` | RadialConvBlock (constrained) | Standard channel-spatial |
| `circular` | MS-DWBlock (unconstrained) | CircularAttentionGate (constrained) |
| `combined` | RadialConvBlock (constrained) | CircularAttentionGate (constrained) |

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{rcunet2026,
  title   = {RC-UNet: A Geometry-Aware Framework with Radial Convolution
             and Circular Attention for Optic Disc Segmentation
             in Retinopathy of Prematurity},
  author  = {Muhammad Awais et al.},
  journal = {Under Review},
  year    = {2026}
}
```

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

This work uses the ROP screening dataset collected at the **University Hospital Ostrava, Czech Republic**, published by Timkovič et al. (*Scientific Data*, 2024, DOI: [10.1038/s41597-024-03409-7](https://doi.org/10.1038/s41597-024-03409-7)).
Experiments were conducted on the **Karolina HPC cluster** (IT4innovations national supercomputing center, VSB-TUO, Czech Republic).
