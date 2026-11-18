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
- **Three operating modes** — `radial` (constrained convolution + standard attention), `combined` (standard convolution + constrained attention), `circular` (both constraints active)
- **Uncertainty-aware inference** — MC-Dropout (`Dropout2d`) at the bottleneck enables predictive entropy and variance maps
- **Multi-camera support** — trained and evaluated across Clarity RetCam 3, RetCam Envision, and Phoenix ICON
- **Patient-level data splitting** — no patient leakage across train/validation/test splits; subject IDs inferred from filename prefixes or supplied via CSV
- **Modular design** — pipeline split across four files for readability and maintainability

---

## Repository Structure

```
RC_UNet_modular_swapped/
├── model.py                 # GeometryRCUNet, RadialConvBlock, CircularAttentionGate, ring_basis()
├── dataset_preprocess.py    # OpticDiscDataset, patient-grouped splitting, augmentation
├── utils.py                 # geometry_audit(), segmentation_loss(), evaluate(), HD95/ASSD
└── train.py                 # Configuration block, training loop, threshold tuning, plots
```

---

## Architecture Overview

```
Input (3 × H × W)
    │
    ├── Encoder 1: ConvBlock          (3 → β)
    ├── Encoder 2: ConvBlock          (β → 2β)
    ├── Encoder 3: RadialConvBlock    (2β → 4β)   ← geometric constraint (radial / circular modes)
    └── Bottleneck: RadialConvBlock   (4β → 8β)   ← geometric constraint (radial / circular modes)
                 + CircularAttentionGate           ← ring-constrained spatial gate (combined / circular modes)
                 + MC-Dropout2d
    ┌── Decoder 3: ConvBlock + skip   (8β → 4β)
    ├── Decoder 2: ConvBlock + skip   (4β → 2β)
    ├── Decoder 1: ConvBlock + skip   (2β → β)
    └── Output: 1×1 Conv → logits
```

Base channel width `β = 64` (Standard). Small (`β = 32`) and Large (`β = 128`) variants available.

---

## Modes

| Mode | Convolution | Attention | Notes |
|------|-------------|-----------|-------|
| `radial` | RadialConvBlock (constrained) | OriginalChannelSpatialAttention (standard) | Radial constraint only |
| `combined` | OriginalMultiScaleDepthwiseBlock (standard) | CircularAttentionGate (constrained) | Circular constraint only |
| `circular` | RadialConvBlock (constrained) | CircularAttentionGate (constrained) | Both constraints active — corresponds to **G3** (best configuration) |

> **Note on naming:** The labels `combined` and `circular` are swapped relative to earlier versions of this codebase. `circular` now refers to the fully constrained model (both radial convolution and circular attention active).

---

## Dataset

This code was developed and evaluated on the **Retinal Image Dataset of Infants and Retinopathy of Prematurity** (Timkovič et al., *Scientific Data*, 2024).

| Split | Images |
|-------|--------|
| Train | 907 |
| Validation | 195–199 |
| Test | 195–204 |

Split sizes vary slightly depending on whether a pre-recorded manifest (`EXISTING_SPLIT_CSV`) is used or splits are generated from scratch.

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
matplotlib
```

Install dependencies:

```bash
pip install torch torchvision numpy Pillow matplotlib
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

Open `train.py` and edit the top section:

```python
DATA_DIR              = '/path/to/your/dataset'
OUTPUT_DIR            = '/path/to/output'
EXISTING_SPLIT_CSV    = None          # or path to a saved split manifest
MODE                  = 'circular'    # 'radial' | 'combined' | 'circular'
IMAGE_SIZE            = 384
BASE_FEATURES         = 64
BATCH_SIZE            = 4
EPOCHS                = 50
```

**3. Run training**

```bash
python train.py
```

Outputs are saved to `OUTPUT_DIR/<mode>/<timestamp>/`:

| File | Description |
|------|-------------|
| `best_model.pt` | Best validation Dice checkpoint |
| `training_history.csv` | Per-epoch train/val metrics and learning rate |
| `test_metrics.json` | Full test set evaluation including HD95, ASSD, and geometry audit |
| `split_manifest.csv` | Recorded train/val/test split for reproducibility |
| `config.json` | Full configuration snapshot |
| `training_curves.png` | Loss and Dice curves |
| `worst_test_examples.png` | Worst-N failure cases (diagnostic) |
| `learned_geometric_kernels.png` | Visualisation of learned radial/circular kernel weights |

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
| Loss | 0.35 × BCE + 0.40 × Soft Dice + 0.25 × Tversky |
| Normalisation | ImageNet (mean/std) |
| Seed | 42 |
| Threshold tuning | Validation-based grid search over {0.30 … 0.70} |

---

## Geometric Constraint Verification

After training, verify that all geometric constraints hold:

```python
from utils import geometry_audit
report = geometry_audit(model)
# Raises AssertionError if any constraint is violated
# Returns per-layer report: ring structure, disk support, rotational and reflection symmetry
```

The audit checks:
- **Disk support** — no weight outside the circular kernel mask
- **Rotational symmetry** — 90° rotation invariance of effective weights
- **Reflection symmetry** — horizontal flip invariance of effective weights

---

## Patient-Level Splitting

Subject identity is inferred automatically from filename prefixes (e.g. `pacient_001_serie_...`) or supplied via a CSV with columns `image_name, patient_id`. A saved `split_manifest.csv` can be passed as `EXISTING_SPLIT_CSV` to reproduce an exact split across runs.

```python
PATIENT_CSV          = None      # or '/path/to/patient_ids.csv'
GROUP_FROM_FILENAMES = True      # infer subject IDs from filename prefixes
EXISTING_SPLIT_CSV   = None      # or '/path/to/split_manifest.csv'
```

> A runtime warning is raised when subject IDs are inferred from filenames. Verify the grouping before publication.

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
Experiments were conducted on the **Karolina HPC cluster** (IT4Innovations National Supercomputing Center, VSB-TUO, Czech Republic).
