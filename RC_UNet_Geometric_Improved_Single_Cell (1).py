"""
RC-UNet geometry-constrained optic-disc segmentation — ONE unified code cell.

No argparse. Edit the configuration at the top and run main().
Train ONE mode per execution: 'radial', 'circular', or 'combined'.

CHANGES relative to the previous single-cell version:
  1. Patient-grouped train/validation/test splitting (no silent image-level fallback).
  2. 384x384 default input, batch=4 (similar pixel budget to 256x256/batch=8).
  3. Mild paired anatomical augmentations; none on validation/test.
  4. BCE + soft Dice + recall-sensitive Tversky loss.
  5. AdamW, AMP on CUDA, gradient clipping, plateau LR schedule.
  6. Threshold selected exclusively on validation data AFTER checkpoint selection.
  7. Optional horizontal-flip test-time averaging applied equally to val and test.
  8. Per-image failure inspection, Dice/IoU/precision/recall, optional HD95/ASSD.

These choices may help, but improved performance is NOT guaranteed.
Geometric constraints in radial/circular modules are enforced exactly.
"""
from __future__ import annotations

import csv
import json
import math
import random
import re
import time
import warnings
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================================
# EDIT HERE ONLY: hardcoded paths and settings. NO argparse.
# ============================================================================
# HPC paths from the last uploaded config.json:
DATA_DIR = '/mnt/proj3/dd-24-125/awa0003/lungs_dataset/final'
OUTPUT_DIR = '/mnt/proj3/dd-24-125/awa0003/lungs_dataset/experiment_retina_v2'
# Colab alternative:
# DATA_DIR = '/content/drive/MyDrive/ret_dataset'
# OUTPUT_DIR = '/content/drive/MyDrive/uq_unet_out/geometry_runs_v2'

PATIENT_CSV = None  # optional CSV: image_name,patient_id; preferred over inferred IDs
# Without a CSV, extract grouped subject identifiers from BOTH known filename styles.
# Verify these inferred identifiers against dataset metadata BEFORE manuscript reporting.
GROUP_FROM_FILENAMES = True
EXISTING_SPLIT_CSV = None  # optional previously VERIFIED grouped split_manifest.csv

MODE = 'radial'       # choose ONE: 'radial', 'circular', 'combined'
IMAGE_SIZE = 384      # higher resolution may improve small optic-disc predictions
BASE_FEATURES = 64
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 1e-5
DROPOUT = 0.20
EARLY_STOP_PATIENCE = 20
SCHEDULER_PATIENCE = 7
VAL_FRAC = 0.15
TEST_FRAC = 0.15
NUM_WORKERS = 2
SEED = 42

AUGMENT_TRAIN = True      # mild ±12° rotation, ≤6% scale, 3% translations, color jitter
ALLOW_HORIZONTAL_FLIP = True
TTA_HORIZONTAL_FLIP = True  # average normal and horizontal-flip predictions at val/test
TUNE_THRESHOLD_ON_VAL = True
THRESHOLD_GRID = (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
DEFAULT_THRESHOLD = 0.50
USE_AMP = True
GRAD_CLIP_NORM = 5.0
SAVE_WORST_N = 8

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ring_basis(kernel_size):
    if kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError('Geometric kernel size must be an odd integer >=3.')
    radius = kernel_size // 2
    y, x = torch.meshgrid(torch.arange(-radius, radius + 1),
                          torch.arange(-radius, radius + 1), indexing='ij')
    d2 = x.square() + y.square()
    disk = d2 <= radius * radius
    rings = torch.stack([(d2 == r).float() for r in torch.unique(d2[disk])])
    rings = rings / torch.sqrt(rings.sum(dim=(-1, -2), keepdim=True))
    return rings, disk.float()


class RadialDepthwiseConv2d(nn.Module):
    """Disk-only depthwise filters with EXACT equal-distance ring weight sharing."""
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        if out_ch % in_ch:
            raise ValueError('out_ch must be divisible by in_ch for depthwise filtering.')
        basis, support = ring_basis(kernel_size)
        self.register_buffer('basis', basis)
        self.register_buffer('support', support)
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.theta = nn.Parameter(torch.empty(out_ch, len(basis)))
        nn.init.normal_(self.theta, std=math.sqrt(2.0 / len(basis)))
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def effective_weight(self):
        return torch.einsum('cr,rxy->cxy', self.theta, self.basis).unsqueeze(1)

    def forward(self, x):
        return F.conv2d(x, self.effective_weight(), self.bias,
                        padding=self.kernel_size // 2, groups=self.in_ch)


class RadialConvBlock(nn.Module):
    """3/5/7 parallel depthwise branches + fusion, with constrained kernels."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branches = nn.ModuleList([RadialDepthwiseConv2d(in_ch, out_ch, k)
                                       for k in (3, 5, 7)])
        self.fuse = nn.Sequential(nn.Conv2d(out_ch * 3, out_ch, 1),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.post_bn = nn.BatchNorm2d(out_ch)
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.fuse(torch.cat([conv(x) for conv in self.branches], dim=1))
        return self.post_relu(self.post_bn(y + x)) if x.shape[1] == y.shape[1] else y


class OriginalMultiScaleDepthwiseBlock(nn.Module):
    """Original unconstrained block: retained only when MODE='circular'."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv2d(in_ch, out_ch, k, padding=k // 2,
                                                   groups=in_ch) for k in (3, 5, 7)])
        self.fuse = nn.Sequential(nn.Conv2d(out_ch * 3, out_ch, 1),
                                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.post_bn = nn.BatchNorm2d(out_ch)
        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        y = self.fuse(torch.cat([conv(x) for conv in self.branches], dim=1))
        return self.post_relu(self.post_bn(y + x)) if x.shape[1] == y.shape[1] else y


class CircularRingSpatialFilter(nn.Module):
    """Two-input 7x7 spatial attention filter, constrained to radial rings."""
    def __init__(self, kernel_size=7):
        super().__init__()
        basis, support = ring_basis(kernel_size)
        self.register_buffer('basis', basis)
        self.register_buffer('support', support)
        self.kernel_size = kernel_size
        self.theta = nn.Parameter(torch.empty(2, len(basis)))
        nn.init.normal_(self.theta, std=0.15)
        self.bias = nn.Parameter(torch.zeros(1))

    def effective_weight(self):
        return torch.einsum('cr,rxy->cxy', self.theta, self.basis).unsqueeze(0)

    def forward(self, x):
        return F.conv2d(x, self.effective_weight(), self.bias,
                        padding=self.kernel_size // 2)


class CircularAttentionGate(nn.Module):
    """SE channel gate + genuinely ring-constrained LOCAL spatial gate."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hid = max(4, channels // reduction)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, hid, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hid, channels, 1), nn.Sigmoid())
        self.circular_spatial = CircularRingSpatialFilter(7)

    def forward(self, x):
        z = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        spatial = torch.sigmoid(self.circular_spatial(z))
        return x * self.channel_gate(x) * spatial


class OriginalChannelSpatialAttention(nn.Module):
    """Original unconstrained attention: retained only when MODE='radial'."""
    def __init__(self, channels):
        super().__init__()
        h = max(channels // 8, 1)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, h, 1), nn.ReLU(inplace=True),
            nn.Conv2d(h, channels, 1), nn.Sigmoid())
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=5, padding=2, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.channel_gate(x) * self.spatial_gate(x)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class GeometryRCUNet(nn.Module):
    """Preserves Standard U-Net topology, changing ONLY selected geometric modules."""
    def __init__(self, mode='radial', base=64, dropout=0.2):
        super().__init__()
        if mode not in ('radial', 'circular', 'combined'):
            raise ValueError('MODE must be radial, circular, or combined.')
        conv = RadialConvBlock if mode in ('radial', 'combined') else OriginalMultiScaleDepthwiseBlock
        attention = CircularAttentionGate if mode in ('circular', 'combined') else OriginalChannelSpatialAttention
        self.mode = mode
        self.enc1 = ConvBlock(3, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = conv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck_conv = conv(base * 4, base * 8)
        self.bottleneck_attn = attention(base * 8)
        self.dropout = nn.Dropout2d(dropout)  # OFF during evaluation
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        if x.shape[-2] % 8 or x.shape[-1] % 8:
            raise ValueError('Height and width must be divisible by 8.')
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        d = self.dropout(self.bottleneck_attn(self.bottleneck_conv(self.pool(c))))
        y = self.dec3(torch.cat([self.up3(d), c], dim=1))
        y = self.dec2(torch.cat([self.up2(y), b], dim=1))
        y = self.dec1(torch.cat([self.up1(y), a], dim=1))
        return self.head(y)  # logits


def geometry_audit(model):
    report = {}
    for name, module in model.named_modules():
        if isinstance(module, (RadialDepthwiseConv2d, CircularRingSpatialFilter)):
            w = module.effective_weight().detach().float().cpu()
            support = module.support.detach().float().cpu()
            outside = (w * (1 - support)).abs().max().item()
            rotate = (w - torch.rot90(w, 1, (-2, -1))).abs().max().item()
            reflect = (w - torch.flip(w, (-1,))).abs().max().item()
            if max(outside, rotate, reflect) > 1e-5:
                raise AssertionError(f'Geometric constraint failed at {name}.')
            report[name] = dict(kernel_size=module.kernel_size,
                                number_of_rings=len(module.basis),
                                max_outside_disk=outside,
                                max_rot90_deviation=rotate,
                                max_reflection_deviation=reflect)
    return report


def find_pairs(folder):
    folder = Path(folder).expanduser()
    if not folder.is_dir():
        raise FileNotFoundError(f'DATA_DIR does not exist: {folder}')
    imgs, masks = {}, {}
    for f in sorted(folder.iterdir()):
        if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
            continue
        key = re.sub(r'_gt$', '', f.stem, flags=re.IGNORECASE).lower()
        dest = masks if re.search(r'_gt$', f.stem, flags=re.IGNORECASE) else imgs
        if key in dest:
            raise RuntimeError(f'Duplicate image/mask stem: {key}')
        dest[key] = str(f)
    matched = sorted(imgs.keys() & masks.keys())
    unmatched = sorted(imgs.keys() ^ masks.keys())
    if unmatched:
        raise RuntimeError(f'{len(unmatched)} unpaired files detected. Example: {unmatched[:5]}')
    if len(matched) < 3:
        raise RuntimeError(f'Only {len(matched)} image/_gt pairs found in {folder}.')
    return [(imgs[k], masks[k]) for k in matched]


def filename_group(image_path):
    """Heuristic grouping by apparent patient ID, NOT authoritative metadata."""
    name = Path(image_path).stem
    m = re.match(r'^pacient\s*0*(\d+)\s*_s[eé]rie\b', name, re.IGNORECASE)
    if m:
        return 'pacient_cohort_' + m.group(1).zfill(3)
    m = re.match(r'^0*(\d+)\s*_s[eé]rie\b', name, re.IGNORECASE)
    if m:
        return 'numeric_cohort_' + m.group(1).zfill(3)
    raise ValueError(f'Unknown filename pattern; cannot infer subject for {name!r}. '
                     'Provide PATIENT_CSV with verified image_name,patient_id.')


def get_patient_mapping(pairs, patient_csv=None):
    if patient_csv is not None:
        with open(patient_csv, newline='', encoding='utf-8-sig') as fh:
            rd = csv.DictReader(fh)
            if not {'image_name', 'patient_id'}.issubset(rd.fieldnames or []):
                raise ValueError('PATIENT_CSV must contain columns image_name,patient_id.')
            mapping = {}
            for row in rd:
                key = Path(row['image_name']).stem.lower()
                pid = row['patient_id'].strip()
                if not pid:
                    raise ValueError(f'Empty patient ID for {key}')
                if key in mapping and mapping[key] != pid:
                    raise ValueError(f'Conflicting patient ID for {key}')
                mapping[key] = pid
        pids = {}
        for ip, _ in pairs:
            key = Path(ip).stem.lower()
            if key not in mapping:
                raise ValueError(f'Missing patient ID for {key}')
            pids[key] = mapping[key]
        source = 'verified_mapping_CSV'
    elif GROUP_FROM_FILENAMES:
        pids = {Path(ip).stem.lower(): filename_group(ip) for ip, _ in pairs}
        source = 'filename_inferred_UNVERIFIED'
        warnings.warn('Subject IDs inferred from filename prefixes. Check these match the original '
                      'clinical patient IDs before any publication claims.', stacklevel=2)
    else:
        raise ValueError('Patient grouping is mandatory. Supply PATIENT_CSV or set '
                         'GROUP_FROM_FILENAMES=True after checking your filenames.')
    return pids, source


def make_grouped_splits(pairs, patient_map, seed, val_frac, test_frac):
    """Greedy patient-disjoint split, approximately targets IMAGE fractions."""
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError('Invalid val/test fractions.')
    groups = defaultdict(list)
    for pair in pairs:
        groups[patient_map[Path(pair[0]).stem.lower()]].append(pair)
    if len(groups) < 3:
        raise ValueError('At least three distinct subjects are required.')
    items = sorted(groups.items())
    rng = random.Random(seed)
    rng.shuffle(items)
    targets = {'test': len(pairs) * test_frac,
               'val': len(pairs) * val_frac}
    allocated = {'train': [], 'val': [], 'test': []}
    counts = {'train': 0, 'val': 0, 'test': 0}
    # Seed each split with one whole patient, then fill test and val by image count.
    for destination, (pid, sample_list) in zip(('test', 'val', 'train'), items[:3]):
        allocated[destination].extend(sample_list)
        counts[destination] += len(sample_list)
    for pid, sample_list in items[3:]:
        if counts['test'] < targets['test']:
            dst = 'test'
        elif counts['val'] < targets['val']:
            dst = 'val'
        else:
            dst = 'train'
        allocated[dst].extend(sample_list)
        counts[dst] += len(sample_list)
    splits = tuple(allocated[d] for d in ('train', 'val', 'test'))
    assert all(splits)
    return splits


def load_verified_manifest(csv_path, pairs, patient_map):
    expected = {str(Path(ip).resolve()): (ip, mp) for ip, mp in pairs}
    assignment = {'train': [], 'val': [], 'test': []}
    seen = set()
    with open(csv_path, newline='', encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh):
            split = row['split'].strip().lower()
            if split not in assignment:
                raise ValueError(f'Unknown split name {split}')
            ip = str(Path(row['image_path']).resolve())
            if ip not in expected or ip in seen:
                raise ValueError('Verified manifest has absent or duplicated image: ' + ip)
            pid = row.get('patient_id', '').strip()
            if pid and pid != patient_map[Path(ip).stem.lower()]:
                raise ValueError('Manifest patient_id inconsistent for ' + ip)
            assignment[split].append(expected[ip])
            seen.add(ip)
    if seen != set(expected) or not all(assignment.values()):
        raise ValueError('Verified manifest must cover every image and all three splits.')
    return tuple(assignment[s] for s in ('train', 'val', 'test'))


def ensure_no_subject_leak(splits, patient_map):
    id_sets = []
    for split in splits:
        id_sets.append({patient_map[Path(ip).stem.lower()] for ip, _ in split})
    if any(id_sets[i] & id_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))):
        raise RuntimeError('Detected subject overlap between splits. STOPPING.')
    return dict(zip(('train', 'val', 'test'), [len(s) for s in id_sets]))


def save_manifest(path, splits, patient_map):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.writer(fh)
        wr.writerow(('split', 'image_path', 'mask_path', 'patient_id'))
        for name, split in zip(('train', 'val', 'test'), splits):
            for ip, mp in split:
                wr.writerow((name, ip, mp, patient_map[Path(ip).stem.lower()]))


def affine_pair(img, mask):
    """Small paired affine transform, no flips vertically or large angle rotations."""
    width, height = img.size
    if ALLOW_HORIZONTAL_FLIP and random.random() < 0.5:
        img, mask = ImageOps.mirror(img), ImageOps.mirror(mask)
    angle = random.uniform(-12, 12)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
    scale = random.uniform(0.94, 1.06)
    dx = random.uniform(-0.03, 0.03) * width
    dy = random.uniform(-0.03, 0.03) * height
    cx, cy = (width - 1) / 2, (height - 1) / 2
    inv = (1 / scale, 0, cx - (cx + dx) / scale,
           0, 1 / scale, cy - (cy + dy) / scale)
    img = img.transform((width, height), Image.Transform.AFFINE, inv,
                        resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
    mask = mask.transform((width, height), Image.Transform.AFFINE, inv,
                          resample=Image.Resampling.NEAREST, fillcolor=0)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.90, 1.10))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.90, 1.10))
    img = ImageEnhance.Color(img).enhance(random.uniform(0.92, 1.08))
    return img, mask


class OpticDiscDataset(Dataset):
    def __init__(self, pairs, size, augment=False):
        self.pairs = list(pairs)
        self.size = size
        self.augment = augment
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        with Image.open(ip) as source:
            img = source.convert('RGB').resize((self.size, self.size), Image.Resampling.BILINEAR)
        with Image.open(mp) as source:
            mask = source.convert('L').resize((self.size, self.size), Image.Resampling.NEAREST)
        if self.augment:
            img, mask = affine_pair(img, mask)
        image = torch.from_numpy(np.asarray(img, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.0
        image = (image - self.mean) / self.std
        binary_mask = torch.from_numpy((np.asarray(mask, dtype=np.uint8) > 127).astype(np.float32).copy())[None]
        return image, binary_mask


def segmentation_loss(logits, labels):
    """Foreground-aware objective to address high-precision / lower-recall pattern."""
    prob = logits.sigmoid()
    dimensions = (1, 2, 3)
    tp = (prob * labels).sum(dimensions)
    fp = (prob * (1 - labels)).sum(dimensions)
    fn = ((1 - prob) * labels).sum(dimensions)
    dice = 1 - ((2 * tp + 1) / (2 * tp + fp + fn + 1)).mean()
    # beta > alpha increases penalty for missed optic-disc pixels.
    tversky = 1 - ((tp + 1) / (tp + 0.35 * fp + 0.65 * fn + 1)).mean()
    bce = F.binary_cross_entropy_with_logits(logits, labels)
    return 0.35 * bce + 0.40 * dice + 0.25 * tversky


def score_tensors(prob, masks, threshold):
    pred = (prob >= threshold).flatten(1)
    true = (masks >= 0.5).flatten(1)
    tp = (pred & true).sum(1).float()
    fp = (pred & ~true).sum(1).float()
    fn = (~pred & true).sum(1).float()
    gt_area = true.sum(1).float()
    pr_area = pred.sum(1).float()
    # Correct convention: precision=0 for an empty prediction when GT is nonempty.
    dice = torch.where((2*tp+fp+fn) > 0, (2*tp)/(2*tp+fp+fn).clamp_min(1), torch.ones_like(tp))
    iou = torch.where((tp+fp+fn) > 0, tp/(tp+fp+fn).clamp_min(1), torch.ones_like(tp))
    prec = torch.where((tp+fp)>0, tp/(tp+fp).clamp_min(1), torch.where(gt_area>0, torch.zeros_like(tp), torch.ones_like(tp)))
    rec = torch.where((tp+fn)>0, tp/(tp+fn).clamp_min(1), torch.ones_like(tp))
    return {'dice': dice, 'iou': iou, 'precision': prec, 'recall': rec,
            'tp': tp, 'fp': fp, 'fn': fn, 'gt_pixels': gt_area, 'pred_pixels': pr_area}


def predict_prob(model, images, tta=False):
    p = torch.sigmoid(model(images))
    if tta:
        mirrored = torch.flip(images, dims=(-1,))
        p_flip = torch.sigmoid(model(mirrored))
        p = 0.5 * (p + torch.flip(p_flip, dims=(-1,)))
    return p


@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5, tta=False):
    model.eval()
    totals = defaultdict(list)
    loss_total = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        p = predict_prob(model, images, tta=tta)
        # Validation loss is only diagnostic here; logits based on averaged p.
        logits = torch.logit(p.clamp(min=1e-6, max=1-1e-6))
        loss_total += segmentation_loss(logits, labels).item() * images.shape[0]
        scores = score_tensors(p, labels, threshold)
        for key, tensor in scores.items():
            totals[key].extend(tensor.detach().cpu().tolist())
    n = len(loader.dataset)
    result = {'loss': loss_total/n, 'n_images': n,
              'threshold': threshold, 'tta_horizontal_flip': bool(tta)}
    for key in ('dice', 'iou', 'precision', 'recall'):
        result[key] = float(np.mean(totals[key]))
        result[key+'_std'] = float(np.std(totals[key], ddof=1)) if n > 1 else 0.0
    tp, fp, fn = [sum(totals[key]) for key in ('tp', 'fp', 'fn')]
    result['pooled_dice'] = float(2*tp/max(1,2*tp+fp+fn))
    result['pooled_iou'] = float(tp/max(1,tp+fp+fn))
    result['empty_predictions'] = int(sum(area == 0 for area in totals['pred_pixels']))
    result['n_below_0_5_dice'] = sum(x < 0.5 for x in totals['dice'])
    return result, totals


@torch.no_grad()
def tune_threshold(model, val_loader, device, tta=False):
    """Choose threshold solely on validation; test set is never examined here."""
    model.eval()
    score_sums = {float(t): 0.0 for t in THRESHOLD_GRID}
    n = 0
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        probs = predict_prob(model, images, tta=tta)
        for t in score_sums:
            score_sums[t] += score_tensors(probs, labels, t)['dice'].sum().item()
        n += images.shape[0]
    scores = {f'{t:.2f}': round(v/n, 6) for t, v in score_sums.items()}
    best = max(score_sums, key=lambda t: (score_sums[t], -abs(t - 0.5)))
    return best, scores


def train_one_model(model, train_loader, val_loader, device, out):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=SCHEDULER_PATIENCE, min_lr=1e-6)
    use_amp = (USE_AMP and device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    best_dice, best_epoch, no_improve = -1, 0, 0
    hist = []
    ckpt_path = out / 'best_model.pt'
    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        model.train()
        epoch_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            ctx = torch.autocast(device_type='cuda', dtype=torch.float16) if use_amp else nullcontext()
            with ctx:
                logits = model(images)
                loss = segmentation_loss(logits, masks)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item() * images.size(0)
        train_loss = epoch_loss / len(train_loader.dataset)
        # Epoch checkpoint selection uses deterministic single-pass val at 0.50.
        val, _ = evaluate(model, val_loader, device, threshold=DEFAULT_THRESHOLD, tta=False)
        scheduler.step(val['loss'])
        curr_lr = optimizer.param_groups[0]['lr']
        hist.append(dict(epoch=epoch, train_loss=train_loss,
                         val_loss=val['loss'], val_dice=val['dice'],
                         val_iou=val['iou'], lr=curr_lr,
                         seconds=round(time.time()-start, 2)))
        print(f'[{MODE}] {epoch:03d}/{EPOCHS}: train={train_loss:.4f}, '
              f'val={val["loss"]:.4f}, Dice={val["dice"]:.4f}, '
              f'lr={curr_lr:.2g}, {hist[-1]["seconds"]:.1f}s', flush=True)
        if val['dice'] > best_dice + 1e-7:
            best_dice, best_epoch, no_improve = val['dice'], epoch, 0
            torch.save(dict(state=model.state_dict(), mode=MODE, epoch=epoch,
                            best_val_dice=best_dice, image_size=IMAGE_SIZE,
                            base=BASE_FEATURES, dropout=DROPOUT), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PATIENCE:
                print(f'Early stopping at epoch {epoch}; best was {best_epoch}.')
                break
    with open(out / 'training_history.csv', 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(hist[0]))
        wr.writeheader(); wr.writerows(hist)
    payload = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(payload['state'])
    return model, hist, best_epoch, best_dice


def hd95_assd(pred, gt):
    """Optional 2D boundary distances in RESIZED IMAGE PIXELS, never in mm."""
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except ImportError:
        return float('nan'), float('nan')
    pred = pred.astype(bool); gt = gt.astype(bool)
    if not pred.any() or not gt.any():
        return float('nan'), float('nan')
    ps = pred ^ binary_erosion(pred)
    gs = gt ^ binary_erosion(gt)
    p2g = distance_transform_edt(~gs)[ps]
    g2p = distance_transform_edt(~ps)[gs]
    samples = np.concatenate((p2g, g2p))
    return float(np.percentile(samples, 95)), float(samples.mean())


@torch.no_grad()
def save_per_image_results(model, pairs, loader, device, threshold, out, tta):
    model.eval()
    rows = []
    img_no = 0
    for images, labels in loader:
        images = images.to(device)
        prob = predict_prob(model, images, tta=tta).cpu()
        scores = score_tensors(prob, labels, threshold)
        for j in range(len(images)):
            ip = pairs[img_no][0]
            pred = (prob[j, 0].numpy() >= threshold)
            mask = (labels[j, 0].numpy() >= 0.5)
            hd95, assd = hd95_assd(pred, mask)
            rows.append({
                'image': Path(ip).name,
                'dice': scores['dice'][j].item(),
                'iou': scores['iou'][j].item(),
                'precision': scores['precision'][j].item(),
                'recall': scores['recall'][j].item(),
                'gt_pixels': int(scores['gt_pixels'][j].item()),
                'pred_pixels': int(scores['pred_pixels'][j].item()),
                'HD95_resized_pixels': hd95,
                'ASSD_resized_pixels': assd,
            })
            img_no += 1
    assert img_no == len(pairs)
    with open(out / 'per_image_test_metrics.csv', 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    return rows


def save_plots(model, hist, test_pairs, device, threshold, out, mode, worst_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    epochs = [r['epoch'] for r in hist]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs[0].plot(epochs, [r['train_loss'] for r in hist], label='Train')
    axs[0].plot(epochs, [r['val_loss'] for r in hist], label='Validation')
    axs[0].set(xlabel='Epoch', ylabel='BCE + Dice + Tversky', title='Training dynamics')
    axs[0].legend()
    axs[1].plot(epochs, [r['val_dice'] for r in hist], label='Validation Dice')
    axs[1].set(xlabel='Epoch', ylabel='Mean per-image Dice', title='Checkpoint selection')
    axs[1].legend()
    fig.tight_layout(); fig.savefig(out / 'training_curves.png', dpi=170); plt.close(fig)

    # Inspect worst test results without altering/training on their labels.
    n = min(SAVE_WORST_N, len(worst_rows))
    if n > 0:
        from matplotlib.colors import ListedColormap
        by_name = {Path(ip).name: (ip, mp) for ip, mp in test_pairs}
        fig, axes = plt.subplots(n, 4, figsize=(14, max(3, 3*n)), squeeze=False)
        model.eval()
        for row_index, row in enumerate(worst_rows[:n]):
            ip, mp = by_name[row['image']]
            ds = OpticDiscDataset([(ip, mp)], IMAGE_SIZE, augment=False)
            x, y = ds[0]
            with torch.no_grad():
                prob = predict_prob(model, x[None].to(device), tta=TTA_HORIZONTAL_FLIP)[0, 0].cpu().numpy()
            raw = (x * torch.tensor(STD).view(3,1,1) + torch.tensor(MEAN).view(3,1,1)).clamp(0,1)
            raw = raw.permute(1,2,0).numpy()
            gt = y[0].numpy()
            pred = (prob >= threshold).astype(np.float32)
            for col, (title, data, cmap) in enumerate([
                ('Input', raw, None), ('Ground truth', gt, 'gray'),
                ('Probability', prob, 'viridis'), ('Prediction', pred, 'gray')]):
                axes[row_index, col].imshow(data, cmap=cmap, vmin=0 if cmap else None,
                                            vmax=1 if cmap else None)
                axes[row_index, col].set_title(title + (f' | Dice {row["dice"]:.3f}' if col==0 else ''), fontsize=9)
                axes[row_index, col].axis('off')
        fig.suptitle('Worst test examples — DIAGNOSTIC ONLY (not for model selection)')
        fig.tight_layout(); fig.savefig(out / 'worst_test_examples.png', dpi=150); plt.close(fig)

    kernels = []
    if mode in ('radial', 'combined'):
        for label, block in (('Encoder 3', model.enc3), ('Bottleneck', model.bottleneck_conv)):
            for branch in block.branches:
                kernels.append((f'{label}: {branch.kernel_size}×{branch.kernel_size}',
                                branch.effective_weight()[0, 0].detach().float().cpu().numpy()))
    if mode in ('circular', 'combined'):
        z = model.bottleneck_attn.circular_spatial.effective_weight().detach().float().cpu().numpy()
        kernels.extend([('Circular attention: mean',z[0,0]), ('Circular attention: max',z[0,1])])
    if kernels:
        fig, axes = plt.subplots(1,len(kernels),figsize=(3*len(kernels),3.3),squeeze=False)
        for ax,(title,w) in zip(axes[0],kernels):
            limit=max(float(abs(w).max()),1e-6)
            im=ax.imshow(w,cmap='coolwarm',vmin=-limit,vmax=limit)
            ax.set_title(title,fontsize=9)
            fig.colorbar(im,ax=ax,shrink=.75)
        fig.tight_layout();fig.savefig(out/'learned_geometric_kernels.png',dpi=160);plt.close(fig)


def main():
    if DATA_DIR.startswith('/content/drive/') and not Path('/content/drive/MyDrive').exists():
        from google.colab import drive
        drive.mount('/content/drive')
    if MODE not in ('radial','circular','combined'):
        raise ValueError('Invalid MODE.')
    if IMAGE_SIZE < 64 or IMAGE_SIZE % 8:
        raise ValueError('IMAGE_SIZE must be >=64 and divisible by eight.')
    seed_everything(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pairs = find_pairs(DATA_DIR)
    patient_map, group_source = get_patient_mapping(pairs, PATIENT_CSV)
    if EXISTING_SPLIT_CSV:
        splits = load_verified_manifest(EXISTING_SPLIT_CSV, pairs, patient_map)
    else:
        splits = make_grouped_splits(pairs, patient_map, SEED, VAL_FRAC, TEST_FRAC)
    ids_per_split = ensure_no_subject_leak(splits, patient_map)
    out = Path(OUTPUT_DIR).expanduser()/MODE/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    settings = {k: v for k,v in globals().copy().items() if k in {
        'DATA_DIR','OUTPUT_DIR','PATIENT_CSV','GROUP_FROM_FILENAMES','EXISTING_SPLIT_CSV',
        'MODE','IMAGE_SIZE','BASE_FEATURES','BATCH_SIZE','EPOCHS','LEARNING_RATE',
        'WEIGHT_DECAY','DROPOUT','EARLY_STOP_PATIENCE','SCHEDULER_PATIENCE',
        'VAL_FRAC','TEST_FRAC','NUM_WORKERS','SEED','AUGMENT_TRAIN',
        'ALLOW_HORIZONTAL_FLIP','TTA_HORIZONTAL_FLIP','TUNE_THRESHOLD_ON_VAL',
        'THRESHOLD_GRID','DEFAULT_THRESHOLD','USE_AMP','GRAD_CLIP_NORM'}}
    settings.update(device=str(device),torch_version=torch.__version__,
                    group_source=group_source, patient_groups=ids_per_split)
    with open(out/'config.json','w') as fh: json.dump(settings,fh,indent=2)
    save_manifest(out/'split_manifest.csv',splits,patient_map)
    print(f'Found {len(pairs)} image/mask pairs. Subject IDs from {group_source}.')
    print('Split image counts (train/val/test):', *(len(s) for s in splits))
    print('Split subject counts:',ids_per_split,'| Overlap: ZERO')
    loaders=[]
    for idx, split in enumerate(splits):
        dataset=OpticDiscDataset(split, IMAGE_SIZE, augment=(idx==0 and AUGMENT_TRAIN))
        loaders.append(DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=(idx==0),
                                  num_workers=NUM_WORKERS,pin_memory=(device.type=='cuda'),
                                  drop_last=False))
    train_loader,val_loader,test_loader=loaders
    model=GeometryRCUNet(MODE,BASE_FEATURES,DROPOUT).to(device)
    num_params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    before=geometry_audit(model)
    print(f'Mode={MODE} | Device={device} | Parameters={num_params:,} | '
          f'Geometric kernels audited={len(before)}',flush=True)
    start=time.time()
    model,hist,best_epoch,best_val_dice=train_one_model(model,train_loader,val_loader,device,out)
    if TUNE_THRESHOLD_ON_VAL:
        threshold, threshold_scores=tune_threshold(model,val_loader,device,TTA_HORIZONTAL_FLIP)
    else:
        threshold, threshold_scores=DEFAULT_THRESHOLD, {}
    val_final,_=evaluate(model,val_loader,device,threshold,TTA_HORIZONTAL_FLIP)
    test_result,_=evaluate(model,test_loader,device,threshold,TTA_HORIZONTAL_FLIP)
    rows=save_per_image_results(model,splits[2],test_loader,device,threshold,out,TTA_HORIZONTAL_FLIP)
    finite_hd=[r['HD95_resized_pixels'] for r in rows if np.isfinite(r['HD95_resized_pixels'])]
    finite_assd=[r['ASSD_resized_pixels'] for r in rows if np.isfinite(r['ASSD_resized_pixels'])]
    test_result.update({
        'mode':MODE,'params':num_params,'best_epoch':best_epoch,
        'best_val_dice_at_0_5_no_tta':best_val_dice,
        'val_metrics_selected_threshold':val_final,
        'validation_threshold_scores':threshold_scores,
        'patient_group_source':group_source,
        'patient_group_counts':ids_per_split,
        'geometry_checks':geometry_audit(model),
        'mean_HD95_resized_pixels':float(np.mean(finite_hd)) if finite_hd else None,
        'mean_ASSD_resized_pixels':float(np.mean(finite_assd)) if finite_assd else None,
        'boundary_evaluable_images':len(finite_hd),
        'elapsed_seconds':round(time.time()-start,1),
        'interpretation':'Single seed. No performance improvement/significance claim without matched repeats.'
    })
    with open(out/'test_metrics.json','w') as fh:json.dump(test_result,fh,indent=2)
    rows=sorted(rows,key=lambda r:r['dice'])
    save_plots(model,hist,splits[2],device,threshold,out,MODE,rows)
    print('----- COMPLETED ONE MODEL RUN -----')
    print(f'Best epoch: {best_epoch} | threshold chosen on validation: {threshold:.2f}')
    print(f'TEST: Dice={test_result["dice"]:.4f}, IoU={test_result["iou"]:.4f}, '
          f'precision={test_result["precision"]:.4f}, recall={test_result["recall"]:.4f}, '
          f'failed (<0.5 Dice)={test_result["n_below_0_5_dice"]}')
    print('Outputs:',out)
    return test_result


# The only execution entry point. In Colab, simply RUN THIS CELL.
if __name__ == '__main__':
    main()
