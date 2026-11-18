from __future__ import annotations

import csv
import random
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


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
    name = Path(image_path).stem
    m = re.match(r'^pacient\s*0*(\d+)\s*_s[eé]rie\b', name, re.IGNORECASE)
    if m:
        return 'pacient_cohort_' + m.group(1).zfill(3)
    m = re.match(r'^0*(\d+)\s*_s[eé]rie\b', name, re.IGNORECASE)
    if m:
        return 'numeric_cohort_' + m.group(1).zfill(3)
    raise ValueError(f'Unknown filename pattern; cannot infer subject for {name!r}. Provide PATIENT_CSV.')


def get_patient_mapping(pairs, patient_csv=None, group_from_filenames=True):
    if patient_csv is not None:
        with open(patient_csv, newline='', encoding='utf-8-sig') as fh:
            rd = csv.DictReader(fh)
            if not {'image_name', 'patient_id'}.issubset(rd.fieldnames or []):
                raise ValueError('PATIENT_CSV must contain columns image_name,patient_id.')
            mapping = {}
            for row in rd:
                key = Path(row['image_name']).stem.lower(); pid = row['patient_id'].strip()
                if not pid: raise ValueError(f'Empty patient ID for {key}')
                if key in mapping and mapping[key] != pid: raise ValueError(f'Conflicting patient ID for {key}')
                mapping[key] = pid
        pids = {}
        for ip, _ in pairs:
            key = Path(ip).stem.lower()
            if key not in mapping: raise ValueError(f'Missing patient ID for {key}')
            pids[key] = mapping[key]
        source = 'verified_mapping_CSV'
    elif group_from_filenames:
        pids = {Path(ip).stem.lower(): filename_group(ip) for ip, _ in pairs}
        source = 'filename_inferred_UNVERIFIED'
        warnings.warn('Subject IDs inferred from filename prefixes. Verify before publication.', stacklevel=2)
    else:
        raise ValueError('Patient grouping is mandatory.')
    return pids, source


def make_grouped_splits(pairs, patient_map, seed, val_frac, test_frac):
    if not (0 < val_frac < 1 and 0 < test_frac < 1 and val_frac + test_frac < 1):
        raise ValueError('Invalid val/test fractions.')
    groups = defaultdict(list)
    for pair in pairs: groups[patient_map[Path(pair[0]).stem.lower()]].append(pair)
    if len(groups) < 3: raise ValueError('At least three distinct subjects are required.')
    items = sorted(groups.items()); rng = random.Random(seed); rng.shuffle(items)
    targets = {'test': len(pairs) * test_frac, 'val': len(pairs) * val_frac}
    allocated = {'train': [], 'val': [], 'test': []}; counts = {'train': 0, 'val': 0, 'test': 0}
    for destination, (_, sample_list) in zip(('test', 'val', 'train'), items[:3]):
        allocated[destination].extend(sample_list); counts[destination] += len(sample_list)
    for _, sample_list in items[3:]:
        dst = 'test' if counts['test'] < targets['test'] else ('val' if counts['val'] < targets['val'] else 'train')
        allocated[dst].extend(sample_list); counts[dst] += len(sample_list)
    splits = tuple(allocated[d] for d in ('train', 'val', 'test'))
    assert all(splits)
    return splits


def load_verified_manifest(csv_path, pairs, patient_map):
    expected = {str(Path(ip).resolve()): (ip, mp) for ip, mp in pairs}
    assignment = {'train': [], 'val': [], 'test': []}; seen = set()
    with open(csv_path, newline='', encoding='utf-8-sig') as fh:
        for row in csv.DictReader(fh):
            split = row['split'].strip().lower()
            if split not in assignment: raise ValueError(f'Unknown split name {split}')
            ip = str(Path(row['image_path']).resolve())
            if ip not in expected or ip in seen: raise ValueError('Manifest has absent/duplicated image: ' + ip)
            pid = row.get('patient_id', '').strip()
            if pid and pid != patient_map[Path(ip).stem.lower()]: raise ValueError('Manifest patient_id inconsistent for ' + ip)
            assignment[split].append(expected[ip]); seen.add(ip)
    if seen != set(expected) or not all(assignment.values()): raise ValueError('Manifest must cover every image/all splits.')
    return tuple(assignment[s] for s in ('train', 'val', 'test'))


def ensure_no_subject_leak(splits, patient_map):
    id_sets = [{patient_map[Path(ip).stem.lower()] for ip, _ in split} for split in splits]
    if any(id_sets[i] & id_sets[j] for i, j in ((0,1),(0,2),(1,2))):
        raise RuntimeError('Detected subject overlap between splits. STOPPING.')
    return dict(zip(('train','val','test'), [len(s) for s in id_sets]))


def save_manifest(path, splits, patient_map):
    with open(path, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.writer(fh); wr.writerow(('split','image_path','mask_path','patient_id'))
        for name, split in zip(('train','val','test'), splits):
            for ip, mp in split: wr.writerow((name, ip, mp, patient_map[Path(ip).stem.lower()]))


def affine_pair(img, mask, allow_horizontal_flip=True):
    width, height = img.size
    if allow_horizontal_flip and random.random() < 0.5: img, mask = ImageOps.mirror(img), ImageOps.mirror(mask)
    angle = random.uniform(-12, 12)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0,0,0)); mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
    scale = random.uniform(0.94, 1.06); dx = random.uniform(-0.03,0.03)*width; dy = random.uniform(-0.03,0.03)*height
    cx, cy = (width-1)/2, (height-1)/2
    inv = (1/scale,0,cx-(cx+dx)/scale,0,1/scale,cy-(cy+dy)/scale)
    img = img.transform((width,height), Image.Transform.AFFINE, inv, resample=Image.Resampling.BILINEAR, fillcolor=(0,0,0))
    mask = mask.transform((width,height), Image.Transform.AFFINE, inv, resample=Image.Resampling.NEAREST, fillcolor=0)
    img = ImageEnhance.Brightness(img).enhance(random.uniform(.90,1.10)); img = ImageEnhance.Contrast(img).enhance(random.uniform(.90,1.10)); img = ImageEnhance.Color(img).enhance(random.uniform(.92,1.08))
    return img, mask


class OpticDiscDataset(Dataset):
    def __init__(self, pairs, size, augment=False, allow_horizontal_flip=True):
        self.pairs=list(pairs); self.size=size; self.augment=augment; self.allow_horizontal_flip=allow_horizontal_flip
        self.mean=torch.tensor(MEAN).view(3,1,1); self.std=torch.tensor(STD).view(3,1,1)
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        with Image.open(ip) as source: img=source.convert('RGB').resize((self.size,self.size), Image.Resampling.BILINEAR)
        with Image.open(mp) as source: mask=source.convert('L').resize((self.size,self.size), Image.Resampling.NEAREST)
        if self.augment: img, mask = affine_pair(img, mask, self.allow_horizontal_flip)
        image=torch.from_numpy(np.asarray(img,dtype=np.float32).transpose(2,0,1).copy())/255.0; image=(image-self.mean)/self.std
        binary_mask=torch.from_numpy((np.asarray(mask,dtype=np.uint8)>127).astype(np.float32).copy())[None]
        return image,binary_mask
