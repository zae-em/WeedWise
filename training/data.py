"""Folder discovery and reproducible splits with exact-duplicate leakage checks."""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _scan(root, classes):
    actual = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if actual != classes:
        raise ValueError(f"Class folders differ at {root}: expected {classes}, got {actual}")
    rows = []
    for label, name in enumerate(classes):
        files = sorted(p for p in (root / name).rglob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        if not files:
            raise ValueError(f"No supported images in class folder: {root / name}")
        rows.extend({"path": p, "label": label, "sha256": file_hash(p)} for p in files)
    return rows


def discover(data_dir, val_fraction=0.15, test_fraction=0.15, seed=42):
    root = Path(data_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}. Extract the ZIP later and configure data_dir.")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("Validation and test fractions must be positive and sum to less than 1.")
    folders = {p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")}
    explicit = "train" in folders
    val_name = "val" if "val" in folders else "validation"
    if explicit and not {"train", val_name, "test"}.issubset(folders):
        raise ValueError("Pre-split data must contain train/, val/ (or validation/), and test/. Otherwise use one class-folder root.")
    class_root = root / "train" if explicit else root
    classes = sorted(p.name for p in class_root.iterdir() if p.is_dir() and not p.name.startswith("."))
    if len(classes) < 2:
        raise ValueError("At least two species class folders are required. Point data_dir at the actual class-folder root.")
    if explicit:
        splits = {name: _scan(root / sub, classes) for name, sub in [("train", "train"), ("val", val_name), ("test", "test")]}
    else:
        rows = _scan(root, classes)
        splits = {"train": [], "val": [], "test": []}
        rng = random.Random(seed)
        for label in range(len(classes)):
            # Identical images stay together even when filenames differ.
            groups = defaultdict(list)
            for row in rows:
                if row["label"] == label:
                    groups[row["sha256"]].append(row)
            units = list(groups.values())
            if len(units) < 3:
                raise ValueError(f"Class {classes[label]!r} needs >=3 distinct image contents for three splits.")
            rng.shuffle(units)
            n_test = max(1, min(round(len(units) * test_fraction), len(units) - 2))
            n_val = max(1, min(round(len(units) * val_fraction), len(units) - n_test - 1))
            for name, selected in [("test", units[:n_test]), ("val", units[n_test:n_test+n_val]), ("train", units[n_test+n_val:])]:
                splits[name].extend(row for group in selected for row in group)
    seen = {}
    for split, rows in splits.items():
        for row in rows:
            identity = (split, row["label"])
            previous = seen.setdefault(row["sha256"], identity)
            if previous != identity:
                raise ValueError(f"Duplicate image content crosses splits or class labels: {row['path']}. Clean the dataset before training.")
    manifest = {"classes": classes, "splits": {
        split: [{**r, "path": r["path"].relative_to(root).as_posix()} for r in rows]
        for split, rows in splits.items()
    }}
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return classes, splits, manifest, fingerprint


def image_transforms(size, training=False):
    if training:
        ops = [Letterbox(size), transforms.RandomResizedCrop(size, scale=(0.70, 1.0)), transforms.RandomHorizontalFlip(),
               transforms.RandomRotation(20), transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)]
    else:
        ops = [Letterbox(size)]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)])


class Letterbox:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return ImageOps.pad(image.convert("RGB"), (self.size,self.size), method=Image.Resampling.LANCZOS, color=(127,127,127))


class FolderImages(Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            value = self.transform(image.convert("RGB"))
        return value, row["label"]
