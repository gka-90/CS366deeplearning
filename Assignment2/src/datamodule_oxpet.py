"""LightningDataModule for Oxford-IIIT Pet trimap segmentation (3 classes).

Trimap PNG values -> training labels:
    1 (pet)        -> 0
    2 (background) -> 1
    3 (border)     -> 2

Images are resized with BILINEAR interpolation, masks always with NEAREST so
no new label values are invented at object edges.
"""
from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

CLASS_NAMES = ("pet", "background", "border")
NUM_CLASSES = len(CLASS_NAMES)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def trimap_to_labels(mask: np.ndarray) -> np.ndarray:
    """Map trimap values {1,2,3} to class ids {0,1,2}."""
    labels = mask.astype(np.int64) - 1
    if labels.min() < 0 or labels.max() >= NUM_CLASSES:
        raise ValueError(f"Unexpected trimap values: {np.unique(mask)}")
    return labels

# Paired transforms (the same geometric op is applied to image and mask)
class PairTransform:
    """Joint image/mask transform.

    aug="none":   resize only (validation / test)
    aug="light":  resize + horizontal flip (baseline)
    aug="strong": random resized crop + flip + small rotation + color jitter
    """

    def __init__(self, img_size: int, aug: str = "none"):
        if aug not in {"none", "light", "strong"}:
            raise ValueError(f"unknown aug '{aug}'")
        self.size = [img_size, img_size]
        self.aug = aug

    def __call__(self, img: Image.Image, mask: Image.Image):
        if self.aug == "strong":
            img, mask = self._random_resized_crop(img, mask)
            if random.random() < 0.5:
                angle = random.uniform(-15, 15)
                img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=2)
        else:
            img = TF.resize(img, self.size, interpolation=InterpolationMode.BILINEAR, antialias=True)
            mask = TF.resize(mask, self.size, interpolation=InterpolationMode.NEAREST)

        if self.aug in {"light", "strong"} and random.random() < 0.5:
            img, mask = TF.horizontal_flip(img), TF.horizontal_flip(mask)

        if self.aug == "strong":
            img = TF.adjust_brightness(img, random.uniform(0.75, 1.25))
            img = TF.adjust_contrast(img, random.uniform(0.75, 1.25))
            img = TF.adjust_saturation(img, random.uniform(0.75, 1.25))
            img = TF.adjust_hue(img, random.uniform(-0.03, 0.03))

        x = TF.to_image(img)
        x = TF.to_dtype(x, torch.float32, scale=True)
        x = TF.normalize(x, IMAGENET_MEAN, IMAGENET_STD)
        y = torch.from_numpy(trimap_to_labels(np.array(mask)))
        return x, y

    def _random_resized_crop(self, img, mask, scale=(0.5, 1.0), ratio=(3 / 4, 4 / 3)):
        w, h = img.size
        area = w * h
        for _ in range(10):
            target = area * random.uniform(*scale)
            r = np.exp(random.uniform(np.log(ratio[0]), np.log(ratio[1])))
            cw, ch = int(round(np.sqrt(target * r))), int(round(np.sqrt(target / r)))
            if 0 < cw <= w and 0 < ch <= h:
                top, left = random.randint(0, h - ch), random.randint(0, w - cw)
                break
        else:
            top, left, ch, cw = 0, 0, h, w
        img = TF.resized_crop(img, top, left, ch, cw, self.size,
                              interpolation=InterpolationMode.BILINEAR, antialias=True)
        mask = TF.resized_crop(mask, top, left, ch, cw, self.size,
                               interpolation=InterpolationMode.NEAREST)
        return img, mask



# Dataset
def read_split(pet_root: Path, split: str) -> list[str]:
    """Return image basenames listed in annotations/{trainval,test}.txt."""
    names = []
    with open(pet_root / "annotations" / f"{split}.txt") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line.split()[0])
    return names


class OxfordPetSeg(Dataset):
    def __init__(self, pet_root: str | Path, names: list[str], transform: PairTransform):
        self.root = Path(pet_root)
        self.names = names
        self.transform = transform

    def __len__(self):
        return len(self.names)

    def load_raw(self, idx):
        name = self.names[idx]
        img = Image.open(self.root / "images" / f"{name}.jpg").convert("RGB")
        mask = Image.open(self.root / "annotations" / "trimaps" / f"{name}.png")
        return img, mask

    def __getitem__(self, idx):
        img, mask = self.load_raw(idx)
        return self.transform(img, mask)

# DataModule
class OxfordPetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        img_size: int = 256,
        batch_size: int = 8,
        val_fraction: float = 0.2,
        aug: str = "light",
        num_workers: int = 4,
        seed: int = 42,
    ):
        super().__init__()
        self.save_hyperparameters()

    @property
    def pet_root(self) -> Path:
        return Path(os.path.expanduser(self.hparams.data_dir)) / "oxford-iiit-pet"

    def prepare_data(self):
        """Download once (same folder layout as the provided exploration script)."""
        if not (self.pet_root / "annotations" / "trimaps").exists():
            from torchvision.datasets import OxfordIIITPet

            root = os.path.expanduser(self.hparams.data_dir)
            OxfordIIITPet(root=root, split="trainval", target_types="segmentation", download=True)

    def setup(self, stage: str | None = None):
        hp = self.hparams
        trainval = read_split(self.pet_root, "trainval")
        # Deterministic 80/20 split of the official trainval list.
        rng = np.random.default_rng(hp.seed)
        order = rng.permutation(len(trainval))
        n_val = int(round(len(trainval) * hp.val_fraction))
        val_names = sorted(trainval[i] for i in order[:n_val])
        train_names = sorted(trainval[i] for i in order[n_val:])

        self.train_ds = OxfordPetSeg(self.pet_root, train_names, PairTransform(hp.img_size, hp.aug))
        self.val_ds = OxfordPetSeg(self.pet_root, val_names, PairTransform(hp.img_size, "none"))
        self.test_ds = OxfordPetSeg(self.pet_root, read_split(self.pet_root, "test"),
                                    PairTransform(hp.img_size, "none"))

    def _loader(self, ds, shuffle):
        nw = self.hparams.num_workers
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            drop_last=shuffle,
            num_workers=nw,
            persistent_workers=nw > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def train_dataloader(self):
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_ds, shuffle=False)
