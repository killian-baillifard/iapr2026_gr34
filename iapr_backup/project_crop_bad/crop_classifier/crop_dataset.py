"""
SyntheticCropDataset: loads single-card RYGB sectors from the synthesizer.
Each sample = one isolated card → perfect 54-class label, no ambiguity.
"""

import os
import random
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from project_crop_bad.scripts.dataset import PARENT_PATH

CACHE_DIR       = os.path.join(PARENT_PATH, "synthesized_cache")
SYNTHESIZED_DIR = os.path.join(PARENT_PATH, "synthesized")
CROP_SIZE       = (224, 224)


class SyntheticCropDataset(Dataset):

    def __init__(self, crop_size=CROP_SIZE, augment=True):
        use_cache   = os.path.isdir(CACHE_DIR)
        labels_path = os.path.join(
            CACHE_DIR if use_cache else SYNTHESIZED_DIR, "labels.npy"
        )
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                "Synthesized data not found. Run: python -m project.scripts.dataset.synthesizer"
            )
        labels_all       = np.load(labels_path)        # (N, 54)
        self.use_cache   = use_cache
        self.labels      = labels_all
        self.crop_size   = crop_size
        self.augment     = augment
        # Only single-card sectors — ground truth is unambiguous
        self.valid_idx   = [i for i in range(len(labels_all)) if labels_all[i].sum() == 1]
        print(f"SyntheticCropDataset: {len(self.valid_idx)} single-card samples")

    def __len__(self):
        return len(self.valid_idx)

    def __getitem__(self, idx):
        i = self.valid_idx[idx]

        if self.use_cache:
            img = np.load(os.path.join(CACHE_DIR, f"{i}.npy"))   # (H, W, 4) uint8
        else:
            raw = np.load(os.path.join(SYNTHESIZED_DIR, f"{i}.npy"))
            import cv2 as _cv2
            from project_crop_bad.scripts.preprocessing.rygb import hsv2rygb
            img = hsv2rygb(_cv2.cvtColor(raw, _cv2.COLOR_RGB2HSV))

        img = cv2.resize(img, self.crop_size)                     # (H, W, 4) uint8
        x   = torch.from_numpy(img).float() / 255.0              # (H, W, 4)

        if self.augment:
            x = _augment(x)

        y = torch.tensor(int(np.argmax(self.labels[i])), dtype=torch.long)
        return x, y


def _augment(x: torch.Tensor) -> torch.Tensor:
    """x: (H, W, 4) float in [0, 1]"""
    x = x.permute(2, 0, 1)   # (4, H, W)
    if random.random() > 0.5:
        x = TF.hflip(x)
    angle = random.uniform(-15, 15)
    x = TF.rotate(x, angle)
    x = (x + torch.randn_like(x) * 0.02).clamp(0, 1)
    # Grayscale: force shape learning
    if random.random() > 0.8:
        lum = x.mean(dim=0, keepdim=True).expand_as(x).clone()
        x   = lum
    return x.permute(1, 2, 0)  # (H, W, 4)
