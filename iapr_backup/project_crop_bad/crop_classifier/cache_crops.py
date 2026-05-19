"""
Pre-generate perspective-augmented crops and save to disk.
Run this ONCE before training (takes ~15-25 min).

Creates: project/crop_classifier/crop_cache/crops.npy   (N, 224, 224, 4) uint8
         project/crop_classifier/crop_cache/labels.npy  (N,) int64

Usage:
    python -m project.crop_classifier.cache_crops
"""

import os
import numpy as np
from torch.utils.data import DataLoader
from project_crop_bad.crop_classifier.card_crop_synthesizer import PerspectiveCropDataset, CARDS_LIST

CACHE_DIR   = os.path.join(os.path.dirname(__file__), "crop_cache")
N_PER_CLASS = 400   # 400 × 54 = 21 600 crops


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    total = N_PER_CLASS * len(CARDS_LIST)
    print(f"Generating {total} crops ({N_PER_CLASS} per class, {len(CARDS_LIST)} classes) …")

    # Generate WITHOUT augmentation — augmentation is applied fresh each training epoch
    ds     = PerspectiveCropDataset(n_per_class=N_PER_CLASS, augment=False)
    loader = DataLoader(ds, batch_size=256, shuffle=False,
                        num_workers=8, pin_memory=False)

    crops_list  = []
    labels_list = []
    seen = 0

    for x, y in loader:
        crops_list.append((x.numpy() * 255).astype(np.uint8))
        labels_list.append(y.numpy())
        seen += len(y)
        print(f"  {seen}/{total}", end="\r", flush=True)

    crops  = np.concatenate(crops_list,  axis=0)   # (N, 224, 224, 4) uint8
    labels = np.concatenate(labels_list, axis=0)   # (N,) int64

    crops_path  = os.path.join(CACHE_DIR, "crops.npy")
    labels_path = os.path.join(CACHE_DIR, "labels.npy")

    np.save(crops_path,  crops)
    np.save(labels_path, labels)

    gb = crops.nbytes / 1e9
    print(f"\nSaved {len(crops)} crops → {CACHE_DIR}")
    print(f"  crops.npy  : {gb:.2f} GB")
    print(f"  labels.npy : {labels.nbytes / 1e6:.1f} MB")
    print(f"\nClass distribution: {np.bincount(labels).min()}-{np.bincount(labels).max()} samples per class")


if __name__ == "__main__":
    main()
