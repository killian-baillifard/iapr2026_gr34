"""
Pre-converts all synthesized sectors from raw RGB (1000x2000x3) to
resized RYGB (448x448x4) and saves them in project/synthesized_cache/.

Run once before training:
    python -m project.scripts.cache_synthesized
"""

import os
import numpy as np
import cv2
import torch

from project_crop_bad.scripts.dataset import PARENT_PATH
from project_crop_bad.scripts.preprocessing.rygb import hsv2rygb

SYNTHESIZED_DIR  = os.path.join(PARENT_PATH, "synthesized")
CACHE_DIR        = os.path.join(PARENT_PATH, "synthesized_cache")
LABELS_PATH      = os.path.join(SYNTHESIZED_DIR,  "labels.npy")
TARGET_SIZE      = (448, 448)

os.makedirs(CACHE_DIR, exist_ok=True)

labels = np.load(LABELS_PATH)
n = len(labels)
print(f"Caching {n} sectors to {CACHE_DIR} ...")

for i in range(n):
    out_path = os.path.join(CACHE_DIR, f"{i}.npy")
    if os.path.exists(out_path):
        continue

    img = np.load(os.path.join(SYNTHESIZED_DIR, f"{i}.npy"))   # (H, W, 3) RGB uint8
    hsv  = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    rygb = hsv2rygb(hsv)                                        # (H, W, 4) uint8

    # Resize to target size on CPU via torch
    t = torch.from_numpy(rygb).float().permute(2, 0, 1).unsqueeze(0)  # (1,4,H,W)
    t = torch.nn.functional.interpolate(t, size=TARGET_SIZE, mode="bilinear", align_corners=False)
    small = t.squeeze(0).permute(1, 2, 0).byte().numpy()               # (448,448,4) uint8

    np.save(out_path, small)

    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{n}")

# Copy labels alongside cache
np.save(os.path.join(CACHE_DIR, "labels.npy"), labels)
print("Done.")
