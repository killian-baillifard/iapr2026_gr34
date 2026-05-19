"""
Cache raw RGB sectors (no color masking) for each training image.

Output layout:
    project/data/preprocessed_rgb/
        <image_id>.npy   # (5, 1000, 2000, 3) uint8

Run:
    python -m project.scripts.cache_rgb_sectors
"""

import os
import numpy as np
import pandas as pd
import cv2

from project.scripts.dataset import TRAIN_FILE, TRAIN_IMAGES_PATH, Label
from project.scripts.preprocessing import slice_sectors

CACHE_DIR = os.path.join(os.path.dirname(TRAIN_FILE), "preprocessed_rgb")
os.makedirs(CACHE_DIR, exist_ok=True)

if __name__ == "__main__":

    csv = pd.read_csv(TRAIN_FILE)
    labels = [Label.from_row(row) for _, row in csv.iterrows()]

    for i, label in enumerate(labels):
        out_path = os.path.join(CACHE_DIR, label.image_id + ".npy")

        if os.path.exists(out_path):
            print(f"[{i+1}/{len(labels)}] {label.image_id} already cached, skipping")
            continue

        image_path = os.path.join(TRAIN_IMAGES_PATH, label.image_id + ".jpg")
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        sectors = slice_sectors(image[np.newaxis])[0]  # (5, H, W, 3) uint8
        np.save(out_path, sectors)
        print(f"[{i+1}/{len(labels)}] saved {label.image_id} {sectors.shape}")

    print(f"\nDone. {len(labels)} samples cached in {CACHE_DIR}")
