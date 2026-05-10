import os
import numpy as np
import pandas as pd
import cv2

from project.scripts.dataset import TRAIN_FILE, TRAIN_IMAGES_PATH, Label
from project.scripts.preprocessing import preprocess

CACHE_DIR = os.path.join(os.path.dirname(TRAIN_FILE), "preprocessed")
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
        preprocessed = preprocess(image[np.newaxis])[0]  # (5, H, W, 4)
        np.save(out_path, preprocessed)
        print(f"[{i+1}/{len(labels)}] saved {label.image_id} {preprocessed.shape}")

    print(f"\nDone. {len(labels)} samples cached in {CACHE_DIR}")