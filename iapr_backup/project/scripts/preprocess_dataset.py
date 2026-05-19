"""
Preprocesses all training images and saves sector masks + labels.json
for use with baseline.py's UNOSectorDataset.

Output layout:
    project/data/preprocessed/
        labels.json
        <image_id>/
            sector_0/masks.npy   # center  (4, H, W) float32 in [0, 1]
            sector_1/masks.npy   # p1
            sector_2/masks.npy   # p2
            sector_3/masks.npy   # p3
            sector_4/masks.npy   # p4

Run:
    cd project/scripts
    python preprocess_dataset.py
"""

import os
import sys
import json
import contextlib
import io
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import TRAIN_FILE, TRAIN_IMAGES_PATH
from preprocessing import preprocess

BATCH_SIZE = 4
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "preprocessed")


def build_labels_json(csv_df: pd.DataFrame) -> dict:
    labels = {}
    for _, row in csv_df.iterrows():
        image_id = str(row["image_id"])

        def parse_cards(val):
            s = str(val)
            return [] if s == "EMPTY" else s.split(";")

        labels[image_id] = {
            "center_card":   str(row["center_card"]),
            "active_player": str(row["active_player"]),
            "p1_cards": parse_cards(row["player_1_cards"]),
            "p2_cards": parse_cards(row["player_2_cards"]),
            "p3_cards": parse_cards(row["player_3_cards"]),
            "p4_cards": parse_cards(row["player_4_cards"]),
        }
    return labels


def preprocess_and_save(output_dir: str = OUTPUT_DIR, batch_size: int = BATCH_SIZE):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_df = pd.read_csv(TRAIN_FILE)
    image_ids = [str(x) for x in csv_df["image_id"].tolist()]
    n = len(image_ids)
    print(f"Preprocessing {n} images -> {output_dir}")

    for start in range(0, n, batch_size):
        batch_ids = image_ids[start:start + batch_size]

        imgs = []
        for img_id in batch_ids:
            path = os.path.join(TRAIN_IMAGES_PATH, img_id + ".jpg")
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            imgs.append(img)
        imgs = np.stack(imgs)  # (B, H, W, 3)

        with contextlib.redirect_stdout(io.StringIO()):
            masks = preprocess(imgs)  # (B, 5, H, W, 4) uint8 [0, 255]

        for b, img_id in enumerate(batch_ids):
            for s in range(5):
                sector_dir = output_dir / img_id / f"sector_{s}"
                sector_dir.mkdir(parents=True, exist_ok=True)
                # (H, W, 4) -> (4, H, W) float32 [0, 1]
                m = masks[b, s].transpose(2, 0, 1).astype(np.float32) / 255.0
                np.save(sector_dir / "masks.npy", m)

        print(f"  [{start + len(batch_ids)}/{n}]", end="\r")

    print(f"\nSaved sector masks.")

    labels = build_labels_json(csv_df)
    with open(output_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)
    print(f"Saved labels.json ({len(labels)} entries).")


if __name__ == "__main__":
    preprocess_and_save()
