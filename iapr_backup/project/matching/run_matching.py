"""
Evaluate ORB card matcher on the labeled dataset, tune the min_score threshold,
and optionally generate a test submission CSV.

Usage:
    python -m project.matching.run_matching              # evaluate + tune
    python -m project.matching.run_matching --test       # also generate test CSV
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kwargs): return it

from project.scripts.dataset import (
    Label, Card, TRAIN_FILE,
    PREPROCESSED_RGB_CACHE_PATH,
    PREPROCESSED_CACHE_PATH,
    TEST_IMAGES_PATH,
)
from project.matching.card_matcher import CardMatcher


# ------------------------------------------------------------------
# Data helpers
# ------------------------------------------------------------------

def _load_rgb(label, sector_idx):
    path = os.path.join(PREPROCESSED_RGB_CACHE_PATH, label.image_id + ".npy")
    return np.load(path)[sector_idx]          # (H, W, 3) uint8


def _load_rygb(label, sector_idx):
    path = os.path.join(PREPROCESSED_CACHE_PATH, label.image_id + ".npy")
    return np.load(path)[sector_idx]          # (H, W, 4) uint8


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------

def evaluate(matcher, labels, min_score=5, silent=False):
    center_correct = 0
    all_f1         = []

    for label in tqdm(labels, desc=f"eval min_score={min_score}", disable=silent):
        # Center
        pred_c = matcher.predict_center(_load_rgb(label, 0))
        if pred_c == label.center_card.value:
            center_correct += 1

        # Players
        tp = fp = fn = 0
        for p in range(4):
            preds = set(matcher.predict_player(
                _load_rgb(label, p + 1),
                _load_rygb(label, p + 1),
                min_score=min_score,
            ))
            gt = set(c.value for c in label.players_cards[p])
            tp += len(preds & gt)
            fp += len(preds - gt)
            fn += len(gt  - preds)

        pr  = tp / (tp + fp + 1e-8)
        rc  = tp / (tp + fn + 1e-8)
        all_f1.append(2 * pr * rc / (pr + rc + 1e-8))

    ca  = center_correct / len(labels)
    mf1 = float(np.mean(all_f1))
    sc  = 0.1 * ca + 0.8 * mf1
    if not silent:
        print(f"  center_acc={ca:.4f}  mean_F1={mf1:.4f}  score={sc:.4f}")
    return ca, mf1, sc


def tune_threshold(matcher, labels, lo=1, hi=30):
    print("Sweeping min_score threshold on labeled set …")
    best_sc, best_t = 0.0, 5
    for t in range(lo, hi + 1, 2):
        _, _, sc = evaluate(matcher, labels, min_score=t, silent=True)
        print(f"  min_score={t:2d}  score={sc:.4f}")
        if sc > best_sc:
            best_sc, best_t = sc, t
    print(f"\nBest min_score={best_t}  (score={best_sc:.4f})")
    return best_t


# ------------------------------------------------------------------
# Test prediction
# ------------------------------------------------------------------

def predict_test(matcher, min_score=5, out_csv="submission_matching.csv"):
    """Run matcher on all test images and write a submission CSV."""
    from project.scripts.preprocessing import slice_sectors
    from project.scripts.preprocessing.rygb import hsv2rygb
    import cv2

    test_files = sorted(f for f in os.listdir(TEST_IMAGES_PATH) if f.lower().endswith(".jpg"))
    rows = []

    for fname in tqdm(test_files, desc="Test prediction"):
        image_id = os.path.splitext(fname)[0]
        img_bgr  = cv2.imread(os.path.join(TEST_IMAGES_PATH, fname))
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        sectors_rgb = slice_sectors(img_rgb[np.newaxis])[0]   # (5, H, W, 3)

        # Build RYGB for color filtering
        sectors_rygb = []
        for s in range(5):
            hsv  = cv2.cvtColor(sectors_rgb[s], cv2.COLOR_RGB2HSV)
            sectors_rygb.append(hsv2rygb(hsv))

        center_card = matcher.predict_center(sectors_rgb[0])

        player_preds = []
        for p in range(4):
            cards = matcher.predict_player(
                sectors_rgb[p + 1],
                sectors_rygb[p + 1],
                min_score=min_score,
            )
            player_preds.append(cards)

        row = {"image_id": image_id, "center_card": center_card}
        for p, cards in enumerate(player_preds):
            row[f"player_{p+1}_cards"] = ";".join(cards) if cards else "EMPTY"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved {len(rows)} predictions → {out_csv}")
    return df


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Also generate test submission CSV")
    parser.add_argument("--min_score", type=int, default=None, help="Skip tuning and use fixed threshold")
    args = parser.parse_args()

    csv    = pd.read_csv(TRAIN_FILE)
    labels = [Label.from_row(row) for _, row in csv.iterrows()]
    print(f"Loaded {len(labels)} labeled images.")

    print("Building CardMatcher …")
    matcher = CardMatcher()
    print(f"  {len(matcher.templates)} templates loaded.\n")

    if args.min_score is not None:
        best_t = args.min_score
    else:
        best_t = tune_threshold(matcher, labels)

    print(f"\nFinal evaluation (min_score={best_t}):")
    evaluate(matcher, labels, min_score=best_t)

    if args.test:
        print("\nGenerating test predictions …")
        predict_test(matcher, min_score=best_t)
