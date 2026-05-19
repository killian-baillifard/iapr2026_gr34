"""
Evaluate the CropClassifier pipeline on the labeled training set.

Pipeline per image:
  center  → resize center sector to CROP_SIZE → classify (54-class argmax)
  players → detect contours in RYGB → classify each crop → keep unique cards
            that match the detected color channel
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from project_crop_bad.scripts.dataset import (
    Label, Card, TRAIN_FILE,
    PREPROCESSED_CACHE_PATH,
)
from project_crop_bad.crop_classifier.model     import CropClassifier, CROP_SIZE
from project_crop_bad.crop_classifier.card_detector import detect_crops

CARDS_LIST = list(Card)
device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _resize_sector(sector_np: np.ndarray) -> torch.Tensor:
    """(H, W, 4) uint8 → (1, crop_H, crop_W, 4) float on device."""
    x = torch.from_numpy(sector_np).float().unsqueeze(0) / 255.0  # (1, H, W, 4)
    x = F.interpolate(
        x.permute(0, 3, 1, 2),
        size=CROP_SIZE, mode='bilinear', align_corners=False,
    ).permute(0, 2, 3, 1).contiguous()
    return x.to(device)


@torch.no_grad()
def predict_image(model: CropClassifier, label: Label):
    rygb = np.load(os.path.join(PREPROCESSED_CACHE_PATH, label.image_id + ".npy"))

    # --- Center ---
    center_t    = _resize_sector(rygb[0])
    pred_center = CARDS_LIST[model(center_t).argmax(1).item()]

    # --- Players ---
    player_preds = []
    for p in range(4):
        crops = detect_crops(rygb[p + 1])
        cards = set()
        for crop_arr, color_name in crops:
            crop_t = torch.from_numpy(crop_arr).unsqueeze(0).to(device)
            logits = model(crop_t)
            probs  = logits.softmax(1)
            conf, card_idx = probs.max(1)
            card = CARDS_LIST[card_idx.item()]
            # only accept if confident enough AND color matches
            if conf.item() > 0.1 and (str(card).startswith(color_name) or str(card) in ('draw_4', 'wild')):
                cards.add(str(card))
        player_preds.append(list(cards))

    return pred_center, player_preds


def evaluate(model: CropClassifier, labels: list[Label]):
    model.eval()
    center_correct = 0
    all_f1 = []

    for label in labels:
        pred_c, pred_players = predict_image(model, label)

        if str(pred_c) == str(label.center_card):
            center_correct += 1

        tp = fp = fn = 0
        for p in range(4):
            preds = set(pred_players[p])
            gt    = {c.value for c in label.players_cards[p]}
            tp += len(preds & gt)
            fp += len(preds - gt)
            fn += len(gt  - preds)

        pr = tp / (tp + fp + 1e-8)
        rc = tp / (tp + fn + 1e-8)
        all_f1.append(2 * pr * rc / (pr + rc + 1e-8))

    ca  = center_correct / len(labels)
    mf1 = float(np.mean(all_f1))
    sc  = 0.1 * ca + 0.8 * mf1
    print(f"center_acc={ca:.4f}  mean_F1={mf1:.4f}  score={sc:.4f}")
    return ca, mf1, sc


if __name__ == "__main__":
    model = CropClassifier().to(device)
    model.load_state_dict(torch.load("best_crop_classifier.pth", map_location=device))

    csv    = pd.read_csv(TRAIN_FILE)
    labels = [Label.from_row(row) for _, row in csv.iterrows()]
    print(f"Evaluating on {len(labels)} images …")
    evaluate(model, labels)
