"""
Generate a submission CSV from test images.

Usage:
    # Sector-level EfficientNet (color-edges model, 640px, 4 channels)
    python -m project.predict_test --model sector --weights best_model_color_edges_640.pth

    # Sector-level EfficientNet with Sobel (5 channels)
    python -m project.predict_test --model sector --weights best_model_448_sobel.pth --in_channels 5 --res 448

    # Crop classifier
    python -m project.predict_test --model crop --weights best_crop_classifier.pth

    # Custom output path
    python -m project.predict_test --model sector --weights best_model_color_edges_640.pth --out my_submission.csv
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import cv2

from project.scripts.dataset import TEST_IMAGES_PATH, Card
from project.scripts.dataset import add_sobel_channel, rygb_to_color_edges
from project.scripts.preprocessing.sectors import slice_sectors
from project.scripts.preprocessing.rygb import hsv2rygb
from project.scripts.preprocessing.filter import area_bandpass_filter
from project.scripts.token import detect_active_player

CARDS_LIST = list(Card)
CROP_SIZE  = (224, 224)   # must match CropClassifier training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_rygb_sectors(img_rgb: np.ndarray) -> np.ndarray:
    """img_rgb: (H, W, 3) uint8 → (5, H, W, 4) uint8 RYGB sectors"""
    sectors_rgb = slice_sectors(img_rgb)   # (5, H, W, 3)
    sectors_rygb = np.stack([
        area_bandpass_filter(hsv2rygb(cv2.cvtColor(s, cv2.COLOR_RGB2HSV)))
        for s in sectors_rgb
    ])
    return sectors_rygb                    # (5, H, W, 4) uint8


# ------------------------------------------------------------------
# Model loaders
# ------------------------------------------------------------------

def load_crop_model(weights_path):
    from project.crop_classifier.model import CropClassifier
    model = CropClassifier().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


def load_sector_model(weights_path, in_channels=4):
    from project.models.efficientnet import UnoEfficientNet
    model = UnoEfficientNet(in_channels=in_channels, variant="b0").to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model


# ------------------------------------------------------------------
# Prediction helpers
# ------------------------------------------------------------------

@torch.no_grad()
def predict_crop_model(model, rygb_sectors):
    """rygb_sectors: (5, H, W, 4) uint8"""
    from project.crop_classifier.card_detector import detect_crops

    center_t    = _resize_rygb(rygb_sectors[0], CROP_SIZE)
    pred_center = CARDS_LIST[model(center_t).argmax(1).item()].value

    player_preds = []
    for p in range(4):
        crops = detect_crops(rygb_sectors[p + 1])
        cards = set()
        for crop_arr, color_name in crops:
            crop_t = torch.from_numpy(crop_arr).unsqueeze(0).to(device)
            logits = model(crop_t)
            probs  = logits.softmax(1)
            conf, card_idx = probs.max(1)
            card = CARDS_LIST[card_idx.item()]
            if conf.item() > 0.1 and (str(card).startswith(color_name) or str(card) in ('draw_4', 'wild')):
                cards.add(str(card))
        player_preds.append(sorted(cards))

    return pred_center, player_preds


@torch.no_grad()
def predict_sector_model(model, rygb_sectors, res, threshold, color_edges, use_sobel):
    """rygb_sectors: (5, H, W, 4) uint8"""
    x = torch.from_numpy(rygb_sectors).float() / 255.0    # (5, H, W, 4)

    # Resize to training resolution first
    x = F.interpolate(
        x.permute(0, 3, 1, 2), size=(res, res), mode='bilinear', align_corners=False
    ).permute(0, 2, 3, 1).contiguous()                     # (5, H, W, 4)

    # Apply the same transform used during training
    if color_edges:
        x = rygb_to_color_edges(x)                         # (5, H, W, 4)
    elif use_sobel:
        x = add_sobel_channel(x)                           # (5, H, W, 5)

    x = x.unsqueeze(0).to(device)                          # (1, 5, H, W, C)

    center_logits, player_logits = model(x)

    pred_center = CARDS_LIST[center_logits.argmax(1).item()].value

    player_probs = player_logits.sigmoid().squeeze(0)       # (4, 54)
    player_preds = []
    for p in range(4):
        cards = [CARDS_LIST[j].value for j in range(54) if player_probs[p, j] > threshold]
        player_preds.append(sorted(cards))

    return pred_center, player_preds


def _resize_rygb(sector_np, size):
    """(H, W, 4) uint8 → (1, H, W, 4) float tensor on device, resized."""
    x = torch.from_numpy(sector_np).float().unsqueeze(0) / 255.0
    x = F.interpolate(
        x.permute(0, 3, 1, 2), size=size, mode='bilinear', align_corners=False
    ).permute(0, 2, 3, 1).contiguous()
    return x.to(device)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",       choices=["crop", "sector"], default="sector")
    parser.add_argument("--weights",     required=True, help="Path to .pth weights file")
    parser.add_argument("--out",         default="submission.csv",   help="Output CSV path")
    parser.add_argument("--threshold",   type=float, default=0.5,    help="Sigmoid threshold (sector model)")
    parser.add_argument("--res",         type=int,   default=640,    help="Resize resolution (sector model)")
    parser.add_argument("--in_channels", type=int,   default=4,      help="Model input channels: 4 or 5")
    parser.add_argument("--color_edges", action="store_true",        help="Apply per-color Sobel transform (for color_edges models)")
    parser.add_argument("--sobel",       action="store_true",        help="Append Sobel channel as 5th channel")
    args = parser.parse_args()

    print(f"Device      : {device}")
    print(f"Model       : {args.model}")
    print(f"Weights     : {args.weights}")
    if args.model == "sector":
        print(f"Resolution  : {args.res}x{args.res}")
        print(f"In channels : {args.in_channels}")
        print(f"Color edges : {args.color_edges}")
        print(f"Sobel       : {args.sobel}")

    if args.model == "crop":
        model = load_crop_model(args.weights)
        predict_fn = lambda s: predict_crop_model(model, s)
    else:
        model = load_sector_model(args.weights, in_channels=args.in_channels)
        predict_fn = lambda s: predict_sector_model(
            model, s,
            res=args.res,
            threshold=args.threshold,
            color_edges=args.color_edges,
            use_sobel=args.sobel,
        )

    test_files = sorted(f for f in os.listdir(TEST_IMAGES_PATH) if f.lower().endswith(".jpg"))
    print(f"\nRunning on {len(test_files)} test images …\n")

    rows = []
    for i, fname in enumerate(test_files):
        image_id = os.path.splitext(fname)[0]
        img_bgr  = cv2.imread(os.path.join(TEST_IMAGES_PATH, fname))
        img_rgb  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        rygb_sectors = get_rygb_sectors(img_rgb)            # (5, H, W, 4) uint8

        pred_center, player_preds = predict_fn(rygb_sectors)

        active = detect_active_player(img_rgb[np.newaxis])[0]
        active_str = str(active) if active is not None else "p1"

        row = {
            "image_id"      : image_id,
            "center_card"   : pred_center,
            "active_player" : active_str,
            "player_1_cards": ";".join(player_preds[0]) if player_preds[0] else "EMPTY",
            "player_2_cards": ";".join(player_preds[1]) if player_preds[1] else "EMPTY",
            "player_3_cards": ";".join(player_preds[2]) if player_preds[2] else "EMPTY",
            "player_4_cards": ";".join(player_preds[3]) if player_preds[3] else "EMPTY",
        }
        rows.append(row)

        if (i + 1) % 20 == 0 or (i + 1) == len(test_files):
            print(f"  [{i+1}/{len(test_files)}] {image_id}  center={pred_center}")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(rows)} predictions → {args.out}")


if __name__ == "__main__":
    main()
