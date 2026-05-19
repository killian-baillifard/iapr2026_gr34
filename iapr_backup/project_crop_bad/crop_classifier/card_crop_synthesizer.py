"""
PerspectiveCropDataset: per-card RYGB crops with perspective distortion.

Each sample:
  1. Pick a random card PNG
  2. Scale down (simulate viewing distance)
  3. Rotate randomly
  4. Apply perspective warp (simulate camera angle)
  5. Composite on random background with illumination
  6. Apply RYGB color masking (same pipeline as inference)
  7. Crop to card content + pad, resize to 224x224

After RYGB, the background is zeroed out — the model only sees the
color-masked card shape. Perspective distortion is the key fix:
synthetic cards without it look flat, unlike real photos.
"""

import os
import random
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from project_crop_bad.scripts.dataset import PARENT_PATH, Card
from project_crop_bad.scripts.preprocessing.rygb import hsv2rygb

CACHE_DIR = os.path.join(os.path.dirname(__file__), "crop_cache")

SAMPLES_DIR     = os.path.join(PARENT_PATH, "samples")
CARDS_DIR       = os.path.join(SAMPLES_DIR, "cards")
BACKGROUNDS_DIR = os.path.join(SAMPLES_DIR, "backgrounds")

CARDS_LIST = list(Card)
CROP_SIZE  = (224, 224)


class PerspectiveCropDataset(Dataset):
    """
    n_per_class x 54 samples total, perfectly balanced across all card types.
    """

    def __init__(self, n_per_class: int = 600, crop_size: tuple = CROP_SIZE, augment: bool = True):
        self.n_per_class = n_per_class
        self.crop_size   = crop_size
        self.augment     = augment
        self._total      = n_per_class * len(CARDS_LIST)

        # Load card PNGs: BGRA -> RGBA, 1024x1024
        self.cards = []
        for card in CARDS_LIST:
            path = os.path.join(CARDS_DIR, f"{card.value}.png")
            img  = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise FileNotFoundError(f"Card image missing: {path}")
            self.cards.append(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))

        # Load backgrounds: BGRA -> RGBA
        self.backgrounds = []
        for fname in sorted(os.listdir(BACKGROUNDS_DIR)):
            if fname.lower().endswith(".png"):
                bg = cv2.imread(os.path.join(BACKGROUNDS_DIR, fname), cv2.IMREAD_UNCHANGED)
                if bg is not None:
                    self.backgrounds.append(cv2.cvtColor(bg, cv2.COLOR_BGRA2RGBA))

        print(f"PerspectiveCropDataset: {len(CARDS_LIST)} classes x {n_per_class} = {self._total} samples")

    def __len__(self):
        return self._total

    def __getitem__(self, idx):
        card_idx  = idx % len(CARDS_LIST)
        card_rgba = self.cards[card_idx].copy()

        # 1. Scale down (simulate card at viewing distance)
        scale     = random.uniform(0.18, 0.48)
        h0, w0    = card_rgba.shape[:2]
        card_rgba = cv2.resize(card_rgba,
                               (int(w0 * scale), int(h0 * scale)),
                               interpolation=cv2.INTER_AREA)

        # 2. Random rotation
        card_rgba = _rotate_rgba(card_rgba, random.uniform(-25, 25))

        # 3. Perspective warp
        card_rgba = _perspective_warp(card_rgba, max_shift=0.15)

        # 4. Composite on random background with lighting
        composite = _composite_card(card_rgba, self.backgrounds, canvas_size=512)

        # 5. RYGB preprocessing (no bandpass filter — isolated card, no background clutter)
        hsv  = cv2.cvtColor(composite, cv2.COLOR_RGB2HSV)
        rygb = hsv2rygb(hsv)    # (H, W, 4) uint8

        # 6. Crop to non-zero content, resize to crop_size
        rygb = _tight_crop(rygb, pad=20, size=self.crop_size)

        # 7. Normalise
        x = torch.from_numpy(rygb).float() / 255.0    # (H, W, 4)

        # 8. Augment
        if self.augment:
            x = _augment(x)

        return x, torch.tensor(card_idx, dtype=torch.long)


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------

def _rotate_rgba(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M    = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh   = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += (nw - w) / 2
    M[1, 2] += (nh - h) / 2
    return cv2.warpAffine(img, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


def _perspective_warp(img: np.ndarray, max_shift: float = 0.15) -> np.ndarray:
    h, w = img.shape[:2]
    s    = max_shift * min(h, w)
    src  = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst  = (src + np.random.uniform(-s, s, src.shape)).astype(np.float32)
    M    = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))


# ------------------------------------------------------------------
# Compositing + illumination
# ------------------------------------------------------------------

def _composite_card(card_rgba: np.ndarray, backgrounds: list, canvas_size: int = 512) -> np.ndarray:
    ch, cw = card_rgba.shape[:2]
    S = canvas_size

    # Background patch
    if backgrounds:
        bg   = random.choice(backgrounds)
        bh, bw = bg.shape[:2]
        y0   = random.randint(0, max(0, bh - S))
        x0   = random.randint(0, max(0, bw - S))
        patch = bg[y0:y0+S, x0:x0+S, :3]
        if patch.shape[:2] != (S, S):
            patch = cv2.resize(patch, (S, S))
        canvas = patch.astype(np.float32)
    else:
        c = random.randint(30, 180)
        canvas = np.full((S, S, 3), c, dtype=np.float32)

    # Ambient background lighting
    canvas += random.gauss(0, 8)

    # Place card at random position near centre — compute canvas/card intersection
    cx = S // 2 + random.randint(-40, 40)
    cy = S // 2 + random.randint(-40, 40)
    x1 = max(0, cx - cw // 2);  x2 = min(S, cx + cw - cw // 2)
    y1 = max(0, cy - ch // 2);  y2 = min(S, cy + ch - ch // 2)
    ox1 = x1 - (cx - cw // 2)
    oy1 = y1 - (cy - ch // 2)
    ox2 = ox1 + (x2 - x1)
    oy2 = oy1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1:
        return np.clip(canvas, 0, 255).astype(np.uint8)

    alpha = card_rgba[oy1:oy2, ox1:ox2, 3:4].astype(np.float32) / 255.0
    fg    = card_rgba[oy1:oy2, ox1:ox2, :3].astype(np.float32)

    # Directional lighting on card
    brightness = random.gauss(0, 12)
    dx, dy     = random.gauss(0, 0.5), random.gauss(0, 0.5)
    sh, sw     = fg.shape[:2]
    xs = np.linspace(-dx * 10, dx * 10, sw, dtype=np.float32)
    ys = np.linspace(-dy * 10, dy * 10, sh, dtype=np.float32)
    gradient = np.add.outer(ys, xs)[:, :, np.newaxis]
    fg = np.clip(fg + brightness + gradient, 0, 255)

    canvas[y1:y2, x1:x2] = (canvas[y1:y2, x1:x2] * (1 - alpha) + fg * alpha)

    # Mild global noise
    canvas += np.random.normal(0, 5, canvas.shape).astype(np.float32)
    return np.clip(canvas, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------
# Crop + augment
# ------------------------------------------------------------------

def _tight_crop(rygb: np.ndarray, pad: int, size: tuple) -> np.ndarray:
    mask = np.any(rygb > 10, axis=-1)
    if not mask.any():
        return cv2.resize(rygb, size)
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    H, W   = rygb.shape[:2]
    r0 = max(0, r0 - pad);  r1 = min(H, r1 + pad + 1)
    c0 = max(0, c0 - pad);  c1 = min(W, c1 + pad + 1)
    crop = rygb[r0:r1, c0:c1]
    return cv2.resize(crop, size) if crop.size > 0 else cv2.resize(rygb, size)


def _augment(x: torch.Tensor) -> torch.Tensor:
    """x: (H, W, 4) float [0,1]"""
    x = x.permute(2, 0, 1)                          # (4, H, W)
    if random.random() > 0.5:
        x = TF.hflip(x)
    if random.random() > 0.5:
        x = TF.vflip(x)
    x = TF.rotate(x, random.uniform(-15, 15))
    x = (x + torch.randn_like(x) * 0.02).clamp(0, 1)
    # Random erasing — simulate occlusion from overlapping cards
    if random.random() > 0.65:
        _, h, w = x.shape
        eh = random.randint(h // 8, h // 3)
        ew = random.randint(w // 8, w // 3)
        ey = random.randint(0, h - eh)
        ex = random.randint(0, w - ew)
        x[:, ey:ey+eh, ex:ex+ew] = 0.0
    # Grayscale 10% — force shape learning over colour
    if random.random() > 0.90:
        lum = x.mean(0, keepdim=True).expand_as(x).clone()
        x   = lum
    return x.permute(1, 2, 0)                       # (H, W, 4)


# ------------------------------------------------------------------
# Fast cached dataset (load pre-generated crops from disk)
# ------------------------------------------------------------------

class CachedCropDataset(Dataset):
    """
    Loads pre-generated RYGB crops from disk (created by cache_crops.py).
    Much faster than on-the-fly generation — each epoch takes seconds.
    Augmentation is applied fresh every epoch for variety.
    """

    def __init__(self, cache_dir: str = CACHE_DIR, augment: bool = True):
        crops_path  = os.path.join(cache_dir, "crops.npy")
        labels_path = os.path.join(cache_dir, "labels.npy")
        if not os.path.exists(crops_path):
            raise FileNotFoundError(
                f"Cache not found at {cache_dir}. "
                "Run: python -m project.crop_classifier.cache_crops"
            )
        print("Loading crop cache into RAM …")
        self.crops  = np.load(crops_path)                   # (N, 224, 224, 4) uint8 — fully in RAM
        self.labels = np.load(labels_path)                  # (N,)
        self.augment = augment
        print(f"CachedCropDataset: {len(self.labels)} samples, "
              f"{len(np.unique(self.labels))} classes")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rygb = self.crops[idx].copy()                       # (224, 224, 4) uint8
        x    = torch.from_numpy(rygb).float() / 255.0
        if self.augment:
            x = _augment(x)
        return x, torch.tensor(int(self.labels[idx]), dtype=torch.long)
