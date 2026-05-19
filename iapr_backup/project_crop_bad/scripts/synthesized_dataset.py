import os
import numpy as np
import torch
from torch.utils.data import Dataset

from project_crop_bad.scripts.dataset import PARENT_PATH, add_sobel_channel, rygb_to_color_edges

SYNTHESIZED_DIR   = os.path.join(PARENT_PATH, "synthesized")
CACHE_DIR         = os.path.join(PARENT_PATH, "synthesized_cache")

# Labels live in the cache dir once caching is done, else fall back to raw dir
def _labels_path():
    cached = os.path.join(CACHE_DIR, "labels.npy")
    return cached if os.path.exists(cached) else os.path.join(SYNTHESIZED_DIR, "labels.npy")

def _use_cache():
    return os.path.isdir(CACHE_DIR) and os.path.exists(os.path.join(CACHE_DIR, "labels.npy"))


class SynthesizedDataset(Dataset):
    """Dataset built from pre-generated synthetic sectors.

    Loads from synthesized_cache/ (fast, 448x448 RYGB) if available,
    otherwise falls back to raw synthesized/ with on-the-fly conversion (slow).

    Each sample groups 5 sectors into a game state matching PreprocessedDataset:
      x        : (5, 448, 448, 4) float32 RYGB in [0,1]
      y_center : () long
      y_player : (4, 54) long

    Run `python -m project.scripts.cache_synthesized` once to build the fast cache.
    """

    def __init__(self, target_size: tuple[int, int] | None = (448, 448), use_sobel: bool = False, use_color_edges: bool = False):
        labels_path = _labels_path()
        if not os.path.exists(labels_path):
            raise FileNotFoundError(
                "Synthesized data not found. "
                "Run: python -m project.scripts.dataset.synthesizer"
            )
        self.labels          = np.load(labels_path)   # (N, 54) float
        self.n               = len(self.labels)
        self.target_size     = target_size
        self.use_sobel       = use_sobel
        self.use_color_edges = use_color_edges
        self.cached          = _use_cache()

        if not self.cached:
            import cv2
            from project_crop_bad.scripts.preprocessing.rygb import hsv2rygb
            self._cv2      = cv2
            self._hsv2rygb = hsv2rygb

        self.center_pool = [i for i in range(self.n) if int(self.labels[i].sum()) == 1]
        if not self.center_pool:
            self.center_pool = list(range(self.n))

    def __len__(self):
        return self.n // 4

    def _load_sector(self, i: int) -> np.ndarray:
        if self.cached:
            return np.load(os.path.join(CACHE_DIR, f"{i}.npy"))   # (448,448,4) uint8
        else:
            img = np.load(os.path.join(SYNTHESIZED_DIR, f"{i}.npy"))
            hsv = self._cv2.cvtColor(img, self._cv2.COLOR_RGB2HSV)
            return self._hsv2rygb(hsv)                             # (H,W,4) uint8

    def __getitem__(self, idx):
        player_ids = [(idx * 4 + p) % self.n for p in range(4)]
        center_id  = self.center_pool[idx % len(self.center_pool)]

        sectors = [self._load_sector(center_id)] + [self._load_sector(p) for p in player_ids]
        x = np.stack(sectors, axis=0)                              # (5, H, W, 4)
        x = torch.from_numpy(x).float() / 255.0

        if self.target_size is not None:
            x = torch.nn.functional.interpolate(
                x.permute(0, 3, 1, 2),
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1).contiguous()

        if self.use_color_edges:
            x = rygb_to_color_edges(x)                             # (5, H, W, 4) per-color edges
        if self.use_sobel:
            x = add_sobel_channel(x)                               # +1 channel

        y_center = torch.tensor(int(np.argmax(self.labels[center_id])), dtype=torch.long)
        player_labels = np.stack([self.labels[p] for p in player_ids]).astype(np.int64)
        y_player = torch.from_numpy(player_labels)
        return x, y_center, y_player
