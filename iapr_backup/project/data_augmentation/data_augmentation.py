import random
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


def _augment_sector(x: torch.Tensor, channel_permute: bool = False) -> torch.Tensor:
    """
    x : (H, W, C) float tensor in [0, 1], C=4 (RYGB) or 5 (RYGB+Sobel)
    Returns augmented (H, W, C) tensor.
    """
    x = x.permute(2, 0, 1)  # (C, H, W)
    C = x.shape[0]
    n_color = min(C, 4)      # only the RYGB channels get color-specific transforms

    if random.random() > 0.5:
        x = TF.hflip(x)

    if random.random() > 0.5:
        x = TF.vflip(x)

    angle = random.uniform(-15, 15)
    x = TF.rotate(x, angle)

    # Gaussian noise on color channels only
    x[:n_color] = (x[:n_color] + torch.randn_like(x[:n_color]) * 0.02).clamp(0, 1)

    # Random grayscale (15%): collapse RYGB → luminance, keep Sobel channel intact
    if random.random() > 0.85:
        lum = x[:n_color].mean(dim=0, keepdim=True).expand(n_color, -1, -1).clone()
        x = torch.cat([lum, x[n_color:]], dim=0) if C > 4 else lum.expand_as(x).clone()

    # Random channel permutation: only shuffles RYGB, never touches Sobel
    if channel_permute and random.random() > 0.5:
        perm = torch.randperm(n_color)
        x = torch.cat([x[:n_color][perm], x[n_color:]], dim=0) if C > 4 else x[perm]

    # Random erasing — simulate occlusion
    if random.random() > 0.5:
        _, H, W = x.shape
        eh = random.randint(H // 8, H // 3)
        ew = random.randint(W // 8, W // 3)
        top  = random.randint(0, H - eh)
        left = random.randint(0, W - ew)
        x[:, top:top + eh, left:left + ew] = 0.0

    return x.permute(1, 2, 0)  # back to (H, W, C)


class AugmentedDataset(Dataset):
    """Wraps any dataset returning (x, y_center, y_player) and augments x.

    multiplier: each epoch sees (multiplier * len(base)) samples, each with
    independently sampled random transforms — effectively expanding the dataset.
    channel_permute: if True, randomly shuffles RYGB channel order per sector.
    """

    def __init__(self, base_dataset: Dataset, multiplier: int = 5, channel_permute: bool = False):
        self.base = base_dataset
        self.multiplier = multiplier
        self.channel_permute = channel_permute

    def __len__(self):
        return self.multiplier * len(self.base)

    def __getitem__(self, idx):
        x, y_center, y_player = self.base[idx % len(self.base)]
        # x is (5, H, W, C) — augment each sector independently
        x = torch.stack([_augment_sector(x[s], self.channel_permute) for s in range(x.shape[0])])
        return x, y_center, y_player
