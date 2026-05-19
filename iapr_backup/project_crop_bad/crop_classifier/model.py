import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

NUM_CARDS = 54
CROP_SIZE = (224, 224)


class CropClassifier(nn.Module):
    """Single-card 54-class classifier.
    Input : (B, H, W, 4) RYGB float32 in [0, 1]
    Output: (B, 54) logits
    """

    def __init__(self, in_channels: int = 4):
        super().__init__()
        backbone = efficientnet_b0(weights=None)
        old_conv = backbone.features[0][0]
        backbone.features[0][0] = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        self.backbone = backbone.features
        self.pool     = backbone.avgpool
        self.head     = nn.Sequential(
            nn.Linear(1280, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, NUM_CARDS),
        )

    def forward(self, x):
        # x: (B, H, W, 4) → (B, 4, H, W)
        x = x.permute(0, 3, 1, 2).contiguous()
        f = self.pool(self.backbone(x)).flatten(1)
        return self.head(f)
