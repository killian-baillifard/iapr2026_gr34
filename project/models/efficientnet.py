import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

INPUT_SIZE = (224, 224)  # EfficientNet expected input size
EMBED_DIM  = 1280        # EfficientNet-B0 output features
NUM_CARDS  = 54
MAX_COUNT  = 4           # predict 0 / 1 / 2 / 3 copies per card


class UnoEfficientNet(nn.Module):
    """
    Input  : (B, 5, H, W, 4)  — 5 sectors, 4 color mask channels (RYGB)
    Output : center_logits (B, 54), player_logits (B, 4, 54, 4)
    """

    def __init__(self):
        super().__init__()

        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Adapt first conv layer: 3 RGB channels -> 4 RYGB channels
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(
            4, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight          # copy RGB weights
            new_conv.weight[:, 3]  = old_conv.weight[:, 0]   # init 4th with red
        backbone.features[0][0] = new_conv

        self.backbone = backbone.features   # convolutional feature extractor
        self.pool     = backbone.avgpool    # AdaptiveAvgPool2d(1) -> (B, 1280, 1, 1)

        self.center_head = nn.Sequential(
            nn.Linear(EMBED_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, NUM_CARDS),
        )

        self.player_head = nn.Sequential(
            nn.Linear(EMBED_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, NUM_CARDS),
        )

    def forward(self, x: torch.Tensor):
        B, S, H, W, C = x.shape

        # Flatten sectors into batch, put channels first
        x = x.view(B * S, H, W, C).permute(0, 3, 1, 2).contiguous()   # (B*S, C, H, W)

        # Resize to EfficientNet expected input size
        x = F.interpolate(x, size=INPUT_SIZE, mode="bilinear", align_corners=False)

        # Shared backbone — same weights across all sectors
        feats = self.pool(self.backbone(x)).flatten(1)  # (B*S, 1280)
        feats = feats.view(B, S, -1)                    # (B, 5, 1280)

        center_emb  = feats[:, 0, :]        # (B, 1280)
        player_embs = feats[:, 1:, :]       # (B, 4, 1280)

        center_logits = self.center_head(center_emb)    # (B, 54)

        player_logits = self.player_head(
            player_embs.reshape(B * 4, -1)
        ).view(B, 4, NUM_CARDS)                         # (B, 4, 54)

        return center_logits, player_logits
