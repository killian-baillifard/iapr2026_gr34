import torch.nn as nn
from torchvision.models import efficientnet_b0, efficientnet_b2, efficientnet_b3

NUM_CARDS = 54

_VARIANTS = {
    "b0": (efficientnet_b0, 1280),
    "b2": (efficientnet_b2, 1408),
    "b3": (efficientnet_b3, 1536),
}


class UnoEfficientNet(nn.Module):
    """
    Input : (B, 5, H, W, C)  — 5 sectors, C color-mask channels (4=RYGB, 3=RGB)
    Output: center_logits (B, 54), player_logits (B, 4, 54)
    """

    def __init__(self, in_channels: int = 4, variant: str = "b2"):
        super().__init__()
        build_fn, embed_dim = _VARIANTS[variant]
        backbone = build_fn(weights=None)
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

        self.center_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Linear(256, NUM_CARDS),
        )
        self.player_head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, NUM_CARDS),
        )

    def forward(self, x):
        B, S, H, W, C = x.shape
        x = x.view(B * S, H, W, C).permute(0, 3, 1, 2).contiguous()  # (B*S, C, H, W)
        feats = self.pool(self.backbone(x)).flatten(1).view(B, S, -1)  # (B, 5, embed_dim)

        center_logits = self.center_head(feats[:, 0, :])               # (B, 54)
        player_logits = self.player_head(
            feats[:, 1:, :].reshape(B * 4, -1)
        ).view(B, 4, NUM_CARDS)                                         # (B, 4, 54)

        return center_logits, player_logits
