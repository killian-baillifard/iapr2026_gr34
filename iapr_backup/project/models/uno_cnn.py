import torch
import torch.nn as nn
from project.models.main import SectorDetection


class UnoCNNSector(nn.Module):
    """
    Input : (B, H, W, 4)  — single preprocessed sector
    Output: (B, 54)        — raw logits (CE for center, BCE for players)
    """

    def __init__(self, embed_dim=256, num_cards=54):
        super().__init__()
        self.encoder = SectorDetection(in_channels=4, embed_dim=embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_cards),
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, 4, H, W)
        return self.head(self.encoder(x))          # (B, 54)
