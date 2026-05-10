# Class for predictor

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# Basic building block
class ConvBNReLU(nn.Module): 
    def __init__(self,c_in,c_out,stride):
        super().__init__() 
        self.net = nn.Sequential(
            nn.Conv2d(c_in,c_out,3,stride=stride,bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU()
        )

    def forward(self,x) : 
        return self.net(x)
    

# Basic block for detection in each sector

class SectorDetection (nn.Module) : 
    def __init__(self,in_channels = 4,embed_dim=256):
        super().__init__() 
        self.net = nn.Sequential(
        ConvBNReLU(in_channels,32,stride=2), # (B, 4, H, W) --> (B, 32, H/2, W/2)
        ConvBNReLU(32,64,stride=2), # (B, 32, H/2, W/2) --> (B, 64, H/4, W/4)
        ConvBNReLU(64,128,stride=2),    # (B, 64, H/4, W/4) -->(B, 128, H/8, W/8)
        ConvBNReLU(128,embed_dim,stride=1) # (B, 128, H/8, W/8) --> (B, embed_dim, H/8, W/8)
        )
        
        # Pooling layer
        self.pool = nn.AdaptiveAvgPool2d(1)  # (B, embed_dim, H/8, W/8) --> (B, embed_dim, 1, 1)

    def forward(self,x):
        return self.pool(self.net(x)).flatten(1)  # (B, embed_dim)



    
class CardCount(nn.Module):
    """
    For each player sector, predict binary presence for each card type.
    """
    def __init__(self, embed_dim=256, num_cards=54):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_cards)
        )

    def forward(self, x):
        return self.net(x)   # (B, 54) — raw logits, apply sigmoid for presence prob
    
class CenterPred(nn.Module):
    """
    For the center sector: single card prediction (argmax, no counting needed).
    """
    def __init__(self, embed_dim=256, num_types=54):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_types)
        )

    def forward(self, x):
        return self.net(x)   # (B, 54) — raw logits, apply softmax/argmax
    


class UnoCNN(nn.Module): 
    """
    Input : (B, 5, H, W, RYGB)
    Output : (B, 5, 54)
    """

    def __init__(self,embed_dim=256):
        super().__init__()
        self.sector = SectorDetection(in_channels=4,embed_dim=embed_dim)
        self.center = CenterPred(embed_dim=embed_dim,num_types=54)
        self.count = CardCount(embed_dim=embed_dim, num_cards=54)

    def forward(self,x):
        """
        Each input is the form (sectors, height, width, rygb)
        """
        B, S, H, W, C = x.shape

        x = x.view(B*S,H,W,C)
        x = x.permute(0, 3, 1, 2).contiguous() # (B*S,C,H,W)

        # Shared backbone — same weights for all sectors
        embs = self.sector(x)                       # (B*5, embed_dim)
        embs = embs.view(B, S, -1)                  # (B, 5, embed_dim)

        # Split center from players
        center_emb  = embs[:, 0, :]                 # (B, embed_dim)
        player_embs = embs[:, 1:, :]                # (B, 4, embed_dim)

        # Center logits — single card
        center_logits = self.center(center_emb)     # (B, 54)

        # Player head — apply to all 4 players at once
        player_logits = self.count(
            player_embs.reshape(B * 4, -1)
        ).view(B, 4, 54)                            # (B, 4, 54)

        return center_logits, player_logits