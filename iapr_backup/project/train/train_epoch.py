import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from project.losses.loss import loss_fn, focal_loss_fn

def train_epoch(model, train_loader, optimizer, device, pos_weight, use_focal=False, center_weight=0.1):
    """One epoch of training. Returns average losses."""
    model.train()
    total_loss, total_lc, total_lp = 0.0, 0.0, 0.0

    for x, yc, yp in train_loader:
        x  = x.to(device)
        yc = yc.to(device)
        yp = yp.to(device)

        optimizer.zero_grad()
        center_logits, player_logits = model(x)

        if use_focal:
            loss, lc, lp = focal_loss_fn(center_logits, player_logits, yc, yp,
                                          pos_weight=pos_weight, center_weight=center_weight)
        else:
            loss, lc, lp = loss_fn(center_logits, player_logits, yc, yp, pos_weight)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_lc   += lc
        total_lp   += lp

    n = len(train_loader)
    return total_loss / n, total_lc / n, total_lp / n

