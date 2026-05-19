import torch
import torch.nn.functional as F


def loss_fn(center_logits, player_logits, center_gt, player_gt, pos_weight):
    """
    BCE-based loss. pos_weight compensates for class imbalance but easy
    negatives still dominate. Prefer focal_loss_fn for better F1 alignment.

    center_logits : (B, 54)
    player_logits : (B, 4, 54)
    center_gt     : (B,)        long — card index
    player_gt     : (B, 4, 54) long — card counts
    """
    pos_weight = pos_weight.to(player_logits.device)

    lc = F.cross_entropy(center_logits, center_gt)
    lp = [
        F.binary_cross_entropy_with_logits(
            player_logits[:, p, :],
            player_gt[:, p, :].clamp(0, 1).float(),
            pos_weight=pos_weight,
        )
        for p in range(4)
    ]

    total = (lc + sum(lp)) / 5
    lp_mean = sum(l.item() for l in lp) / 4
    return total, lc.item(), lp_mean


def _focal_bce(logits, targets, gamma=2.0, pos_weight=None):
    """
    Focal loss for binary classification.
    Down-weights easy correct predictions so the model focuses on hard cases.

    FL(p_t) = -(1 - p_t)^gamma * log(p_t)

    For absent cards the model already predicts absent (p_t → 1):
      focal weight (1-p_t)^2 → 0  — nearly ignored
    For present cards the model misses (p_t → 0):
      focal weight (1-p_t)^2 → 1  — full loss kept
    """
    bce = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    p_t = torch.exp(-bce)                          # probability of correct class
    focal_weight = (1 - p_t) ** gamma
    return (focal_weight * bce).mean()


def focal_loss_fn(center_logits, player_logits, center_gt, player_gt,
                  gamma=2.0, pos_weight=None, center_weight=0.1):
    """
    Focal loss for player cards + cross-entropy for center.

    center_weight controls the balance between center (cross-entropy, large
    magnitude) and player (focal, small magnitude). Default 0.1 matches the
    score formula: 10% center accuracy, 80% player F1.

    center_logits : (B, 54)
    player_logits : (B, 4, 54)
    center_gt     : (B,)        long — card index
    player_gt     : (B, 4, 54) long — card counts
    """
    if pos_weight is not None:
        pos_weight = pos_weight.to(player_logits.device)

    lc = F.cross_entropy(center_logits, center_gt)
    lp = [
        _focal_bce(
            player_logits[:, p, :],
            player_gt[:, p, :].clamp(0, 1).float(),
            gamma=gamma,
            pos_weight=pos_weight,
        )
        for p in range(4)
    ]

    lp_mean = sum(lp) / 4
    total = center_weight * lc + (1.0 - center_weight) * lp_mean
    lp_scalar = sum(l.item() for l in lp) / 4
    return total, lc.item(), lp_scalar
