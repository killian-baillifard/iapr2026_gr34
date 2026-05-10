import torch
import torch.nn.functional as F


def loss_fn(center_logits, player_logits, center_gt, player_gt):
    """
    center_logits : (B, 54)      — logits for center card class
    player_logits : (B, 4, 54)  — logits for binary card presence per player
    center_gt     : (B,)         — int class index for center card
    player_gt     : (B, 4, 54)  — int count per card type per player
    """
    loss_center = F.cross_entropy(center_logits, center_gt)

    # Binary presence per card type — sigmoid independent per card
    loss_player = F.binary_cross_entropy_with_logits(
        player_logits,
        player_gt.float()
    )

    total = 0.1 * loss_center + 0.9 * loss_player
    return total, loss_center.item(), loss_player.item()
