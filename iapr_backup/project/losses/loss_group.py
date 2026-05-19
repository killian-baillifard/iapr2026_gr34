import torch
import torch.nn.functional as F

# ~2 cards per player out of 54 → positive rate ≈ 3.7% → neg/pos ratio ≈ 26
# pos_weight=15 makes missing a card 15x more costly than a false positive,
# forcing the model out of the all-zeros local minimum
_POS_WEIGHT = torch.full((54,), 15.0)


def loss_fn(center_logits, player_logits, center_gt, player_gt):
    """
    center_logits : (B, 54)      — logits for center card class
    player_logits : (B, 4, 54)  — logits for binary card presence per player
    center_gt     : (B,)         — int class index for center card
    player_gt     : (B, 4, 54)  — int count per card type per player
    """
    loss_center = F.cross_entropy(center_logits, center_gt)

    pos_weight = _POS_WEIGHT.to(player_logits.device)
    loss_player = F.binary_cross_entropy_with_logits(
        player_logits,
        player_gt.float(),
        pos_weight=pos_weight,
    )

    total = 0.1 * loss_center + 0.9 * loss_player
    return total, loss_center.item(), loss_player.item()
