import numpy as np
import torch


def evaluate(model, val_loader, device):
    model.eval()
    center_correct = 0
    total_images   = 0
    all_f1         = []

    with torch.no_grad():
        for x, yc, yp in val_loader:
            x  = x.to(device)
            yc = yc.to(device)
            yp = yp.to(device)

            center_logits, player_logits = model(x)
            B = x.shape[0]

            # Center accuracy
            pred_center     = center_logits.argmax(dim=-1)          # (B,)
            center_correct += (pred_center == yc).sum().item()
            total_images   += B

            # Player: sigmoid > 0.5 gives binary presence per card type
            pred_counts = (player_logits.sigmoid() > 0.5).long()    # (B, 4, 54)
            gt_counts   = yp.long()                                  # (B, 4, 54)

            # Multiset F1
            for i in range(B):
                tp, fp, fn = 0, 0, 0
                for p in range(4):
                    tp += torch.minimum(pred_counts[i, p], gt_counts[i, p]).sum().item()
                    fp += (pred_counts[i, p] - gt_counts[i, p]).clamp(min=0).sum().item()
                    fn += (gt_counts[i, p] - pred_counts[i, p]).clamp(min=0).sum().item()

                precision = tp / (tp + fp + 1e-8)
                recall    = tp / (tp + fn + 1e-8)
                f1        = 2 * precision * recall / (precision + recall + 1e-8)
                all_f1.append(f1)

    center_acc = center_correct / total_images
    mean_f1    = np.mean(all_f1)
    score      = 0.1 * center_acc + 0.8 * mean_f1

    return center_acc, mean_f1, score
