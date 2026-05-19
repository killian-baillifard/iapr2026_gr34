import numpy as np
import torch
from project.scripts.dataset import Card

_CARDS_LIST = list(Card)


def _print_predictions(center_logits, player_logits, yc, yp, n=3, threshold=0.25):
    pred_center  = center_logits.argmax(dim=-1)
    center_probs = center_logits.softmax(dim=-1)
    player_probs = player_logits.sigmoid()
    pred_player  = (player_probs > threshold).long()
    B = min(n, center_logits.shape[0])

    for i in range(B):
        gt_c   = str(_CARDS_LIST[yc[i].item()])
        pd_c   = str(_CARDS_LIST[pred_center[i].item()])
        pd_p   = center_probs[i, pred_center[i]].item()
        mark   = "✓" if yc[i] == pred_center[i] else "✗"
        print(f"  [{i+1}] CENTER  gt={gt_c:<14} pred={pd_c:<14} ({pd_p:.2f}) {mark}")

        for p in range(4):
            gt_cards = [str(_CARDS_LIST[j]) for j in range(len(_CARDS_LIST)) if yp[i, p, j] > 0]
            pd_cards = [(str(_CARDS_LIST[j]), player_probs[i, p, j].item())
                        for j in range(len(_CARDS_LIST)) if pred_player[i, p, j] > 0]
            gt_str = ", ".join(gt_cards) if gt_cards else "EMPTY"
            pd_str = ", ".join(f"{c}({prob:.2f})" for c, prob in pd_cards) if pd_cards else "EMPTY"
            print(f"      P{p+1}  gt=[{gt_str}]  pred=[{pd_str}]")


def evaluate(model, val_loader, device, sigmoid_param=0.25, print_samples=3):
    model.eval()
    center_correct = 0
    total_images   = 0
    all_f1         = []
    printed        = False

    with torch.no_grad():
        for x, yc, yp in val_loader:
            x  = x.to(device)
            yc = yc.to(device)
            yp = yp.to(device)

            center_logits, player_logits = model(x)
            B = x.shape[0]

            if not printed and print_samples > 0:
                _print_predictions(
                    center_logits.cpu(), player_logits.cpu(),
                    yc.cpu(), yp.cpu(),
                    n=print_samples, threshold=sigmoid_param,
                )
                printed = True

            # Center accuracy
            pred_center     = center_logits.argmax(dim=-1)
            center_correct += (pred_center == yc).sum().item()
            total_images   += B

            # Player: sigmoid > threshold gives binary presence per card type
            pred_counts = (player_logits.sigmoid() > sigmoid_param).long()  # (B, 4, 54)
            gt_counts   = yp.long()                                          # (B, 4, 54)

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


def calibrate_threshold(model, val_loader, device, thresholds=None):
    """Sweep sigmoid thresholds on val set; return the one with best score."""
    if thresholds is None:
        thresholds = np.arange(0.05, 1.00, 0.05)

    # Single forward pass — collect all logits/labels
    model.eval()
    all_cl, all_pl, all_yc, all_yp = [], [], [], []
    with torch.no_grad():
        for x, yc, yp in val_loader:
            cl, pl = model(x.to(device))
            all_cl.append(cl.cpu())
            all_pl.append(pl.cpu())
            all_yc.append(yc)
            all_yp.append(yp)

    all_cl = torch.cat(all_cl)
    all_pl = torch.cat(all_pl).sigmoid()
    all_yc = torch.cat(all_yc)
    all_yp = torch.cat(all_yp).long()

    center_acc = (all_cl.argmax(dim=-1) == all_yc).float().mean().item()

    best_thresh, best_score = 0.25, 0.0
    print("Threshold calibration:")
    for t in thresholds:
        pred = (all_pl > t).long()
        all_f1 = []
        for i in range(len(all_yc)):
            tp = fp = fn = 0
            for p in range(4):
                tp += torch.minimum(pred[i, p], all_yp[i, p]).sum().item()
                fp += (pred[i, p] - all_yp[i, p]).clamp(0).sum().item()
                fn += (all_yp[i, p] - pred[i, p]).clamp(0).sum().item()
            pr = tp / (tp + fp + 1e-8)
            rc = tp / (tp + fn + 1e-8)
            all_f1.append(2 * pr * rc / (pr + rc + 1e-8))
        score = 0.1 * center_acc + 0.8 * float(np.mean(all_f1))
        print(f"  t={t:.2f}  F1={np.mean(all_f1):.4f}  score={score:.4f}")
        if score > best_score:
            best_score, best_thresh = score, float(t)

    print(f"Best threshold: {best_thresh:.2f}  (score={best_score:.4f})")
    return best_thresh
