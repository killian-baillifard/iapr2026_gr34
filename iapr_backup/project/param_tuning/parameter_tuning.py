import torch

# 
# 6.  THRESHOLD CALIBRATION
# 
@torch.no_grad()
def calibrate_threshold(model, loader, thresholds=None, device="cpu"):
    """
    Sweep thresholds on the validation set; pick the one with best F1.
    Call after training; returns best threshold.
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.05).tolist()

    model.eval()
    all_logits, all_labels = [], []

    for masks, labels, _, _ in loader:
        logits = model(masks.to(device)).cpu()
        all_logits.append(logits)
        all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = decode_predictions(all_logits, threshold=t)
        f1s = []
        for pred, lv in zip(preds, all_labels):
            gt = [IDX2CARD[i] for i, v in enumerate(lv.tolist()) if v > 0]
            f1s.append(multiset_f1_score(pred, gt))
        mean_f1 = float(np.mean(f1s))
        print(f"  threshold={t:.2f}  F1={mean_f1:.4f}")
        if mean_f1 > best_f1:
            best_f1, best_t = mean_f1, t

    print(f"\nBest threshold: {best_t:.2f}  (F1={best_f1:.4f})")
    return best_t