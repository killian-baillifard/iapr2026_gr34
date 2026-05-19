import numpy
import torch
from project.evaluation.evaluate import evaluate, calibrate_threshold
from project.train.train_epoch import train_epoch

def train(epochs, model, train_loader, val_loader, optimizer, scheduler, device, sigmoid_param, print_samples, pos_weight, use_focal=False, center_weight=0.1, calibrate_every=5, save_path="best_model.pth"):
    """Outer loop — calls train_epoch and evaluate each epoch, logs and saves."""
    best_score = 0.0

    for epoch in range(epochs):

        loss, lc, lp = train_epoch(model, train_loader, optimizer, device, pos_weight, use_focal=use_focal, center_weight=center_weight)
        center_acc, mean_f1, score = evaluate(model, val_loader, device, sigmoid_param, print_samples)
        scheduler.step()

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"loss {loss:.4f} | "
            f"lc {lc:.4f} | "
            f"lp {lp:.4f} | "
            f"val_center_acc {center_acc:.4f} | "
            f"val_F1 {mean_f1:.4f} | "
            f"score {score:.4f}"
        )

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), save_path)
            print(f"  → best model saved to {save_path} (score {score:.4f})")

        if calibrate_every and (epoch + 1) % calibrate_every == 0:
            best_t = calibrate_threshold(model, val_loader, device)
            sigmoid_param = best_t
            print(f"  [epoch {epoch+1}] threshold updated to {best_t:.2f}")