import numpy
import torch
from project.evaluation.evaluate import evaluate
from project.train.train_epoch import train_epoch

def train(epochs, model, train_loader, val_loader, optimizer, scheduler, device):
    """Outer loop — calls train_epoch and evaluate each epoch, logs and saves."""
    best_score = 0.0

    for epoch in range(epochs):

        loss, lc, lp = train_epoch(model, train_loader, optimizer, device)
        center_acc, mean_f1, score = evaluate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"loss {loss:.4f} | "
            f"lc {lc:.4f} | "
            f"lp {lp:.4f} | "
            f"val_acc {center_acc:.4f} | "
            f"val_F1 {mean_f1:.4f} | "
            f"score {score:.4f}"
        )

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"  → best model saved (score {score:.4f})")