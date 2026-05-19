"""
Train the CropClassifier on synthetic single-card RYGB crops.

Why this is better than sector-level classification:
  - Each training sample = one isolated card with an unambiguous label
  - 54-class cross-entropy is simpler than multi-label over whole sectors
  - Synthetic data is perfect here: every sample is a clean single-card render
  - At inference, RYGB contour detection extracts per-card crops from real sectors
"""

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from project.crop_classifier.model       import CropClassifier
from project.crop_classifier.crop_dataset import SyntheticCropDataset

EPOCHS     = 80
BATCH_SIZE = 32
LR         = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_ds = SyntheticCropDataset(augment=True)
loader   = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True,
)

model     = CropClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

print(f"Device : {device}")
print(f"Params : {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
print(f"Samples: {len(train_ds)}")

if __name__ == "__main__":
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = correct = total = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.argmax(1) == y).sum().item()
            total      += y.size(0)

        scheduler.step()
        acc = correct / total
        avg_loss = total_loss / len(loader)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "best_crop_classifier.pth")

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"loss={avg_loss:.4f} | acc={acc:.4f} | best={best_acc:.4f}"
        )

    print(f"\nDone. Best train acc: {best_acc:.4f}")
    print("Saved: best_crop_classifier.pth")
    print("\nEvaluate with:")
    print("  python -m project.crop_classifier.evaluate")
