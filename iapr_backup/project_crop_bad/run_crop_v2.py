"""
Train CropClassifier v2 on perspective-augmented synthetic crops.

Prerequisites:
    python -m project.crop_classifier.cache_crops   # one-time, ~20 min

Usage:
    python -m project.run_crop_v2

Submit:
    python -m project.predict_test --model crop --weights best_crop_v2.pth
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from project_crop_bad.crop_classifier.model                 import CropClassifier
from project_crop_bad.crop_classifier.card_crop_synthesizer import CachedCropDataset
from project_crop_bad.crop_classifier.evaluate              import evaluate
from project_crop_bad.scripts.dataset                       import TRAIN_FILE, Label

EPOCHS     = 150
BATCH_SIZE = 64
LR         = 5e-4
SAVE_PATH  = "best_crop_v2.pth"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Training data (synthetic) ---
train_ds = CachedCropDataset(augment=True)
loader   = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True,
)

# --- Validation data (real labeled images) ---
csv    = pd.read_csv(TRAIN_FILE)
labels = [Label.from_row(row) for _, row in csv.iterrows()]

model     = CropClassifier().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=2)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

print(f"Device     : {device}")
print(f"Params     : {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
print(f"Train size : {len(train_ds)} synthetic crops")
print(f"Val size   : {len(labels)} real labeled images")
print(f"Batches    : {len(loader)}")

if __name__ == "__main__":
    best_val_score = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = correct = total = 0

        for x, y in loader:
            x, y   = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct    += (logits.detach().argmax(1) == y).sum().item()
            total      += y.size(0)

        scheduler.step(epoch)
        train_acc = correct / total
        avg_loss  = total_loss / len(loader)

        ca, mf1, score = evaluate(model, labels)

        if score > best_val_score:
            best_val_score = score
            torch.save(model.state_dict(), SAVE_PATH)
            tag = " ← best"
        else:
            tag = ""

        print(
            f"Epoch {epoch+1:03d}/{EPOCHS} | "
            f"loss={avg_loss:.4f} | train_acc={train_acc:.4f} | "
            f"center={ca:.3f} f1={mf1:.3f} score={score:.4f}{tag}",
            flush=True,
        )

    print(f"\nDone. Best val score: {best_val_score:.4f}")
    print(f"Saved: {SAVE_PATH}")
