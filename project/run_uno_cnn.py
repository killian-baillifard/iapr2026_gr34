import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from project.scripts.dataset import PreprocessedDataset, Label, TRAIN_FILE
from project.models.main import UnoCNN
from project.train.training import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
print("Building dataset...")
csv    = pd.read_csv(TRAIN_FILE)
labels = [Label.from_row(row) for _, row in csv.iterrows()]

idx    = np.random.permutation(len(labels))
split  = int(0.9 * len(labels))
train_labels = [labels[i] for i in idx[:split]]
val_labels   = [labels[i] for i in idx[split:]]

train_loader = DataLoader(PreprocessedDataset(train_labels), batch_size=8, shuffle=True,  num_workers=0)
val_loader   = DataLoader(PreprocessedDataset(val_labels),   batch_size=8, shuffle=False, num_workers=0)
print(f"  -> train {len(train_loader.dataset)} | val {len(val_loader.dataset)} samples")

# Model
model     = UnoCNN(embed_dim=256).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

print(f"Device : {device}")
print(f"Params : {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

if __name__ == "__main__":
    train(
        epochs       = 50,
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        optimizer    = optimizer,
        scheduler    = scheduler,
        device       = device,
    )
