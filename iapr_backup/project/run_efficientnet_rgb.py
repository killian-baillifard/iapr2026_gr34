import torch
from torch.utils.data import DataLoader, random_split

from project.scripts.dataset import RawSectorDataset, Label
from project.models.efficientnet import UnoEfficientNet
from project.train.training import train
from project.evaluation.evaluate import calibrate_threshold
from project.data_augmentation.data_augmentation import AugmentedDataset

import pandas as pd
from project.scripts.dataset import TRAIN_FILE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

csv    = pd.read_csv(TRAIN_FILE)
labels = [Label.from_row(row) for _, row in csv.iterrows()]

dataset = RawSectorDataset(labels)

n_val   = max(1, int(0.1 * len(dataset)))
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(AugmentedDataset(train_ds), batch_size=4, shuffle=True,  num_workers=4, pin_memory=True,
                          worker_init_fn=lambda wid: torch.manual_seed(torch.initial_seed() + wid))
val_loader   = DataLoader(val_ds,                     batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

print(f"Train: {n_train} samples | Val: {n_val} samples")

model     = UnoEfficientNet(in_channels=3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
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
        sigmoid_param= 0.35,
        print_samples= 5,
        pos_weight   = torch.full((54,), 8.0),
    )
    best_t = calibrate_threshold(model, val_loader, device)
    print(f"\nUse this threshold for inference: {best_t:.2f}")
