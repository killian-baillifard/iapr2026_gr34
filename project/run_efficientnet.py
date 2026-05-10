import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from project.scripts.dataset import load_train_images
from project.scripts.preprocessing import preprocess
from project.models.efficientnet import UnoEfficientNet
from project.train.training import train

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data
print("Loading images...")
train_images, labels = load_train_images()
print(f"  -> {len(labels)} images loaded, shape {train_images.shape}")

print("Preprocessing...")
preprocessed = preprocess(train_images)
print(f"  -> preprocessed shape {preprocessed.shape}")

print("Building ground truth tensors...")
center_array = np.array([label.center_idx()          for label in labels])  # (N,)
player_array = np.array([label.probabilities()[1:, :] for label in labels])  # (N, 4, 54)
print(f"  -> center {center_array.shape} | players {player_array.shape}")

print("Converting to tensors...")
X        = torch.from_numpy(preprocessed).float() / 255.0
Y_center = torch.from_numpy(center_array).long()
Y_player = torch.from_numpy(player_array).float()
print(f"  -> X {X.shape} | Y_center {Y_center.shape} | Y_player {Y_player.shape}")

print("Building loaders...")
idx       = np.random.permutation(len(X))
split     = int(0.9 * len(X))
idx_train = idx[:split]
idx_val   = idx[split:]
train_loader = DataLoader(
    TensorDataset(X[idx_train], Y_center[idx_train], Y_player[idx_train]),
    batch_size=8, shuffle=True, num_workers=4,
)
val_loader = DataLoader(
    TensorDataset(X[idx_val], Y_center[idx_val], Y_player[idx_val]),
    batch_size=8, shuffle=False, num_workers=4,
)
print(f"  -> train {len(train_loader.dataset)} samples | val {len(val_loader.dataset)} samples")

# Model
model     = UnoEfficientNet().to(device)
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
    )
