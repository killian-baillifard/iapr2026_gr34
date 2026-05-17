import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, RandomSampler
import matplotlib.pyplot as plt
from project.scripts.preprocessing.cache import Cache, load_labels, load_image
from sklearn.metrics import f1_score

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        # 1×1 projection to match channel dims for the skip addition
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))


class UNOCNNClassifier(nn.Module):
    def __init__(self, num_classes=54):
        super().__init__()

        # ── Spatial feature extraction ──────────────────────────────────────
        # Block 1: 4 → 32,  MaxPool ÷2  → (32, 125, 250)
        self.block1 = nn.Sequential(
            ResidualBlock(4, 32),
            nn.MaxPool2d(2, 2),
        )
        # Block 2: 32 → 64, MaxPool ÷2  → (64, 62, 125)
        self.block2 = nn.Sequential(
            ResidualBlock(32, 64),
            nn.MaxPool2d(2, 2),
        )
        # Block 3: 64 → 128, MaxPool ÷2 → (128, 31, 62)
        self.block3 = nn.Sequential(
            ResidualBlock(64, 128),
            nn.MaxPool2d(2, 2),
        )
        # Block 4: 128 → 256, NO MaxPool → (256, 31, 62)
        # Keeps spatial resolution high before pooling
        self.block4 = ResidualBlock(128, 256)

        # ── Spatial → vector ────────────────────────────────────────────────
        # (4×4) instead of (2×2): 8× more spatial info than original
        self.pool = nn.AdaptiveAvgPool2d((4, 4))   # → (256, 4, 4) = 4096-d

        # ── Classifier head ─────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.pool(x)
        return self.classifier(x)

class UNODataset(Dataset):

    def __init__(self, cache: Cache):
        """
        Parameters
        ----------

        cache : Cache
            Cache from which to load the dataset
        """
        self.cache = cache
        self.labels = load_labels(self.cache)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, i):
        image = load_image(self.cache, i)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1) # (c, h, w)
        return image, self.labels[i]

def compute_pos_weights(labels: np.ndarray, device: torch.device):
    """
    pos_weight[i] = (# negative samples for label i) / (# positive samples for label i)
    This is the standard formulation recommended by PyTorch docs.

    labels : np.ndarray of shape (N, 54), binary
    """
    labels     = torch.from_numpy(labels).float()
    pos_counts = labels.sum(dim=0)
    neg_counts = len(labels) - pos_counts
    pos_weight = neg_counts / pos_counts.clamp(min=1)
    pos_weight = pos_weight.clamp(max=20) # NOTE Increase when model predict 0 everywhere, decrease when model predict false positives
    return pos_weight.to(device)

def train_epoch(model: UNOCNNClassifier, loader: DataLoader, optimizer: torch.optim.AdamW, criterion: nn.BCEWithLogitsLoss, device: torch.device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)   # pos_weight applied automatically per label
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def val_epoch(model: UNOCNNClassifier, loader: DataLoader, criterion: nn.BCEWithLogitsLoss, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_f1 = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            total_loss += criterion(outputs, labels).item()

            # Threshold to get binary predictions
            THRESHOLD = 0.2
            preds = (torch.sigmoid(outputs) > THRESHOLD).float()
            f1 = f1_score(labels, preds, average='micro', zero_division=0)
            
            total_f1 += f1

    avg_f1 = total_f1 / len(loader)
    return total_loss / len(loader), avg_f1

def train_model() -> None:

    # Split dataset into train and validations sets
    print("Creating datasets")
    train_dataset = UNODataset(Cache.TRAINING)
    val_dataset = UNODataset(Cache.VALIDATION)
    sampler = RandomSampler(train_dataset, replacement=True, num_samples=512)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Instantiate model
    print("Instantiating new model")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = UNOCNNClassifier(num_classes=54).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    pos_weight = compute_pos_weights(train_dataset.labels, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Run epochs
    print("Running training epochs")
    NUM_EPOCHS = 100
    PATIENCE = 10
    patience = 0
    best_val_loss = np.inf
    train_losses = []
    val_losses = []
    f1s = []
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, f1 = val_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1:02d}/{NUM_EPOCHS} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | F1 score: {f1:.2%}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        f1s.append(f1)

        if val_loss < best_val_loss:
            patience = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ Saved new best model (val loss: {val_loss:.4f})")
        else:
            patience += 1
            if patience > PATIENCE:
                break

        scheduler.step(val_loss)

    # Plot results
    print("Training complete, generating plots")
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.plot(epochs, f1s, label="F1 scores")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.title("Training evolution")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_model()
