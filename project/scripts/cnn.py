import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from cache import load_cached_preprocessed_train
import matplotlib.pyplot as plt

class UNOCNNClassifier(nn.Module):
    def __init__(self, num_classes=54):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1 — 4 → 32
            nn.Conv2d(4, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),          # /2

            # Block 2 — 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),          # /4

            # Block 3 — 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),          # /8

            # Block 4 — 128 → 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),          # /16
        )

        # Head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 54)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class UNOSectorizedDataset(Dataset):
    def __init__(self, file_paths, labels):
        """
        file_paths : list of (n*s) paths to individual sector images on disk
        labels     : (n*s, 54) binary numpy array, small enough to keep in RAM
        """
        self.file_paths = file_paths
        self.labels     = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = np.load(self.file_paths[idx])              # load single sample
        image = torch.tensor(image, dtype=torch.float32)
        image = image.permute(2, 0, 1)                     # (c, h, w)
        image = F.interpolate(
            image.unsqueeze(0),
            size=(256, 512),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        return image, self.labels[idx]

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for images, labels in loader:
        images = images.to(device)   # (batch, 4, 1000, 2000)
        labels = labels.to(device)   # (batch, 54)

        optimizer.zero_grad()
        outputs = model(images)      # (batch, 54) — raw logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss    = 0.0
    total_correct = 0
    total_cards   = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs     = model(images)
            loss        = criterion(outputs, labels)
            total_loss += loss.item()

            # Threshold at 0.5
            preds         = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (preds == labels).sum().item()
            total_cards   += labels.numel()

    accuracy = total_correct / total_cards
    return total_loss / len(loader), accuracy

def stratified_split_multilabel(labels: np.ndarray, train_ratio: float = 0.7):
    """
    labels      : (n, C) binary numpy array
    train_ratio : fraction of data for training (default 0.7)
    Returns train_indices, val_indices guaranteeing each card
    appears at least once in the training set.
    """
    n, num_cards = labels.shape
    target_train = int(n * train_ratio)

    assigned      = np.zeros(n, dtype=bool)
    train_indices = []

    # Phase 1 — guarantee each card appears at least once
    for card in range(num_cards):
        candidates = np.where((labels[:, card] == 1) & ~assigned)[0]

        if len(candidates) == 0:
            continue

        chosen = np.random.choice(candidates)
        train_indices.append(chosen)
        assigned[chosen] = True

    # Phase 2 — fill up to target_train with random unassigned samples
    remaining = np.where(~assigned)[0]
    np.random.shuffle(remaining)

    still_needed = max(0, target_train - len(train_indices))
    extra = remaining[:still_needed]

    train_indices.extend(extra.tolist())
    assigned[extra] = True

    val_indices = np.where(~assigned)[0].tolist()

    return train_indices, val_indices

if __name__ == "__main__":

    # Load dataset
    print("Loading preprocessed data")
    images, labels = load_cached_preprocessed_train()

    # Split dataset into train and validations sets
    print("Stratifying data")
    train_indices, val_indices = stratified_split_multilabel(labels)
    train_dataset = UNOSectorizedDataset([images[i] for i in train_indices], labels[train_indices])
    val_dataset   = UNOSectorizedDataset([images[i] for i in val_indices],   labels[val_indices])
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4)
    val_loader    = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4)

    # Instantiate model
    print("Loading model")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = UNOCNNClassifier(num_classes=54).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Run epochs
    print("Running training epochs")
    NUM_EPOCHS = 30
    PATIENCE = 2
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses   = []
    val_losses     = []
    val_accuracies = []

    for epoch in range(NUM_EPOCHS):
        train_loss          = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc   = val_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1:03d}/{NUM_EPOCHS} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.2%}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss              = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ Saved new best model (val loss: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print("Training complete, generating plots")
    epochs = range(1, len(train_losses) + 1)  # use actual epochs ran, not num_epochs
    plt.plot(epochs, train_losses,   label="Train loss")
    plt.plot(epochs, val_losses,     label="Val loss")
    plt.plot(epochs, val_accuracies, label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.title("Training evolution")
    plt.legend()
    plt.tight_layout()
    plt.show()
