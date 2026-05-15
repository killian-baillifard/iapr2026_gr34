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
            nn.MaxPool2d(2, 2),
            # Block 2 — 32 → 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            # Block 3 — 64 → 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
            nn.Linear(128 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

class UNODataset(Dataset):

    def __init__(self, file_paths, labels):
        """
        Parameters
        ----------
        file_paths : list[str]
            List of paths to  of (n*s) paths to individual sector images on disk

        labels : np.ndarray
            (n*s, 54) binary numpy array, small enough to keep in RAM
        """
        self.file_paths = file_paths
        self.labels     = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        image = np.load(self.file_paths[idx])              # load single sample
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)                     # (c, h, w)
        image = F.interpolate(
            image.unsqueeze(0),
            size=(250, 500),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        return image, self.labels[idx]

def compute_pos_weights(loader, num_labels, device):
    """
    pos_weight[i] = (# negative samples for label i) / (# positive samples for label i)
    This is the standard formulation recommended by PyTorch docs.
    """
    pos_counts = torch.zeros(num_labels)
    total = 0
    for _, labels in loader:
        pos_counts += labels.sum(dim=0).cpu()
        total += labels.shape[0]
    neg_counts = total - pos_counts
    pos_weight = neg_counts / pos_counts.clamp(min=1)
    pos_weight = pos_weight.clamp(max=20) # NOTE Increase when model predict 0 everywhere, decrease when model predict false positives
    return pos_weight.to(device)

def train_epoch(model, loader, optimizer, criterion, device):
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

def val_epoch(model, loader, criterion, device):
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

            # Calculate True Positives, False Positives, False Negatives
            tp = (preds * labels).sum().item()
            fp = (preds * (1 - labels)).sum().item()
            fn = ((1 - preds) * labels).sum().item()

            # Calculate F1 for this batch (adding epsilon to avoid div by zero)
            precision = tp / (tp + fp + 1e-7)
            recall = tp / (tp + fn + 1e-7)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
            
            total_f1 += f1

    avg_f1 = total_f1 / len(loader)
    return total_loss / len(loader), avg_f1

def stratified_split_multilabel(labels: np.ndarray, train_ratio: float = 0.8):
    """
    labels      : (n, C) binary numpy array
    train_ratio : fraction of data for training (default 0.8)
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
    train_dataset = UNODataset([images[i] for i in train_indices], labels[train_indices])
    val_dataset   = UNODataset([images[i] for i in val_indices],   labels[val_indices])
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True,  num_workers=4)
    val_loader    = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4)

    # Instantiate model
    print("Loading model")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = UNOCNNClassifier(num_classes=54).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )
    """
    optimizer = torch.optim.Adagrad(
        model.parameters(),
        lr=0.01,           # AdaGrad typically needs a higher lr than Adam (0.01–0.1)
        lr_decay=0,        # optional: decays lr over time to combat the vanishing lr problem
        eps=1e-10,         # numerical stability term
        weight_decay=0     # L2 regularization if needed
    )
    """
    pos_weight = compute_pos_weights(train_loader, num_labels=54, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    # Run epochs
    print("Running training epochs")
    NUM_EPOCHS = 30
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    f1_scores = []
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, f1_score = val_epoch(model, val_loader, criterion, device)
        print(f"Epoch {epoch + 1:d}/{NUM_EPOCHS} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | F1 score: {f1_score:.2%}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        f1_scores.append(f1_score)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"  ✓ Saved new best model (val loss: {val_loss:.4f})")

    # Plot results
    print("Training complete, generating plots")
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.plot(epochs, f1_scores, label="F1 scores")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Accuracy")
    plt.title("Training evolution")
    plt.legend()
    plt.tight_layout()
    plt.show()
