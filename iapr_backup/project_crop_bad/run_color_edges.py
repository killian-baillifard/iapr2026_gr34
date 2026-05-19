import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset

from project_crop_bad.scripts.dataset import PreprocessedDataset, Label
from project_crop_bad.scripts.synthesized_dataset import SynthesizedDataset
from project.models.efficientnet import UnoEfficientNet
from project.train.training import train
from project.evaluation.evaluate import calibrate_threshold
from project.data_augmentation.data_augmentation import AugmentedDataset

import pandas as pd
from project_crop_bad.scripts.dataset import TRAIN_FILE

# Per-color Sobel edges at high resolution:
# each of the 4 input channels = Sobel of one color channel (R/Y/G/B)
# the model must learn number/symbol shapes, not just color presence
RES = (640, 640)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

csv    = pd.read_csv(TRAIN_FILE)
labels = [Label.from_row(row) for _, row in csv.iterrows()]

dataset = PreprocessedDataset(labels, target_size=RES, use_color_edges=True)

n_val   = max(1, int(0.5 * len(dataset)))
n_train = len(dataset) - n_val
train_ds, val_ds = random_split(
    dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)

synth_ds    = SynthesizedDataset(target_size=RES, use_color_edges=True)
combined_ds = ConcatDataset([
    AugmentedDataset(train_ds, multiplier=5),
    AugmentedDataset(synth_ds, multiplier=10),
])

train_loader = DataLoader(combined_ds, batch_size=4, shuffle=True, num_workers=4, pin_memory=True,
                          worker_init_fn=lambda wid: torch.manual_seed(torch.initial_seed() + wid))
val_loader   = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

def compute_pos_weight(dataset, num_cards=54):
    pos = torch.zeros(num_cards)
    total = 0
    for _, _, yp in dataset:
        pos   += yp.float().clamp(0, 1).sum(dim=0)
        total += 4
    neg = total - pos
    return (neg / pos.clamp(min=1)).clamp(max=8)

print(f"[color_edges {RES[0]}px] train={n_train}×5aug | synth={len(synth_ds)}×10aug | val={n_val}")

dyn_pos_weight = compute_pos_weight(train_ds)
print(f"pos_weight  mean={dyn_pos_weight.mean():.1f}  min={dyn_pos_weight.min():.1f}  max={dyn_pos_weight.max():.1f}")

# in_channels=4: R-edges / Y-edges / G-edges / B-edges
model     = UnoEfficientNet(in_channels=4, variant="b0").to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

print(f"Device : {device}")
print(f"Params : {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

if __name__ == "__main__":
    train(
        epochs        = 50,
        model         = model,
        train_loader  = train_loader,
        val_loader    = val_loader,
        optimizer     = optimizer,
        scheduler     = scheduler,
        device        = device,
        sigmoid_param = 0.55,
        print_samples = 5,
        pos_weight    = dyn_pos_weight,
        use_focal     = True,
        center_weight = 0.1,
        save_path     = "best_model_color_edges_640.pth",
    )
    best_t = calibrate_threshold(model, val_loader, device)
    print(f"\nUse this threshold for inference: {best_t:.2f}")
