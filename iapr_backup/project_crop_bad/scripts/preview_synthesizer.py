"""
Quick preview of the synthesizer output.
Generates N samples and saves them as PNG files in project/synthesized/preview/.

Usage:
    python -m project.scripts.preview_synthesizer        # 8 previews
    python -m project.scripts.preview_synthesizer 16     # 16 previews
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # works without a display
import matplotlib.pyplot as plt

from project_crop_bad.scripts.dataset.synthesizer import Synthesizer
from project_crop_bad.scripts.dataset import PARENT_PATH, CARDS_COUNT, Card

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
OUT_DIR = os.path.join(PARENT_PATH, "synthesized", "preview")
os.makedirs(OUT_DIR, exist_ok=True)

synthesizer = Synthesizer()

for i in range(N):
    image, label = synthesizer.generate()
    present = [str(c) for j, c in enumerate(Card) if label[j] == 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    axes[0].imshow(image)
    axes[0].set_title(f"Sample {i+1}  —  cards: {present if present else ['(empty)']}", fontsize=9)
    axes[0].axis("off")

    axes[1].bar(np.arange(CARDS_COUNT), label, width=0.6)
    axes[1].set_xlim(-0.5, CARDS_COUNT - 0.5)
    axes[1].set_ylim(0, 1.2)
    axes[1].set_xticks(np.arange(CARDS_COUNT))
    axes[1].set_xticklabels([str(c) for c in Card], rotation=90, fontsize=7)
    axes[1].set_yticks([])
    axes[1].set_title("Card labels")

    plt.tight_layout()
    out_path = os.path.join(OUT_DIR, f"preview_{i+1:02d}.png")
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Saved {out_path}")

print(f"\nDone — {N} previews saved to {OUT_DIR}")
