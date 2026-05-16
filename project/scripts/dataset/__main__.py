import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from project.scripts.dataset import PARENT_PATH, CARDS_COUNT, load_random_train_images, Card

if __name__ == "__main__":

    # Print path and cards count
    print(f"Parent path : {PARENT_PATH}")
    print(f"Number of unique cards : {CARDS_COUNT}")

    # Load random set of images
    N = 2
    images, labels = load_random_train_images(N)
    probabilities = np.array([label.as_binary_vector() for label in labels])

    # Print images and their labels
    fig = plt.figure(f"{N} random samples from train dataset")
    gs = GridSpec(6, N, figure=fig, height_ratios=[5, 1, 1, 1, 1, 1], hspace=0)

    for i in range(N):
        # Image
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(images[i])
        ax.axis("off")

        # Histograms
        for j in range(5):
            ax = fig.add_subplot(gs[j + 1, i])
            ax.bar(np.arange(CARDS_COUNT), probabilities[i, j], width=0.6)
            ax.set_xlim(-0.5, CARDS_COUNT - 0.5)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            if j < 4:
                ax.set_xticks([])
            else:
                ax.set_xticks(np.arange(CARDS_COUNT))
                ax.set_xticklabels([str(c) for c in Card], rotation=90, fontsize=8)

    # Show figure
    plt.tight_layout()
    plt.show()
