import numpy as np
import cv2
from scripts.dataset import CARDS_COUNT, Card, load_random_train_images
from matplotlib import pyplot as plt
from scripts.preprocessing.sectors import slice_sectors
from scripts.preprocessing.rygb import hsv2rygb, rygb2rgb
from scripts.preprocessing.filter import area_bandpass_filter

if __name__ == "__main__":

    # Load random train sample
    N = 4
    images, labels = load_random_train_images(N)

    # For each image
    for i in range(N):

        # Apply pipeline for each sector
        print(f"Preprocessing image {i + 1} / {N}")
        preprocessed_preview = []
        filtered_preview = []
        for sector in slice_sectors(images[i]):

            hsv = cv2.cvtColor(sector, cv2.COLOR_RGB2HSV)

            preprocessed = hsv2rygb(hsv)
            preprocessed_preview.append(rygb2rgb(preprocessed))

            filtered = area_bandpass_filter(preprocessed)
            filtered_preview.append(rygb2rgb(filtered))

        # Prepare plot
        COLS = 4
        plt.figure(f"Preprocessing {i}")
        plt.subplot(1, COLS, 1)
        plt.imshow(images[i])
        plt.axis('off')

        # Plot sectors
        for s, label in enumerate(labels[i].as_binary_vector()):

            # Show preprocessed images
            plt.subplot(5, COLS, 2 + COLS * s)
            plt.imshow(preprocessed_preview[s])
            plt.axis('off')

            # Show filtered images
            plt.subplot(5, COLS, 3 + COLS * s)
            plt.imshow(filtered_preview[s])
            plt.axis('off')

            # Show labels
            plt.subplot(5, COLS, 4 + COLS * s)
            plt.bar(np.arange(CARDS_COUNT), label, width=0.6)
            plt.xlim(-0.5, CARDS_COUNT - 0.5)
            plt.ylim(0, 1)
            plt.yticks([])
            if s < 4:
                plt.xticks([])
            else:
                plt.xticks(np.arange(CARDS_COUNT), [str(c) for c in Card], rotation=90, fontsize=8)

        # Finialize figure
        plt.tight_layout()

    # Show all figures
    plt.show()
