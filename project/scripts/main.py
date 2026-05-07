import numpy as np
from matplotlib import pyplot as plt
from dataset import CARDS_COUNT, Card, load_train_images, load_test_images, load_random_train_images
from preprocessing import preprocess

"""
Note :
- download the dataset from https://www.kaggle.com/competitions/iapr-26-uno-vision-challenge/data
- unzip it's content into the 'data' folder
"""

if __name__ == "__main__":

    # Load random train sample
    N = 4
    train_images, labels = load_random_train_images(N)
    
    # Load full dataset
    # train_images, labels = load_train_images()
    # test_images = load_test_images()

    # Apply preprocessing to batch, only keep preview
    _, previews = preprocess(train_images, True) # Vector of (N, [Center, P1, P2, P3, P4], height, width, [R, G, B])

    # Apply preprocessing to batch
    # preprocessed = preprocess(train_images) # Vector of (N, [Center, P1, P2, P3, P4], height, width, [R, Y, G, B]) with black encoded as R=255, Y=255, G=255, B=255 for +4 and color cards
    probabilities = np.array([label.probabilities() for label in labels]) # Vector of (N, [Center, P1, P2, P3, P4], 54)

    # Create a figure for each image
    print("Plotting preview")
    for n in range(len(labels)):
        plt.figure(f"Sample {n}")

        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(train_images[n])
        plt.axis('off')

        # Show results sector by sector
        for sector in range(5):

            # Show preprocessing
            plt.subplot(5, 3, 2 + 3 * sector)
            plt.imshow(previews[n, sector])
            plt.axis('off')

            # Show train probabilities
            plt.subplot(5, 3, 3 + 3 * sector)
            plt.bar(np.arange(CARDS_COUNT), probabilities[n, sector], width=0.6)
            plt.xlim(-0.5, CARDS_COUNT - 0.5)
            plt.ylim(0, 1)
            plt.yticks([])
            if sector < 4:
                plt.xticks([])
            else:
                plt.xticks(np.arange(CARDS_COUNT), [str(c) for c in Card], rotation=90, fontsize=8)

        # Finalize figure
        plt.tight_layout()

    # Show all figures
    plt.show()

