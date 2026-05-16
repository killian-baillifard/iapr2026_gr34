import numpy as np
from skimage.morphology import closing, disk
from matplotlib import pyplot as plt
from project.scripts.dataset import CARDS_COUNT, Card
from project.scripts.preprocessing.rygb import rygb2rgb
from project.scripts.preprocessing.cache import load_cached_preprocessing

if __name__ == "__main__":

    # Load data and select random sample
    train = load_cached_preprocessing()
    #n = labels.shape[0]
    #i = np.random.randint(0, n)
    i = 0

    # Load, binarize and (morphologically) close image
    preprocessed = np.array(np.load(train[i][0]))
    binarized = preprocessed > 0
    footprint = disk(2)
    closed = np.zeros_like(binarized)
    for i in range(4):
        closed[:, :, i] = closing(binarized[:, :, i], footprint)


    #num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)

    # Create image preview
    closed = 255 * closed
    image_preview = rygb2rgb(closed)

    # Show image and label
    plt.figure()

    plt.subplot(121)
    plt.imshow(image_preview)
    plt.axis("off")

    plt.subplot(122)
    plt.bar(np.arange(CARDS_COUNT), train[i][1], width=0.6)
    plt.xlim(-0.5, CARDS_COUNT - 0.5)
    plt.ylim(0, 1)
    plt.yticks([])
    plt.xticks(np.arange(CARDS_COUNT), [str(c) for c in Card], rotation=90, fontsize=8)
        
    plt.show()
