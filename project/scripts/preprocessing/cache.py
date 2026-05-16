import os, cv2
import numpy as np
from matplotlib import pyplot as plt
from project.scripts.dataset import CARDS_COUNT
from project.scripts.preprocessing.sectors import slice_sectors
from project.scripts.dataset import PARENT_PATH, Card, load_train_images_paths_and_labels
from project.scripts.preprocessing import preprocess
from project.scripts.preprocessing.rygb import rygb2rgb

CACHE_DIRECTORY   = os.path.join(PARENT_PATH, "cache")
SECTORS_DIRECTORY = os.path.join(CACHE_DIRECTORY, "sectors")
LABELS_CACHE_FILE = os.path.join(CACHE_DIRECTORY, "labels.npy")

def rebuild_preprocessing_cache() -> None:

    # Create cache directories
    print("Rebuilding caches")
    os.makedirs(CACHE_DIRECTORY,   exist_ok=True)
    os.makedirs(SECTORS_DIRECTORY, exist_ok=True)

    # Load train dataset paths and labels
    train_dataset = load_train_images_paths_and_labels()
    N = len(train_dataset)

    # For each image
    labels: np.ndarray = np.zeros((0, CARDS_COUNT))
    for i, (path, label) in enumerate(train_dataset):
        labels = np.concatenate([labels, label.as_binary_vector()])
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        # For each sector
        for j, sector in enumerate(slice_sectors(image)):
            k = 5 * i + j
            print(f"Preprocessing sector {k + 1} / {N * 5}")
            np.save(os.path.join(SECTORS_DIRECTORY, f"{k}.npy"), preprocess(sector))

    # Save labels
    np.save(LABELS_CACHE_FILE, np.array(labels))
    print("Caching done")

def load_cached_preprocessing() -> tuple[list[str], np.ndarray]:
    """
    Returns
    -------

        paths : list[str]
            Paths to the cached preprocessed sectors
        label : np.ndarray
            Sectors labels
    """

    labels = np.array(np.load(LABELS_CACHE_FILE))
    paths = [os.path.join(SECTORS_DIRECTORY, f"{k}.npy") for k in range(labels.shape[0])]
    return paths, labels

if __name__ == "__main__":

    RECOMPUTE_CACHE = True

    if RECOMPUTE_CACHE:
        rebuild_preprocessing_cache()
    else:
        paths, labels = load_cached_preprocessing()
        
        i = np.random.randint(len(paths))
        label = labels[i]
        path = paths[i]

        preprocessed = np.load(path)
        preview = rygb2rgb(preprocessed)
        
        plt.figure()

        plt.subplot(121)
        plt.imshow(preview)
        plt.axis("off")

        plt.subplot(122)
        plt.bar(np.arange(CARDS_COUNT), label, width=0.6)
        plt.xlim(-0.5, CARDS_COUNT - 0.5)
        plt.ylim(0, 1)
        plt.yticks([])
        plt.xticks(np.arange(CARDS_COUNT), [str(c) for c in Card], rotation=90, fontsize=8)
        plt.tight_layout()
        plt.show()
