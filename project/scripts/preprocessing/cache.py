import os, cv2
import numpy as np
from matplotlib import pyplot as plt
from enum import StrEnum
from project.scripts.dataset import CARDS_COUNT
from project.scripts.preprocessing.sectors import slice_sectors
from project.scripts.dataset import PARENT_PATH, Card, load_train_images_paths_and_labels
from project.scripts.dataset.synthesizer import SYNTHESIZED_DIRECTORY, SYNTHESIZED_LABELS_PATH, SYNTHESIZED_SECTORS, synthesized_image_path
from project.scripts.preprocessing import preprocess
from project.scripts.preprocessing.rygb import rygb2rgb

class Cache(StrEnum):
    VALIDATION = "validation"
    TRAINING = "training"

CACHE_DIRECTORY = os.path.join(PARENT_PATH, "cache")
VALIDATION_CACHE_DIRECTORY = os.path.join(CACHE_DIRECTORY, str(Cache.VALIDATION))
VALIDATION_LABELS_PATH = os.path.join(VALIDATION_CACHE_DIRECTORY, "labels.npy")
TRAINING_CACHE_DIRECTORY = os.path.join(CACHE_DIRECTORY, str(Cache.TRAINING))
TRAINING_LABELS_PATH = os.path.join(TRAINING_CACHE_DIRECTORY, "labels.npy")

validation_image_path = lambda i: os.path.join(VALIDATION_CACHE_DIRECTORY, f"{i}.npy")
training_image_path = lambda i: os.path.join(TRAINING_CACHE_DIRECTORY, f"{i}.npy")

def rebuild_cache(cache: Cache) -> None:
    """
    Parameters
    ----------

        cache : Cache
            Cache to rebuild
    """

    # Create cache directories
    print(f"Rebuilding {cache} cache")
    os.makedirs(CACHE_DIRECTORY, exist_ok=True)
    match cache:

        case Cache.VALIDATION:

            # Rebuild validation cache
            os.makedirs(VALIDATION_CACHE_DIRECTORY, exist_ok=True)

            # Load validation dataset paths and labels
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
                    np.save(validation_image_path(k), preprocess(sector))

            # Save labels
            np.save(VALIDATION_LABELS_PATH, np.array(labels))
            print("Caching done")

        case Cache.TRAINING:

            # Rebuild training cache
            os.makedirs(TRAINING_CACHE_DIRECTORY, exist_ok=True)

            # Copy labels
            labels = np.load(SYNTHESIZED_LABELS_PATH)
            n = labels.shape[0]
            np.save(TRAINING_LABELS_PATH, labels)

            # For each sector
            for i in range(n):
                print(f"Preprocessing sector {i + 1} / {n}")
                path = synthesized_image_path(i)
                sector = np.array(np.load(path))
                np.save(training_image_path(i), preprocess(sector))
            
            print("Caching done")

def load_labels(cache: Cache) -> np.ndarray:
    """
    Parameters
    ----------

        cache : Cache
            Cache to load labels from

    Returns
    -------

        label : np.ndarray
            Sectors labels
    """

    match cache:
        case Cache.VALIDATION:
            return np.array(np.load(VALIDATION_LABELS_PATH))
        case Cache.TRAINING:
            return np.array(np.load(TRAINING_LABELS_PATH))
        
def load_image(cache: Cache, i: int) -> np.ndarray:
    match cache:
        case Cache.VALIDATION:
            return np.load(validation_image_path(i))
        case Cache.TRAINING:
            return np.load(training_image_path(i))

if __name__ == "__main__":

    REBUILD_CACHE = True
    CACHE = Cache.TRAINING

    if REBUILD_CACHE:
        rebuild_cache(CACHE)
    else:

        labels = load_labels(CACHE)
        i = np.random.randint(labels.shape[0])

        image = load_image(CACHE, i)
        label = labels[i]

        preview = rygb2rgb(image)
        
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
