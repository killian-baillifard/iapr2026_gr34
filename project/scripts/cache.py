import os
import numpy as np
from dataset import PARENT_PATH, load_train_images
from preprocessing import preprocess

CACHE_DIRECTORY   = os.path.join(PARENT_PATH, "cache")
SECTORS_DIRECTORY = os.path.join(CACHE_DIRECTORY, "sectors")
LABELS_CACHE_FILE = os.path.join(CACHE_DIRECTORY, "labels.npy")

RELOAD_TRAIN  = True
RELOAD_LABELS = True

def load_cached_preprocessed_train() -> tuple[list[str], np.ndarray]:
    """
    Returns
    -------
    file_paths : list[str]
        Sorted paths to individual sector images (n*s,)
    labels : np.ndarray
        Binary vectors associated with each sector (n*s, 54)
    """
    file_paths = sorted([
        os.path.join(SECTORS_DIRECTORY, f)
        for f in os.listdir(SECTORS_DIRECTORY)
        if f.endswith(".npy")
    ])
    labels = np.load(LABELS_CACHE_FILE)
    # Reshape labels from (n, s, 54) to (n*s, 54) to match file_paths
    labels = labels.reshape(-1, labels.shape[-1])
    return file_paths, labels

if __name__ == "__main__":
    # Create cache directories
    os.makedirs(CACHE_DIRECTORY,   exist_ok=True)
    os.makedirs(SECTORS_DIRECTORY, exist_ok=True)

    images, labels_list = load_train_images()

    if RELOAD_TRAIN:
        BATCH_SIZE = 5
        ROUNDS     = int(np.ceil(len(images) / BATCH_SIZE))

        global_idx = 0  # tracks game index across batches
        for i in range(ROUNDS):
            print(f"Preprocessing batch {i + 1} of {ROUNDS}")
            slice_size   = min(BATCH_SIZE, len(images))
            preprocessed = preprocess(images[:slice_size])  # (batch, s, h, w, c)

            # Save each sector independently
            for game in range(preprocessed.shape[0]):
                for sector in range(preprocessed.shape[1]):
                    path = os.path.join(SECTORS_DIRECTORY, f"sector_{global_idx + game:04d}_{sector}.npy")
                    np.save(path, preprocessed[game, sector])

            global_idx += slice_size
            images      = images[slice_size:]

    if RELOAD_LABELS:
        print("Parsing and saving labels")
        # Save as (n, s, 54) — reshaping to (n*s, 54) is done at load time
        labels = np.array([label.as_binary_vector() for label in labels_list])
        np.save(LABELS_CACHE_FILE, labels)
