import numpy as np
from matplotlib import pyplot as plt
from scripts.dataset import load_random_test_image
from scripts.preprocessing.sectors import slice_sectors, SECTOR_END_WIDTH, SECTOR_END_HEIGHT

CROP_SQUARE_SIZE = 64
NUM_CROPS = 16
CROP_X_GRID_SIZE = SECTOR_END_WIDTH // CROP_SQUARE_SIZE
CROP_Y_GRID_SIZE = SECTOR_END_HEIGHT // CROP_SQUARE_SIZE
assert CROP_X_GRID_SIZE * CROP_SQUARE_SIZE == SECTOR_END_WIDTH
assert CROP_Y_GRID_SIZE * CROP_SQUARE_SIZE == SECTOR_END_HEIGHT

def crop(image: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------

    image : np.ndarray
        Full image (height, width, channels)
    
    Returns
    -------

    cropped_image : np.ndarray
        Same image cropped in random positions (height, width, channels)
    """

    # Generate crops positions
    x_starts = np.random.randint(0, CROP_X_GRID_SIZE, (NUM_CROPS)) * CROP_SQUARE_SIZE
    y_starts = np.random.randint(0, CROP_Y_GRID_SIZE, (NUM_CROPS)) * CROP_SQUARE_SIZE

    # Add square range to indices
    span = np.arange(CROP_SQUARE_SIZE)
    y_indices = (y_starts[:, np.newaxis] + span)[..., np.newaxis]
    x_indices = (x_starts[:, np.newaxis] + span)[:, np.newaxis, :]

    # Broadcast then flatten
    y_indices = np.broadcast_to(y_indices, (NUM_CROPS, CROP_SQUARE_SIZE, CROP_SQUARE_SIZE)).reshape(-1)
    x_indices = np.broadcast_to(x_indices, (NUM_CROPS, CROP_SQUARE_SIZE, CROP_SQUARE_SIZE)).reshape(-1)

    # Copy and crop image
    cropped = image.copy()
    cropped[y_indices, x_indices, :] = 0
    return cropped

if __name__ == "__main__":
    sector = slice_sectors(load_random_test_image())[0]
    plt.figure()
    plt.subplot(211)
    plt.imshow(sector)
    plt.axis("off")
    plt.subplot(212)
    plt.imshow(crop(sector))
    plt.axis("off")
    plt.tight_layout()
    plt.show()
