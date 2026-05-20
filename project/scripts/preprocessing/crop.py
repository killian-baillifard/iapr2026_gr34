import numpy as np
from scripts.preprocessing.sectors import SECTOR_SRC_WIDTH, SECTOR_SRC_HEIGHT

CROP_SQUARE_SIZE = 250
NUM_CROPS = 16
CROP_X_GRID_SIZE = SECTOR_SRC_WIDTH // CROP_SQUARE_SIZE
CROP_Y_GRID_SIZE = SECTOR_SRC_HEIGHT // CROP_SQUARE_SIZE
assert CROP_X_GRID_SIZE * CROP_SQUARE_SIZE == SECTOR_SRC_WIDTH
assert CROP_Y_GRID_SIZE * CROP_SQUARE_SIZE == SECTOR_SRC_HEIGHT

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
    x_indices = np.random.randint(0, CROP_X_GRID_SIZE, (NUM_CROPS)) * CROP_SQUARE_SIZE
    y_indices = np.random.randint(0, CROP_Y_GRID_SIZE, (NUM_CROPS)) * CROP_SQUARE_SIZE

    # Add square range to indices
    offsets = np.arange(CROP_SQUARE_SIZE)
    x_indices += offsets
    y_indices += offsets

    # Copy, crop and return
    cropped = image.copy()
    cropped[y_indices, x_indices, :] = 0
    return cropped
