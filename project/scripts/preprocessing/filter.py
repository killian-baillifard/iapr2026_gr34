import numpy as np
import cv2

AREA_BANDS = [(3_000, 90_000)]

def area_bandpass_filter(image: np.ndarray) -> np.ndarray:
    """
    Remove connected components whose area does not fall within any of the specified bands.
    The mask is computed as the logical OR across all channels before filtering.
    
    Parameters
    ----------
    image : np.ndarray
        image (height, width, channels)
    
    Returns
    -------
    filtered : np.ndarray
        Filtered image, same shape (height, width, channels)
    """

    # Logical or across all binarized channels
    combined = np.any(image > 0, axis=-1).astype(np.uint8) * 255

    # Build mask from area constraints
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)
    mask = np.zeros_like(combined)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if any(min_area <= area <= max_area for min_area, max_area in AREA_BANDS):
            mask[labels == label] = 255

    # Apply the spatial mask to all channels
    filtered = image.copy()
    for c in range(filtered.shape[-1]):
        filtered[:, :, c] = np.where(mask > 0, image[:, :, c], 0)

    # Return filterd image
    return filtered
