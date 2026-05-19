import numpy as np
import cv2
try:
    from project.scripts.dataset import CARDS_COUNT, Card, Player, load_random_train_images
except ImportError:
    from dataset import CARDS_COUNT, Card, Player, load_random_train_images
from matplotlib import pyplot as plt

FULL_IMAGE_WIDTH = 4000
FULL_IMAGE_HEIGHT = 2662
SECTOR_WIDTH = 2000
SECTOR_HEIGHT = 1000

CROP_SQUARE_SIZE = 250
NUM_CROPS = 16
CROP_X_GRID_SIZE = SECTOR_WIDTH // CROP_SQUARE_SIZE
CROP_Y_GRID_SIZE = SECTOR_HEIGHT // CROP_SQUARE_SIZE
assert CROP_X_GRID_SIZE * CROP_SQUARE_SIZE == SECTOR_WIDTH
assert CROP_Y_GRID_SIZE * CROP_SQUARE_SIZE == SECTOR_HEIGHT

class Sector:

    def __init__(self, x: int, y: int, width: int, height: int) -> None:

        # Assert sector is contained in image bounds
        assert 0 <= x and x <= FULL_IMAGE_WIDTH
        assert 0 <= y and y <= FULL_IMAGE_HEIGHT
        assert 0 <= x + width and x + width <= FULL_IMAGE_WIDTH
        assert 0 <= y + height and y + height <= FULL_IMAGE_HEIGHT

        # Save sector dimensions
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)

    def slice(self, images: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------

        images : np.ndarray
            RGB images of whole table (n, height, width, 3)
        
        Returns
        -------

        sectors : np.ndarray 
            RGB images of this sector (n, sector, sector_height, sector_width, 3)
        """
        return images[:, self.y:(self.y + self.height), self.x:(self.x + self.width), :]

SECTORS = {
    "center": Sector(
        x       = FULL_IMAGE_WIDTH / 2 - SECTOR_WIDTH / 2,
        y       = FULL_IMAGE_HEIGHT / 2 - SECTOR_HEIGHT / 2,
        width   = SECTOR_WIDTH,
        height  = SECTOR_HEIGHT
    ),
    str(Player.P1): Sector(
        x       = FULL_IMAGE_WIDTH / 2 - SECTOR_WIDTH / 2,
        y       = FULL_IMAGE_HEIGHT - SECTOR_HEIGHT,
        width   = SECTOR_WIDTH,
        height  = SECTOR_HEIGHT
    ),
    str(Player.P2): Sector(
        x       = FULL_IMAGE_WIDTH - SECTOR_HEIGHT,
        y       = FULL_IMAGE_HEIGHT / 2 - SECTOR_WIDTH / 2,
        width   = SECTOR_HEIGHT,
        height  = SECTOR_WIDTH
    ),
    str(Player.P3): Sector(
        x       = FULL_IMAGE_WIDTH / 2 - SECTOR_WIDTH / 2,
        y       = 0,
        width   = SECTOR_WIDTH,
        height  = SECTOR_HEIGHT
    ),
    str(Player.P4): Sector(
        x       = 0,
        y       = FULL_IMAGE_HEIGHT / 2 - SECTOR_WIDTH / 2,
        width   = SECTOR_HEIGHT,
        height  = SECTOR_WIDTH
    )
}

def slice_sectors(images: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------

    images : np.ndarray
        Whole RGB images (n, height, width, 3)
    
    Returns
    -------

    sectors : np.ndarray
        RGB images sliced in sectors (n, sector, sector_height, sector_width, 3)
    """

    # Print current step
    print(f"Slicing sectors {images.shape}", end="")

    # Slice each sector
    center_sector = SECTORS["center"].slice(images)
    player_1_sector = SECTORS[str(Player.P1)].slice(images)
    player_2_sector = SECTORS[str(Player.P2)].slice(images)
    player_3_sector = SECTORS[str(Player.P3)].slice(images)
    player_4_sector = SECTORS[str(Player.P4)].slice(images)

    # Rotate players 2 to 4 sectors
    player_2_sector = np.rot90(player_2_sector, k=-1, axes=(1, 2))
    player_3_sector = np.rot90(player_3_sector, k=2, axes=(1, 2))
    player_4_sector = np.rot90(player_4_sector, k=1, axes=(1, 2))

    # Return stacked sectors
    sectors = np.stack([
        center_sector,
        player_1_sector,
        player_2_sector,
        player_3_sector,
        player_4_sector
    ], axis=1)
    print(f" -> {sectors.shape}")
    return sectors

class HSV:

    def __init__(self, h: int, s: int, v: int) -> None:
        self.h = np.clip(h, 0, 179)
        self.s = np.clip(s, 0, 255)
        self.v = np.clip(v, 0, 255)

# Mean colors
COLORS: dict[str, HSV] = {
    "red":      HSV(1,     184,    239),
    "yellow":   HSV(25,    215,    247),
    "green":    HSV(56,    131,    185),
    "blue":     HSV(98,    211,    217),
    "black":    HSV(17,    46,     43)
}

# Tolerances in multiple of standard deviation
N_SIGMAS = 6
TOLERANCES: dict[str, HSV] = {
    "red":      HSV(N_SIGMAS * 1,   N_SIGMAS * 16,  N_SIGMAS * 14),
    "yellow":   HSV(N_SIGMAS * 1,   N_SIGMAS * 33,  N_SIGMAS * 8),
    "green":    HSV(N_SIGMAS * 4,   N_SIGMAS * 22,  N_SIGMAS * 13),
    "blue":     HSV(N_SIGMAS * 2,   N_SIGMAS * 50,  N_SIGMAS * 13),
    "black":    HSV(179,                       64,  N_SIGMAS * 14)
}

def rgb_to_hsv_batch(rgb: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------

    rgb : np.ndarray
        RGB images (n, height, width, 3)
    
    Returns
    -------

    hsv : np.ndarray
        HSV images (n, height, width, 3)
    """

    # Print current step
    print(f"Converting from RGB to HSV {rgb.shape}", end="")

    img = rgb.astype(np.float32) / 255.0
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    Cmax = np.maximum(np.maximum(r, g), b)
    Cmin = np.minimum(np.minimum(r, g), b)
    delta = Cmax - Cmin

    # Value
    v = Cmax

    # Saturation
    s = np.where(Cmax == 0, 0.0, delta / Cmax)

    # Hue
    h = np.zeros_like(r)
    mask_r = (Cmax == r) & (delta != 0)
    mask_g = (Cmax == g) & (delta != 0)
    mask_b = (Cmax == b) & (delta != 0)

    h[mask_r] = (60 * ((g[mask_r] - b[mask_r]) / delta[mask_r])) % 360
    h[mask_g] = (60 * ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 120)
    h[mask_b] = (60 * ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 240)

    # Scale to OpenCV ranges: H in [0, 180], S and V in [0, 255]
    h = (h / 2).astype(np.uint8)
    s = (s * 255).astype(np.uint8)
    v = (v * 255).astype(np.uint8)

    # Return HSV conversion
    hsv = np.stack([h, s, v], axis=-1)
    print(f" -> {hsv.shape}")
    return hsv

def segment_color(images: np.ndarray, component: str) -> np.ndarray:
    """
    Parameters
    ----------

    images : np.ndarray
        HSV images (n, 5, height, width, 3)
    
    component: str
        Code of color to segment
    
    Returns
    -------

    progressive color mask
        (n, 5, height, width) with 255 at perfect match, 0 at tolerance boundary
    """

    # Print current step
    print(f"Segmenting {component} component {images.shape}", end="")

    center = np.array([COLORS[component].h, COLORS[component].s, COLORS[component].v], dtype=np.float32)
    tol    = np.array([TOLERANCES[component].h, TOLERANCES[component].s, TOLERANCES[component].v], dtype=np.float32)

    # Normalized distance per channel: 0 = perfect match, 1 = at boundary
    dist = np.abs(images.astype(np.float32) - center) / tol  # (..., 3)

    # Handle hue wrap around 0 <=> 180
    if COLORS[component].h <= 2 * TOLERANCES[component].h:
        h_dist = images[..., 0].astype(np.float32)
        wrapped_dist = np.abs(h_dist - 180 - COLORS[component].h) / TOLERANCES[component].h  # distance via the wrap
        dist[..., 0] = np.minimum(dist[..., 0], wrapped_dist)

    # Worst-channel distance determines overall match (1 = all channels match)
    score = 1.0 - np.max(dist, axis=-1)  # (..., H, W)
    score = np.clip(score, 0.0, 1.0)
    segmentations = (score * 255).astype(np.uint8)
    print(f" -> {segmentations.shape}")
    return segmentations

def bandpass_area_filter(preprocessed: np.ndarray, area_bands: list[tuple[int, int]]) -> np.ndarray:
    """
    Remove connected components whose area does not fall within any of the specified bands.
    The mask is computed as the logical OR across all channels before filtering.
    
    Parameters
    ----------
    preprocessed : np.ndarray
        RYGB images (n, 5, height, width, rygb)
    area_bands : list[tuple[int, int]]
        List of (min_area, max_area) bands. Components whose area falls within
        any band are kept, all others are removed.
    
    Returns
    -------
    np.ndarray
        Filtered images, same shape (n, 5, height, width, rygb)
    """
    n, sectors, _, _, n_channels = preprocessed.shape
    output = preprocessed.copy()

    for img_idx in range(n):
        for cap_idx in range(sectors):

            # OR across all channels: a pixel is foreground if any channel detects it
            combined = np.any(preprocessed[img_idx, cap_idx] > 0, axis=-1).astype(np.uint8) * 255  # (H, W)

            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(combined, connectivity=8)

            mask = np.zeros_like(combined)
            for label in range(1, num_labels):
                area = stats[label, cv2.CC_STAT_AREA]
                if any(min_area <= area <= max_area for min_area, max_area in area_bands):
                    mask[labels == label] = 255

            # Apply the same spatial mask to all channels
            for ch_idx in range(n_channels):
                output[img_idx, cap_idx, :, :, ch_idx] = np.where(mask > 0, preprocessed[img_idx, cap_idx, :, :, ch_idx], 0)

    print(f"remove_small_objects: {preprocessed.shape}, bands={area_bands}")
    return output
    
def preprocess(images: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------

    images : np.ndarray
        RGB raw images (n, height, width, 3)
    
    Returns
    -------

    preprocessed : np.ndarray
        RYGB preprocessed images (n, sectors, height, width, 4)
    """

    # Print current step
    print(f"Preprocessing batch of {images.shape[0]} images")

    # Split images into HSV sectors
    rgb_sectors = slice_sectors(images)
    hsv_sectors = rgb_to_hsv_batch(rgb_sectors)

    # Apply color segmentation on all sectors
    r_mask = segment_color(hsv_sectors, "red")
    y_mask = segment_color(hsv_sectors, "yellow")
    g_mask = segment_color(hsv_sectors, "green")
    b_mask = segment_color(hsv_sectors, "blue")
    k_mask = segment_color(hsv_sectors, "black")

    # Add black component to all channels as the white of this new color space (for +4 and color cards)
    print(f"Embedding black as white into RYGB color components")
    y_mask = np.clip(y_mask.astype(np.int16) + k_mask.astype(np.int16), 0, 255).astype(np.uint8)
    r_mask = np.clip(r_mask.astype(np.int16) + k_mask.astype(np.int16), 0, 255).astype(np.uint8)
    g_mask = np.clip(g_mask.astype(np.int16) + k_mask.astype(np.int16), 0, 255).astype(np.uint8)
    b_mask = np.clip(b_mask.astype(np.int16) + k_mask.astype(np.int16), 0, 255).astype(np.uint8)

    # Recombine components
    print("Recombining components", end="")
    combined = np.stack([r_mask, y_mask, g_mask, b_mask], axis=-1)
    print(f" -> {combined.shape}")

    # Filter area bands to remove background
    print("Filtering area bands", end="")
    preprocessed = bandpass_area_filter(combined, [(5_000, 90_000)])
    print(f" -> {preprocessed.shape}")

    return preprocessed

def preview(preprocessed: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------

    preprocessed : np.ndarray
        RYGB preprocessed images (n, sectors, height, width, rygb)
    
    Returns
    -------

    preview : np.ndarray
        RGB preview images (n, height, width, rgb)
    """

    print(f"Building previews of {preprocessed.shape[0]} RYGB images into RGB images", end="")
    RGB_COLORS = {
        "R": np.array([255,   0,   0], dtype=np.float32),
        "Y": np.array([255, 255,   0], dtype=np.float32),
        "G": np.array([  0, 255,   0], dtype=np.float32),
        "B": np.array([  0,   0, 255], dtype=np.float32)
    }
    r_mask = preprocessed[:, :, :, :, 0]
    y_mask = preprocessed[:, :, :, :, 1]
    g_mask = preprocessed[:, :, :, :, 2]
    b_mask = preprocessed[:, :, :, :, 3]
    previews = sum((mask[..., np.newaxis] / 255) * rgb_color for mask, rgb_color in zip([r_mask, y_mask, g_mask, b_mask], RGB_COLORS.values()))
    previews = np.clip(previews, 0, 255).astype(np.uint8)
    print(f" -> {previews.shape}")
    return previews

def crop(images: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------

    RYGB preprocessed images : np.ndarray
        (n, sectors, height, width, 4)
    
    Returns
    -------

    RYGB cropped images : np.ndarray
        (n, sectors, height, width, rygb)
    """

    # Generate crops positions
    n, sectors, _, _, _ = images.shape
    print(f"Cropping batch of {n} images")
    x_crop_positions = np.random.randint(0, CROP_X_GRID_SIZE, (n, sectors, NUM_CROPS)) * CROP_SQUARE_SIZE
    y_crop_positions = np.random.randint(0, CROP_Y_GRID_SIZE, (n, sectors, NUM_CROPS)) * CROP_SQUARE_SIZE

    # Add square range to indices
    offsets = np.arange(CROP_SQUARE_SIZE)
    x_indices = x_crop_positions[..., np.newaxis] + offsets
    y_indices = y_crop_positions[..., np.newaxis] + offsets

    # Reshape x and y and channel indices
    x_indices = x_indices[:, :, :, np.newaxis, :]
    y_indices = y_indices[:, :, :, :, np.newaxis]

    # Combine into one indexing array
    n_idx = np.arange(n)[:, np.newaxis, np.newaxis, np.newaxis, np.newaxis]
    sector_idx = np.arange(sectors)[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
    crop_indices = (n_idx, sector_idx, y_indices, x_indices)

    # Copy, crop and return
    cropped = images.copy()
    cropped[crop_indices] = 0
    return cropped

if __name__ == "__main__":

    # Load random train sample
    N = 4
    train_images, labels = load_random_train_images(N)

    # Apply preprocessing to batch
    preprocessed = preprocess(train_images)
    probabilities = np.array([label.probabilities() for label in labels])
    cropped = crop(preprocessed)

    # Create RGB preview from RYGB images for display
    preprocessed_preview = preview(preprocessed)
    cropped_preview = preview(cropped)

    # Create a figure for each image
    print("Plotting preview")
    for n in range(len(labels)):
        plt.figure(f"Sample {n}")

        # Plot original image
        plt.subplot(1, 4, 1)
        plt.imshow(train_images[n])
        plt.axis('off')

        # Show results sector by sector
        for sector in range(5):

            # Show preprocessed images
            plt.subplot(5, 4, 2 + 4 * sector)
            plt.imshow(preprocessed_preview[n, sector])
            plt.axis('off')

            # Show cropped images
            plt.subplot(5, 4, 3 + 4 * sector)
            plt.imshow(cropped_preview[n, sector])
            plt.axis('off')

            # Show labels
            plt.subplot(5, 4, 4 + 4 * sector)
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
