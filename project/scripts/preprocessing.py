import numpy as np
from project.scripts.dataset import Player, load_random_train_images
from matplotlib import pyplot as plt

FULL_IMAGE_WIDTH = 4000
FULL_IMAGE_HEIGHT = 2662
SECTOR_WIDTH = 2000
SECTOR_HEIGHT = 1000

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
        Input:
            Images (n, height, width, rgb)
        Output:
            Sectors (n, sector, sector_height, sector_width, rgb)
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
    Input:
        Images (n, height, width, rgb)
    Output:
        Sectors (n, sector, sector_height, sector_width, rgb)
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
TOLERANCES: dict[str, HSV] = {
    "red":      HSV(6 * 1,     6 * 16,     6 * 14),
    "yellow":   HSV(6 * 1,     6 * 33,     6 * 8),
    "green":    HSV(6 * 4,     6 * 22,     6 * 13),
    "blue":     HSV(6 * 2,     6 * 50,     6 * 13),
    "black":    HSV(179,       64,         6 * 14)
}

def rgb_to_hsv_batch(images: np.ndarray) -> np.ndarray:
    """
    Input:
        RGB images (n, height, width, 3)
    Output:
        HSV images (n, height, width, 3)
    """

    # Print current step
    print(f"Converting from RGB to HSV {images.shape}", end="")

    img = images.astype(np.float32) / 255.0
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
    Input:
        Images (n, 5, height, width, 3) in HSV
    Output:
        Masks (n, 5, height, width) with 255 at perfect match, 0 at tolerance boundary
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

def preprocess(images: np.ndarray, preview: bool = False) -> np.ndarray:
    """
    Input:
        Images (n, height, width, rgb)
    Output:
        Preprocessed images (n, sectors, height, width, rygb)
        Previews (n, sectors, height, width, rgb) [Optionnal]
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

    print("Recombining components", end="")
    preprocessed = np.stack([r_mask, y_mask, g_mask, b_mask], axis=-1)
    print(f" -> {preprocessed.shape}")

    # Build previews from masks using fixed RGB colors
    if preview:
        print(f"Building previews {preprocessed.shape}", end="")
        RGB_COLORS = {
            "R": np.array([255,   0,   0], dtype=np.float32),
            "Y": np.array([255, 255,   0], dtype=np.float32),
            "G": np.array([  0, 255,   0], dtype=np.float32),
            "B": np.array([  0,   0, 255], dtype=np.float32),
            "K": np.array([255, 255, 255], dtype=np.float32),
        }
        previews = sum(
            (mask[..., np.newaxis] / 255) * rgb_color
            for mask, rgb_color in zip(
                [r_mask, y_mask, g_mask, b_mask, k_mask],
                RGB_COLORS.values()
            )
        )
        previews = np.clip(previews, 0, 255).astype(np.uint8)
        print(f" -> {previews.shape}")
        return preprocessed, previews
    
    # Or return only preprocessed images
    else:
        return preprocessed

if __name__ == "__main__":

    # Load random train sample
    images, labels = load_random_train_images(4)

    # Preprocess whole batch
    _, previews = preprocess(images, True)

    # Create a figure for each image
    for n in range(len(labels)):
        plt.figure(f"Sample {n}")
        plt.subplot(1, 2, 1)
        plt.imshow(images[n])
        plt.axis('off')

        # Show results sector by sector
        for sector in range(5):
            plt.subplot(5, 2, 2 + 2 * sector)
            plt.imshow(previews[n, sector])
            plt.axis('off')

        # Finalize figure
        plt.tight_layout()

    # Show all figures
    plt.show()
