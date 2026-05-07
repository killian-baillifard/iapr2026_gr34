import cv2
import numpy as np
from dataset import Player, load_random_train_image
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

    def slice(self, image: np.ndarray) -> np.ndarray:
        """
        Input:
            Image (height, width, rgb)
        Output:
            Sectors (sector, sector_height, sector_width, rgb)
        """
        return image[self.y:(self.y + self.height), self.x:(self.x + self.width), :]

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

def slice_sectors(image: np.ndarray) -> np.ndarray:
    """
    Input:
        Image (height, width, rgb)
    Output:
        Sectors (sector, sector_height, sector_width, rgb)
    """
    center_sector = SECTORS["center"].slice(image)
    player_1_sector = SECTORS[str(Player.P1)].slice(image)
    player_2_sector = SECTORS[str(Player.P2)].slice(image)
    player_3_sector = SECTORS[str(Player.P3)].slice(image)
    player_4_sector = SECTORS[str(Player.P4)].slice(image)

    player_2_sector = np.rot90(player_2_sector, k=-1)
    player_3_sector = np.rot90(player_3_sector, k=2)
    player_4_sector = np.rot90(player_4_sector, k=1)

    return np.array([
        center_sector,
        player_1_sector,
        player_2_sector,
        player_3_sector,
        player_4_sector
    ]).reshape((5, SECTOR_HEIGHT, SECTOR_WIDTH, 3))

class HSV:

    def __init__(self, h: int, s: int, v: int) -> None:
        self.h = np.clip(h, 0, 180)
        self.s = np.clip(s, 0, 255)
        self.v = np.clip(v, 0, 255)

# Mean colors
COLORS: dict[str, HSV] = {
    "R": HSV(1,     184,    239),
    "Y": HSV(25,    215,    247),
    "G": HSV(56,    131,    185),
    "B": HSV(98,    211,    217),
    "K": HSV(17,    46,     43)
}

# Tolerances in multiple of standard deviation
TOLERANCES: dict[str, HSV] = {
    "R": HSV(3 * 1,     6 * 16,     6 * 14),
    "Y": HSV(3 * 1,     6 * 33,     6 * 8),
    "G": HSV(4 * 4,     6 * 22,     6 * 13),
    "B": HSV(3 * 2,     6 * 50,     6 * 13),
    "K": HSV(180,       64,         3 * 14)
}

def segment_color(image: np.ndarray, color: HSV, tolerance: HSV) -> np.ndarray:
    """
    Input:
        Image (height, width, rgb)
        Color (hsv)
        Tolerance (hsv)
    Output:
        Masked image (height, width, rgb)
        Color mask (height, width)
    """

    # Compute color mask
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower = np.array([max(color.h - tolerance.h, 0), max(color.s - tolerance.s, 0), max(color.v - tolerance.v, 0)])
    upper = np.array([min(color.h + tolerance.h, 179), min(color.s + tolerance.s, 255), min(color.v + tolerance.v, 255)])
    mask = cv2.inRange(hsv, lower, upper)

    # Wrap hue around 0 <=> 180
    if color.h <= 2 * tolerance.h:
        wraped_lower = np.array([max(179 - tolerance.h, 0), max(color.s - tolerance.s, 0), max(color.v - tolerance.v, 0)])
        wraped_upper = np.array([179, min(color.s + tolerance.s, 255), min(color.v + tolerance.v, 255)])
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, wraped_lower, wraped_upper))
    
    masked = cv2.bitwise_and(image, image, mask = mask)
    return masked, mask

def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Input:
        Image (height, width, rgb)
    Output:
        Preprocessed image (height, width, rygb)
    """

    height, width = image.shape[:2]
    masked_r, r_mask = segment_color(image, COLORS["R"], TOLERANCES["R"])
    masked_y, y_mask = segment_color(image, COLORS["Y"], TOLERANCES["Y"])
    masked_g, g_mask = segment_color(image, COLORS["G"], TOLERANCES["G"])
    masked_b, b_mask = segment_color(image, COLORS["B"], TOLERANCES["B"])
    masked_k, k_mask = segment_color(image, COLORS["K"], TOLERANCES["K"])

    reversed_k = 255 - masked_k
    preview = masked_r + masked_y + masked_g + masked_b + cv2.bitwise_and(reversed_k, reversed_k, mask = k_mask)
    preprocessed = np.array([r_mask + k_mask, y_mask + k_mask, g_mask + k_mask, b_mask + k_mask]).reshape(height, width, 4)

    return preprocessed, preview

if __name__ == "__main__":

    # Load random train sample
    image, label = load_random_train_image()
    plt.figure(str(label))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.axis('off')

    # Split sectors
    sectors = slice_sectors(image)
    for i in range(5):
        plt.subplot(5, 3, 2 + 3 * i)
        plt.imshow(sectors[i])
        plt.axis('off')

        # Apply color segmentation on sector
        _, preview = preprocess(sectors[i])
        plt.subplot(5, 3, 3 + 3 * i)
        plt.imshow(preview)
        plt.axis('off')
    
    plt.show()
