import numpy as np

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

def filter_color(image: np.ndarray, component: str) -> np.ndarray:
    """
    Parameters
    ----------

    image : np.ndarray
        HSV image (height, width, 3)
    
    component: str
        Color code
    
    Returns
    -------

    progressive_mask
        (height, width) with 255 at perfect match, 0 at tolerance boundary
    """

    # First pass
    center = np.array([COLORS[component].h, COLORS[component].s, COLORS[component].v], dtype=np.float32)
    tolerance = np.array([TOLERANCES[component].h, TOLERANCES[component].s, TOLERANCES[component].v], dtype=np.float32)
    distance = np.abs(image.astype(np.float32) - center) / tolerance

    # Second pass, handle hue wrap around
    if COLORS[component].h <= 2 * TOLERANCES[component].h:
        hue_distance = image[:, :, 0].astype(np.float32)
        wrapped_dist = np.abs(hue_distance - 180 - COLORS[component].h) / TOLERANCES[component].h  # distance via the wrap
        distance[:, :, 0] = np.minimum(distance[:, :, 0], wrapped_dist)

    # Compute distance score
    score = 1.0 - np.max(distance, axis=-1)
    score = np.clip(score, 0.0, 1.0)
    progressive_mask = (score * 255).astype(np.uint8)
    return progressive_mask

def hsv2rygb(hsv: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------

    images : np.ndarray
        HSV image (height, width, 3)
    
    Returns
    -------

    image : np.ndarray
        RYGB image (height, width, 4)
    """

    # Apply color segmentation on all sectors
    r = filter_color(hsv, "red")
    y = filter_color(hsv, "yellow")
    g = filter_color(hsv, "green")
    b = filter_color(hsv, "blue")
    k = filter_color(hsv, "black")

    # Add black component to all channels as the white of this new color space
    r = np.clip(r.astype(np.int16) + k.astype(np.int16), 0, 255).astype(np.uint8)
    y = np.clip(y.astype(np.int16) + k.astype(np.int16), 0, 255).astype(np.uint8)
    g = np.clip(g.astype(np.int16) + k.astype(np.int16), 0, 255).astype(np.uint8)
    b = np.clip(b.astype(np.int16) + k.astype(np.int16), 0, 255).astype(np.uint8)

    # Recombine and return components
    return np.stack([r, y, g, b], axis=-1)

def rygb2rgb(rygb: np.ndarray) -> np.ndarray:
    """
    Parameters
    ----------

    rygb : np.ndarray
        RYGB image (height, width, 4)
    
    Returns
    -------

    rgb : np.ndarray
        RGB image (height, width, 3)
    """

    # Create conversion dictionnary
    RGB = {
        "R": np.array([255,   0,   0], dtype=np.float32),
        "Y": np.array([255, 255,   0], dtype=np.float32),
        "G": np.array([  0, 255,   0], dtype=np.float32),
        "B": np.array([  0,   0, 255], dtype=np.float32)
    }

    # Split components
    r = rygb[:, :, 0]
    y = rygb[:, :, 1]
    g = rygb[:, :, 2]
    b = rygb[:, :, 3]
    masks = [r, y, g, b]

    # Sum components and return conversion
    rgb = sum(color * mask[:, :, np.newaxis] / 255 for mask, color in zip(masks, RGB.values()))
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb
