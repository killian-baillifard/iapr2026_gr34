import numpy as np
import cv2
import os
from cv2.typing import MatLike
from matplotlib import pyplot as plt
from project.scripts.dataset import PARENT_PATH, CARD_LOOKUP, CARDS_COUNT, Card
from project.scripts.preprocessing.sectors import SECTOR_WIDTH, SECTOR_HEIGHT
from scipy.stats import truncnorm

SAMPLES_PATH = os.path.join(PARENT_PATH, "samples")
BACKGROUNDS_PATH = os.path.join(SAMPLES_PATH, "backgrounds")
CARDS_PATH = os.path.join(SAMPLES_PATH, "cards")
TOKENS_PATH = os.path.join(SAMPLES_PATH, "tokens")

gray_background_path = lambda id: os.path.join(BACKGROUNDS_PATH, f"gray_{id}.png")
flower_background_path = lambda id: os.path.join(BACKGROUNDS_PATH, f"flower_{id}.png")
card_path = lambda id: os.path.join(CARDS_PATH, f"{id}.png")
black_token_path = lambda id: os.path.join(TOKENS_PATH, f"k_{id}.png")
yellow_token_path = lambda id: os.path.join(TOKENS_PATH, f"y_{id}.png")

def centered_truncated_normal(center: float, spread: float, std: float = 100) -> float:
    """
    Return clamped normal distribution

    Parameters
    ----------

    center : float
        Distribution center

    spread : float
        Distribution one-sided edges

    std : float
        Shape, 100 is standard, 50 is tight, 200 is wide
    """

    a = -spread / std
    return truncnorm.rvs(a, -a, loc=center, scale=std)

class Synthesizer:

    def __init__(self) -> None:

        # Load backgrounds
        self.gray_backgrounds = [cv2.imread(gray_background_path(i), cv2.IMREAD_UNCHANGED) for i in range(1, 5)]
        self.flower_backgrounds = [cv2.imread(flower_background_path(i), cv2.IMREAD_UNCHANGED) for i in range(1, 5)]

        # Load cards
        self.cards = [cv2.imread(card_path(str(card)), cv2.IMREAD_UNCHANGED) for card in CARD_LOOKUP]

        # Load tokens
        self.black_tokens = [cv2.imread(black_token_path(i), cv2.IMREAD_UNCHANGED) for i in range(1, 5)]
        self.yellow_tokens = [cv2.imread(yellow_token_path(i), cv2.IMREAD_UNCHANGED) for i in range(1, 5)]

        # Allocate canvas
        self.canvas = np.zeros((SECTOR_HEIGHT, SECTOR_WIDTH, 4))

    def alpha_blend(self, overlay: np.ndarray, x: int, y: int, angle: float = 0.0) -> None:

        # Rotate image
        if angle != 0.0:
            h, w = overlay.shape[:2]
            cx, cy = w / 2, h / 2
            M = cv2.getRotationMatrix2D((cx, cy), -angle, scale=1.0)
            cos, sin = np.abs(M[0, 0]), np.abs(M[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            M[0, 2] += (new_w - w) / 2
            M[1, 2] += (new_h - h) / 2
            overlay = cv2.warpAffine(overlay, M, (new_w, new_h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))

        # Clamp overlay over canvas
        oh, ow = overlay.shape[:2]
        bh, bw = self.canvas.shape[:2]
        x1, y1 = max(x - ow // 2, 0), max(y - oh // 2, 0)
        x2, y2 = min(x + ow // 2, bw), min(y + oh // 2, bh)
        ox1, oy1 = x1 - (x - ow // 2), y1 - (y - oh // 2)
        ox2, oy2 = ox1 + (x2 - x1), oy1 + (y2 - y1)
        if x2 <= x1 or y2 <= y1:
            return
        
        # Compute alpha blending
        bg = self.canvas[y1:y2, x1:x2].astype(np.float64) / 255
        fg = overlay[oy1:oy2, ox1:ox2].astype(np.float64) / 255
        alpha_fg = fg[:, :, 3:4]
        alpha_bg = bg[:, :, 3:4]
        alpha_out = alpha_fg + alpha_bg * (1 - alpha_fg)
        rgb_out   = (fg[:, :, :3] * alpha_fg + bg[:, :, :3] * alpha_bg * (1 - alpha_fg)) / np.clip(alpha_out, 1e-6, 1)

        # Write canvas
        self.canvas[y1:y2, x1:x2, :3] = (rgb_out   * 255).astype(np.uint8)
        self.canvas[y1:y2, x1:x2, 3:] = (alpha_out * 255).astype(np.uint8)

    def generate(self) -> tuple[MatLike, np.ndarray]:
        """
        Returns
        -------

        image : MatLike

        label : np.ndarray
        """

        # Select random background and sector and initialze new canvas
        flower = np.random.randint(0, 2)
        sector = np.random.randint(0, 4)
        self.canvas = self.flower_backgrounds[sector].copy() if flower else self.gray_backgrounds[sector].copy()

        # Select random number of cards
        nb_cards = np.random.randint(0, 5)
        label = np.zeros(54)
        if nb_cards:

            # Select random cards
            indices = [np.random.randint(0, CARDS_COUNT) for _ in range(nb_cards)]
            label[indices] = 1.0

            # Select random horizontal, vertical and angle cards centerline placement
            centerline_x = centered_truncated_normal(SECTOR_WIDTH / 2, 100)
            centerline_y = centered_truncated_normal(SECTOR_HEIGHT / 2, 50)
            centerline_angle = centered_truncated_normal(0, np.deg2rad(15))

            # Select if cards are stacked and randomize direction
            stacked_cards = np.random.randint(0, 2)
            stride = centered_truncated_normal(150, 50) if stacked_cards else centered_truncated_normal(380, 20)
            direction = 1 if np.random.randint(0, 2) else -1
            x_stride = direction * stride * np.cos(centerline_angle)
            y_stride = direction * stride * np.sin(centerline_angle)

            # Overlay cards on image
            x = centerline_x - (nb_cards - 1) * x_stride / 2
            y = centerline_y + (nb_cards - 1) * y_stride / 2
            for index in indices:

                # Randomize card placement over centerline
                card_x = int(centered_truncated_normal(x, 10))
                card_y = int(centered_truncated_normal(y, 10))
                card_angle = centered_truncated_normal(centerline_angle, 45 if stacked_cards else 10)
                self.alpha_blend(self.cards[index], card_x, card_y, card_angle)

                # Increment centerline position
                x += x_stride
                y -= y_stride

            # Overlay token on image with random placement
            token_x = int(centered_truncated_normal(0.95 * SECTOR_WIDTH, 50))
            token_y = int(centered_truncated_normal(SECTOR_HEIGHT / 4, 100))
            token = self.yellow_tokens[sector] if flower else self.black_tokens[sector]
            self.alpha_blend(token, token_x, token_y, 0.0)

        # Convert to RGB and return result
        return cv2.cvtColor(self.canvas.copy(), cv2.COLOR_BGRA2RGB), label

if __name__ == "__main__":

    # Synthesize N image
    N = 5
    synthesizer = Synthesizer()
    for i in range(N):

        # Generate new image
        image, label = synthesizer.generate()
        plt.figure(f"Synthesized image {i}")
        
        # Plot image
        plt.subplot(211)
        plt.imshow(image)
        plt.axis("off")

        # Plot labels
        plt.subplot(212)
        plt.bar(np.arange(CARDS_COUNT), label, width=0.6)
        plt.xlim(-0.5, CARDS_COUNT - 0.5)
        plt.ylim(0, 1)
        plt.yticks([])
        plt.xticks(np.arange(CARDS_COUNT), [str(c) for c in Card], rotation=90, fontsize=8)
        
        # Finish plot
        plt.tight_layout()

    # Show plot
    plt.show()
