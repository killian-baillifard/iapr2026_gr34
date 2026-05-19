import os, cv2, numpy as np
from matplotlib import pyplot as plt
from project.scripts.dataset import Card, CARDS_COUNT
from project.scripts.dataset.synthesizer import CARDS_DIRECTORY
from project.scripts.preprocessing.cache import Cache, load_labels, load_image
from project.scripts.preprocessing.rygb import hsv2rygb, rygb2rgb

def bgra_to_rgb_white_bg(image):
    bgr = image[:, :, :3]
    alpha = image[:, :, 3]
    white_bg = np.ones_like(bgr, dtype=np.uint8) * 255
    alpha_factor = alpha[:, :, np.newaxis] / 255.0
    composited = (bgr * alpha_factor + white_bg * (1 - alpha_factor)).astype(np.uint8)
    return cv2.cvtColor(composited, cv2.COLOR_BGR2RGB)

class PatternMatcher:

    def __init__(self) -> None:
        
        # Load patterns
        self.patterns = []
        for card in list(Card):
            bgra = cv2.imread(os.path.join(CARDS_DIRECTORY, f"{card}.png"), cv2.IMREAD_UNCHANGED)
            w = bgra.shape[1] // 4
            h = bgra.shape[0] // 4
            downscaled = cv2.resize(bgra, (w, h), interpolation=cv2.INTER_AREA)
            rygb = hsv2rygb(cv2.cvtColor(bgra_to_rgb_white_bg(downscaled), cv2.COLOR_RGB2HSV))
            self.patterns.append(rygb)

    def match(self, rygb_image: np.ndarray) -> np.ndarray:

        # Compute match score for each pattern
        scores = np.zeros(len(self.patterns))
        for i, pattern in enumerate(self.patterns):
            for c in range(4):
                pattern_channel = pattern[:, :, c]
                image_channel = rygb_image[:, :, c]

                # Sweep across all rotations and small scalings
                score = 0
                for angle in np.linspace(-np.pi, np.pi, 32, endpoint=False):
                    for scale in np.linspace(0.9, 1.1, 8, endpoint=False):

                        # Rotate the pattern around its center
                        h, w = pattern_channel.shape
                        cx, cy = w / 2, h / 2
                        M = cv2.getRotationMatrix2D((cx, cy), np.degrees(angle), scale)
                        rotated = cv2.warpAffine(
                            pattern_channel, M, (w, h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0
                        )

                        # Match the rotated pattern against the image channel
                        result = cv2.matchTemplate(
                            image_channel.astype(np.float32),
                            rotated.astype(np.float32),
                            cv2.TM_CCOEFF_NORMED  # Score in [-1, 1], 1 = perfect match
                        )

                        # Keep best match score across all positions
                        score = np.max([score, np.max(result)])

                scores[i] += score / 4

        return scores

if __name__ == "__main__":

    print("Loading patterns")
    pattern_matcher = PatternMatcher()
    
    # Load random train image
    print("Loading random train image")
    labels = load_labels(Cache.VALIDATION)
    i = np.random.randint(0, labels.shape[0])
    rygb_image = load_image(Cache.VALIDATION, i)
    label = labels[i]

    # Run predictor
    print("Computing predictions")
    predictions = pattern_matcher.match(rygb_image)

    # Plot probabilities for each card
    plt.figure()
    plt.subplot(121)
    plt.imshow(rygb2rgb(rygb_image))
    plt.axis("off")

    plt.subplot(222)
    plt.bar(np.arange(CARDS_COUNT), label, width=0.6)
    plt.xticks(np.arange(CARDS_COUNT), [str(c) for c in Card], rotation=90, fontsize=8)

    plt.subplot(224)
    plt.bar(np.arange(CARDS_COUNT), predictions, width=0.6)
    plt.xticks(np.arange(CARDS_COUNT), [str(c) for c in Card], rotation=90, fontsize=8)
    plt.tight_layout()

    plt.show()
