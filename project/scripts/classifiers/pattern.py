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
        # Load patterns and precompute rotated FFTs
        # We need to know the padded shape first — defer until match() sees the image,
        # OR fix a canonical padded shape here if image size is known in advance.
        # We'll store rotated templates and compute FFTs lazily on first match() call.

        self.angles = np.linspace(-np.pi, np.pi, 32, endpoint=False)
        self.scales = np.linspace(0.9, 1.1, 8, endpoint=False)

        # rotated_templates[angle_idx][scale_idx] -> (52, H, W, 4) normalized array
        self.rotated_templates = []

        patterns = []
        for card in list(Card):
            bgra = cv2.imread(os.path.join(CARDS_DIRECTORY, f"{card}.png"), cv2.IMREAD_UNCHANGED)
            w = bgra.shape[1] // 4
            h = bgra.shape[0] // 4
            downscaled = cv2.resize(bgra, (w, h), interpolation=cv2.INTER_AREA)
            rygb = hsv2rygb(cv2.cvtColor(bgra_to_rgb_white_bg(downscaled), cv2.COLOR_RGB2HSV))
            patterns.append(rygb)

        # patterns: list of 52 arrays, each (h, w, 4)
        template_h, template_w = patterns[0].shape[:2]

        # Precompute and normalize all rotated templates
        # Shape after stacking: (52, h, w, 4)
        for angle in self.angles:
            angle_list = []
            for scale in self.scales:
                cx, cy = template_w / 2, template_h / 2
                M = cv2.getRotationMatrix2D((cx, cy), np.degrees(angle), scale)
                rotated_cards = []
                for pattern in patterns:
                    rotated = np.stack([
                        cv2.warpAffine(
                            pattern[:, :, c], M, (template_w, template_h),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0
                        )
                        for c in range(4)
                    ], axis=-1).astype(np.float32)  # (h, w, 4)

                    # Normalize each channel: zero-mean, unit-std
                    for c in range(4):
                        ch = rotated[:, :, c]
                        rotated[:, :, c] = (ch - ch.mean()) / (ch.std() + 1e-8)

                    rotated_cards.append(rotated)

                # Stack to (52, h, w, 4)
                angle_list.append(np.stack(rotated_cards, axis=0))
            self.rotated_templates.append(angle_list)

        self.template_shape = (template_h, template_w)
        self._F_templates = None  # populated on first match() call

    def _precompute_ffts(self, padded_shape):
        """Precompute FFTs of all rotated templates for the given padded shape."""
        # F_templates[angle_idx][scale_idx] -> (52, 4, pH, pW//2+1) complex array
        self._F_templates = []
        for angle_idx in range(len(self.angles)):
            angle_list = []
            for scale_idx in range(len(self.scales)):
                # (52, h, w, 4) -> (52, 4, h, w) for rfft2 over last two axes
                templates = self.rotated_templates[angle_idx][scale_idx]
                templates_chw = templates.transpose(0, 3, 1, 2)  # (52, 4, h, w)
                # rfft2 over spatial dims, zero-padded to padded_shape
                F = np.fft.rfft2(templates_chw, s=padded_shape)  # (52, 4, pH, pW//2+1)
                angle_list.append(F)
            self._F_templates.append(angle_list)
        self._padded_shape = padded_shape

    def match(self, rygb_image: np.ndarray) -> np.ndarray:
        image_h, image_w = rygb_image.shape[:2]
        template_h, template_w = self.template_shape
        padded_shape = (image_h + template_h - 1, image_w + template_w - 1)

        # Normalize image globally: per-channel zero-mean unit-std
        image = rygb_image.astype(np.float32)
        for c in range(4):
            ch = image[:, :, c]
            image[:, :, c] = (ch - ch.mean()) / (ch.std() + 1e-8)

        # Compute image FFTs: (4, pH, pW//2+1)
        F_image = np.fft.rfft2(image.transpose(2, 0, 1), s=padded_shape)

        scores = np.full(len(self.rotated_templates[0][0]), -np.inf)

        for angle_idx in range(len(self.angles)):
            for scale_idx in range(len(self.scales)):
                # Recompute template FFTs on the fly: (52, h, w, 4) -> (52, 4, pH, pW//2+1)
                templates = self.rotated_templates[angle_idx][scale_idx]
                F_templates = np.fft.rfft2(templates.transpose(0, 3, 1, 2), s=padded_shape)

                # Batched correlation
                product = F_templates.conj() * F_image[None, :, :, :]  # (52, 4, pH, pW//2+1)
                corr = np.fft.irfft2(product, s=padded_shape)           # (52, 4, pH, pW)

                # Max over spatial positions, mean over channels
                best = corr.max(axis=(-2, -1)).mean(axis=-1)  # max over spatial, mean over channels
                scores = np.maximum(scores, best)

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
