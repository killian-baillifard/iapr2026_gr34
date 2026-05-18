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

class Pattern:

    # Kernel definition
    KERNEL_SIDE     = 256
    KERNEL_SIZE     = (
        KERNEL_SIDE,
        KERNEL_SIDE
    )
    KERNEL_CENTER = (
        KERNEL_SIDE / 2,
        KERNEL_SIDE / 2
    )

    # Input definitions
    INPUT_HEIGHT    = 256
    INPUT_WIDTH     = 512
    PADDED_SIZE     = (
        INPUT_HEIGHT + KERNEL_SIDE - 1,
        INPUT_WIDTH + KERNEL_SIDE - 1
    )

    # Sweep definitions
    ANGLES = np.linspace(-np.pi, np.pi, 32, endpoint=False)
    SCALES = np.linspace(0.9, 1.1, 8, endpoint=False)

    def __init__(self) -> None:

        # Allocate kernels
        self.kernels = np.zeros((
            CARDS_COUNT,
            len(Pattern.ANGLES),
            len(Pattern.SCALES),
            Pattern.PADDED_SIZE[1],
            Pattern.PADDED_SIZE[0],
            4
        ), dtype=np.complex64)

        # Load each kernel
        for n, card in enumerate(list(Card)):
            bgra = cv2.imread(os.path.join(CARDS_DIRECTORY, f"{card}.png"), cv2.IMREAD_UNCHANGED)

            # Preprocess kernels
            downscaled = cv2.resize(bgra, Pattern.KERNEL_SIZE, interpolation=cv2.INTER_AREA)
            rygb = hsv2rygb(cv2.cvtColor(bgra_to_rgb_white_bg(downscaled), cv2.COLOR_RGB2HSV))

            # Sweep angles and scales
            for a, angle in enumerate(Pattern.ANGLES):
                for s, scale in enumerate(Pattern.SCALES):

                    # Compute linear transform matrix for rotation and scaling
                    rotation = cv2.getRotationMatrix2D(
                        self.KERNEL_CENTER,
                        np.degrees(angle),
                        scale
                    )
                    transform = np.stack([
                        cv2.warpAffine(
                            rygb[:, :, c],
                            rotation,
                            Pattern.KERNEL_SIZE,
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0
                        )
                        for c in range(4)
                    ], axis=-1).astype(np.float32)

                    # Normalize channels
                    for c in range(4):
                        ch = transform[:, :, c]
                        transform[:, :, c] = (ch - ch.mean()) / (ch.std() + 1e-8)
                    
                    # Compute kernel RFFT
                    fft = np.fft.rfft2(transform.transpose(2, 0, 1), Pattern.PADDED_SIZE)
                    self.kernels[n, a, s] = fft.transpose(1, 2, 0)

    def match(self, rygb_image: np.ndarray) -> np.ndarray:

        # Normalize channels
        image = rygb_image.astype(np.float32)
        for c in range(4):
            ch = image[:, :, c]
            image[:, :, c] = (ch - ch.mean()) / (ch.std() + 1e-8)

        # Compute input RFFT
        image_chw = image.transpose(2, 0, 1)
        fft = np.fft.rfft2(image_chw, s=Pattern.PADDED_SIZE)

        # Accumulate scores across all angles, scales and channels
        scores = np.zeros(52)
        for a in range(len(self.angles)):
            for s in range(len(self.scales)):
                kernel = self.kernels[:, a, s, :, :, :]

                # Batched normalized cross-correlation
                product = kernel.conj() * fft[np.newaxis, :, :, :]

                # IRFFT back to spatial domain
                correlation = np.fft.irfft2(product, s=Pattern.PADDED_SIZE)

                # Max over spatial positions, mean over channels
                scores += correlation.reshape(52, 4, -1).max(axis=-1).mean(axis=-1)

        # Normalize by number of angles and scales and return
        scores /= (Pattern.ANGLES.size * Pattern.SCALES.size)
        return scores

if __name__ == "__main__":

    print("Loading patterns")
    pattern_matcher = Pattern()
    
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
