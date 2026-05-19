import os, cv2, numpy as np
from matplotlib import pyplot as plt
from project.scripts.dataset import load_random_test_image
from project.scripts.preprocessing.sectors import slice_sectors
from project.scripts.preprocessing import preprocess

KERNEL_DIRECTORY = os.path.join("project", "samples", "center_kernels")

KERNELS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "reverse", "skip", "draw_2", "draw_4", "wild"]

class Pattern:

    # Kernel definition
    KERNEL_SIDE = 256
    KERNEL_SIZE = (
        KERNEL_SIDE,
        KERNEL_SIDE
    )
    KERNEL_CENTER = (
        KERNEL_SIDE / 2,
        KERNEL_SIDE / 2
    )

    # Input definitions
    INPUT_HEIGHT = 256
    INPUT_WIDTH = 512
    INPUT_SIZE = (
        INPUT_WIDTH,
        INPUT_HEIGHT
        
    )
    PADDED_SIZE = (
        INPUT_HEIGHT + KERNEL_SIDE - 1,
        INPUT_WIDTH + KERNEL_SIDE - 1
    )

    # Sweep definitions
    ANGLES = np.linspace(-np.pi, np.pi, 32, endpoint=False)
    SCALES = np.linspace(0.9, 1.1, 8, endpoint=False)

    def __init__(self) -> None:
        # Allocate kernels
        self.kernels = np.zeros((
            len(KERNELS),
            Pattern.ANGLES.size,
            Pattern.SCALES.size,
            Pattern.PADDED_SIZE[0],
            Pattern.PADDED_SIZE[1] // 2 + 1
        ), dtype=np.complex64)
        # Load each kernel
        for n, kernel in enumerate(KERNELS):
            print(f"Loading kernel '{kernel}' {n + 1} / {len(KERNELS)}")
            bgr = cv2.imread(os.path.join(KERNEL_DIRECTORY, f"{kernel}.jpg"))
            # Preprocess kernels
            downscaled = cv2.resize(bgr, Pattern.KERNEL_SIZE, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(downscaled, cv2.COLOR_BGR2GRAY).astype(np.float32)
            # Sweep angles and scales
            for a, angle in enumerate(Pattern.ANGLES):
                for s, scale in enumerate(Pattern.SCALES):
                    rotation = cv2.getRotationMatrix2D(
                        Pattern.KERNEL_CENTER,
                        np.degrees(angle),
                        scale
                    )
                    transform = cv2.warpAffine(
                        gray, rotation, Pattern.KERNEL_SIZE,
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=0
                    ).astype(np.float32)  # (h, w)
                    # Normalize: zero-mean, unit-std
                    transform = (transform - transform.mean()) / (transform.std() + 1e-8)
                    # Compute kernel RFFT: (h, w) -> (pH, pW//2+1)
                    self.kernels[n, a, s] = np.fft.rfft2(transform, s=Pattern.PADDED_SIZE)

    def match(self, image: np.ndarray) -> np.ndarray:
        # Normalize: zero-mean, unit-std
        image = image.astype(np.float32)
        image = (image - image.mean()) / (image.std() + 1e-8)
        # Compute input RFFT: (H, W) -> (pH, pW//2+1)
        fft = np.fft.rfft2(image, s=Pattern.PADDED_SIZE)
        # Accumulate scores across all angles and scales
        scores = np.full(len(KERNELS), -np.inf)
        positions = np.zeros((len(KERNELS), 2), dtype=int)
        for a in range(Pattern.ANGLES.size):
            for s in range(Pattern.SCALES.size):
                kernel = self.kernels[:, a, s, :, :]  # (N, pH, pW//2+1)
                product = kernel.conj() * fft[np.newaxis, :, :]  # (N, pH, pW//2+1)
                correlation = np.fft.irfft2(product, s=Pattern.PADDED_SIZE)  # (N, pH, pW)
                flat_idx = correlation.reshape(len(KERNELS), -1).argmax(axis=-1)  # (N,)
                best_scores = correlation.reshape(len(KERNELS), -1)[np.arange(len(KERNELS)), flat_idx]  # (N,)
                improved = best_scores > scores
                scores[improved] = best_scores[improved]
                positions[improved] = np.stack(
                    np.unravel_index(flat_idx[improved], correlation.shape[1:]), axis=-1
                )

        return scores, positions

if __name__ == "__main__":
    
    # Load random train image
    print("Loading random test image")
    rgb = slice_sectors(load_random_test_image())[0]
    downscaled = cv2.resize(rgb, Pattern.INPUT_SIZE, interpolation=cv2.INTER_AREA)
    preprocessed = preprocess(downscaled)
    monochrome = np.max(preprocessed, -1)
    #_, binarized = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    # Run pattern matching
    pattern = Pattern()
    #print(pattern.match(monochrome))

    # Show kernels
    fig, axes = plt.subplots(1, len(KERNELS), figsize=(len(KERNELS) * 2, 3))
    for ax, name, kernel in zip(axes, KERNELS, pattern.kernels[:, 0, 0]):
        ax.imshow(np.fft.irfft2(kernel).real, cmap="gray")
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle("Kernels (angle=0, scale=0)")

    plt.figure()

    # Show rgb image
    plt.subplot(211)
    plt.imshow(downscaled)
    plt.axis("off")

    # Show gray image
    plt.subplot(212)
    plt.imshow(monochrome)
    plt.axis("off")

    plt.show()
