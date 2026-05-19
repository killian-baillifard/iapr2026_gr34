import numpy as np
import cv2
from skimage.morphology import closing, disk
from matplotlib import pyplot as plt
from project.scripts.dataset import CARDS_COUNT, Card
from project.scripts.preprocessing.rygb import rygb2rgb
from project.scripts.preprocessing.cache import load_labels

def fourier_descriptors(contour: np.ndarray, n_descriptors: int = 32) -> np.ndarray:
    """
    Parameters
    ----------
    contour       : (N, 1, 2) array as returned by cv2.findContours
    n_descriptors : how many coefficients to keep (truncate high frequencies)

    Returns
    -------
    descriptors : (n_descriptors,) real-valued, translation/scale/rotation invariant
    """
    # Flatten (N, 1, 2) → (N, 2) then encode as complex numbers
    pts     = contour[:, 0, :].astype(np.float64)          # (N, 2)
    complex_signal = pts[:, 0] + 1j * pts[:, 1]            # (N,) complex

    # FFT of the contour signal
    coeffs = np.fft.fft(complex_signal)                     # (N,) complex

    # --- Invariances ---
    # Translation: zero out the DC component (index 0)
    coeffs[0] = 0

    # Truncate to n_descriptors (keep low frequencies = coarse shape)
    #coeffs = coeffs[:n_descriptors]

    # Scale + rotation: normalize by the magnitude of the first AC coefficient
    coeffs /= np.abs(coeffs[1])

    # Return magnitudes only → starting-point invariance
    phase_offset = np.angle(coeffs[1])
    coeffs *= np.exp(-1j * phase_offset)

    return coeffs

if __name__ == "__main__":

    # Load data and select random sample
    paths, labels = load_labels()
    i = np.random.randint(0, labels.shape[0])
    path = paths[i]
    label = labels[i]
    preprocessed = np.array(np.load(path))

    # Load, binarize and (morphologically) close image
    binarized = preprocessed > 0
    footprint = disk(2)
    closed = np.zeros_like(binarized)
    for i in range(4):
        closed[:, :, i] = closing(binarized[:, :, i], footprint)
    closed = (255 * closed).astype(np.uint8)

    # Create image preview
    preview = rygb2rgb(closed)

    # For each channel component
    all_descriptors = []  # collect across channels if needed
    for c in range(closed.shape[-1]):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed[:, :, c], connectivity=8)
        channel_contours = []

        for label_id in range(1, num_labels):
            mask = np.uint8(labels == label_id) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            channel_contours.extend(contours)

        for contour in channel_contours:
            if len(contour) < 16:          # too few points → unstable FFT, skip
                continue
            fd = fourier_descriptors(contour, n_descriptors=32)
            all_descriptors.append(fd)

        cv2.drawContours(preview, channel_contours, -1, (255, 255, 255), 20)

    # Show image and label
    plt.figure()

    # Plot preview with overlayed contours
    plt.subplot(121)
    plt.imshow(preview)
    plt.axis("off")

    # Plot labels
    #plt.subplot(122)
    #plt.bar(np.arange(CARDS_COUNT), label, width=0.6)
    #plt.xlim(-0.5, CARDS_COUNT - 0.5)
    #plt.ylim(0, 1)
    #plt.yticks([])
    #plt.xticks(np.arange(CARDS_COUNT), [str(c) for c in Card], rotation=90, fontsize=8)

    # Plot fourier descriptors real part
    plt.subplot(222)
    for fd in all_descriptors:
        plt.plot(np.real(fd))

    # Plot fourier descriptors imaginary part
    plt.subplot(224)
    for fd in all_descriptors:
        plt.plot(np.imag(fd))

    plt.show()
