import os
import cv2
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from dataset import load_random_train_images, PARENT_PATH
from preprocessing import preprocess, preview

class CardFilter:
    def __init__(self, angle_step: int = 30):
        MASK_IMAGE = cv2.imread(os.path.join(PARENT_PATH, "manual_segmentation", "card_mask.png"))
        MASK_IMAGE = cv2.cvtColor(MASK_IMAGE, cv2.COLOR_RGB2GRAY)
        _, mask_bin = cv2.threshold(MASK_IMAGE, 127, 255, cv2.THRESH_BINARY)

        # Precompute rotated versions of the mask
        self.templates = []
        h, w = mask_bin.shape
        center = (w // 2, h // 2)
        for angle in range(0, 180, angle_step):
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(mask_bin, M, (w, h),
                                     flags=cv2.INTER_NEAREST,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0)
            self.templates.append(rotated)

        print(f"Mask shape: {mask_bin.shape}, "
            f"unique values: {np.unique(mask_bin)}, "
            f"white pixels: {np.count_nonzero(mask_bin)}")
        print(f"Loaded {len(self.templates)} rotated templates")

    def _match_channel(self, channel: np.ndarray) -> np.ndarray:
        h, w = channel.shape
        response = np.full((h, w), -1.0, dtype=np.float32)

        for template in self.templates:
            th, tw = template.shape
            result = cv2.matchTemplate(channel, template, cv2.TM_CCOEFF_NORMED)
            # result top-left maps to template top-left corner in image
            # resize back to full image via upsampling to avoid offset confusion
            result_full = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
            np.maximum(response, result_full, out=response)

        # Apply as soft mask: keep original pixels weighted by correlation score
        response = np.clip(response, 0, None)  # [-1,1] -> [0,1]
        return (channel.astype(np.float32) * response).astype(np.float32)

    def filter(self, images: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        images : np.ndarray
            RYGB preprocessed images (n, sectors, height, width, 4)
        Returns
        -------
        filtered : np.ndarray
            Correlation maps (n, sectors, height, width, 4), float32 in [-1, 1]
            High values = likely UNO card location
        """
        n, sectors, height, width, channels = images.shape
        flat = images.reshape(-1, height, width, channels)  # (n*sectors, H, W, 4)
        total = flat.shape[0] * channels

        results = Parallel(n_jobs=-1)(
            delayed(self._match_channel)(flat[i, :, :, c])
            for i in range(flat.shape[0])
            for c in range(channels)
        )

        filtered = np.stack(results).reshape(n, sectors, channels, height, width)
        filtered = filtered.transpose(0, 1, 3, 4, 2)
        return filtered

if __name__ == "__main__":

    # Load images
    N = 1
    images, labels = load_random_train_images(N)

    # Preprocess images
    preprocessed = preprocess(images)
    preprocessed_preview = preview(preprocessed)

    # Filter images
    card_filter = CardFilter()
    filtered = card_filter.filter(preprocessed)
    filtered_preview = preview(filtered)

    # Create a figure for each image
    for n in range(len(labels)):
        plt.figure(f"Sample {n}")

        # Plot original image
        plt.subplot(1, 3, 1)
        plt.imshow(images[n])
        plt.axis('off')

        # Show results sector by sector
        for sector in range(5):

            # Show preprocessed images
            plt.subplot(5, 3, 2 + 3 * sector)
            plt.imshow(preprocessed_preview[n, sector])
            plt.axis('off')

            # Show filtered images
            plt.subplot(5, 3, 3 + 3 * sector)
            plt.imshow(filtered_preview[n, sector])
            plt.axis('off')

        # Finalize figure
        plt.tight_layout()

    # Show all figures
    plt.show()
