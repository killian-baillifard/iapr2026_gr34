"""
Detect individual card crops from a RYGB sector using contour detection.

For each color channel, threshold → morphological clean → find contours →
extract bounding-box crops. Returns RYGB crops ready for the CropClassifier.
"""

import cv2
import numpy as np

CROP_SIZE        = (224, 224)
MIN_CARD_PIXELS  = 4000    # filter tiny noise blobs
MAX_N_PER_COLOR  = 1       # take only the largest crop per color — avoids duplicates
THRESHOLD        = 25      # RYGB channel value to consider "active"
MIN_COVERAGE     = 0.01    # color channel must cover at least 1% of sector pixels


def detect_crops(rygb_sector: np.ndarray, crop_size=CROP_SIZE) -> list[tuple[np.ndarray, str]]:
    """
    rygb_sector : (H, W, 4) uint8
    returns     : list of (crop float32 [0,1] shape crop_size+(4,), color_char 'r'|'y'|'g'|'b')
    """
    H, W   = rygb_sector.shape[:2]
    crops  = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    for c_idx, color_name in enumerate(['r', 'y', 'g', 'b']):
        channel = rygb_sector[:, :, c_idx]
        if int(channel.max()) < THRESHOLD:
            continue
        # require meaningful coverage to avoid bleeding from adjacent sectors
        if (channel > THRESHOLD).sum() < MIN_COVERAGE * H * W:
            continue

        _, binary = cv2.threshold(channel, THRESHOLD, 255, cv2.THRESH_BINARY)
        binary    = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary    = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours    = sorted(contours, key=cv2.contourArea, reverse=True)

        n = 0
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CARD_PIXELS:
                break
            x, y, w, h = cv2.boundingRect(cnt)
            # small padding for context
            x1 = max(0, x - 8);  y1 = max(0, y - 8)
            x2 = min(W, x+w+8);  y2 = min(H, y+h+8)
            crop = rygb_sector[y1:y2, x1:x2, :]
            if crop.size == 0:
                continue
            crop_r = cv2.resize(crop, crop_size).astype(np.float32) / 255.0
            crops.append((crop_r, color_name))
            n += 1
            if n >= MAX_N_PER_COLOR:
                break

    return crops
