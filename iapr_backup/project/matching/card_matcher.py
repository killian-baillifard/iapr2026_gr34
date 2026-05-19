"""
ORB-based UNO card matcher. No training required.

Strategy:
  - Apply Canny edge detection to both templates and sectors to bridge the
    render-vs-photo domain gap (edges are invariant to lighting/color style).
  - Center card: match the full card's edge-ORB descriptors against the sector.
  - Player cards: use RYGB mask to narrow candidates by color, then match
    corner-crop descriptors (top-left ~30% where the number lives).
"""

import cv2
import numpy as np
from pathlib import Path

CARDS_DIR   = Path(__file__).parent.parent / "samples" / "cards"
N_FEATURES  = 2000
LOWE_RATIO  = 0.75
CANNY_LOW   = 30
CANNY_HIGH  = 120


class CardMatcher:

    def __init__(self, n_features=N_FEATURES, lowe_ratio=LOWE_RATIO):
        self.orb        = cv2.ORB_create(nfeatures=n_features)
        self.bf         = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.lowe_ratio = lowe_ratio
        self._load_templates()

    # ------------------------------------------------------------------
    # Template loading
    # ------------------------------------------------------------------

    def _load_templates(self):
        from project.scripts.dataset import Card
        self.card_names = [c.value for c in Card]
        self.templates  = {}

        for name in self.card_names:
            bgr, alpha = _load_card_png(CARDS_DIR / f"{name}.png")
            gray  = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

            h, w = gray.shape
            # Corner crop: top-left 30%×25% — contains the number/symbol
            ch, cw          = int(h * 0.30), int(w * 0.25)
            corner_edges    = edges[:ch, :cw]
            corner_mask     = alpha[:ch, :cw] if alpha is not None else None

            self.templates[name] = dict(
                des_full   = self._describe(edges,        mask=alpha),
                des_corner = self._describe(corner_edges, mask=corner_mask),
            )

    def _describe(self, edge_img, mask=None):
        _, des = self.orb.detectAndCompute(edge_img, mask)
        return des

    # ------------------------------------------------------------------
    # Core matching
    # ------------------------------------------------------------------

    def _match_score(self, des_templ, des_scene):
        if des_templ is None or des_scene is None:
            return 0
        if len(des_templ) < 2 or len(des_scene) < 2:
            return 0
        matches = self.bf.knnMatch(des_templ, des_scene, k=2)
        return sum(
            1 for pair in matches
            if len(pair) == 2
            and pair[0].distance < self.lowe_ratio * pair[1].distance
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_center(self, sector_rgb):
        """Return the predicted card name for the center sector."""
        edges = _sector_edges(sector_rgb)
        des   = self._describe(edges)
        scores = {n: self._match_score(t['des_full'], des) for n, t in self.templates.items()}
        return max(scores, key=scores.get)

    def predict_player(self, sector_rgb, rygb_sector=None, min_score=5, top_k=7):
        """Return list of predicted card names for a player sector."""
        edges      = _sector_edges(sector_rgb)
        des        = self._describe(edges)
        candidates = _filter_by_color(rygb_sector, self.card_names)
        scores     = {n: self._match_score(self.templates[n]['des_corner'], des)
                      for n in candidates}
        sorted_c   = sorted(scores.items(), key=lambda x: -x[1])
        return [n for n, s in sorted_c if s >= min_score][:top_k]

    def all_scores(self, sector_rgb, rygb_sector=None, use_corner=True):
        """Return raw score dict — useful for threshold tuning."""
        edges      = _sector_edges(sector_rgb)
        des        = self._describe(edges)
        candidates = _filter_by_color(rygb_sector, self.card_names)
        key        = 'des_corner' if use_corner else 'des_full'
        return {n: self._match_score(self.templates[n][key], des) for n in candidates}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _sector_edges(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    # Mild blur to reduce photo noise before edge detection
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)


def _load_card_png(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Card template not found: {path}")
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3]
        a     = alpha[:, :, None] / 255.0
        bgr   = (img[:, :, :3] * a + 255 * (1 - a)).astype(np.uint8)
        return bgr, alpha
    return img, None


def _filter_by_color(rygb_sector, card_names):
    """Keep only cards whose color is present in the RYGB sector."""
    if rygb_sector is None:
        return card_names
    presence  = rygb_sector.reshape(-1, 4).max(axis=0)   # (4,) RYGB
    color_map = {0: 'r', 1: 'y', 2: 'g', 3: 'b'}
    active    = {color_map[i] for i in range(4) if presence[i] > 15}
    return [
        n for n in card_names
        if n.split('_')[0] in active or n in ('draw_4', 'wild')
    ]
