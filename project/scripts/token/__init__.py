import cv2
import numpy as np
from project.scripts.dataset import WIDTH, HEIGHT, Player
from project.scripts.preprocessing.rygb import filter_color
from project.scripts.preprocessing.sectors import SECTOR_SRC_WIDTH, SECTOR_SRC_HEIGHT
from matplotlib import pyplot as plt

# Player zone centers
PLAYER_CENTERS = {
    Player.P1: (WIDTH // 2,                  HEIGHT - SECTOR_SRC_HEIGHT // 2),
    Player.P2: (WIDTH  - SECTOR_SRC_HEIGHT // 2, HEIGHT // 2),
    Player.P3: (WIDTH // 2,                  SECTOR_SRC_HEIGHT // 2),
    Player.P4: (SECTOR_SRC_HEIGHT // 2,          HEIGHT // 2),
}

PLAYER_ZONES = {
    Player.P1: (WIDTH//2 - SECTOR_SRC_WIDTH//2,  HEIGHT - SECTOR_SRC_HEIGHT,
                WIDTH//2 + SECTOR_SRC_WIDTH//2,  HEIGHT),
    Player.P2: (WIDTH - SECTOR_SRC_HEIGHT,       HEIGHT//2 - SECTOR_SRC_WIDTH//2,
                WIDTH,                       HEIGHT//2 + SECTOR_SRC_WIDTH//2),
    Player.P3: (WIDTH//2 - SECTOR_SRC_WIDTH//2,  0,
                WIDTH//2 + SECTOR_SRC_WIDTH//2,  SECTOR_SRC_HEIGHT),
    Player.P4: (0,                           HEIGHT//2 - SECTOR_SRC_WIDTH//2,
                SECTOR_SRC_HEIGHT,               HEIGHT//2 + SECTOR_SRC_WIDTH//2),
}

# Token blob parameters 
MIN_AREA               = 300
MAX_AREA               = 80000  
MIN_SOLIDITY           = 0.65
MIN_CIRCULARITY_YELLOW = 0.65  
MAX_CIRCULARITY_BLACK  = 0.82
MIN_ASPECT_BLACK       = 1.1

# Yellow token area range
YELLOW_MIN_AREA = 500
YELLOW_MAX_AREA = 40000

# Black token HSV range —> includes dark grey nuances
BLACK_V_MAX = 130
BLACK_S_MAX = 90


# Step 1: detect background 

def is_white_background(image_rgb: np.ndarray) -> bool:
    """Sample 4 corners — if mean brightness > 200 it's a white background."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    corners = [
        gray[:h//6,    :w//6],
        gray[:h//6,    w*5//6:],
        gray[h*5//6:,  :w//6],
        gray[h*5//6:,  w*5//6:],
    ]
    return np.mean([r.mean() for r in corners]) > 200


# Step 2a: black mask (white background) 

def compute_black_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    HSV-based dark mask that captures both pure black and dark grey nuances of the token. 
    Followed by morphological cleanup.
    """
    hsv  = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    mask = ((hsv[:, :, 2] < BLACK_V_MAX) &
            (hsv[:, :, 1] < BLACK_S_MAX)).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
    return mask


# Step 2b: yellow mask (flower background) 

def compute_yellow_mask(image_rgb: np.ndarray) -> np.ndarray:
    """
    Use preprocessing soft mask for yellow, then threshold to binary.
    """
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    soft = filter_color(hsv, "yellow")

    _, binary = cv2.threshold(soft, 80, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k)
    return binary, soft

# Step 3: find token blob

def find_token_blob(binary_mask: np.ndarray, token_type: str) -> dict | None:
    """
    Find the blob that best matches the token shape.
      yellow → solid circle  (high circularity + solidity + density)
      black  → solid rectangle (high solidity + aspect ratio)
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    min_a = YELLOW_MIN_AREA if token_type == "yellow" else MIN_AREA
    max_a = YELLOW_MAX_AREA if token_type == "yellow" else MAX_AREA

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (min_a < area < max_a):
            continue

        perim  = cv2.arcLength(cnt, True)
        circ   = (4 * np.pi * area / perim**2) if perim > 0 else 0
        hull   = cv2.convexHull(cnt)
        sol    = area / cv2.contourArea(hull) if cv2.contourArea(hull) > 0 else 0
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = max(w, h) / (min(w, h) + 1e-5)
        cx, cy = x + w//2, y + h//2

        if sol < MIN_SOLIDITY:
            continue

        if token_type == "yellow":
            if circ < MIN_CIRCULARITY_YELLOW:
                continue
            # Check yellow pixel density inside blob (token=solid, skip card=hollow)
            blob_mask_tmp = np.zeros(binary_mask.shape, dtype=np.uint8)
            cv2.drawContours(blob_mask_tmp, [cnt], -1, 255, -1)
            yellow_pixels = np.sum((binary_mask > 0) & (blob_mask_tmp > 0))
            total_pixels  = np.sum(blob_mask_tmp > 0)
            yellow_density = yellow_pixels / total_pixels if total_pixels > 0 else 0
            if yellow_density < 0.55:
                continue
            score = circ * sol * yellow_density

        else:  # black
            if circ > MAX_CIRCULARITY_BLACK or aspect < MIN_ASPECT_BLACK:
                continue

            # Rectangle fit
            # A solid rectangle has rect_fill close to 1.0
            # Card symbol outlines have rect_fill much lower (hollow)
            rect       = cv2.minAreaRect(cnt)
            rect_area  = rect[1][0] * rect[1][1]
            rect_fill  = area / rect_area if rect_area > 0 else 0
            if rect_fill < 0.65:   # must fill at least 65% of its bounding rectangle
                continue

            score = sol * rect_fill

        candidates.append({
            "area": area, "circ": round(circ, 3), "sol": round(sol, 3),
            "aspect": round(aspect, 2), "score": round(score, 4),
            "bbox": (x, y, w, h), "center": (cx, cy), "contour": cnt,
        })

    if not candidates:
        return None

    # Debug: print all candidates
    #print(f"  [{token_type}] {len(candidates)} candidates:")
    #for c in sorted(candidates, key=lambda b: -b["score"])[:5]:
    #    print(f"    score={c['score']:.3f} area={int(c['area']):6d} "
    #          f"circ={c['circ']} sol={c['sol']} aspect={c['aspect']} "
    #          f"pos={c['center']}")

    if token_type == "black":
        return max(candidates, key=lambda b: b["area"])  # biggest = token
    else:
        return max(candidates, key=lambda b: b["score"])  # density already in score

# Step 4: assign to nearest player 

def assign_player(cx: int, cy: int) -> Player:
    return min(PLAYER_CENTERS,
               key=lambda p: (cx - PLAYER_CENTERS[p][0])**2
                           + (cy - PLAYER_CENTERS[p][1])**2)


# Main pipeline 

def detect_active_player(images: np.ndarray, debug: bool = False) -> list[Player]:
    results = []

    for n in range(len(images)):
        img   = images[n]
        white = is_white_background(img)

        if white:
            # White background → black token
            mask       = compute_black_mask(img)
            soft       = mask  
            token      = find_token_blob(mask, "black")
            token_type = "black"
        else:
            # Flower background → yellow token
            mask, soft = compute_yellow_mask(img)
            token      = find_token_blob(mask, "yellow")
            token_type = "yellow"

        active_player = assign_player(*token["center"]) if token else None

        if debug:
            _debug_plot(n, img, white, soft, mask,
                        token, token_type, active_player)

        results.append(active_player)
        #bg  = "WHITE" if white else "FLOWER"
        #det = f"score={token['score']} area={token['area']}" if token else "✗ not found"
        #print(f"Image {n}: bg={bg}  predicted={active_player}  {det}")

    return results


# Debug 

ZONE_COLORS = {
    Player.P1: (0,   200, 255),
    Player.P2: (0,   255, 100),
    Player.P3: (255, 100,   0),
    Player.P4: (200,   0, 255),
}

def _debug_plot(n, img, white, soft, binary,
                token, token_type, active_player):

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    bg = "WHITE" if white else "FLOWER"
    fig.suptitle(f"Image {n} → bg={bg}  token={token_type}  "
                 f"predicted={active_player}", fontsize=14)

    # Full image
    vis = img.copy()
    for player, (x1, y1, x2, y2) in PLAYER_ZONES.items():
        c = ZONE_COLORS[player]
        cv2.rectangle(vis, (x1, y1), (x2, y2), c, 12)
        cv2.putText(vis, str(player), (x1+20, y1+100),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, c, 6)
    if token:
        cx, cy = token["center"]
        cv2.circle(vis, (cx, cy), 120, (255, 0, 0), -1)
        cv2.circle(vis, (cx, cy), 140, (255, 255, 0), 15)
        cv2.drawContours(vis, [token["contour"]], -1, (255, 0, 0), 15)
        cv2.putText(vis, f"TOKEN ({token_type})",
                    (max(cx-350, 0), max(cy-160, 50)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 0, 0), 5)
    axes[0].imshow(vis)
    axes[0].set_title("Full image + detection")
    axes[0].axis("off")

    # Soft / raw mask
    cmap = "YlOrBr" if token_type == "yellow" else "gray"
    axes[1].imshow(soft, cmap=cmap)
    axes[1].set_title(f"{token_type} soft mask")
    axes[1].axis("off")

    # Binary mask
    axes[2].imshow(binary, cmap="gray")
    axes[2].set_title(f"{token_type} binary  "
                      f"({'✓ found' if token else '✗ not found'})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(f"/tmp/debug_image_{n}.png", dpi=60)
    plt.close()
    print(f"  Debug saved to /tmp/debug_image_{n}.png")
