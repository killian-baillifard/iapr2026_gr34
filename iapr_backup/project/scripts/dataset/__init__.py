import os
import random
from typing import Self
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from enum import StrEnum
from cv2.typing import MatLike

_SOBEL_KX = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
_SOBEL_KY = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)

def add_sobel_channel(x: torch.Tensor) -> torch.Tensor:
    """x: (S, H, W, 4) float32 in [0,1] → (S, H, W, 5), Sobel edge magnitude as 5th channel."""
    S, H, W, C = x.shape
    gray  = x.max(dim=-1).values.unsqueeze(1)              # (S, 1, H, W)
    gx    = F.conv2d(gray, _SOBEL_KX, padding=1)
    gy    = F.conv2d(gray, _SOBEL_KY, padding=1)
    edges = (gx**2 + gy**2).sqrt().squeeze(1)              # (S, H, W)
    edges = edges / (edges.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    return torch.cat([x, edges.unsqueeze(-1)], dim=-1)     # (S, H, W, 5)


def rygb_to_color_edges(x: torch.Tensor) -> torch.Tensor:
    """Replace RYGB channels with per-color Sobel edge maps.

    Instead of raw color presence, each channel now encodes WHERE the edges
    of that color's cards are — forcing the model to learn symbol shapes,
    not just color blobs.

    x      : (S, H, W, 4) float32 RYGB in [0, 1]
    returns: (S, H, W, 4) float32 — per-color edge magnitude, each normalised to [0,1]
    """
    S, H, W, C = x.shape
    # (S, 4, H, W) for grouped conv
    x_chw = x.permute(0, 3, 1, 2).reshape(S * C, 1, H, W)
    gx    = F.conv2d(x_chw, _SOBEL_KX, padding=1)
    gy    = F.conv2d(x_chw, _SOBEL_KY, padding=1)
    edges = (gx**2 + gy**2).sqrt()                        # (S*4, 1, H, W)
    edges = edges.view(S, C, H, W)                        # (S, 4, H, W)
    # normalise each channel independently
    edges = edges / (edges.amax(dim=(-2, -1), keepdim=True) + 1e-8)
    return edges.permute(0, 2, 3, 1).contiguous()         # (S, H, W, 4)

CURRENT_FILE = os.path.abspath(__file__)
CURRENT_PATH = os.path.dirname(CURRENT_FILE)
PARENT_PATH = os.path.dirname(os.path.dirname(CURRENT_PATH))
TRAIN_FILE = os.path.join(PARENT_PATH, "data", "train.csv")
TRAIN_IMAGES_PATH = os.path.join(PARENT_PATH, "data", "train_images")
TEST_IMAGES_PATH = os.path.join(PARENT_PATH, "data", "test_images")
REF_IMAGES_PATH = os.path.join(PARENT_PATH, "data", "reference_images")
MANUAL_SEGMENTATION_PATH = os.path.join(PARENT_PATH, "manual_segmentation")
PREPROCESSED_CACHE_PATH     = os.path.join(PARENT_PATH, "data", "preprocessed")
PREPROCESSED_RGB_CACHE_PATH = os.path.join(PARENT_PATH, "data", "preprocessed_rgb")

WIDTH = 4000
HEIGHT = 2662

class Player(StrEnum):
    P1 = "p1"
    P2 = "p2"
    P3 = "p3"
    P4 = "p4"

class Card(StrEnum):

    R0 = "r_0"
    R1 = "r_1"
    R2 = "r_2"
    R3 = "r_3"
    R4 = "r_4"
    R5 = "r_5"
    R6 = "r_6"
    R7 = "r_7"
    R8 = "r_8"
    R9 = "r_9"
    R_REVERSE = "r_reverse"
    R_SKIP = "r_skip"
    R_DRAW_2 = "r_draw_2"

    Y0 = "y_0"
    Y1 = "y_1"
    Y2 = "y_2"
    Y3 = "y_3"
    Y4 = "y_4"
    Y5 = "y_5"
    Y6 = "y_6"
    Y7 = "y_7"
    Y8 = "y_8"
    Y9 = "y_9"
    Y_REVERSE = "y_reverse"
    Y_SKIP = "y_skip"
    Y_DRAW_2 = "y_draw_2"

    G0 = "g_0"
    G1 = "g_1"
    G2 = "g_2"
    G3 = "g_3"
    G4 = "g_4"
    G5 = "g_5"
    G6 = "g_6"
    G7 = "g_7"
    G8 = "g_8"
    G9 = "g_9"
    G_REVERSE = "g_reverse"
    G_SKIP = "g_skip"
    G_DRAW_2 = "g_draw_2"

    B0 = "b_0"
    B1 = "b_1"
    B2 = "b_2"
    B3 = "b_3"
    B4 = "b_4"
    B5 = "b_5"
    B6 = "b_6"
    B7 = "b_7"
    B8 = "b_8"
    B9 = "b_9"
    B_REVERSE = "b_reverse"
    B_SKIP = "b_skip"
    B_DRAW_2 = "b_draw_2"

    DRAW_4 = "draw_4"
    WILD = "wild"

CARD_LOOKUP = []
CARDS_IDX_DICT = {}
for i, card in enumerate(list(Card)):
    CARD_LOOKUP.append(card)
    CARDS_IDX_DICT[str(card)] = i
CARDS_COUNT = len(Card)

def cards_list_to_binary_vector(cards: list[Card]) -> np.ndarray:
    vector = np.zeros(CARDS_COUNT, dtype=np.float32)
    for card in cards:
        vector[CARDS_IDX_DICT[str(card)]] = 1.0
    return vector

class Label:

    def __init__(self, image_id: str, center_card: Card, active_player: Player, players_cards: list[list[Card]]) -> None:
        self.image_id: str = image_id
        self.center_card: Card = center_card
        self.active_player: Player = active_player
        self.players_cards: list[list[Card]] = players_cards

    def __str__(self) -> str:

        player_cards = []
        for i in range(4):
            if len(self.players_cards[i]) == 0:
                player_cards.append(" EMPTY")
            else:
                cards = ""
                for card in self.players_cards[i]:
                    cards += " "
                    cards += card
                player_cards.append(cards)

        return "[id: " + self.image_id + ", " + \
            "center: " + self.center_card + ", " \
            "active: " + self.active_player + ", " \
            "p1_cards: [" + player_cards[0] + "], " \
            "p2_cards: [" + player_cards[1] + "], " \
            "p3_cards: [" + player_cards[2] + "], " \
            "p4_cards: [" + player_cards[3] + "]]"
    
    @staticmethod
    def from_row(row: pd.Series) -> Self:
        image_id: str = str(row["image_id"])
        center_card: Card = Card(row["center_card"])
        active_player: Player = Player(row["active_player"])
        players_cards: list[list[Card]] = []
        for i in range(1, 5):
            raw_cards_strings = str(row[f"player_{i}_cards"]).split(";")
            if raw_cards_strings[0] == 'EMPTY':
                players_cards.append([])
            else:
                cards_list = [Card(card_id) for card_id in raw_cards_strings]
                players_cards.append(cards_list)
        return Label(image_id, center_card, active_player, players_cards)
    
    def as_binary_vector(self) -> np.ndarray:
        """
        Returns
        -------

            labels : np.ndarray
                Binary vector of cards presence for each player and center [center, p1, p2, p3, p4]
        """
        return np.stack([
            cards_list_to_binary_vector([self.center_card]),
            cards_list_to_binary_vector(self.players_cards[0]),
            cards_list_to_binary_vector(self.players_cards[1]),
            cards_list_to_binary_vector(self.players_cards[2]),
            cards_list_to_binary_vector(self.players_cards[3])
        ])

    def center_idx(self) -> int:
        return CARDS_IDX_DICT[str(self.center_card)]

    def counts(self) -> np.ndarray:
        result = np.zeros((5, CARDS_COUNT), dtype=np.int64)
        for card in [self.center_card]:
            result[0, CARDS_IDX_DICT[str(card)]] += 1
        for i, hand in enumerate(self.players_cards):
            for card in hand:
                result[i + 1, CARDS_IDX_DICT[str(card)]] += 1
        return result

class SectorDataset(Dataset):
    """One sample per sector — 5 samples per image.
    Returns (H, W, 4) mask, (54,) float label, bool is_center."""

    def __init__(self, labels: list[Label]):
        self.samples = []
        for label in labels:
            path = os.path.join(PREPROCESSED_CACHE_PATH, label.image_id + ".npy")
            center_vec = np.zeros(CARDS_COUNT, dtype=np.float32)
            center_vec[label.center_idx()] = 1.0
            self.samples.append((path, 0, True, center_vec))
            counts = label.counts()  # (5, 54)
            for p in range(4):
                self.samples.append((path, p + 1, False, counts[p + 1].astype(np.float32)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, sector_idx, is_center, label = self.samples[idx]
        x = torch.from_numpy(np.load(path)[sector_idx]).float() / 255.0  # (H, W, 4)
        y = torch.from_numpy(label)
        return x, y, torch.tensor(is_center, dtype=torch.bool)


class PreprocessedDataset(Dataset):
    """Loads preprocessed .npy files on demand — one sample per __getitem__ call."""

    def __init__(self, labels: list[Label], target_size: tuple[int, int] | None = (448, 448), use_sobel: bool = False, use_color_edges: bool = False):
        self.labels = labels
        self.cache_dir = PREPROCESSED_CACHE_PATH
        self.target_size = target_size
        self.use_sobel = use_sobel
        self.use_color_edges = use_color_edges

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        path = os.path.join(self.cache_dir, label.image_id + ".npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cache missing for {label.image_id}. Run: python -m project.scripts.cache_preprocessing"
            )
        x = torch.from_numpy(np.load(path)).float() / 255.0  # (5, H, W, 4)
        if self.target_size is not None:
            x = torch.nn.functional.interpolate(
                x.permute(0, 3, 1, 2),
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1).contiguous()
        if self.use_color_edges:
            x = rygb_to_color_edges(x)                       # (5, H, W, 4) — replaces RYGB with per-color edges
        if self.use_sobel:
            x = add_sobel_channel(x)                         # (5, H, W, 5)
        y_center = torch.tensor(label.center_idx(), dtype=torch.long)
        y_player = torch.from_numpy(label.counts()[1:, :]).long()   # (4, 54)
        return x, y_center, y_player

class RawSectorDataset(Dataset):
    """Loads raw RGB sectors (no color masking) cached by cache_rgb_sectors.py.

    Returns x as (5, H, W, 3) float32 in [0, 1] — same label format as PreprocessedDataset.
    Run project.scripts.cache_rgb_sectors first to populate data/preprocessed_rgb/.
    """

    def __init__(self, labels: list[Label], target_size: tuple[int, int] | None = (448, 448)):
        self.labels = labels
        self.cache_dir = PREPROCESSED_RGB_CACHE_PATH
        self.target_size = target_size

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        path = os.path.join(self.cache_dir, label.image_id + ".npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cache missing for {label.image_id}. Run: python -m project.scripts.cache_rgb_sectors"
            )
        x = torch.from_numpy(np.load(path)).float() / 255.0  # (5, H, W, 3)
        if self.target_size is not None:
            x = torch.nn.functional.interpolate(
                x.permute(0, 3, 1, 2),
                size=self.target_size,
                mode="bilinear",
                align_corners=False,
            ).permute(0, 2, 3, 1).contiguous()
        y_center = torch.tensor(label.center_idx(), dtype=torch.long)
        y_player = torch.from_numpy(label.counts()[1:, :]).long()   # (4, 54)
        return x, y_center, y_player


def load_preprocessed_train() -> tuple[np.ndarray, list[Label]]:
    """
    Output:
        Preprocessed images (n, 5, height, width, 4)
        Labels (n)
    """
    cache_dir = os.path.join(PARENT_PATH, "data", "preprocessed")
    csv = pd.read_csv(TRAIN_FILE)
    labels = [Label.from_row(row) for _, row in csv.iterrows()]
    preprocessed = []
    for label in labels:
        path = os.path.join(cache_dir, label.image_id + ".npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cache missing for {label.image_id}. Run: python -m project.scripts.cache_preprocessing"
            )
        preprocessed.append(np.load(path))
    return np.array(preprocessed), labels


def load_train_images_paths_and_labels() -> list[tuple[str, Label]]:
    """
    Returns
    -------

        train : list[tuple[str, Label]]
            - path
                Path to the train image
            - label
                Image label
    """

    # Print current step
    print(f"Loading all train images paths with their labels")

    # Load labels and images paths
    csv = pd.read_csv(TRAIN_FILE)
    labels: list[Label] = [Label.from_row(row) for _, row in csv.iterrows()]
    paths = [os.path.join(TRAIN_IMAGES_PATH, label.image_id + ".jpg") for label in labels]

    # Return both
    return [(path, label) for (label, path) in zip(labels, paths)]

def load_train_images() -> tuple[np.ndarray, list[Label]]:
    """
    Returns
    -------

        images: np.ndarray
            RGB train images (n, height, width, 3)
        labels: list[Label]
            list of label objects (n)
    """

    # Print current step
    print(f"Loading all train images")

    # Load labels
    csv = pd.read_csv(TRAIN_FILE)
    labels: list[Label] = [Label.from_row(row) for _, row in csv.iterrows()]
    
    # Load associated images
    images = []
    for label in labels:
        image_path = os.path.join(TRAIN_IMAGES_PATH, label.image_id + ".jpg")
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        images.append(image)

    # Return both
    return np.array(images), labels

def load_random_train_images(n: int) -> tuple[np.ndarray, list[Label]]:
    """
    Parameters
    ----------

    n : int
        Number of images to load (selected at random)

    Returns
    -------

        images : np.ndarray
            Set of n random RGB images (n, height, width, 3)
        labels: list[Label]
            list of label objects (n)
    """

    # Print current step
    print(f"Loading {n} random images from train set")

    # Load labels and pick n at random
    csv = pd.read_csv(TRAIN_FILE)
    rows = csv.sample(n)
    labels = [Label.from_row(row) for _, row in rows.iterrows()]

    # Load all corresponding images
    images = []
    for label in labels:
        image_path = os.path.join(TRAIN_IMAGES_PATH, label.image_id + ".jpg")
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        images.append(image)

    return np.array(images), labels

def load_test_images_paths() -> list[str]:
    """
    Returns
    -------

        paths : list[str]
            List of paths to test images
    """

    return os.listdir(TEST_IMAGES_PATH)

def load_test_images() -> np.ndarray:
    """
    Returns
    -------

        images : np.ndarray
            Test RGB images (n, height, width, 3)
    """

    # Print current step
    print(f"Loading all test images")

    # Load all test images
    images = []
    for filename in os.listdir(TEST_IMAGES_PATH):
        image_path = os.path.join(TEST_IMAGES_PATH, filename)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        images.append(image)
    return np.array(images)

def load_random_test_image() -> np.ndarray:
    """
    Selects a random image from the test directory and loads it.

    Returns
    -------
    image : np.ndarray
        A single RGB image (height, width, 3)
    """
    # Get list of all files in the directory
    files = os.listdir(TEST_IMAGES_PATH)
    
    # Filter for common image extensions if necessary
    # files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not files:
        raise FileNotFoundError(f"No images found in {TEST_IMAGES_PATH}")

    # Pick one random filename
    random_filename = random.choice(files)
    image_path = os.path.join(TEST_IMAGES_PATH, random_filename)

    # Print current step
    print(f"Loading random image: {random_filename}")

    # Load the image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    return image

def load_manually_segmented_images() -> dict[str, MatLike]:
    """
    Returns
    -------

        images : np.ndarray
            Manually segmented RGB images (5, height, width, 3)
    """

    images = {}
    for color in ["r", "y", "g", "b", "k"]:
        path = os.path.join(MANUAL_SEGMENTATION_PATH, f"{color}.png")
        images[color] = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return images

def load_reference_images() -> np.ndarray:
    """
    Returns
    -------

        images : np.ndarray
            Reference images (4, height, width, 3)
    """

    # Load all reference images
    images = []
    for filename in os.listdir(REF_IMAGES_PATH):
        image_path = os.path.join(REF_IMAGES_PATH, filename)
        image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        images.append(image)
    return np.array(images)
