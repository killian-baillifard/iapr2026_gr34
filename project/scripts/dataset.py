import os
from typing import Self
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from enum import StrEnum
from cv2.typing import MatLike
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

CURRENT_FILE = os.path.abspath(__file__)
CURRENT_PATH = os.path.dirname(CURRENT_FILE)
PARENT_PATH = os.path.dirname(CURRENT_PATH)
TRAIN_FILE = os.path.join(PARENT_PATH, "data", "train.csv")
TRAIN_IMAGES_PATH = os.path.join(PARENT_PATH, "data", "train_images")
TEST_IMAGES_PATH = os.path.join(PARENT_PATH, "data", "test_images")
MANUAL_SEGMENTATION_PATH = os.path.join(PARENT_PATH, "manual_segmentation")
PREPROCESSED_CACHE_PATH = os.path.join(PARENT_PATH, "data", "preprocessed")

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

CARDS_IDX_DICT = {}
for i, card in enumerate(list(Card)):
    CARDS_IDX_DICT[str(card)] = i
CARDS_COUNT = len(Card)

def to_probability_vector(cards: list[Card]) -> np.ndarray:
    vector = np.zeros(CARDS_COUNT, dtype=np.float32)
    for card in cards:
        vector[CARDS_IDX_DICT[str(card)]] += 1
    if vector.sum() > 0:
        vector /= vector.sum()
    else:
        vector[:] = 1 / CARDS_COUNT
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
    
    def probabilities(self) -> np.ndarray:
        """
        Output:
            Probability vectors [center, p1, p2, p3, p4]
        """
        return np.array([
            to_probability_vector([self.center_card]),
            to_probability_vector(self.players_cards[0]),
            to_probability_vector(self.players_cards[1]),
            to_probability_vector(self.players_cards[2]),
            to_probability_vector(self.players_cards[3])
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

class PreprocessedDataset(Dataset):
    """Loads preprocessed .npy files on demand — one sample per __getitem__ call."""

    def __init__(self, labels: list[Label]):
        self.labels = labels
        self.cache_dir = PREPROCESSED_CACHE_PATH

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        path = os.path.join(self.cache_dir, label.image_id + ".npy")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cache missing for {label.image_id}. Run: python -m project.scripts.cache_preprocessing"
            )
        x       = torch.from_numpy(np.load(path)).float() / 255.0  # (5, H, W, 4)
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

def load_train_images() -> tuple[np.ndarray, list[Label]]:
    """
    Output:
        Train images (n, height, width, 3)
        Labels (n)
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
    Output:
        Random set of train images (n, height, width, 3)
        Labels (n)
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

def load_test_images() -> np.ndarray:
    """
    Output:
        Test images (n, height, width, 3)
    """

    # Print current step
    print(f"Loading all test images")

    # Load all test images
    images = []
    for filename in os.listdir(TEST_IMAGES_PATH):
        image_path = os.path.join(TEST_IMAGES_PATH, filename)
        image = cv2.imread(image_path)
        images.append(image)
    return np.array(images)

def load_manually_segmented_images() -> dict[str, MatLike]:
    images = {}
    for color in ["r", "y", "g", "b", "k"]:
        path = os.path.join(MANUAL_SEGMENTATION_PATH, f"{color}.png")
        images[color] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return images

if __name__ == "__main__":

    # Print path and cards count
    print(f"Parent path : {PARENT_PATH}")
    print(f"Number of unique cards : {CARDS_COUNT}")

    # Load random set of images
    N = 2
    images, labels = load_random_train_images(N)
    probabilities = np.array([label.probabilities() for label in labels])

    # Print images and their labels
    fig = plt.figure(f"{N} random samples from train dataset")
    gs = GridSpec(6, N, figure=fig, height_ratios=[5, 1, 1, 1, 1, 1], hspace=0)

    for i in range(N):
        # Image
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(images[i])
        ax.axis("off")

        # Histograms
        for j in range(5):
            ax = fig.add_subplot(gs[j + 1, i])
            ax.bar(np.arange(CARDS_COUNT), probabilities[i, j], width=0.6)
            ax.set_xlim(-0.5, CARDS_COUNT - 0.5)
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            if j < 4:
                ax.set_xticks([])
            else:
                ax.set_xticks(np.arange(CARDS_COUNT))
                ax.set_xticklabels([str(c) for c in Card], rotation=90, fontsize=8)

    # Show figure
    plt.tight_layout()
    plt.show()
