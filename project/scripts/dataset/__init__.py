import os
import random
from typing import Self
import cv2
import numpy as np
import pandas as pd
from enum import StrEnum
from cv2.typing import MatLike

CURRENT_FILE = os.path.abspath(__file__)
CURRENT_PATH = os.path.dirname(CURRENT_FILE)
PARENT_PATH = os.path.dirname(os.path.dirname(CURRENT_PATH))
TRAIN_FILE = os.path.join(PARENT_PATH, "data", "train.csv")
TRAIN_IMAGES_PATH = os.path.join(PARENT_PATH, "data", "train_images")
TEST_IMAGES_PATH = os.path.join(PARENT_PATH, "data", "test_images")
REF_IMAGES_PATH = os.path.join(PARENT_PATH, "data", "reference_images")
COLOR_SAMPLES_PATH = os.path.join(PARENT_PATH, "samples", "colors")

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

def load_train_image(image_id: str) -> tuple[np.ndarray, Label]:
    image_path = os.path.join(TRAIN_IMAGES_PATH, f"{image_id}.jpg")
    print(f"Loading train image: {image_path}")

    # Load the image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Load the label matching this image_id
    csv = pd.read_csv(TRAIN_FILE, index_col=0)
    row = csv.loc[[image_id]].reset_index().iloc[0]
    label = Label.from_row(row)

    return image, label

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
    print(f"Loading random test image: {random_filename}")

    # Load the image
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    if image is None:
        raise ValueError(f"Could not read image at {image_path}")

    return image

def load_test_image(image_id: str) -> np.ndarray:
    image_path = os.path.join(TEST_IMAGES_PATH, f"{image_id}.jpg")

    # Print current step
    print(f"Loading test image: {image_path}")

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
        path = os.path.join(COLOR_SAMPLES_PATH, f"{color}.png")
        images[color] = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGRA2RGBA)
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
