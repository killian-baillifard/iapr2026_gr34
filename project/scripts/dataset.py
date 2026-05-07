import os
from typing import Self
import cv2
import numpy as np
import pandas as pd
from enum import StrEnum
from cv2.typing import MatLike
from matplotlib import pyplot as plt

CURRENT_FILE = os.path.abspath(__file__)
CURRENT_PATH = os.path.dirname(CURRENT_FILE)
PARENT_PATH = os.path.dirname(CURRENT_PATH)
TRAIN_FILE = os.path.join(PARENT_PATH, "data", "train.csv")
TRAIN_IMAGES_PATH = os.path.join(PARENT_PATH, "data", "train_images")
TEST_IMAGES_PATH = os.path.join(PARENT_PATH, "data", "test_images")
MANUAL_SEGMENTATION_PATH = os.path.join(PARENT_PATH, "manual_segmentation")

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

def load_train_images() -> tuple[list[MatLike], list[Label]]:

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
    return images, labels

def load_random_train_image() -> tuple[MatLike, Label]:

    # Load labels and pick one at random
    csv = pd.read_csv(TRAIN_FILE)
    row = csv.sample(1).iloc[0]
    label = Label.from_row(row)

    # Load only the corresponding image
    image_path = os.path.join(TRAIN_IMAGES_PATH, label.image_id + ".jpg")
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    return image, label

def load_test_images() -> list[MatLike]:
    images = []
    for filename in os.listdir(TEST_IMAGES_PATH):
        image_path = os.path.join(TEST_IMAGES_PATH, filename)
        image = cv2.imread(image_path)
        images.append(image)
    return images

def load_manually_segmented_images() -> dict[str, MatLike]:
    images = {}
    for color in ["r", "y", "g", "b", "k"]:
        path = os.path.join(MANUAL_SEGMENTATION_PATH, f"{color}.png")
        images[color] = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return images

if __name__ == "__main__":
    print(f"Number of unique cards : {len(Card)}")
    print(f"Parent path : {PARENT_PATH}")
    train_images, labels = load_train_images()
    plt.figure(figsize=(10, 10))
    for i in range(5):
        random_index = np.random.randint(len(train_images))
        plt.subplot(1, 5, i + 1)
        plt.imshow(train_images[random_index])
        plt.axis("off")
        print(labels[random_index])
    plt.show()
