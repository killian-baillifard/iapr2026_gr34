
import csv
import os
from typing import Optional

import numpy as np
import pandas as pd

from dataset import Card, Player, Label


# String helpers

def cards_to_str(cards: list[Card]) -> str:
    """
    Convert a list of Cards to the Kaggle CSV format.
    """
    if not cards:
        return "EMPTY"
    return ";".join(str(c) for c in cards)


def str_to_cards(s: str) -> list[Card]:
    """
    Parse a Kaggle CSV cell back into a list of Cards.
    """
    s = s.strip()
    if s.upper() == "EMPTY":
        return []
    return [Card(token.strip()) for token in s.split(";")]


def player_to_str(player: Optional[Player]) -> str:
    """Player -> Kaggle string ("p1" … "p4"). Falls back to "p1" if None."""
    return str(player) if player is not None else "p1"

# F1 metric  (2TP / (2TP + FP + FN))

def _f1_single(pred: list[Card], true: list[Card]) -> float:
    """F1 for one player's hand in one image (multiset matching)."""
    if not pred and not true:
        return 1.0          # both empty → perfect

    # Count occurrences of each card string
    pred_counts: dict[str, int] = {}
    for c in pred:
        pred_counts[str(c)] = pred_counts.get(str(c), 0) + 1

    true_counts: dict[str, int] = {}
    for c in true:
        true_counts[str(c)] = true_counts.get(str(c), 0) + 1

    tp = sum(min(pred_counts.get(k, 0), v) for k, v in true_counts.items())
    fp = sum(pred_counts.values()) - tp
    fn = sum(true_counts.values()) - tp

    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def _f1_image(pred_players: list[list[Card]], true_players: list[list[Card]]) -> float:
    """Average F1 across the 4 players for a single image."""
    return float(np.mean([_f1_single(pred_players[p], true_players[p]) for p in range(4)]))


# Main evaluation function

def compute_metrics(
    labels:              list[Label],
    pred_center_cards:   list[Card],
    pred_active_players: list[Optional[Player]],
    pred_player_cards:   list[list[list[Card]]],  # shape: (n_images, 4 players)
) -> dict:
    """
    Compute all metrics from the project slides.

    Parameters
    ----------
    labels : list[Label]
        Ground-truth labels (from load_train_images).
    pred_center_cards : list[Card]
        One predicted center card per image.
    pred_active_players : list[Player | None]
        One predicted active player per image.
    pred_player_cards : list of (4 lists of Card)
        pred_player_cards[i][j] = predicted cards for image i, player j+1.

    Returns
    -------
    dict
        center_acc  : float          — CenterAcc averaged over all images
        active_acc  : float          — ActiveAcc averaged over all images
        f1          : np.ndarray     — per-image F1, shape (n,)
        mean_f1     : float          — F1 averaged over all images
        score       : float          — 0.1*CenterAcc + 0.1*ActiveAcc + 0.8*F1
    """
    n = len(labels)
    assert n == len(pred_center_cards) == len(pred_active_players) == len(pred_player_cards), \
        "All prediction lists must have the same length as labels."

    # CenterAcc_i = 1[c_hat_i == c_i]
    center_acc = float(np.mean([
        str(pred_center_cards[i]) == str(labels[i].center_card)
        for i in range(n)
    ]))

    # ActiveAcc_i = 1[a_hat_i == a_i]
    active_acc = float(np.mean([
        player_to_str(pred_active_players[i]) == str(labels[i].active_player)
        for i in range(n)
    ]))

    # F1_i averaged over 4 players, then over all images
    f1_per_image = np.array([
        _f1_image(pred_player_cards[i], labels[i].players_cards)
        for i in range(n)
    ])
    mean_f1 = float(np.mean(f1_per_image))

    # Final score
    score = 0.1 * center_acc + 0.1 * active_acc + 0.8 * mean_f1

    return {
        "center_acc": center_acc,
        "active_acc": active_acc,
        "f1":         f1_per_image,
        "mean_f1":    mean_f1,
        "score":      score,
    }


def print_metrics(metrics: dict) -> None:
    """Pretty-print the output of compute_metrics()."""
    print("=" * 44)
    print(f"  CenterAcc  : {metrics['center_acc']:.4f}  (weight 0.1)")
    print(f"  ActiveAcc  : {metrics['active_acc']:.4f}  (weight 0.1)")
    print(f"  Mean F1    : {metrics['mean_f1']:.4f}  (weight 0.8)")
    print(f"  {'─' * 38}")
    print(f"  Score      : {metrics['score']:.4f}")
    print("=" * 44)

# Kaggle CSV submission

def to_submission_csv(
    sample_submission_path: str,
    pred_center_cards:      list[Card],
    pred_active_players:    list[Optional[Player]],
    pred_player_cards:      list[list[list[Card]]],  # (n_images, 4 players)
    output_path:            str = "submission.csv",
) -> None:
    """
    Write the Kaggle submission CSV

    """
    # Read image_ids in the exact Kaggle-expected order
    sample_df = pd.read_csv(sample_submission_path)
    image_ids = sample_df["image_id"].astype(str).tolist()

    n = len(image_ids)
    assert n == len(pred_center_cards) == len(pred_active_players) == len(pred_player_cards), \
        f"Expected {n} predictions (matching sample_submission.csv), got {len(pred_center_cards)}."

    fieldnames = [
        "image_id", "center_card", "active_player",
        "player_1_cards", "player_2_cards", "player_3_cards", "player_4_cards",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(n):
            writer.writerow({
                "image_id":       image_ids[i],
                "center_card":    str(pred_center_cards[i]),
                "active_player":  player_to_str(pred_active_players[i]),
                "player_1_cards": cards_to_str(pred_player_cards[i][0]),
                "player_2_cards": cards_to_str(pred_player_cards[i][1]),
                "player_3_cards": cards_to_str(pred_player_cards[i][2]),
                "player_4_cards": cards_to_str(pred_player_cards[i][3]),
            })

    print(f"Submission written -> {output_path}  ({n} rows)")


# Self-test

if __name__ == "__main__":

    # F1 unit tests 
    assert _f1_single([Card.R5, Card.B_SKIP], [Card.R5, Card.B_SKIP]) == 1.0, "perfect match"
    assert _f1_single([Card.R5], [Card.B_SKIP])                        == 0.0, "complete miss"
    assert _f1_single([], [])                                          == 1.0, "both empty"

    f = _f1_single([Card.R5, Card.G2], [Card.R5, Card.B_SKIP])
    assert abs(f - 0.5) < 1e-9, f"partial: expected 0.5, got {f}"

    # 2x R5 predicted, 1x R5 true  →  TP=1 FP=1 FN=0  →  F1 = 2/3
    f = _f1_single([Card.R5, Card.R5], [Card.R5])
    assert abs(f - 2/3) < 1e-9, f"duplicate: expected 2/3, got {f}"

    print("All F1 unit tests passed.")

    #  compute_metrics smoke test 
    dummy = Label(
        image_id      = "0001",
        center_card   = Card.R5,
        active_player = Player.P1,
        players_cards = [[Card.R5], [Card.B_SKIP], [], [Card.G3]],
    )
    metrics = compute_metrics(
        labels              = [dummy],
        pred_center_cards   = [Card.R5],
        pred_active_players = [Player.P1],
        pred_player_cards   = [[[Card.R5], [Card.B_SKIP], [], [Card.G3]]],
    )
    assert abs(metrics["score"] - 1.0) < 1e-9, f"perfect score expected, got {metrics['score']}"
    print("compute_metrics smoke test passed.")
    print_metrics(metrics)

    # string helper round-trips 
    cards = [Card.R5, Card.B_SKIP, Card.WILD]
    assert str_to_cards(cards_to_str(cards)) == cards
    assert cards_to_str([])      == "EMPTY"
    assert str_to_cards("EMPTY") == []
    print("String helper round-trips passed.")