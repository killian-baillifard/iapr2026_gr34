import os, torch, cv2, csv, numpy as np
from project.scripts.dataset import Player, load_test_images_paths, TEST_IMAGES_PATH, CARD_LOOKUP
from project.scripts.dataset.synthesizer import synthesize_train_set
from project.scripts.preprocessing import preprocess
from project.scripts.preprocessing.sectors import slice_sectors
from project.scripts.preprocessing.cache import Cache, rebuild_cache
from project.scripts.token import detect_active_player
from project.scripts.classifiers.cnn import UNOCNNClassifier, train_model
import torch.nn.functional as F

# Dataset from https://www.kaggle.com/competitions/iapr-26-uno-vision-challenge/data

MODEL_PATH = os.path.join("best_model.pth")
SUBMISSION_FILE = os.path.join("submission.csv")

if __name__ == "__main__":

    # Settings
    SYNTHETIZE = False
    REBUILD_TRAIN_PREPROC_CACHE = False
    REBUILD_VAL_PREPROC_CACHE = False
    TRAIN_MODEL = True
    SUBMISSION = False

    # Synthetize data
    if SYNTHETIZE:
        synthesize_train_set(512)

    # Preprocess and cache train set
    if REBUILD_TRAIN_PREPROC_CACHE:
        rebuild_cache(Cache.TRAINING)

    # Preprocess and cache train set
    if REBUILD_TRAIN_PREPROC_CACHE:
        rebuild_cache(Cache.VALIDATION)

    # Train model
    if TRAIN_MODEL:
        train_model()

    # Create submission file
    with open(SUBMISSION_FILE, "w") as submission:
        writer = csv.writer(submission, lineterminator="\n")
        writer.writerow([
            "image_id",
            "center_card",
            "active_player",
            "player_1_cards",
            "player_2_cards",
            "player_3_cards",
            "player_4_cards"
        ])

        # Load latest model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = UNOCNNClassifier().to(device)
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        with torch.no_grad():

            # For each image
            paths = load_test_images_paths()
            N = len(paths)
            for i, path in enumerate(paths):

                # Load image
                print(f"Image {i + 1} / {N}")
                image = cv2.cvtColor(cv2.imread(os.path.join(TEST_IMAGES_PATH, path)), cv2.COLOR_BGR2RGB)
                image_id: str = path.split(".")[0]

                # Find active player
                active_player: list[Player] = str(detect_active_player(np.array([image]))[0])

                # For each sector of the image
                player_cards: list[str] = []
                for j, sector in enumerate(slice_sectors(image)):

                    # Preprocess sector
                    preprocessed = preprocess(sector)

                    # Downscale image
                    tensor = torch.tensor(preprocessed, dtype=torch.float32).permute(2, 0, 1)
                    tensor = F.interpolate(
                        tensor.unsqueeze(0), 
                        size=(256, 512), 
                        mode="bilinear", 
                        align_corners=False
                    ).to(device)

                    # Run inference
                    logits = model(tensor)
                    predicted = torch.sigmoid(logits).squeeze(0).cpu().numpy()

                    # For center card, keep only most likely card
                    if j == 0:
                        center_card = CARD_LOOKUP[np.argmax(predicted)]

                    # For all players keep all probabilites above 0.5
                    else:
                        indices, = np.where(predicted > 0.5)
                        
                        # Keep at most 4, prioritizing highest probabilities
                        if len(indices) > 4:
                            indices = indices[np.argsort(predicted[indices])[::-1][:4]]
                        cards = [CARD_LOOKUP[index] for index in indices]
                        player_cards.append(";".join([str(card) for card in cards]) if cards else "EMPTY")

                # Write result to submission file
                writer.writerow([
                    image_id,
                    center_card,
                    active_player,
                    player_cards[0],
                    player_cards[1],
                    player_cards[2],
                    player_cards[3]
                ])
