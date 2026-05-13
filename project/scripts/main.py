import numpy as np
from matplotlib import pyplot as plt
from dataset import Player, load_train_images, load_test_images
from preprocessing import preprocess, crop

"""
Note :
- download the dataset from https://www.kaggle.com/competitions/iapr-26-uno-vision-challenge/data
- unzip it's content into the 'data' folder
"""

if __name__ == "__main__":
    
    # Load full dataset
    train_images, labels = load_train_images()
    test_images = load_test_images()

    # TODO find active player, return Player.P1 or .P2 ...

    # Apply preprocessing to full dataset
    train_preprocessed = preprocess(train_images)
    train_cropped = crop(train_preprocessed)
    test_preprocessed = preprocess(test_images)
    probabilities = np.array([label.probabilities() for label in labels])

    # TODO train model

    # TODO run inference in test dataset

    # TODO store results in CSV file
