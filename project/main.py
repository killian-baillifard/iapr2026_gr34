import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def main():

    # Get current file path
    current_file_path = os.path.abspath(__file__)

    # Load train images from the "train_images" folder
    train_images_folder = os.path.join(os.path.dirname(current_file_path), "data", "train_images")
    train_images = []
    for filename in os.listdir(train_images_folder):
        image_path = os.path.join(train_images_folder, filename)
        image = np.array(cv2.imread(image_path))
        train_images.append(image)

    # Display randomly 5 images from the training set
    plt.figure(figsize=(10, 10))
    for i in range(5):
        random_index = np.random.randint(len(train_images))
        plt.subplot(1, 5, i + 1)
        plt.imshow(train_images[random_index])
        plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
