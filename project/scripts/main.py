from dataset import load_train_images, load_test_images, load_random_train_image
from preprocessing import preprocess
from matplotlib import pyplot as plt

"""
Note :
- download the dataset from https://www.kaggle.com/competitions/iapr-26-uno-vision-challenge/data
- unzip it's content into the 'data' folder
"""

if __name__ == "__main__":

    # Load random train sample
    train_image, label = load_random_train_image()
    
    # Load datasets
    # train_images, labels = load_train_images()
    # test_images = load_test_images()

    # Apply red segmentation on first train image
    preview = preprocess(train_image)
    plt.figure(str(label))
    plt.subplot(111)
    plt.imshow(preview)
    plt.show()

