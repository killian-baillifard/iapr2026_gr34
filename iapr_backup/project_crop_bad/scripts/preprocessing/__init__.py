import numpy as np
import cv2
from project_crop_bad.scripts.preprocessing.rygb import hsv2rygb
from project_crop_bad.scripts.preprocessing.filter import area_bandpass_filter

def preprocess(image: np.ndarray) -> np.ndarray:
    """
    Preprocess an image of the dataset
    
    Parameters
    ----------
    image : np.ndarray
        Raw RGB image (height, width, 3)
    
    Returns
    -------
    preprocessed : np.ndarray
        RYGB preprocessed image (height, width, 4)
    """

    return area_bandpass_filter(hsv2rygb(cv2.cvtColor(image, cv2.COLOR_RGB2HSV)))
