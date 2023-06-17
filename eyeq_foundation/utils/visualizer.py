import cv2
import numpy as np


def visualizer(window_name: str, image: np.ndarray, wait_key: int =1):
    cv2.imshow(window_name, image)
    cv2.waitKey(wait_key)