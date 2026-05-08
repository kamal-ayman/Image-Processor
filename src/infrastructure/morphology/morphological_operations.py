import cv2
import numpy as np
from ...domain.image_operation import ImageOperation
from ...utils.image_utils import ensure_grayscale

class Dilation(ImageOperation):
    def execute(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        image = ensure_grayscale(image)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.dilate(image, kernel, iterations=1)

class Opening(ImageOperation):
    def execute(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        image = ensure_grayscale(image)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
