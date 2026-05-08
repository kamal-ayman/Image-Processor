import cv2
import numpy as np
from ....domain.image_operation import ImageOperation

class WeightedAverageFilter(ImageOperation):
    """Also known as Mean Filter or Average Filter."""
    def execute(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        return cv2.blur(image, (kernel_size, kernel_size))

class BilateralFilter(ImageOperation):
    def execute(self, image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
