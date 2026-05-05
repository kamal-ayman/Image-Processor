import cv2
import numpy as np
from ...domain.image_operation import ImageOperation
from ...utils.image_utils import to_grayscale

class RgbToGrayscale(ImageOperation):
    def execute(self, image: np.ndarray) -> np.ndarray:
        return to_grayscale(image)

class GrayToBinary(ImageOperation):
    def execute(self, image: np.ndarray, threshold: int = 127) -> np.ndarray:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return bw

class RgbToBinary(ImageOperation):
    def execute(self, image: np.ndarray, threshold: int = 127) -> np.ndarray:
        gray = to_grayscale(image)
        _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return bw
