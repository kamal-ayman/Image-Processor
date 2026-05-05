import cv2
import numpy as np
from ...domain.image_operation import ImageOperation
from ...utils.image_utils import ensure_grayscale

class MorphologicalOperation(ImageOperation):
    """Base class for morphological operations."""
    def _get_kernel(self, size: int = 3):
        return np.ones((size, size), np.uint8)

class Erosion(MorphologicalOperation):
    def execute(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        image = ensure_grayscale(image)
        return cv2.erode(image, self._get_kernel(kernel_size), iterations=1)

class Dilation(MorphologicalOperation):
    def execute(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        image = ensure_grayscale(image)
        return cv2.dilate(image, self._get_kernel(kernel_size), iterations=1)

class Opening(MorphologicalOperation):
    def execute(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        image = ensure_grayscale(image)
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, self._get_kernel(kernel_size))

class Closing(MorphologicalOperation):
    def execute(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        image = ensure_grayscale(image)
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, self._get_kernel(kernel_size))

class BoundaryExtraction(MorphologicalOperation):
    def execute(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        image = ensure_grayscale(image)
        eroded = cv2.erode(image, self._get_kernel(kernel_size), iterations=1)
        return image - eroded
