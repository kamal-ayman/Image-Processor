import cv2
import numpy as np
from scipy.ndimage import minimum_filter, maximum_filter
from ....domain.image_operation import ImageOperation

class MeanFilter(ImageOperation):
    def execute(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        return cv2.blur(image, (kernel_size, kernel_size))

class MedianFilter(ImageOperation):
    def execute(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        if kernel_size % 2 == 0: kernel_size += 1
        return cv2.medianBlur(image, kernel_size)

class MaxFilter(ImageOperation):
    def execute(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for c in range(3):
                result[:, :, c] = maximum_filter(image[:, :, c], size=kernel_size)
            return result
        return maximum_filter(image, size=kernel_size)

class MinFilter(ImageOperation):
    def execute(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        if len(image.shape) == 3:
            result = np.zeros_like(image)
            for c in range(3):
                result[:, :, c] = minimum_filter(image[:, :, c], size=kernel_size)
            return result
        return minimum_filter(image, size=kernel_size)

class MidpointFilter(ImageOperation):
    def execute(self, image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
        img_float = image.astype(np.float64)
        if len(img_float.shape) == 3:
            result = np.zeros_like(img_float)
            for c in range(3):
                ch = img_float[:, :, c]
                mn = minimum_filter(ch, size=kernel_size)
                mx = maximum_filter(ch, size=kernel_size)
                result[:, :, c] = (mn + mx) / 2.0
        else:
            mn = minimum_filter(img_float, size=kernel_size)
            mx = maximum_filter(img_float, size=kernel_size)
            result = (mn + mx) / 2.0
        return np.clip(result, 0, 255).astype(np.uint8)

class CustomWeightFilter(ImageOperation):
    def execute(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return cv2.filter2D(image, -1, mask)
