import cv2
import numpy as np
from ...domain.image_operation import ImageOperation
from ...utils.image_utils import ensure_grayscale

class RayleighNoise(ImageOperation):
    def execute(self, image: np.ndarray, a: float = 0, b: float = 30) -> np.ndarray:
        image = ensure_grayscale(image).astype(np.float64)
        u = np.random.rand(*image.shape)
        noise = a + b * np.sqrt(-2 * np.log(1 - u + 1e-10))
        output = image + noise
        return np.clip(output, 0, 255).astype(np.uint8)

class ImpulseNoise(ImageOperation):
    """Salt and Pepper noise."""
    def execute(self, image: np.ndarray, salt_prob: float = 0.05, pepper_prob: float = 0.05) -> np.ndarray:
        image = ensure_grayscale(image)
        output = image.copy()
        prob = np.random.rand(*image.shape)
        output[prob < pepper_prob] = 0
        output[prob > 1 - salt_prob] = 255
        return output

class ExponentialNoise(ImageOperation):
    def execute(self, image: np.ndarray, a: float = 1.0) -> np.ndarray:
        image = ensure_grayscale(image).astype(np.float64)
        # Exponential distribution: p(z) = a * exp(-a*z) for z >= 0
        noise = np.random.exponential(1.0/a, image.shape)
        output = image + noise
        return np.clip(output, 0, 255).astype(np.uint8)
