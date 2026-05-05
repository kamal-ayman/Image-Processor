import cv2
import numpy as np
from ...domain.image_operation import ImageOperation
from ...utils.image_utils import ensure_grayscale

class GaussianNoise(ImageOperation):
    def execute(self, image: np.ndarray, mean: float = 0, sigma: float = 25) -> np.ndarray:
        image = ensure_grayscale(image).astype(np.float64)
        noise = np.random.normal(mean, sigma, image.shape)
        output = image + noise
        return np.clip(output, 0, 255).astype(np.uint8)

class SaltAndPepperNoise(ImageOperation):
    def execute(self, image: np.ndarray, salt_prob: float = 0.05, pepper_prob: float = 0.05) -> np.ndarray:
        image = ensure_grayscale(image)
        output = image.copy()
        prob = np.random.rand(*image.shape)
        output[prob < pepper_prob] = 0
        output[prob > 1 - salt_prob] = 255
        return output

class RayleighNoise(ImageOperation):
    def execute(self, image: np.ndarray, a: float = 0, b: float = 30) -> np.ndarray:
        image = ensure_grayscale(image).astype(np.float64)
        # Rayleigh distribution: R(x) = (x/b^2) * exp(-x^2 / 2b^2) for x >= 0
        # In practice, we can generate it using: b * sqrt(-2 * ln(1 - U)) + a
        u = np.random.rand(*image.shape)
        noise = a + b * np.sqrt(-2 * np.log(1 - u + 1e-10))
        output = image + noise
        return np.clip(output, 0, 255).astype(np.uint8)
