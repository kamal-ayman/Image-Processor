import cv2
import numpy as np
from ...domain.image_operation import ImageOperation
from ...utils.image_utils import ensure_grayscale

class LaplacianFilter(ImageOperation):
    def execute(self, image: np.ndarray) -> np.ndarray:
        image = ensure_grayscale(image)
        # Using a standard Laplacian mask
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        return cv2.convertScaleAbs(laplacian)

class RobertCrossFilter(ImageOperation):
    def execute(self, image: np.ndarray) -> np.ndarray:
        image = ensure_grayscale(image).astype(np.float64)
        kernel_x = np.array([[1, 0], [0, -1]])
        kernel_y = np.array([[0, 1], [-1, 0]])
        
        gx = cv2.filter2D(image, -1, kernel_x)
        gy = cv2.filter2D(image, -1, kernel_y)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        return np.clip(magnitude, 0, 255).astype(np.uint8)

class UnsharpMasking(ImageOperation):
    def execute(self, image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.5) -> np.ndarray:
        image_float = image.astype(np.float64)
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        mask = image_float - blurred.astype(np.float64)
        sharpened = image_float + amount * mask
        return np.clip(sharpened, 0, 255).astype(np.uint8)
