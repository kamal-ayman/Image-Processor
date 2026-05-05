import cv2
import numpy as np
from ...domain.image_operation import ImageOperation
from ...utils.image_utils import ensure_grayscale

class PointDetection(ImageOperation):
    def execute(self, image: np.ndarray, threshold: float = 100) -> np.ndarray:
        image = ensure_grayscale(image)
        mask = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        filtered = cv2.filter2D(image.astype(np.float64), -1, mask)
        result = np.zeros_like(image)
        result[np.abs(filtered) > threshold] = 255
        return result

class PointSharpening(ImageOperation):
    def execute(self, image: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        img_float = image.astype(np.float64)
        laplacian_mask = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
        edges = cv2.filter2D(img_float, -1, laplacian_mask)
        # Original - (alpha * edges) because Laplacian is negative at peak
        sharpened = img_float - (alpha * edges) 
        return np.clip(sharpened, 0, 255).astype(np.uint8)

class LineDetection(ImageOperation):
    def execute(self, image: np.ndarray, direction: str = 'H', mask_type: str = 'sobel') -> np.ndarray:
        image = ensure_grayscale(image)
        if mask_type == 'sobel':
            if direction == 'H':
                mask = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            elif direction == 'V':
                mask = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            elif direction == 'DL':
                mask = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])
            else: # DR
                mask = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
        else: # Prewitt-like or other
            if direction == 'H':
                mask = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
            elif direction == 'V':
                mask = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
            elif direction == 'DL':
                mask = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])
            else: # DR
                mask = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]])

        result = cv2.filter2D(image, -1, mask)
        return np.clip(result, 0, 255).astype(np.uint8)

class LineSharpening(ImageOperation):
    def execute(self, image: np.ndarray, direction: str = 'H', alpha: float = 1.0) -> np.ndarray:
        image = ensure_grayscale(image)
        if direction == 'H':
            mask = np.array([[-1, -1, -1], [alpha+2, alpha+2, alpha+2], [-1, -1, -1]])
        elif direction == 'V':
            mask = np.array([[-1, alpha+2, -1], [-1, alpha+2, -1], [-1, alpha+2, -1]])
        elif direction == 'DL':
            mask = np.array([[-1, -1, alpha+2], [-1, alpha+2, -1], [alpha+2, -1, -1]])
        else: # DR
            mask = np.array([[alpha+2, -1, -1], [-1, alpha+2, -1], [-1, -1, alpha+2]])
        
        result = cv2.filter2D(image, -1, mask)
        return np.clip(result, 0, 255).astype(np.uint8)
