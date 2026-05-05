import cv2
import numpy as np

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts a BGR image to grayscale using luminosity method."""
    if len(image.shape) == 3:
        # 0.299*R + 0.587*G + 0.114*B
        # OpenCV uses BGR order
        weights = np.array([0.114, 0.587, 0.299])
        gray = np.dot(image[..., :3], weights)
        return gray.astype(np.uint8)
    return image

def ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Ensures the image is grayscale. Converts if necessary."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalizes image to 0-255 range and converts to uint8."""
    img_min, img_max = image.min(), image.max()
    if img_max - img_min == 0:
        return image.astype(np.uint8)
    normalized = 255 * (image - img_min) / (img_max - img_min)
    return normalized.astype(np.uint8)
