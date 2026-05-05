import cv2
import numpy as np
from ...domain.image_operation import ImageOperation
from ...utils.image_utils import ensure_grayscale

class HistogramEqualization(ImageOperation):
    def execute(self, image: np.ndarray) -> np.ndarray:
        if len(image.shape) == 3:
            img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return cv2.equalizeHist(image)

class ContrastStretching(ImageOperation):
    def execute(self, image: np.ndarray) -> np.ndarray:
        img_float = image.astype(np.float64)
        low, high = np.min(img_float), np.max(img_float)
        if high - low == 0: return image
        out = (img_float - low) * (255.0 / (high - low))
        return out.astype(np.uint8)

class BrightnessAdjustment(ImageOperation):
    def execute(self, image: np.ndarray, offset: float = 30) -> np.ndarray:
        return np.clip(image.astype(np.float64) + offset, 0, 255).astype(np.uint8)

class GammaCorrection(ImageOperation):
    def execute(self, image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        table = np.array([((i / 255.0) ** gamma) * 255 
                          for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

class ImageNegative(ImageOperation):
    def execute(self, image: np.ndarray) -> np.ndarray:
        return (255 - image).astype(np.uint8)

class LogTransformation(ImageOperation):
    def execute(self, image: np.ndarray) -> np.ndarray:
        img_float = image.astype(np.float64)
        c = 255 / np.log(1 + np.max(img_float))
        out = c * np.log(1 + img_float)
        return out.astype(np.uint8)
