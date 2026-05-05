import cv2
import numpy as np
from typing import Optional, Dict, Any
from ..domain.image_operation import ImageOperation

class ImageService:
    def __init__(self):
        self._original_image: Optional[np.ndarray] = None
        self._current_image: Optional[np.ndarray] = None

    def load_image(self, path: str) -> bool:
        image = cv2.imread(path)
        if image is not None:
            self._original_image = image
            self._current_image = image.copy()
            return True
        return False

    def get_current_image(self) -> Optional[np.ndarray]:
        return self._current_image

    def get_original_image(self) -> Optional[np.ndarray]:
        return self._original_image

    def reset(self):
        if self._original_image is not None:
            self._current_image = self._original_image.copy()

    def apply_operation(self, operation: ImageOperation, **kwargs):
        if self._original_image is not None:
            # We usually apply operations on the original image for single filters
            # or on the current image for chained filters.
            # For this project, let's assume single operation from original.
            self._current_image = operation.execute(self._original_image, **kwargs)

    def apply_on_current(self, operation: ImageOperation, **kwargs):
        if self._current_image is not None:
            self._current_image = operation.execute(self._current_image, **kwargs)
