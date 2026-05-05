import numpy as np
from ....domain.image_operation import ImageOperation
from ....utils.image_utils import ensure_grayscale, normalize_image

class FrequencyFilter(ImageOperation):
    """Base class for frequency domain filters."""
    
    def _apply_filter_mask(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # 1. Fourier Transform
        f = np.fft.fft2(image.astype(np.float64))
        f_shift = np.fft.fftshift(f)
        
        # 2. Apply Mask
        filtered = f_shift * mask
        
        # 3. Inverse Fourier Transform
        f_ishift = np.fft.ifftshift(filtered)
        result = np.real(np.fft.ifft2(f_ishift))
        
        return normalize_image(result)

    def _get_distance_grid(self, shape):
        rows, cols = shape
        u = np.linspace(0, rows - 1, rows)
        v = np.linspace(0, cols - 1, cols)
        u_grid, v_grid = np.meshgrid(u, v, indexing='ij')
        return np.sqrt((u_grid - rows/2)**2 + (v_grid - cols/2)**2)

class IdealLowPassFilter(FrequencyFilter):
    def execute(self, image: np.ndarray, d0: float = 30) -> np.ndarray:
        image = ensure_grayscale(image)
        dist = self._get_distance_grid(image.shape)
        mask = np.zeros_like(image, dtype=np.float64)
        mask[dist <= d0] = 1
        return self._apply_filter_mask(image, mask)

class IdealHighPassFilter(FrequencyFilter):
    def execute(self, image: np.ndarray, d0: float = 30) -> np.ndarray:
        image = ensure_grayscale(image)
        dist = self._get_distance_grid(image.shape)
        mask = np.zeros_like(image, dtype=np.float64)
        mask[dist > d0] = 1
        return self._apply_filter_mask(image, mask)

class ButterworthLowPassFilter(FrequencyFilter):
    def execute(self, image: np.ndarray, d0: float = 30, n: int = 2) -> np.ndarray:
        image = ensure_grayscale(image)
        dist = self._get_distance_grid(image.shape)
        mask = 1 / (1 + (dist / d0)**(2 * n))
        return self._apply_filter_mask(image, mask)

class ButterworthHighPassFilter(FrequencyFilter):
    def execute(self, image: np.ndarray, d0: float = 30, n: int = 2) -> np.ndarray:
        image = ensure_grayscale(image)
        dist = self._get_distance_grid(image.shape)
        mask = 1 / (1 + (d0 / (dist + 1e-10))**(2 * n))
        return self._apply_filter_mask(image, mask)

class GaussianLowPassFilter(FrequencyFilter):
    def execute(self, image: np.ndarray, d0: float = 30) -> np.ndarray:
        image = ensure_grayscale(image)
        dist = self._get_distance_grid(image.shape)
        mask = np.exp(-(dist**2) / (2 * (d0**2)))
        return self._apply_filter_mask(image, mask)

class GaussianHighPassFilter(FrequencyFilter):
    def execute(self, image: np.ndarray, d0: float = 30) -> np.ndarray:
        image = ensure_grayscale(image)
        dist = self._get_distance_grid(image.shape)
        mask = 1 - np.exp(-(dist**2) / (2 * (d0**2)))
        return self._apply_filter_mask(image, mask)
