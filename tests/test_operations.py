import os
import cv2
import numpy as np
import pytest
from src.infrastructure.detection.detection_operations import RobertCrossFilter, LaplacianFilter, UnsharpMasking
from src.infrastructure.filters.spatial.spatial_filters import WeightedAverageFilter, BilateralFilter
from src.infrastructure.filters.frequency.frequency_filters import IdealLowPassFilter, GaussianHighPassFilter
from src.infrastructure.morphology.morphological_operations import Dilation, Opening
from src.infrastructure.noise.noise_operations import RayleighNoise, ImpulseNoise, ExponentialNoise

@pytest.fixture(scope="module")
def test_image():
    image_path = os.path.join("assets", "test.jpg")
    if not os.path.exists(image_path):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (25, 25), (75, 75), (255, 255, 255), -1)
        os.makedirs("assets", exist_ok=True)
        cv2.imwrite(image_path, img)
    
    img = cv2.imread(image_path)
    return img

class TestSpatial:
    def test_weighted_average(self, test_image):
        op = WeightedAverageFilter()
        result = op.execute(test_image)
        assert result is not None
        assert result.shape == test_image.shape

    def test_bilateral(self, test_image):
        op = BilateralFilter()
        result = op.execute(test_image)
        assert result is not None
        assert result.shape == test_image.shape

class TestFrequency:
    def test_ideal_lps(self, test_image):
        op = IdealLowPassFilter()
        result = op.execute(test_image)
        assert result is not None
        assert len(result.shape) == 2

    def test_gaussian_hpf(self, test_image):
        op = GaussianHighPassFilter()
        result = op.execute(test_image)
        assert result is not None
        assert len(result.shape) == 2

class TestMorphology:
    def test_dilation(self, test_image):
        op = Dilation()
        result = op.execute(test_image)
        assert result is not None
        assert len(result.shape) == 2

    def test_opening(self, test_image):
        op = Opening()
        result = op.execute(test_image)
        assert result is not None
        assert len(result.shape) == 2

class TestNoise:
    def test_rayleigh(self, test_image):
        op = RayleighNoise()
        result = op.execute(test_image)
        assert result is not None

    def test_impulse(self, test_image):
        op = ImpulseNoise()
        result = op.execute(test_image)
        assert result is not None

    def test_exponential(self, test_image):
        op = ExponentialNoise()
        result = op.execute(test_image)
        assert result is not None

class TestDetection:
    def test_robert_cross(self, test_image):
        op = RobertCrossFilter()
        result = op.execute(test_image)
        assert result is not None

    def test_laplacian(self, test_image):
        op = LaplacianFilter()
        result = op.execute(test_image)
        assert result is not None

    def test_unsharp_masking(self, test_image):
        op = UnsharpMasking()
        result = op.execute(test_image)
        assert result is not None
