import os
import cv2
import numpy as np
import pytest
from src.infrastructure.detection.detection_operations import PointDetection, PointSharpening, LineDetection, LineSharpening
from src.infrastructure.enhancement.enhancement_operations import HistogramEqualization, ContrastStretching, BrightnessAdjustment, GammaCorrection, ImageNegative, LogTransformation
from src.infrastructure.filters.spatial.spatial_filters import MeanFilter, MedianFilter, MaxFilter, MinFilter, MidpointFilter, CustomWeightFilter
from src.infrastructure.filters.frequency.frequency_filters import IdealLowPassFilter, IdealHighPassFilter, ButterworthLowPassFilter, ButterworthHighPassFilter, GaussianLowPassFilter, GaussianHighPassFilter
from src.infrastructure.morphology.morphological_operations import Erosion, Dilation, Opening, Closing, BoundaryExtraction
from src.infrastructure.transform.transform_operations import RgbToGrayscale, GrayToBinary, RgbToBinary

@pytest.fixture(scope="module")
def test_image():
    image_path = os.path.join("assets", "test.jpg")
    if not os.path.exists(image_path):
        # Create a dummy image if not exists
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(img, (25, 25), (75, 75), (255, 255, 255), -1)
        os.makedirs("assets", exist_ok=True)
        cv2.imwrite(image_path, img)
    
    img = cv2.imread(image_path)
    return img

# --- Detection Operations ---
class TestDetection:
    def test_point_detection(self, test_image):
        op = PointDetection()
        result = op.execute(test_image)
        assert result is not None
        assert result.shape[:2] == test_image.shape[:2]

    def test_point_sharpening(self, test_image):
        op = PointSharpening()
        result = op.execute(test_image)
        assert result is not None
        assert result.shape == test_image.shape

    def test_line_detection(self, test_image):
        op = LineDetection()
        for direction in ['H', 'V', 'DL', 'DR']:
            result = op.execute(test_image, direction=direction)
            assert result is not None
            assert result.shape[:2] == test_image.shape[:2]

    def test_line_sharpening(self, test_image):
        op = LineSharpening()
        for direction in ['H', 'V', 'DL', 'DR']:
            result = op.execute(test_image, direction=direction)
            assert result is not None
            assert result.shape[:2] == test_image.shape[:2]

# --- Enhancement Operations ---
class TestEnhancement:
    def test_histogram_equalization(self, test_image):
        op = HistogramEqualization()
        result = op.execute(test_image)
        assert result is not None
        assert result.shape == test_image.shape

    def test_contrast_stretching(self, test_image):
        op = ContrastStretching()
        result = op.execute(test_image)
        assert result is not None
        assert result.shape == test_image.shape

    def test_brightness_adjustment(self, test_image):
        op = BrightnessAdjustment()
        result = op.execute(test_image, offset=50)
        assert result is not None
        assert result.shape == test_image.shape

    def test_gamma_correction(self, test_image):
        op = GammaCorrection()
        result = op.execute(test_image, gamma=1.5)
        assert result is not None

    def test_image_negative(self, test_image):
        op = ImageNegative()
        result = op.execute(test_image)
        assert result is not None
        assert result.shape == test_image.shape

    def test_log_transformation(self, test_image):
        op = LogTransformation()
        result = op.execute(test_image)
        assert result is not None
        assert result.shape == test_image.shape

# --- Spatial Filters ---
class TestSpatialFilters:
    def test_mean_filter(self, test_image):
        op = MeanFilter()
        result = op.execute(test_image, kernel_size=5)
        assert result is not None
        assert result.shape == test_image.shape

    def test_median_filter(self, test_image):
        op = MedianFilter()
        result = op.execute(test_image, kernel_size=3)
        assert result is not None
        assert result.shape == test_image.shape

    def test_max_filter(self, test_image):
        op = MaxFilter()
        result = op.execute(test_image, kernel_size=3)
        assert result is not None
        assert result.shape == test_image.shape

    def test_min_filter(self, test_image):
        op = MinFilter()
        result = op.execute(test_image, kernel_size=3)
        assert result is not None
        assert result.shape == test_image.shape

    def test_midpoint_filter(self, test_image):
        op = MidpointFilter()
        result = op.execute(test_image, kernel_size=3)
        assert result is not None
        assert result.shape == test_image.shape

    def test_custom_weight_filter(self, test_image):
        op = CustomWeightFilter()
        mask = np.ones((3,3)) / 9.0
        result = op.execute(test_image, mask=mask)
        assert result is not None
        assert result.shape == test_image.shape

# --- Frequency Filters ---
class TestFrequencyFilters:
    @pytest.mark.parametrize("filter_class", [
        IdealLowPassFilter, IdealHighPassFilter,
        ButterworthLowPassFilter, ButterworthHighPassFilter,
        GaussianLowPassFilter, GaussianHighPassFilter
    ])
    def test_frequency_filter(self, test_image, filter_class):
        op = filter_class()
        result = op.execute(test_image, d0=30)
        assert result is not None
        assert len(result.shape) == 2 # Frequency filters in this implementation convert to grayscale

# --- Morphological Operations ---
class TestMorphology:
    @pytest.mark.parametrize("op_class", [
        Erosion, Dilation, Opening, Closing, BoundaryExtraction
    ])
    def test_morphology_op(self, test_image, op_class):
        op = op_class()
        result = op.execute(test_image, kernel_size=3)
        assert result is not None
        assert len(result.shape) == 2

# --- Transform Operations ---
class TestTransform:
    def test_rgb_to_grayscale(self, test_image):
        op = RgbToGrayscale()
        result = op.execute(test_image)
        assert result is not None
        assert len(result.shape) == 2

    def test_gray_to_binary(self, test_image):
        op = GrayToBinary()
        # Convert to grayscale first for GrayToBinary
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        result = op.execute(gray)
        assert result is not None
        assert len(result.shape) == 2

    def test_rgb_to_binary(self, test_image):
        op = RgbToBinary()
        result = op.execute(test_image)
        assert result is not None
        assert len(result.shape) == 2
