import numpy as np
import pytest
from ShallowLearn.features import computer_vision_features as cvf

class TestFeatureExtractors:
    @classmethod
    def setup_class(cls):
        # Create test images (adjust sizes as needed)
        cls.color_image = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        cls.gray_image = np.random.randint(0, 256, (64, 64, 1), dtype=np.uint8)

    def test_edge_density(self):
        result = cvf.edge_density(self.color_image)
        assert result.shape == self.color_image.shape[:2]

    def test_texture_features(self):
        texture = cvf.texture_features(self.color_image)
        assert texture.shape == self.color_image.shape[:2]

    def test_color_histogram(self):
        hist = cvf.color_histogram(self.color_image)
        assert hist.shape == (self.color_image.shape[-1], 32)

    def test_sobel_edge_detection(self):
        edges = cvf.sobel_edge_detection(self.color_image)
        assert edges.shape == self.color_image.shape[:2]

    def test_gabor_features(self):
        response = cvf.gabor_features(self.color_image)
        assert response.shape == self.color_image.shape[:2]

    def test_histogram_of_oriented_gradients(self):
        hog_img = cvf.histogram_of_oriented_gradients(self.color_image)
        assert hog_img.shape == self.color_image.shape[:2]
