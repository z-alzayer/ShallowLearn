"""
Test cloud detection functionality.
"""

import numpy as np
import pytest
from pathlib import Path

from ShallowLearn.ml.cloud_detector import (
    detect_clouds,
    add_nan_buffer,
    percentile_without_zeros,
    percentile_without_zeros_and_first,
    load_img_model,
    load_img_planet,
    get_default_model_path,
)


class TestCloudDetection:
    """Test cloud detection functions."""
    
    def test_detect_clouds_basic(self):
        """Test basic cloud detection on time series."""
        # Create synthetic datacube (time, x, y, channels)
        datacube = np.random.rand(5, 10, 10, 3) * 100
        
        # Add some "clouds" (high values)
        datacube[2, 4:7, 4:7, :] = 200  # Cloud in frame 2
        
        cloud_mask = detect_clouds(datacube, threshold=50, window_size=3)
        
        assert cloud_mask.shape == datacube.shape
        assert cloud_mask.dtype == bool
        # Check that high values are detected as clouds
        assert np.any(cloud_mask[2, 4:7, 4:7, :])
    
    def test_detect_clouds_edge_cases(self):
        """Test cloud detection with edge cases."""
        # Single frame
        datacube = np.random.rand(1, 5, 5, 2) * 100
        cloud_mask = detect_clouds(datacube, window_size=1)
        assert cloud_mask.shape == datacube.shape
        
        # Large window size
        datacube = np.random.rand(3, 5, 5, 2) * 100
        cloud_mask = detect_clouds(datacube, window_size=5)
        assert cloud_mask.shape == datacube.shape
    
    def test_add_nan_buffer(self):
        """Test NaN buffer addition."""
        # Create array with some NaN values
        arr = np.ones((10, 10))
        arr[4:6, 4:6] = np.nan
        
        # Add buffer
        result = add_nan_buffer(arr, dilation_size=3)
        
        # Check that NaN region has expanded
        nan_count_original = np.sum(np.isnan(arr))
        nan_count_buffered = np.sum(np.isnan(result))
        assert nan_count_buffered > nan_count_original
    
    def test_percentile_without_zeros(self):
        """Test percentile calculation excluding zeros."""
        # Array with zeros
        arr = np.array([0, 0, 1, 2, 3, 4, 5])
        result = percentile_without_zeros(arr, 50)
        assert result == 3.0  # Median of [1, 2, 3, 4, 5]
        
        # All zeros
        arr = np.zeros(10)
        result = percentile_without_zeros(arr, 50)
        assert np.isnan(result)
        
        # No zeros
        arr = np.array([1, 2, 3, 4, 5])
        result = percentile_without_zeros(arr, 50)
        assert result == 3.0
    
    def test_percentile_without_zeros_and_first(self):
        """Test percentile excluding zeros and repeated smallest values."""
        # Array with repeated smallest values
        arr = np.array([0, 0, 1, 1, 1, 2, 3, 4, 5])
        result = percentile_without_zeros_and_first(arr, 50)
        # Should exclude 0s and the repeated 1s, median of [2, 3, 4, 5]
        assert result == 3.5
        
        # All same non-zero value
        arr = np.array([2, 2, 2, 2])
        result = percentile_without_zeros_and_first(arr, 50)
        assert np.isnan(result)  # After removing duplicates, nothing left
    
    def test_load_img_model(self):
        """Test image loading for Sentinel-2 model."""
        # Create synthetic Sentinel-2 image (10 bands)
        img = np.random.rand(20, 20, 10)
        
        prepared, shape, original = load_img_model(img, processed=False)
        
        # Check shapes
        assert prepared.shape == (400, 4)  # 20*20 pixels, 4 bands
        assert shape == (20, 20, 10)
        assert np.array_equal(original, img)
        
        # Test processed mode
        img_processed = np.random.rand(20, 20, 4)
        prepared, shape, original = load_img_model(img_processed, processed=True)
        assert prepared.shape == (400, 4)
    
    def test_load_img_planet(self):
        """Test image loading for Planet model."""
        # Create synthetic Planet image (4 bands)
        img = np.random.rand(20, 20, 4)
        
        prepared, shape, original = load_img_planet(img, processed=False)
        
        # Check shapes
        assert prepared.shape == (400, 4)  # 20*20 pixels, 4 bands
        assert shape == (20, 20, 4)
        assert np.array_equal(original, img)
    
    def test_get_default_model_path(self):
        """Test default model path retrieval."""
        try:
            path = get_default_model_path()
            assert isinstance(path, Path)
            # Should point to CloudDetectXGB.json
            assert path.name == "CloudDetectXGB.json"
            assert path.exists()
        except FileNotFoundError:
            # Model might not exist in test environment
            pytest.skip("Model file not found in test environment")


class TestCloudRegressorIntegration:
    """Integration tests for cloud regressor (requires model file)."""
    
    @pytest.mark.skipif(
        not Path("/home/zba21/Documents/ShallowLearn/Models/CloudDetectXGB.json").exists(),
        reason="Model file not available"
    )
    def test_cloud_regressor_basic(self):
        """Test basic cloud regressor functionality."""
        from ShallowLearn.ml.cloud_detector import cloud_regressor
        
        # Create synthetic Sentinel-2 image
        img = np.random.rand(50, 50, 10) * 1000
        
        # Test mask generation
        mask = cloud_regressor(img, return_mask=True)
        assert mask.shape == (50, 50, 1)
        
        # Test masked image generation
        masked_img = cloud_regressor(img, return_mask=False)
        assert masked_img.shape == img.shape