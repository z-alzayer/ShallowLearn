"""
Comprehensive tests for the Transform module.
Tests image transformations, scaling, and enhancement techniques.
"""

import pytest
import numpy as np
import warnings

try:
    from ShallowLearn.Transform import (
        BCET, linear_contrast_enhancement, mutliband_standard_scaler, 
        mutliband_robust_scaler
    )
    TRANSFORM_AVAILABLE = True
except ImportError as e:
    TRANSFORM_AVAILABLE = False
    print(f"Warning: Could not import Transform module: {e}")

try:
    from ShallowLearn.Transform import rgb_to_lab, lab_to_rgb, rgb_to_hsv, hsv_to_rgb
    COLOR_TRANSFORMS_AVAILABLE = True
except ImportError:
    COLOR_TRANSFORMS_AVAILABLE = False


@pytest.fixture
def sample_image_uint8():
    """Create a synthetic uint8 image for testing."""
    np.random.seed(42)
    image = np.random.randint(50, 200, (30, 30, 3), dtype=np.uint8)
    
    # Add some patterns
    image[10:20, 10:20, :] = 255  # Bright area
    image[0:5, 0:5, :] = 0        # Dark area
    
    return image


@pytest.fixture
def sample_image_float():
    """Create a synthetic float image for testing."""
    np.random.seed(42)
    image = np.random.rand(30, 30, 3) * 2000 + 1000  # Range [1000, 3000]
    
    # Add some patterns
    image[10:20, 10:20, :] = 4000  # Very bright area
    image[0:5, 0:5, :] = 100       # Very dark area
    
    return image.astype(np.float64)


@pytest.fixture
def sample_multispectral_image():
    """Create a synthetic multispectral image for testing."""
    np.random.seed(42)
    # 6 bands with different value ranges
    image = np.random.rand(40, 40, 6) * 1000
    
    # Make each band have different characteristics
    for band in range(6):
        image[:, :, band] = image[:, :, band] * (band + 1) + band * 500
    
    return image


@pytest.fixture
def sample_image_with_nans():
    """Create an image with NaN values for testing edge cases."""
    np.random.seed(42)
    image = np.random.rand(20, 20, 3) * 1000 + 500
    
    # Add NaN values
    image[0:3, 0:3, :] = np.nan
    image[10:12, 15:17, 1] = np.nan
    
    return image


@pytest.mark.skipif(not TRANSFORM_AVAILABLE, reason="Transform module not available")
class TestBCET:
    """Test Bias Correction and Enhancement Technique."""
    
    def test_bcet_basic(self, sample_image_uint8):
        """Test basic BCET transformation."""
        result = BCET(sample_image_uint8)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image_uint8.shape
        assert result.dtype == np.int32 or result.dtype == int
        
        # Check if transformation actually changes the image
        assert not np.array_equal(result, sample_image_uint8)

    def test_bcet_parameters(self, sample_image_float):
        """Test BCET with custom parameters."""
        result = BCET(sample_image_float, min_value=0, max_value=1000, desired_mean=500)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image_float.shape
        assert result.dtype == np.int32 or result.dtype == int
        
        # Check that output is roughly in the specified range
        assert np.min(result) >= -100  # Allow some tolerance
        assert np.max(result) <= 1100
        
        # Check that mean is approximately the desired mean
        assert abs(np.mean(result) - 500) < 200  # Allow reasonable tolerance

    def test_bcet_single_band(self, sample_image_uint8):
        """Test BCET on single band image."""
        single_band = sample_image_uint8[:, :, 0]
        result = BCET(single_band)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == single_band.shape
        assert result.dtype == np.int32 or result.dtype == int

    def test_bcet_uniform_image(self):
        """Test BCET on uniform image."""
        uniform_image = np.ones((10, 10)) * 128
        
        # BCET might have issues with uniform images (no variation)
        # The function should handle this gracefully or raise appropriate error
        try:
            result = BCET(uniform_image)
            assert isinstance(result, np.ndarray)
        except (ZeroDivisionError, ValueError):
            # This is acceptable for uniform images
            pass

    def test_bcet_extreme_values(self, sample_image_float):
        """Test BCET with extreme parameter values."""
        result = BCET(sample_image_float, min_value=-1000, max_value=5000, desired_mean=2000)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image_float.shape

    def test_bcet_mathematical_correctness(self):
        """Test BCET mathematical properties."""
        # Create simple test case
        test_image = np.array([[100, 150, 200]])
        result = BCET(test_image, min_value=0, max_value=255, desired_mean=127.5)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape
        # The transformation should be monotonic for this simple case
        assert result[0, 0] <= result[0, 1] <= result[0, 2]


@pytest.mark.skipif(not TRANSFORM_AVAILABLE, reason="Transform module not available")
class TestLinearContrastEnhancement:
    """Test Linear Contrast Enhancement (LCE)."""
    
    def test_lce_basic(self, sample_image_uint8):
        """Test basic linear contrast enhancement."""
        result = linear_contrast_enhancement(sample_image_uint8)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image_uint8.shape
        
        # After LCE, image should use full dynamic range
        assert np.min(result) == 0 or np.min(result) <= 10  # Allow small tolerance
        assert np.max(result) >= 245  # Should be close to 255

    def test_lce_custom_max(self, sample_image_float):
        """Test LCE with custom max value."""
        result = linear_contrast_enhancement(sample_image_float, max_value=1000)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_image_float.shape
        
        # Should scale to [0, 1000] range
        assert np.min(result) <= 10  # Close to 0
        assert np.max(result) >= 990  # Close to 1000

    def test_lce_single_band(self, sample_image_uint8):
        """Test LCE on single band image."""
        single_band = sample_image_uint8[:, :, 0]
        result = linear_contrast_enhancement(single_band)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == single_band.shape

    def test_lce_with_nans(self, sample_image_with_nans):
        """Test LCE handling of NaN values."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            result = linear_contrast_enhancement(sample_image_with_nans)
            
            assert isinstance(result, np.ndarray)
            assert result.shape == sample_image_with_nans.shape
            
            # NaN values should remain NaN
            original_nan_mask = np.isnan(sample_image_with_nans)
            result_nan_mask = np.isnan(result)
            assert np.array_equal(original_nan_mask, result_nan_mask)

    def test_lce_uniform_image(self):
        """Test LCE on uniform image (should raise ValueError)."""
        uniform_image = np.ones((10, 10)) * 128
        
        with pytest.raises(ValueError):
            linear_contrast_enhancement(uniform_image)

    def test_lce_mathematical_correctness(self):
        """Test LCE mathematical properties."""
        test_image = np.array([[100, 150, 200]], dtype=np.float64)
        result = linear_contrast_enhancement(test_image, max_value=255)
        
        # Should map 100->0, 200->255, 150->127.5
        expected = np.array([[0, 127.5, 255]])
        assert np.allclose(result, expected, atol=1e-10)


@pytest.mark.skipif(not TRANSFORM_AVAILABLE, reason="Transform module not available")
class TestScalingFunctions:
    """Test multiband scaling functions."""
    
    def test_standard_scaler_basic(self, sample_multispectral_image):
        """Test multiband standard scaler."""
        result = mutliband_standard_scaler(sample_multispectral_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_multispectral_image.shape
        
        # After standard scaling, each band should have mean~0, std~1
        for band in range(result.shape[2]):
            band_data = result[:, :, band].flatten()
            assert abs(np.mean(band_data)) < 1e-10  # Mean should be ~0
            assert abs(np.std(band_data) - 1.0) < 1e-10  # Std should be ~1

    def test_robust_scaler_basic(self, sample_multispectral_image):
        """Test multiband robust scaler."""
        result = mutliband_robust_scaler(sample_multispectral_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_multispectral_image.shape
        
        # After robust scaling, median should be ~0
        for band in range(result.shape[2]):
            band_data = result[:, :, band].flatten()
            assert abs(np.median(band_data)) < 1e-10

    def test_standard_scaler_single_band(self, sample_image_float):
        """Test standard scaler on single band."""
        single_band = sample_image_float[:, :, 0:1]  # Keep 3D shape
        result = mutliband_standard_scaler(single_band)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == single_band.shape
        
        # Check scaling properties
        band_data = result.flatten()
        assert abs(np.mean(band_data)) < 1e-10
        assert abs(np.std(band_data) - 1.0) < 1e-10

    def test_robust_scaler_single_band(self, sample_image_float):
        """Test robust scaler on single band."""
        single_band = sample_image_float[:, :, 0:1]  # Keep 3D shape
        result = mutliband_robust_scaler(single_band)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == single_band.shape

    def test_scalers_preserve_shape(self, sample_multispectral_image):
        """Test that scalers preserve image shape."""
        std_result = mutliband_standard_scaler(sample_multispectral_image)
        robust_result = mutliband_robust_scaler(sample_multispectral_image)
        
        assert std_result.shape == sample_multispectral_image.shape
        assert robust_result.shape == sample_multispectral_image.shape

    def test_scalers_different_results(self, sample_multispectral_image):
        """Test that different scalers produce different results."""
        std_result = mutliband_standard_scaler(sample_multispectral_image)
        robust_result = mutliband_robust_scaler(sample_multispectral_image)
        
        # Results should be different
        assert not np.allclose(std_result, robust_result, atol=1e-5)

    def test_scaler_with_outliers(self):
        """Test that robust scaler handles outliers better than standard scaler."""
        # Create image with outliers
        image = np.random.rand(20, 20, 3) * 100
        image[0, 0, :] = 10000  # Add outliers
        image[19, 19, :] = -5000
        
        std_result = mutliband_standard_scaler(image)
        robust_result = mutliband_robust_scaler(image)
        
        # Robust scaler should be less affected by outliers
        # Check that the bulk of the data is less extreme in robust scaling
        for band in range(3):
            std_band = std_result[:, :, band]
            robust_band = robust_result[:, :, band]
            
            # Remove the outlier pixels for comparison
            std_no_outliers = np.delete(std_band.flatten(), [0, -1])
            robust_no_outliers = np.delete(robust_band.flatten(), [0, -1])
            
            # Robust scaling should have less extreme values for non-outlier data
            assert np.std(robust_no_outliers) <= np.std(std_no_outliers) * 2  # Allow some tolerance


@pytest.mark.skipif(not TRANSFORM_AVAILABLE, reason="Transform module not available")
class TestTransformIntegration:
    """Test integration and combination of transform functions."""
    
    def test_transform_pipeline(self, sample_image_float):
        """Test applying multiple transforms in sequence."""
        # Apply transforms in pipeline
        lce_result = linear_contrast_enhancement(sample_image_float, max_value=255)
        bcet_result = BCET(lce_result.astype(np.uint8))
        scaled_result = mutliband_standard_scaler(bcet_result.astype(np.float64))
        
        assert isinstance(scaled_result, np.ndarray)
        assert scaled_result.shape == sample_image_float.shape

    def test_transform_reversibility_approximation(self, sample_image_uint8):
        """Test approximate reversibility of some transforms."""
        # LCE should be approximately reversible for this test
        original_min, original_max = np.min(sample_image_uint8), np.max(sample_image_uint8)
        
        # Apply LCE
        lce_result = linear_contrast_enhancement(sample_image_uint8.astype(np.float64))
        
        # "Reverse" by scaling back to original range
        reverse_result = (lce_result / 255.0) * (original_max - original_min) + original_min
        
        # Should be approximately equal (allowing for numerical precision)
        assert np.allclose(reverse_result, sample_image_uint8, atol=1.0)

    def test_transforms_preserve_relative_structure(self, sample_image_uint8):
        """Test that transforms preserve relative image structure."""
        # Create image with clear structure
        structured_image = np.zeros((20, 20, 3), dtype=np.uint8)
        structured_image[5:15, 5:15, :] = 128  # Square in middle
        structured_image[8:12, 8:12, :] = 255  # Bright center
        
        # Apply transforms
        lce_result = linear_contrast_enhancement(structured_image.astype(np.float64))
        bcet_result = BCET(structured_image)
        
        # Structure should be preserved (bright areas remain brightest)
        # Check that the bright center is still the brightest area
        center_region = lce_result[8:12, 8:12, :]
        outer_region = lce_result[0:5, 0:5, :]
        
        assert np.mean(center_region) > np.mean(outer_region)
        
        # Same for BCET
        center_region_bcet = bcet_result[8:12, 8:12, :]
        outer_region_bcet = bcet_result[0:5, 0:5, :]
        
        assert np.mean(center_region_bcet) > np.mean(outer_region_bcet)


@pytest.mark.skipif(not TRANSFORM_AVAILABLE, reason="Transform module not available")
class TestTransformEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_image(self):
        """Test transforms with empty image."""
        empty_image = np.array([], dtype=np.float64).reshape(0, 0, 3)
        
        # Transforms should handle empty images gracefully
        try:
            result = mutliband_standard_scaler(empty_image)
            assert result.shape == empty_image.shape
        except (ValueError, IndexError):
            # This is acceptable for empty images
            pass

    def test_single_pixel_image(self):
        """Test transforms with single pixel image."""
        single_pixel = np.array([[[100, 150, 200]]], dtype=np.float64)
        
        # Standard scaler should fail or handle gracefully (no variance)
        try:
            result = mutliband_standard_scaler(single_pixel)
            # If it doesn't fail, result should have same shape
            assert result.shape == single_pixel.shape
        except (ValueError, RuntimeWarning):
            # Acceptable for single pixel (no variance to normalize)
            pass

    def test_transforms_with_infinite_values(self):
        """Test transforms with infinite values."""
        image_with_inf = np.random.rand(10, 10, 3) * 1000
        image_with_inf[0, 0, :] = np.inf
        image_with_inf[5, 5, :] = -np.inf
        
        # Transforms should handle or reject infinite values
        try:
            result = linear_contrast_enhancement(image_with_inf)
            # If it succeeds, check that infinities are handled
            assert not np.any(np.isinf(result[1:, 1:, :]))  # Non-inf areas should be finite
        except (ValueError, RuntimeWarning):
            # This is acceptable behavior for infinite values
            pass

    def test_very_large_images(self):
        """Test transforms with larger images (memory/performance test)."""
        # Create moderately large image
        large_image = np.random.rand(200, 200, 6) * 1000
        
        # Should complete without memory issues
        std_result = mutliband_standard_scaler(large_image)
        assert std_result.shape == large_image.shape
        
        robust_result = mutliband_robust_scaler(large_image)
        assert robust_result.shape == large_image.shape
        
        lce_result = linear_contrast_enhancement(large_image)
        assert lce_result.shape == large_image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])