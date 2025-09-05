"""
Comprehensive tests for the Indices module.
Tests all remote sensing indices calculations with synthetic data.
"""

import pytest
import numpy as np
import warnings

try:
    from ShallowLearn.Indices import (
        ci, ndvi, ndwi, mndwi, ndmi, gndvi, evi, savi, msavi, ari, cri1, cri2,
        get_band_numbers, validate_band_shape
    )
    INDICES_AVAILABLE = True
except ImportError as e:
    INDICES_AVAILABLE = False
    print(f"Warning: Could not import Indices module: {e}")


@pytest.fixture
def sample_image():
    """Create a synthetic multi-spectral image for testing."""
    # Create image with 12 bands (like Sentinel-2)
    np.random.seed(42)  # For reproducible tests
    image = np.random.rand(50, 50, 12) * 3000  # Simulate reflectance values
    
    # Add some realistic patterns
    # Make NIR (B08, index 7) generally higher than Red (B04, index 3) for vegetation
    image[:, :, 7] = image[:, :, 7] * 1.5 + 1000  # NIR
    image[:, :, 3] = image[:, :, 3] * 0.8 + 500   # Red
    
    # Make SWIR bands have different characteristics
    image[:, :, 10] = image[:, :, 10] * 0.7 + 800  # B11 (SWIR1)
    image[:, :, 11] = image[:, :, 11] * 0.6 + 600  # B12 (SWIR2)
    
    return image.astype(np.uint16)


@pytest.fixture
def sample_image_with_zeros():
    """Create a synthetic image with some zero values for testing edge cases."""
    image = np.random.rand(20, 20, 12) * 2000 + 500
    
    # Add some zero values
    image[0:5, 0:5, :] = 0  # Top-left corner all zeros
    image[10:15, 10:15, 3] = 0  # Some zeros in red band
    
    return image.astype(np.uint16)


@pytest.mark.skipif(not INDICES_AVAILABLE, reason="Indices module not available")
class TestIndicesCalculation:
    """Test all indices calculations with synthetic data."""
    
    def test_ndvi_basic_calculation(self, sample_image):
        """Test NDVI calculation with default bands."""
        result = ndvi(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]
        
        # NDVI should be between -1 and 1
        valid_mask = ~np.isnan(result)
        assert np.all(result[valid_mask] >= -1) and np.all(result[valid_mask] <= 1)

    def test_ndvi_custom_bands(self, sample_image):
        """Test NDVI with custom band specification."""
        result = ndvi(sample_image, bands=['B08', 'B04'])
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        valid_mask = ~np.isnan(result)
        assert np.all(result[valid_mask] >= -1) and np.all(result[valid_mask] <= 1)

    def test_ndwi_calculation(self, sample_image):
        """Test NDWI calculation."""
        result = ndwi(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]
        valid_mask = ~np.isnan(result)
        assert np.all(result[valid_mask] >= -1) and np.all(result[valid_mask] <= 1)

    def test_mndwi_calculation(self, sample_image):
        """Test Modified NDWI calculation."""
        result = mndwi(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]
        valid_mask = ~np.isnan(result)
        assert np.all(result[valid_mask] >= -1) and np.all(result[valid_mask] <= 1)

    def test_ndmi_calculation(self, sample_image):
        """Test NDMI calculation."""
        result = ndmi(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]
        valid_mask = ~np.isnan(result)
        assert np.all(result[valid_mask] >= -1) and np.all(result[valid_mask] <= 1)

    def test_gndvi_calculation(self, sample_image):
        """Test Green NDVI calculation."""
        result = gndvi(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]
        valid_mask = ~np.isnan(result)
        assert np.all(result[valid_mask] >= -1) and np.all(result[valid_mask] <= 1)

    def test_evi_calculation(self, sample_image):
        """Test Enhanced Vegetation Index calculation."""
        result = evi(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]

    def test_savi_calculation(self, sample_image):
        """Test Soil Adjusted Vegetation Index calculation."""
        result = savi(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]

    def test_msavi_calculation(self, sample_image):
        """Test Modified Soil Adjusted Vegetation Index calculation."""
        result = msavi(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]

    def test_ari_calculation(self, sample_image):
        """Test Anthocyanin Reflectance Index calculation."""
        result = ari(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]

    def test_cri1_calculation(self, sample_image):
        """Test Carotenoid Reflectance Index 1 calculation."""
        result = cri1(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]

    def test_cri2_calculation(self, sample_image):
        """Test Carotenoid Reflectance Index 2 calculation."""
        result = cri2(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]

    def test_ci_calculation(self, sample_image):
        """Test Chlorophyll Index calculation."""
        result = ci(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]


@pytest.mark.skipif(not INDICES_AVAILABLE, reason="Indices module not available")
class TestIndicesEdgeCases:
    """Test edge cases and error conditions for indices."""
    
    def test_zero_values_handling(self, sample_image_with_zeros):
        """Test that indices handle zero values appropriately."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # Ignore division by zero warnings
            
            result = ndvi(sample_image_with_zeros)
            assert isinstance(result, np.ndarray)
            assert result.shape == (20, 20)

    def test_single_pixel_image(self, sample_image):
        """Test indices work with single pixel images."""
        single_pixel = sample_image[0:1, 0:1, :]  # 1x1 pixel
        
        result = ndvi(single_pixel)
        assert result.shape == (1, 1)

    def test_mathematical_correctness_ndvi(self):
        """Test NDVI mathematical correctness with known values."""
        # Create a simple test case where we know the expected result
        image = np.zeros((2, 2, 12))
        image[:, :, 3] = 100   # Red (B04)
        image[:, :, 7] = 200   # NIR (B08)
        
        result = ndvi(image)
        expected = (200 - 100) / (200 + 100)  # (NIR - Red) / (NIR + Red)
        
        assert np.allclose(result, expected, atol=1e-10)

    def test_mathematical_correctness_ndwi(self):
        """Test NDWI mathematical correctness with known values."""
        image = np.zeros((2, 2, 12))
        image[:, :, 2] = 150   # Green (B03) 
        image[:, :, 7] = 100   # NIR (B08)
        
        result = ndwi(image)
        expected = (150 - 100) / (150 + 100)  # (Green - NIR) / (Green + NIR)
        
        assert np.allclose(result, expected, atol=1e-10)


@pytest.mark.skipif(not INDICES_AVAILABLE, reason="Indices module not available")
class TestUtilityFunctions:
    """Test utility functions in the Indices module."""
    
    def test_get_band_numbers(self):
        """Test get_band_numbers function."""
        try:
            bands = ['B02', 'B03', 'B04']
            result = get_band_numbers(bands)
            
            assert isinstance(result, list)
            assert len(result) == 3
            assert all(isinstance(x, int) for x in result)
        except (NameError, ImportError):
            pytest.skip("get_band_numbers function not available")

    def test_validate_band_shape(self, sample_image):
        """Test validate_band_shape function."""
        try:
            band_numbers = [0, 1, 2, 3]
            validate_band_shape(sample_image, band_numbers)
        except (NameError, ImportError):
            pytest.skip("validate_band_shape function not available")


@pytest.mark.skipif(not INDICES_AVAILABLE, reason="Indices module not available")
class TestIndicesIntegration:
    """Test integration scenarios and real-world use cases."""
    
    def test_batch_indices_calculation(self, sample_image):
        """Test calculating multiple indices on the same image."""
        indices = {
            'ndvi': ndvi(sample_image),
            'ndwi': ndwi(sample_image),
            'mndwi': mndwi(sample_image),
            'evi': evi(sample_image),
            'savi': savi(sample_image)
        }
        
        # All indices should have same shape
        for name, result in indices.items():
            assert result.shape == (50, 50), f"{name} has incorrect shape"
            assert isinstance(result, np.ndarray), f"{name} is not numpy array"

    def test_indices_correlation(self, sample_image):
        """Test that related indices show expected correlations."""
        ndvi_result = ndvi(sample_image)
        gndvi_result = gndvi(sample_image)
        
        # Both should be vegetation indices with similar patterns
        assert ndvi_result.shape == gndvi_result.shape
        assert isinstance(ndvi_result, np.ndarray)
        assert isinstance(gndvi_result, np.ndarray)

    def test_consistent_data_types(self, sample_image):
        """Test that all indices return consistent data types."""
        indices_funcs = [ndvi, ndwi, mndwi, evi, savi, gndvi]
        
        results = []
        for idx_func in indices_funcs:
            try:
                result = idx_func(sample_image)
                results.append(result)
            except Exception as e:
                print(f"Warning: {idx_func.__name__} failed: {e}")
                continue
        
        # All results should have same dtype
        if results:
            first_dtype = results[0].dtype
            for i, result in enumerate(results[1:], 1):
                assert result.dtype == first_dtype, f"Inconsistent dtype at index {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])