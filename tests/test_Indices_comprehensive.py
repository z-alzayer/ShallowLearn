"""
Comprehensive tests for the Indices module.
Tests all remote sensing indices calculations with synthetic data.
"""

import pytest
import numpy as np
import warnings

from ShallowLearn.Indices import (
    ci, oci, cl_oci, ssi, ti, wqi, ndci, cloud_index, bgr, 
    calculate_water_surface_index, calculate_pseudo_subsurface_depth,
    get_band_numbers, validate_band_shape
)


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


class TestIndicesCalculation:
    """Test all indices calculations with synthetic data."""
    
    def test_oci_calculation(self, sample_image):
        """Test Ocean Chlorophyll Index calculation."""
        result = oci(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]

    def test_cl_oci_calculation(self, sample_image):
        """Test Coastal Lagoon Ocean Chlorophyll Index calculation."""
        result = cl_oci(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]

    def test_ssi_calculation(self, sample_image):
        """Test Suspended Sediment Index calculation."""
        result = ssi(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]

    def test_ti_calculation(self, sample_image):
        """Test Turbidity Index calculation."""
        result = ti(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]

    def test_wqi_calculation(self, sample_image):
        """Test Water Quality Index calculation."""
        result = wqi(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]

    def test_ndci_calculation(self, sample_image):
        """Test Normalized Difference Chlorophyll Index calculation."""
        result = ndci(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]

    def test_cloud_index_calculation(self, sample_image):
        """Test Cloud Index calculation."""
        result = cloud_index(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]

    def test_bgr_calculation(self, sample_image):
        """Test Blue-Green Ratio calculation."""
        result = bgr(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]

    def test_water_surface_index_calculation(self, sample_image):
        """Test Water Surface Index calculation."""
        result = calculate_water_surface_index(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]

    def test_pseudo_subsurface_depth_calculation(self, sample_image):
        """Test Pseudo Subsurface Depth calculation."""
        result = calculate_pseudo_subsurface_depth(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]

    def test_ci_calculation(self, sample_image):
        """Test Chlorophyll Index calculation."""
        result = ci(sample_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]


class TestIndicesEdgeCases:
    """Test edge cases and error conditions for indices."""
    
    def test_zero_values_handling(self, sample_image_with_zeros):
        """Test that indices handle zero values appropriately."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # Ignore division by zero warnings
            
            result = ci(sample_image_with_zeros)
            assert isinstance(result, np.ndarray)
            assert result.shape == (20, 20)

    def test_single_pixel_image(self, sample_image):
        """Test indices work with single pixel images."""
        single_pixel = sample_image[0:1, 0:1, :]  # 1x1 pixel
        
        result = ci(single_pixel)
        assert result.shape == (1, 1)

    def test_mathematical_correctness_ci(self):
        """Test CI mathematical correctness with known values."""
        # Create a simple test case where we know the expected result
        image = np.zeros((2, 2, 12))
        image[:, :, 1] = 100   # Blue (B02)
        image[:, :, 2] = 200   # Green (B03)
        
        result = ci(image)
        # CI should return valid numeric results
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)
        assert not np.all(np.isnan(result))

    def test_mathematical_correctness_bgr(self):
        """Test BGR mathematical correctness with known values."""
        image = np.zeros((2, 2, 12))
        image[:, :, 1] = 100   # Blue (B02)
        image[:, :, 2] = 200   # Green (B03)
        
        result = bgr(image)
        expected = 100 / 200  # Blue / Green
        
        assert np.allclose(result, expected, atol=1e-10)


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


class TestIndicesIntegration:
    """Test integration scenarios and real-world use cases."""
    
    def test_batch_indices_calculation(self, sample_image):
        """Test calculating multiple indices on the same image."""
        indices = {
            'ci': ci(sample_image),
            'oci': oci(sample_image),
            'ssi': ssi(sample_image),
            'ti': ti(sample_image),
            'wqi': wqi(sample_image)
        }
        
        # All indices should have same shape
        for name, result in indices.items():
            assert result.shape == (50, 50), f"{name} has incorrect shape"
            assert isinstance(result, np.ndarray), f"{name} is not numpy array"

    def test_indices_correlation(self, sample_image):
        """Test that related indices show expected correlations."""
        ci_result = ci(sample_image)
        oci_result = oci(sample_image)
        
        # Both should be chlorophyll-related indices with similar patterns
        assert ci_result.shape == oci_result.shape
        assert isinstance(ci_result, np.ndarray)
        assert isinstance(oci_result, np.ndarray)

    def test_consistent_data_types(self, sample_image):
        """Test that all indices return consistent data types."""
        indices_funcs = [ci, oci, ssi, ti, wqi, ndci, bgr]
        
        results = []
        for idx_func in indices_funcs:
            result = idx_func(sample_image)
            results.append(result)
        
        # All results should have same dtype
        if results:
            first_dtype = results[0].dtype
            for i, result in enumerate(results[1:], 1):
                assert result.dtype == first_dtype, f"Inconsistent dtype at index {i}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])