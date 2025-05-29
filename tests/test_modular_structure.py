import pytest
import numpy as np
from unittest.mock import Mock, patch
import tempfile

# Test the new modular structure
from ShallowLearn.core.array_utils import (
    clip_array, select_channels, remove_channel, apply_mask,
    validate_band_shape, get_band_numbers, normalize_array
)
from ShallowLearn.spectral.indices import (
    chlorophyll_index, ocean_color_index, suspended_sediment_index,
    water_quality_index, normalized_difference_chlorophyll_index,
    DEFAULT_SENTINEL2_MAPPING
)
from ShallowLearn.segmentation.superpixels import (
    slic_segmentation, felzenszwalb_segmentation, extract_patches,
    pca_segments, cluster_segments, process_superpixel_pipeline
)
from ShallowLearn.visualization.display import (
    create_rgb_image, plot_rgb, plot_histogram, plot_discrete_image
)


class TestCoreArrayUtils:
    """Test core array utilities with minimal dependencies."""
    
    def test_clip_array(self):
        """Test array clipping functionality."""
        arr = np.array([[-100, 0, 5000, 15000]], dtype=float)
        clipped = clip_array(arr, min_val=0, max_val=10000)
        
        assert np.array_equal(clipped, [[0, 0, 5000, 10000]])
        assert clipped.dtype == arr.dtype

    def test_clip_array_custom_range(self):
        """Test array clipping with custom range."""
        arr = np.array([[10, 20, 30, 40]], dtype=float)
        clipped = clip_array(arr, min_val=15, max_val=35)
        
        assert np.array_equal(clipped, [[15, 20, 30, 35]])

    def test_select_channels(self):
        """Test channel selection."""
        img = np.random.rand(10, 10, 5)
        indices = [0, 2, 4]
        
        result = select_channels(img, indices)
        
        assert result.shape == (10, 10, 3)
        assert np.array_equal(result[:, :, 0], img[:, :, 0])
        assert np.array_equal(result[:, :, 1], img[:, :, 2])
        assert np.array_equal(result[:, :, 2], img[:, :, 4])

    def test_select_channels_invalid_length(self):
        """Test channel selection with invalid number of channels."""
        img = np.random.rand(10, 10, 5)
        
        with pytest.raises(ValueError, match="length of indices must be 3"):
            select_channels(img, [0, 1])  # Only 2 indices
        
        with pytest.raises(ValueError, match="length of indices must be 3"):
            select_channels(img, [0, 1, 2, 3])  # 4 indices

    def test_remove_channel(self):
        """Test channel removal."""
        img = np.random.rand(10, 10, 5)
        
        result = remove_channel(img, 2)
        
        assert result.shape == (10, 10, 4)
        assert np.array_equal(result[:, :, 0], img[:, :, 0])
        assert np.array_equal(result[:, :, 1], img[:, :, 1])
        assert np.array_equal(result[:, :, 2], img[:, :, 3])
        assert np.array_equal(result[:, :, 3], img[:, :, 4])

    def test_remove_channel_invalid_index(self):
        """Test channel removal with invalid index."""
        img = np.random.rand(10, 10, 5)
        
        with pytest.raises(ValueError, match="Channel index is out of bounds"):
            remove_channel(img, 5)
        
        with pytest.raises(ValueError, match="Channel index is out of bounds"):
            remove_channel(img, -1)

    def test_apply_mask(self):
        """Test mask application."""
        data = np.array([[1, 2, 3], [4, 5, 6]])
        mask = np.array([[True, False, True], [False, True, False]])
        
        result = apply_mask(data, mask, fill_value=0)
        expected = np.array([[1, 0, 3], [0, 5, 0]])
        
        assert np.array_equal(result, expected)

    def test_validate_band_shape(self):
        """Test band shape validation."""
        img = np.random.rand(10, 10, 5)
        
        # Valid band numbers
        validate_band_shape(img, [0, 1, 2])  # Should not raise
        
        # Invalid band numbers
        with pytest.raises(ValueError, match="out of bounds"):
            validate_band_shape(img, [5])
        
        with pytest.raises(ValueError, match="out of bounds"):
            validate_band_shape(img, [0, 1, 5])

    def test_validate_band_shape_2d_image(self):
        """Test validation with 2D image."""
        img = np.random.rand(10, 10)
        
        with pytest.raises(ValueError, match="must be 3D array"):
            validate_band_shape(img, [0])

    def test_get_band_numbers(self):
        """Test band number conversion."""
        band_mapping = {
            'B02': {'index': 1},
            'B03': {'index': 2}, 
            'B04': {'index': 3}
        }
        
        bands = ['B02', 'B04']
        result = get_band_numbers(bands, band_mapping)
        
        assert result == [1, 3]

    def test_get_band_numbers_invalid_band(self):
        """Test band number conversion with invalid band."""
        band_mapping = {'B02': {'index': 1}}
        
        with pytest.raises(KeyError, match="Band 'B99' not found"):
            get_band_numbers(['B99'], band_mapping)

    def test_normalize_array_minmax(self):
        """Test min-max normalization."""
        arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
        
        result = normalize_array(arr, method='minmax')
        
        assert np.allclose(result.min(), 0.0)
        assert np.allclose(result.max(), 1.0)

    def test_normalize_array_zscore(self):
        """Test z-score normalization."""
        arr = np.array([1, 2, 3, 4, 5], dtype=float)
        
        result = normalize_array(arr, method='zscore')
        
        assert np.allclose(np.mean(result), 0.0, atol=1e-10)
        assert np.allclose(np.std(result), 1.0)

    def test_normalize_array_invalid_method(self):
        """Test normalization with invalid method."""
        arr = np.array([1, 2, 3])
        
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_array(arr, method='invalid')


class TestSpectralIndices:
    """Test spectral indices with configurable band mapping."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample multispectral image."""
        # Create image with 13 bands (Sentinel-2 like)
        return np.random.randint(100, 5000, (50, 50, 13), dtype=np.uint16)

    @pytest.fixture
    def custom_band_mapping(self):
        """Create a custom band mapping for testing."""
        return {
            'B02': {'index': 0},
            'B03': {'index': 1},
            'B04': {'index': 2},
            'B05': {'index': 3},
            'B08': {'index': 4},
            'B11': {'index': 5},
            'B12': {'index': 6}
        }

    def test_chlorophyll_index_default(self, sample_image):
        """Test chlorophyll index with default parameters."""
        result = chlorophyll_index(sample_image)
        
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]
        assert not np.all(np.isnan(result))  # Should have some valid values

    def test_chlorophyll_index_custom_mapping(self, custom_band_mapping):
        """Test chlorophyll index with custom band mapping."""
        # Create smaller image matching custom mapping
        img = np.random.randint(100, 5000, (20, 20, 7), dtype=np.uint16)
        
        result = chlorophyll_index(
            img, 
            bands=['B02', 'B03', 'B04', 'B05'],
            band_mapping=custom_band_mapping
        )
        
        assert result.shape == (20, 20)
        assert not np.all(np.isnan(result))

    def test_ocean_color_index_default(self, sample_image):
        """Test ocean color index with default parameters."""
        result = ocean_color_index(sample_image)
        
        assert result.shape == (50, 50)
        assert result.dtype in [np.float64, np.float32]

    def test_water_quality_index_custom_bands(self, sample_image):
        """Test water quality index with custom bands."""
        result = water_quality_index(
            sample_image,
            bands=['B02', 'B03', 'B04', 'B05']  # Custom band selection
        )
        
        assert result.shape == (50, 50)

    def test_ndci_calculation(self, sample_image):
        """Test NDCI calculation."""
        result = normalized_difference_chlorophyll_index(sample_image)
        
        assert result.shape == (50, 50)
        # NDCI should be in range [-1, 1] for most values
        valid_values = result[~np.isnan(result)]
        assert np.all(valid_values >= -2) and np.all(valid_values <= 2)

    def test_spectral_index_invalid_bands(self, sample_image):
        """Test spectral index with invalid band mapping."""
        invalid_mapping = {'B99': {'index': 0}}
        
        with pytest.raises(KeyError, match="Band 'B04' not found"):
            chlorophyll_index(sample_image, band_mapping=invalid_mapping)

    def test_spectral_index_out_of_bounds(self):
        """Test spectral index with band indices out of bounds."""
        small_img = np.random.rand(10, 10, 3)  # Only 3 bands
        
        with pytest.raises(ValueError, match="out of bounds"):
            chlorophyll_index(small_img)  # Needs bands at indices 3, 4, 5, 6

    def test_zero_handling_in_indices(self):
        """Test that zeros are properly handled in spectral indices."""
        # Create image with some zero values that result in zero calculations
        img = np.ones((10, 10, 13), dtype=float)
        # Set specific bands to create zero results in CI calculation
        img[:, :, 3] = 0  # B04 (r665)
        img[:, :, 6] = 0  # B07 (r783)
        # This should create zeros in the CI calculation: ri = r665 + r783 = 0 + 0
        
        result = chlorophyll_index(img)
        
        # Should handle the calculation without errors
        assert result.shape == (10, 10)
        # Some values might be NaN due to zero handling, but not necessarily all


class TestSegmentationSuperpixels:
    """Test segmentation and superpixel functions."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for segmentation."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

    def test_slic_segmentation(self, sample_image):
        """Test SLIC segmentation."""
        segments = slic_segmentation(sample_image, n_segments=100)
        
        assert segments.shape == (100, 100)
        assert segments.dtype in [np.int32, np.int64]
        assert len(np.unique(segments)) <= 100  # Should have <= n_segments

    def test_felzenszwalb_segmentation(self, sample_image):
        """Test Felzenszwalb segmentation."""
        segments = felzenszwalb_segmentation(sample_image)
        
        assert segments.shape == (100, 100)
        assert len(np.unique(segments)) > 1  # Should produce multiple segments

    def test_extract_patches(self, sample_image):
        """Test patch extraction from segments."""
        segments = slic_segmentation(sample_image, n_segments=50)
        patches = extract_patches(sample_image, segments, min_size=5)
        
        assert isinstance(patches, list)
        assert len(patches) > 0
        assert all(isinstance(patch, np.ndarray) for patch in patches)

    def test_pca_segments(self, sample_image):
        """Test PCA on segmented patches."""
        segments = slic_segmentation(sample_image, n_segments=20)
        patches = extract_patches(sample_image, segments, min_size=5)
        
        if patches:  # Only test if patches were extracted
            features = pca_segments(patches, n_components=3)
            
            assert isinstance(features, np.ndarray)
            if features.size > 0:
                assert features.shape[1] <= 3  # Should have <= n_components

    def test_cluster_segments(self):
        """Test segment clustering."""
        # Create simple test features
        features = np.random.rand(50, 5)
        
        clusters = cluster_segments(features, eps=0.5, min_samples=3)
        
        assert len(clusters) == 50
        assert clusters.dtype in [np.int32, np.int64]

    def test_process_superpixel_pipeline(self, sample_image):
        """Test complete superpixel processing pipeline."""
        result = process_superpixel_pipeline(
            sample_image,
            segmentation_method='slic',
            n_segments=50,
            pca_components=3
        )
        
        assert isinstance(result, dict)
        assert 'segments' in result
        assert 'patches' in result
        assert 'features' in result
        assert 'clusters' in result
        assert 'dii' in result
        
        assert result['segments'].shape == (100, 100)

    def test_empty_patches_handling(self):
        """Test handling of empty patches."""
        empty_patches = []
        
        features = pca_segments(empty_patches)
        clusters = cluster_segments(features) if features.size > 0 else np.array([])
        
        assert features.size == 0
        assert clusters.size == 0


class TestVisualizationDisplay:
    """Test visualization and display functions."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample multispectral image."""
        return np.random.randint(0, 1000, (50, 50, 5), dtype=np.uint16)

    def test_create_rgb_image(self, sample_image):
        """Test RGB image creation."""
        band_indices = [0, 1, 2]
        
        rgb = create_rgb_image(sample_image, band_indices)
        
        assert rgb.shape == (50, 50, 3)
        assert rgb.dtype == np.uint8
        assert np.all(rgb >= 0) and np.all(rgb <= 255)

    def test_create_rgb_image_invalid_bands(self, sample_image):
        """Test RGB creation with invalid band indices."""
        with pytest.raises(ValueError, match="Exactly 3 band indices required"):
            create_rgb_image(sample_image, [0, 1])  # Only 2 bands
        
        with pytest.raises(ValueError, match="out of bounds"):
            create_rgb_image(sample_image, [0, 1, 10])  # Band 10 doesn't exist

    @patch('matplotlib.pyplot.show')
    def test_plot_rgb_no_show(self, mock_show, sample_image):
        """Test RGB plotting without showing."""
        band_indices = [0, 1, 2]
        
        fig = plot_rgb(sample_image, band_indices, show=False)
        
        assert fig is not None
        mock_show.assert_not_called()

    @patch('matplotlib.pyplot.show')  
    def test_plot_rgb_with_show(self, mock_show, sample_image):
        """Test RGB plotting with showing."""
        band_indices = [0, 1, 2]
        
        result = plot_rgb(sample_image, band_indices, show=True)
        
        assert result is None
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_histogram(self, mock_show, sample_image):
        """Test histogram plotting."""
        fig = plot_histogram(
            sample_image, 
            channels=[0, 1, 2], 
            channel_names=['Band 1', 'Band 2', 'Band 3'],
            show=False
        )
        
        assert fig is not None
        mock_show.assert_not_called()

    @patch('matplotlib.pyplot.show')
    def test_plot_discrete_image(self, mock_show):
        """Test discrete image plotting."""
        # Create discrete classification array
        arr = np.random.randint(0, 5, (30, 30))
        value_labels = {0: 'Water', 1: 'Land', 2: 'Vegetation', 3: 'Urban', 4: 'Cloud'}
        
        fig = plot_discrete_image(arr, value_labels=value_labels, show=False)
        
        assert fig is not None
        mock_show.assert_not_called()

    def test_single_channel_histogram(self):
        """Test histogram with single channel image."""
        single_channel = np.random.randint(0, 255, (30, 30))
        
        # Should handle 2D input gracefully
        fig = plot_histogram(single_channel, show=False)
        assert fig is not None


class TestModularIntegration:
    """Test integration between the new modular components."""
    
    def test_spectral_indices_with_custom_preprocessing(self):
        """Test spectral indices with custom preprocessing using core utils."""
        # Create test image
        img = np.random.randint(100, 5000, (20, 20, 13), dtype=np.uint16)
        
        # Apply preprocessing using core utilities
        clipped_img = clip_array(img, min_val=0, max_val=4000)
        
        # Apply spectral index
        ci_result = chlorophyll_index(clipped_img)
        
        assert ci_result.shape == (20, 20)
        assert not np.all(np.isnan(ci_result))

    def test_segmentation_with_spectral_preprocessing(self):
        """Test segmentation pipeline with spectral preprocessing."""
        # Create multispectral image
        img = np.random.randint(0, 1000, (50, 50, 8), dtype=np.uint16)
        
        # Create RGB using core utilities
        rgb_bands = [2, 1, 0]  # R, G, B
        rgb_img = create_rgb_image(img, rgb_bands)
        
        # Apply segmentation
        segments = slic_segmentation(rgb_img, n_segments=100)
        
        assert segments.shape == (50, 50)
        assert len(np.unique(segments)) > 1

    def test_visualization_with_spectral_results(self):
        """Test visualization of spectral index results."""
        # Create test image
        img = np.random.randint(100, 5000, (30, 30, 13), dtype=np.uint16)
        
        # Calculate spectral index
        ndci_result = normalized_difference_chlorophyll_index(img)
        
        # Visualize as discrete classes
        # Convert continuous values to discrete classes
        discrete_classes = np.digitize(ndci_result, bins=[-1, -0.5, 0, 0.5, 1])
        
        fig = plot_discrete_image(
            discrete_classes,
            value_labels={1: 'Very Low', 2: 'Low', 3: 'Medium', 4: 'High', 5: 'Very High'},
            show=False
        )
        
        assert fig is not None

    def test_end_to_end_workflow(self):
        """Test complete workflow using all new modules."""
        # 1. Create test data
        img = np.random.randint(100, 5000, (40, 40, 13), dtype=np.uint16)
        
        # 2. Preprocess using core utilities
        clipped_img = clip_array(img)
        normalized_img = normalize_array(clipped_img)
        
        # 3. Create RGB for visualization
        rgb_img = create_rgb_image(img, [3, 2, 1])  # R, G, B bands
        
        # 4. Apply segmentation
        segments = slic_segmentation(rgb_img, n_segments=50)
        
        # 5. Calculate spectral indices
        wqi_result = water_quality_index(img)
        
        # 6. Process superpixel pipeline
        pipeline_result = process_superpixel_pipeline(rgb_img, n_segments=30)
        
        # Verify all steps completed successfully
        assert normalized_img.shape == img.shape
        assert rgb_img.shape == (40, 40, 3)
        assert segments.shape == (40, 40)
        assert wqi_result.shape == (40, 40)
        assert 'segments' in pipeline_result
        assert pipeline_result['segments'].shape == (40, 40)


class TestBackwardCompatibility:
    """Test that new modules maintain compatibility with expected interfaces."""
    
    def test_spectral_indices_aliases(self):
        """Test that spectral index aliases work for backward compatibility."""
        from ShallowLearn.spectral.indices import ci, oci, ssi, wqi, ndci
        
        img = np.random.randint(100, 5000, (20, 20, 13), dtype=np.uint16)
        
        # Test aliases work
        ci_result = ci(img)
        oci_result = oci(img)
        ssi_result = ssi(img)
        wqi_result = wqi(img)
        ndci_result = ndci(img)
        
        assert ci_result.shape == (20, 20)
        assert oci_result.shape == (20, 20)
        assert ssi_result.shape == (20, 20)
        assert wqi_result.shape == (20, 20)
        assert ndci_result.shape == (20, 20)

    def test_default_band_mapping_still_works(self):
        """Test that default Sentinel-2 band mapping is still functional."""
        img = np.random.randint(100, 5000, (15, 15, 13), dtype=np.uint16)
        
        # Should work without specifying band_mapping (uses default)
        result = chlorophyll_index(img)
        
        assert result.shape == (15, 15)
        assert isinstance(result, np.ndarray)

    def test_core_utils_standalone(self):
        """Test that core utilities can be used independently."""
        # Test array manipulation without any remote sensing context
        data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        
        clipped = clip_array(data, min_val=3, max_val=7)
        normalized = normalize_array(data)
        
        assert np.array_equal(clipped, [[3, 3, 3], [4, 5, 6], [7, 7, 7]])
        assert normalized.min() >= 0 and normalized.max() <= 1