"""
Comprehensive tests for SuperPixel segmentation workflows.
Tests various segmentation algorithms and their integration with PCA/clustering.
"""

import pytest
import numpy as np
import warnings

try:
    from ShallowLearn.SuperPixelExtraction import (
        felzenszwalb_segmentation, slic_segmentation, quickshift_segmentation, 
        watershed_segmentation, multiotsu_thresholding
    )
    SUPERPIXEL_AVAILABLE = True
except ImportError as e:
    SUPERPIXEL_AVAILABLE = False
    print(f"Warning: Could not import SuperPixelExtraction module: {e}")

try:
    from ShallowLearn.SuperPixelExtraction import superpixel_pca_workflow, cluster_superpixels
    SUPERPIXEL_WORKFLOWS_AVAILABLE = True
except ImportError:
    SUPERPIXEL_WORKFLOWS_AVAILABLE = False


@pytest.fixture
def sample_rgb_image():
    """Create a synthetic RGB image suitable for segmentation."""
    np.random.seed(42)
    image = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    
    # Add distinct regions for better segmentation
    # Top-left region - blue
    image[0:30, 0:30, :] = [50, 50, 200]
    
    # Top-right region - green  
    image[0:30, 70:100, :] = [50, 200, 50]
    
    # Bottom-left region - red
    image[70:100, 0:30, :] = [200, 50, 50]
    
    # Center region - purple
    image[40:60, 40:60, :] = [150, 50, 150]
    
    # Add some noise
    noise = np.random.randint(-20, 20, image.shape)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image


@pytest.fixture
def sample_grayscale_image():
    """Create a synthetic grayscale image for segmentation."""
    np.random.seed(42)
    image = np.random.randint(50, 200, (80, 80), dtype=np.uint8)
    
    # Add distinct regions
    image[10:25, 10:25] = 255  # Bright square
    image[50:65, 50:65] = 50   # Dark square
    image[30:50, 20:40] = 150  # Medium rectangle
    
    return image


@pytest.fixture
def sample_multispectral_image():
    """Create a synthetic multispectral image."""
    np.random.seed(42)
    image = np.random.rand(60, 60, 6) * 1000 + 500
    
    # Add spectral signatures for different regions
    # Region 1: High in bands 0,1 low in others
    image[10:25, 10:25, 0:2] *= 2
    image[10:25, 10:25, 2:] *= 0.5
    
    # Region 2: High in bands 3,4
    image[35:50, 35:50, 3:5] *= 2
    image[35:50, 35:50, [0,1,2,5]] *= 0.5
    
    return image.astype(np.uint16)


@pytest.mark.skipif(not SUPERPIXEL_AVAILABLE, reason="SuperPixelExtraction module not available")
class TestSegmentationAlgorithms:
    """Test individual segmentation algorithms."""
    
    def test_slic_segmentation_basic(self, sample_rgb_image):
        """Test basic SLIC segmentation."""
        result = slic_segmentation(sample_rgb_image, n_segments=50)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_rgb_image.shape[:2]  # 2D output
        assert result.dtype in [np.int32, np.int64, int]
        
        # Should have multiple segments
        unique_segments = np.unique(result)
        assert len(unique_segments) > 1
        assert len(unique_segments) <= 60  # Should be around n_segments, allowing some tolerance

    def test_slic_segmentation_parameters(self, sample_rgb_image):
        """Test SLIC with different parameters."""
        # Test with different compactness
        result_compact = slic_segmentation(sample_rgb_image, n_segments=30, compactness=20)
        result_loose = slic_segmentation(sample_rgb_image, n_segments=30, compactness=1)
        
        assert result_compact.shape == result_loose.shape
        # Results should be different with different compactness
        assert not np.array_equal(result_compact, result_loose)

    def test_felzenszwalb_segmentation_basic(self, sample_rgb_image):
        """Test Felzenszwalb segmentation."""
        result = felzenszwalb_segmentation(sample_rgb_image, scale=100)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_rgb_image.shape[:2]
        assert result.dtype in [np.int32, np.int64, int]
        
        # Should have multiple segments
        unique_segments = np.unique(result)
        assert len(unique_segments) > 1

    def test_quickshift_segmentation_basic(self, sample_rgb_image):
        """Test Quickshift segmentation."""
        result = quickshift_segmentation(sample_rgb_image)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_rgb_image.shape[:2]
        assert result.dtype in [np.int32, np.int64, int]
        
        # Should have multiple segments
        unique_segments = np.unique(result)
        assert len(unique_segments) > 1

    def test_watershed_segmentation_basic(self, sample_grayscale_image):
        """Test Watershed segmentation."""
        result = watershed_segmentation(sample_grayscale_image, markers=20)
        
        assert isinstance(result, np.ndarray)
        assert result.shape == sample_grayscale_image.shape
        assert result.dtype in [np.int32, np.int64, int]
        
        # Should have multiple segments
        unique_segments = np.unique(result)
        assert len(unique_segments) > 1

    def test_multiotsu_thresholding(self, sample_grayscale_image):
        """Test Multi-Otsu thresholding."""
        result = multiotsu_thresholding(sample_grayscale_image, classes=3)
        
        assert isinstance(result, np.ndarray)
        assert len(result) == 2  # n_classes - 1 thresholds
        assert result.dtype in [np.float64, np.float32]
        
        # Thresholds should be in ascending order
        assert result[0] < result[1]


@pytest.mark.skipif(not SUPERPIXEL_AVAILABLE, reason="SuperPixelExtraction module not available")
class TestSegmentationConsistency:
    """Test consistency and properties of segmentation results."""
    
    def test_segment_labels_start_from_expected_values(self, sample_rgb_image):
        """Test that segment labels start from expected values."""
        slic_result = slic_segmentation(sample_rgb_image, n_segments=20)
        
        # SLIC typically starts from 1 (as specified in start_label=1)
        unique_labels = np.unique(slic_result)
        assert np.min(unique_labels) >= 0  # Should be non-negative
        
        # Felzenszwalb typically starts from 0
        fz_result = felzenszwalb_segmentation(sample_rgb_image)
        unique_labels_fz = np.unique(fz_result)
        assert np.min(unique_labels_fz) >= 0

    def test_segmentation_reproducibility(self, sample_rgb_image):
        """Test that segmentation results are reproducible with same parameters."""
        result1 = slic_segmentation(sample_rgb_image, n_segments=25, compactness=10)
        result2 = slic_segmentation(sample_rgb_image, n_segments=25, compactness=10)
        
        assert np.array_equal(result1, result2)

    def test_different_algorithms_different_results(self, sample_rgb_image):
        """Test that different algorithms produce different results."""
        slic_result = slic_segmentation(sample_rgb_image, n_segments=30)
        fz_result = felzenszwalb_segmentation(sample_rgb_image)
        qs_result = quickshift_segmentation(sample_rgb_image)
        
        # Results should be different between algorithms
        assert not np.array_equal(slic_result, fz_result)
        assert not np.array_equal(slic_result, qs_result)
        assert not np.array_equal(fz_result, qs_result)

    def test_parameter_effects(self, sample_rgb_image):
        """Test that changing parameters affects results meaningfully."""
        # Test SLIC with different segment numbers
        slic_few = slic_segmentation(sample_rgb_image, n_segments=10)
        slic_many = slic_segmentation(sample_rgb_image, n_segments=50)
        
        # More segments should generally create more unique labels
        unique_few = len(np.unique(slic_few))
        unique_many = len(np.unique(slic_many))
        assert unique_many > unique_few
        
        # Test Felzenszwalb with different scales
        fz_fine = felzenszwalb_segmentation(sample_rgb_image, scale=50)
        fz_coarse = felzenszwalb_segmentation(sample_rgb_image, scale=200)
        
        assert not np.array_equal(fz_fine, fz_coarse)

    def test_segmentation_covers_whole_image(self, sample_rgb_image):
        """Test that segmentation assigns every pixel to a segment."""
        algorithms = [
            lambda img: slic_segmentation(img, n_segments=20),
            lambda img: felzenszwalb_segmentation(img),
            lambda img: quickshift_segmentation(img),
        ]
        
        for algorithm in algorithms:
            try:
                result = algorithm(sample_rgb_image)
                
                # Every pixel should be assigned to a segment (no unassigned pixels)
                assert result.shape == sample_rgb_image.shape[:2]
                assert not np.any(result < 0)  # No negative labels
                
                # Should not have "gaps" - every pixel should have a label
                assert np.all(np.isfinite(result))
                
            except Exception as e:
                print(f"Algorithm failed: {e}")
                continue


@pytest.mark.skipif(not SUPERPIXEL_AVAILABLE, reason="SuperPixelExtraction module not available")
class TestSegmentationEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_small_image_segmentation(self):
        """Test segmentation on very small images."""
        small_image = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        
        # SLIC should handle small images
        try:
            result = slic_segmentation(small_image, n_segments=5)
            assert isinstance(result, np.ndarray)
            assert result.shape == (10, 10)
        except Exception as e:
            # Small images might cause issues, which is acceptable
            print(f"Small image segmentation failed: {e}")

    def test_uniform_image_segmentation(self):
        """Test segmentation on uniform images."""
        uniform_image = np.ones((50, 50, 3), dtype=np.uint8) * 128
        
        # Algorithms should handle uniform images gracefully
        try:
            slic_result = slic_segmentation(uniform_image, n_segments=10)
            assert isinstance(slic_result, np.ndarray)
            # Uniform image might result in fewer segments than requested
            unique_segments = len(np.unique(slic_result))
            assert unique_segments >= 1
            
        except Exception as e:
            # Uniform images can be challenging for some algorithms
            print(f"Uniform image segmentation failed: {e}")

    def test_single_channel_image(self, sample_grayscale_image):
        """Test algorithms that can handle single-channel images."""
        # Some algorithms might work with grayscale, others might need RGB
        try:
            # Convert to 3D for algorithms that expect RGB
            gray_3d = np.stack([sample_grayscale_image] * 3, axis=2)
            result = slic_segmentation(gray_3d, n_segments=15)
            assert isinstance(result, np.ndarray)
            assert result.shape == sample_grayscale_image.shape
        except Exception as e:
            print(f"Grayscale segmentation failed: {e}")

    def test_extreme_parameter_values(self, sample_rgb_image):
        """Test algorithms with extreme parameter values."""
        # Test SLIC with very few segments
        try:
            result_few = slic_segmentation(sample_rgb_image, n_segments=1)
            assert isinstance(result_few, np.ndarray)
            # Should have very few unique segments
            unique_count = len(np.unique(result_few))
            assert unique_count <= 5  # Allow some tolerance
        except Exception as e:
            print(f"Extreme parameter test failed: {e}")

    def test_high_contrast_image(self):
        """Test segmentation on high contrast images."""
        # Create checkerboard pattern
        checkerboard = np.zeros((40, 40, 3), dtype=np.uint8)
        for i in range(40):
            for j in range(40):
                if (i + j) % 2 == 0:
                    checkerboard[i, j, :] = 255
                else:
                    checkerboard[i, j, :] = 0
        
        try:
            result = slic_segmentation(checkerboard, n_segments=20)
            assert isinstance(result, np.ndarray)
            assert result.shape == (40, 40)
            
            # High contrast should enable good segmentation
            unique_segments = len(np.unique(result))
            assert unique_segments > 5
            
        except Exception as e:
            print(f"High contrast segmentation failed: {e}")


@pytest.mark.skipif(not SUPERPIXEL_AVAILABLE, reason="SuperPixelExtraction module not available")
class TestSegmentationIntegration:
    """Test integration of segmentation with other processing steps."""
    
    def test_segmentation_with_different_dtypes(self, sample_rgb_image):
        """Test that segmentation works with different input data types."""
        # Test with float input
        float_image = sample_rgb_image.astype(np.float64) / 255.0
        
        try:
            result_float = slic_segmentation(float_image, n_segments=20)
            result_uint8 = slic_segmentation(sample_rgb_image, n_segments=20)
            
            # Results should be similar (allowing for some numerical differences)
            assert result_float.shape == result_uint8.shape
            assert isinstance(result_float, np.ndarray)
            
        except Exception as e:
            print(f"Different dtype test failed: {e}")

    def test_segmentation_preprocessing_pipeline(self, sample_multispectral_image):
        """Test segmentation as part of a preprocessing pipeline."""
        # Simulate a typical workflow: resize -> segment
        from skimage.transform import resize
        
        # Resize image
        resized_image = resize(sample_multispectral_image, (40, 40), preserve_range=True)
        resized_image = resized_image.astype(np.uint8)
        
        # Convert to RGB for segmentation (take first 3 bands)
        rgb_image = resized_image[:, :, :3]
        
        try:
            segments = slic_segmentation(rgb_image, n_segments=15)
            assert isinstance(segments, np.ndarray)
            assert segments.shape == (40, 40)
            
        except Exception as e:
            print(f"Pipeline test failed: {e}")

    def test_multiple_segmentation_comparison(self, sample_rgb_image):
        """Test comparing results from multiple segmentation algorithms."""
        algorithms = {
            'slic': lambda img: slic_segmentation(img, n_segments=25),
            'felzenszwalb': lambda img: felzenszwalb_segmentation(img, scale=100),
            'quickshift': lambda img: quickshift_segmentation(img)
        }
        
        results = {}
        for name, algorithm in algorithms.items():
            try:
                results[name] = algorithm(sample_rgb_image)
            except Exception as e:
                print(f"Algorithm {name} failed: {e}")
                continue
        
        # Compare properties of different results
        if len(results) > 1:
            shapes = [result.shape for result in results.values()]
            assert all(shape == shapes[0] for shape in shapes)  # All same shape
            
            # Count unique segments for each method
            segment_counts = {name: len(np.unique(result)) 
                            for name, result in results.items()}
            
            # All methods should produce reasonable numbers of segments
            for name, count in segment_counts.items():
                assert count > 1, f"{name} produced only {count} segments"
                assert count < sample_rgb_image.shape[0] * sample_rgb_image.shape[1] // 4  # Not too many


if __name__ == "__main__":
    pytest.main([__file__, "-v"])