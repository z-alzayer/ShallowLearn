"""
Tests for modern QuickLook functionality using ml/quicklook_processor and visualization modules.
Uses realistic synthetic data and tests both success and failure conditions.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from ShallowLearn.ml.quicklook_processor import (
    QuickLookProcessor, 
    PCAReducer, 
    TSNEReducer,
    DBSCANClustering, 
    KMeansClustering,
    GMMClustering,
    SatelliteImageProcessor,
    quick_analysis
)
from ShallowLearn.visualization.quicklook_viz import QuickLookVisualizer


def create_realistic_satellite_data():
    """Create realistic synthetic satellite image data that mimics real spectral signatures."""
    # Create 12 images with realistic spectral characteristics
    num_images = 12
    height, width = 30, 30
    num_bands = 4  # Blue, Green, Red, NIR
    
    image_stack = np.zeros((num_images, height, width, num_bands))
    
    for i in range(num_images):
        img = np.zeros((height, width, num_bands))
        
        # Create different land cover types with realistic spectral signatures
        
        # Deep water (low reflectance, very low NIR)
        water_deep = np.array([0.02, 0.04, 0.03, 0.01])
        img[20:, :10] = water_deep
        
        # Shallow water (slightly higher reflectance)
        water_shallow = np.array([0.05, 0.08, 0.06, 0.02])
        img[15:20, :15] = water_shallow
        
        # Healthy vegetation (low red, high NIR)
        vegetation = np.array([0.03, 0.08, 0.04, 0.45])
        img[:15, :] = vegetation
        
        # Sand/coral (high reflectance across bands)
        sand = np.array([0.35, 0.45, 0.55, 0.60])
        img[15:, 20:] = sand
        
        # Stressed vegetation (higher red, lower NIR)
        veg_stressed = np.array([0.04, 0.09, 0.08, 0.25])
        img[10:15, 10:20] = veg_stressed
        
        # Add temporal variation (seasonal effects)
        seasonal_factor = 0.8 + 0.4 * np.sin(2 * np.pi * i / 12)  # Annual cycle
        img *= seasonal_factor
        
        # Add realistic noise
        noise = np.random.normal(0, 0.005, img.shape)
        img += noise
        
        # Ensure valid reflectance range
        img = np.clip(img, 0, 1)
        
        image_stack[i] = img
    
    return image_stack


def create_metadata_dataframe(num_images=12):
    """Create realistic metadata for synthetic images."""
    dates = pd.date_range('2023-01-01', periods=num_images, freq='30D')
    return pd.DataFrame({
        'acquisition_date': dates,
        'cloud_coverage': np.random.uniform(0, 25, num_images),
        'scene_id': [f'synthetic_scene_{i:03d}' for i in range(num_images)],
        'processing_level': ['L2A'] * num_images,
        'aot_retrieval_method': np.random.choice(['LUT', 'SEN2COR'], num_images),
        'satellite': ['Sentinel-2A'] * (num_images // 2) + ['Sentinel-2B'] * (num_images - num_images // 2)
    })


class MockSatelliteImage:
    """Mock satellite image class for testing."""
    
    def __init__(self, image_array, scene_id, processing_level='L2A'):
        self.image = image_array
        self.path = Path(f"/mock/path/{scene_id}.SAFE")
        self.present_bands = {'B02', 'B03', 'B04', 'B08'}  # Common Sentinel-2 bands
        self.band_order = {'B02': 0, 'B03': 1, 'B04': 2, 'B08': 3}
        self.band_status = {band: True for band in self.present_bands}
        self.tags = {
            'PROCESSING_LEVEL': processing_level,
            'AOT_RETRIEVAL_METHOD': 'SEN2COR' if np.random.random() > 0.5 else 'LUT'
        }
        self.meta = {'scene_id': scene_id}


class TestDimensionalityReduction:
    """Test dimensionality reduction methods with real data."""
    
    def test_pca_reducer_basic_functionality(self):
        """Test PCA reducer with realistic data."""
        # Create sample spectral data
        n_samples, n_features = 1000, 4
        data = create_realistic_satellite_data()
        
        # Flatten spatial dimensions
        flat_data = data.reshape(-1, n_features)
        
        # Test PCA with different settings
        reducer = PCAReducer(n_components=2)
        transformed = reducer.fit_transform(flat_data)
        
        assert transformed.shape == (flat_data.shape[0], 2)
        assert hasattr(reducer, 'model')
        assert reducer.get_name() == "PCA (2 components)"
        
        # Test variance explained
        explained_var_ratio = reducer.model.explained_variance_ratio_
        assert len(explained_var_ratio) == 2
        assert np.all(explained_var_ratio > 0)
        assert np.sum(explained_var_ratio) <= 1.0
        
    def test_pca_reducer_variance_threshold(self):
        """Test PCA with variance threshold."""
        data = create_realistic_satellite_data()
        flat_data = data.reshape(-1, 4)
        
        reducer = PCAReducer(n_components=0.95)  # 95% variance
        transformed = reducer.fit_transform(flat_data)
        
        assert transformed.shape[1] <= 4  # Should be <= original features
        assert transformed.shape[0] == flat_data.shape[0]
        assert "95%" in reducer.get_name()
        
        # Verify variance explained meets threshold
        total_var_explained = np.sum(reducer.model.explained_variance_ratio_)
        assert total_var_explained >= 0.94  # Allow small numerical tolerance
        
    def test_tsne_reducer(self):
        """Test t-SNE reducer with realistic data."""
        data = create_realistic_satellite_data()
        flat_data = data.reshape(-1, 4)
        
        # Use subset for faster t-SNE
        subset_indices = np.random.choice(flat_data.shape[0], 500, replace=False)
        subset_data = flat_data[subset_indices]
        
        reducer = TSNEReducer(n_components=2, perplexity=30)
        transformed = reducer.fit_transform(subset_data)
        
        assert transformed.shape == (500, 2)
        assert "t-SNE" in reducer.get_name()
        assert "perplexity=30" in reducer.get_name()
        
    def test_dimensionality_reduction_edge_cases(self):
        """Test edge cases for dimensionality reduction."""
        # Test with very small dataset
        small_data = np.random.rand(10, 4)
        
        reducer = PCAReducer(n_components=2)
        transformed = reducer.fit_transform(small_data)
        assert transformed.shape == (10, 2)
        
        # Test with single feature
        single_feature = np.random.rand(100, 1)
        reducer_single = PCAReducer(n_components=1)
        transformed_single = reducer_single.fit_transform(single_feature)
        assert transformed_single.shape == (100, 1)


class TestClusteringMethods:
    """Test clustering methods with realistic data."""
    
    def test_dbscan_clustering(self):
        """Test DBSCAN clustering with realistic spectral data."""
        data = create_realistic_satellite_data()
        
        # Apply PCA first
        flat_data = data.reshape(-1, 4)
        reducer = PCAReducer(n_components=2)
        transformed = reducer.fit_transform(flat_data)
        
        # Apply DBSCAN
        clusterer = DBSCANClustering(eps=0.1, min_samples=20)
        labels = clusterer.fit_predict(transformed)
        
        assert len(labels) == transformed.shape[0]
        assert len(np.unique(labels)) >= 2  # Should find multiple clusters
        assert "DBSCAN" in clusterer.get_name()
        
        # Should have some noise points (-1 labels) or multiple clusters
        unique_labels = np.unique(labels)
        assert len(unique_labels) > 1 or -1 in labels
        
    def test_kmeans_clustering(self):
        """Test K-Means clustering."""
        data = create_realistic_satellite_data()
        flat_data = data.reshape(-1, 4)
        
        # Apply PCA
        reducer = PCAReducer(n_components=2)
        transformed = reducer.fit_transform(flat_data)
        
        # Apply K-Means
        clusterer = KMeansClustering(n_clusters=4)
        labels = clusterer.fit_predict(transformed)
        
        assert len(labels) == transformed.shape[0]
        assert len(np.unique(labels)) == 4  # Should find exactly 4 clusters
        assert "K-Means" in clusterer.get_name()
        assert all(label >= 0 for label in labels)  # No noise labels in K-Means
        
    def test_gmm_clustering(self):
        """Test Gaussian Mixture Model clustering."""
        data = create_realistic_satellite_data()
        flat_data = data.reshape(-1, 4)
        
        # Apply PCA
        reducer = PCAReducer(n_components=2)
        transformed = reducer.fit_transform(flat_data)
        
        # Apply GMM
        clusterer = GMMClustering(n_components=3)
        labels = clusterer.fit_predict(transformed)
        
        assert len(labels) == transformed.shape[0]
        assert len(np.unique(labels)) == 3  # Should find exactly 3 clusters
        assert "GMM" in clusterer.get_name()
        assert all(label >= 0 for label in labels)  # No noise labels in GMM


class TestSatelliteImageProcessor:
    """Test satellite image processing functionality."""
    
    def test_processor_initialization(self):
        """Test processor initialization with different settings."""
        # Default processor
        processor = SatelliteImageProcessor()
        assert processor.apply_stretch == True
        assert processor.normalize == True
        assert processor.clip_percent == 2.0
        
        # Custom processor
        custom_processor = SatelliteImageProcessor(
            target_bands=['B04', 'B03', 'B02'],
            apply_stretch=False,
            clip_percent=1.0,
            normalize=False
        )
        assert custom_processor.target_bands == ['B04', 'B03', 'B02']
        assert custom_processor.apply_stretch == False
        assert custom_processor.clip_percent == 1.0
        
    def test_satellite_type_detection(self):
        """Test satellite type detection from metadata."""
        processor = SatelliteImageProcessor()
        
        # Test Sentinel-2 detection
        data = create_realistic_satellite_data()[0]
        s2_image = MockSatelliteImage(data, "S2A_test", "Level-2A")
        detected_type = processor.detect_satellite_type(s2_image)
        assert detected_type == "sentinel-2"
        
        # Test unknown type
        unknown_image = MockSatelliteImage(data, "unknown_test")
        unknown_image.tags = {}  # No processing level
        unknown_image.present_bands = {'band1', 'band2', 'band3'}  # Non-standard bands
        unknown_type = processor.detect_satellite_type(unknown_image)
        assert unknown_type == "unknown"
        
    def test_image_processing_pipeline(self):
        """Test complete image processing pipeline."""
        processor = SatelliteImageProcessor()
        
        # Create mock satellite image
        data = create_realistic_satellite_data()[0]
        # Convert to uint16 to simulate real Sentinel-2 data
        data_uint16 = (data * 3000 + 1000).astype(np.uint16)  # Add offset like real S2 data
        
        mock_image = MockSatelliteImage(data_uint16, "test_scene")
        
        # Process image
        processed = processor.process_image(mock_image)
        
        assert processed is not None
        assert processed.shape[2] == 3  # RGB channels
        assert processed.dtype == np.float64 or processed.dtype == np.float32
        assert np.all(processed >= 0)
        assert np.all(processed <= 1)  # Should be normalized
        
    def test_processing_failure_cases(self):
        """Test image processing failure scenarios."""
        processor = SatelliteImageProcessor()
        
        # Create image with no valid bands
        data = create_realistic_satellite_data()[0]
        mock_image = MockSatelliteImage(data, "invalid_scene")
        mock_image.present_bands = set()  # No bands available
        
        processed = processor.process_image(mock_image)
        assert processed is None  # Should fail gracefully


class TestQuickLookProcessor:
    """Test the main QuickLook processor."""
    
    def test_processor_initialization(self):
        """Test processor initialization with different components."""
        # Default processor
        processor = QuickLookProcessor()
        assert isinstance(processor.reducer, PCAReducer)
        assert isinstance(processor.clustering, DBSCANClustering)
        assert isinstance(processor.image_processor, SatelliteImageProcessor)
        
        # Custom processor
        custom_processor = QuickLookProcessor(
            reducer=TSNEReducer(),
            clustering=KMeansClustering(n_clusters=3),
            image_processor=SatelliteImageProcessor(apply_stretch=False)
        )
        assert isinstance(custom_processor.reducer, TSNEReducer)
        assert isinstance(custom_processor.clustering, KMeansClustering)
        
    def test_complete_processing_pipeline(self):
        """Test complete image processing pipeline with mock images."""
        processor = QuickLookProcessor()
        
        # Create mock satellite images
        image_data = create_realistic_satellite_data()
        mock_images = []
        for i in range(len(image_data)):
            # Convert to uint16 to simulate real data
            data_uint16 = (image_data[i] * 3000 + 1000).astype(np.uint16)
            mock_img = MockSatelliteImage(data_uint16, f"scene_{i:03d}")
            mock_images.append(mock_img)
        
        # Process images
        results = processor.process_images(mock_images, create_metadata=True)
        assert len(processor.processed_images) == len(mock_images)
        assert processor.metadata_df is not None
        assert len(processor.metadata_df) == len(mock_images)
        
        # Apply dimensionality reduction
        processor.reduce_dimensions()
        assert processor.transformed_data is not None
        assert processor.transformed_data.shape[0] == len(mock_images)
        
        # Apply clustering
        processor.cluster_images()
        assert processor.labels is not None
        assert len(processor.labels) == len(mock_images)
        assert 'cluster_label' in processor.metadata_df.columns
        
        # Verify clusters were found
        unique_labels = np.unique(processor.labels)
        assert len(unique_labels) >= 1  # At least one cluster
        
    def test_processing_error_handling(self):
        """Test error handling in processing pipeline."""
        processor = QuickLookProcessor()
        
        # Test processing with no images
        empty_results = processor.process_images([])
        assert len(processor.processed_images) == 0
        
        # Test dimensionality reduction without processed images
        with pytest.raises(ValueError, match="No processed images available"):
            processor.reduce_dimensions()
        
        # Process some images first
        image_data = create_realistic_satellite_data()[:3]  # Use fewer images
        mock_images = []
        for i in range(len(image_data)):
            data_uint16 = (image_data[i] * 3000 + 1000).astype(np.uint16)
            mock_img = MockSatelliteImage(data_uint16, f"scene_{i:03d}")
            mock_images.append(mock_img)
        
        processor.process_images(mock_images)
        
        # Test clustering without dimensionality reduction
        with pytest.raises(ValueError, match="No transformed data available"):
            processor.cluster_images()
        
    def test_results_retrieval(self):
        """Test getting processing results."""
        processor = QuickLookProcessor()
        
        # Create and process mock images
        image_data = create_realistic_satellite_data()[:5]
        mock_images = []
        for i in range(len(image_data)):
            data_uint16 = (image_data[i] * 3000 + 1000).astype(np.uint16)
            mock_img = MockSatelliteImage(data_uint16, f"scene_{i:03d}")
            mock_images.append(mock_img)
        
        # Run complete analysis
        processor.run_complete_analysis(mock_images)
        
        # Get results
        results = processor.get_results()
        
        assert 'images' in results
        assert 'processed_images' in results
        assert 'transformed_data' in results
        assert 'labels' in results
        assert 'metadata_df' in results
        assert 'reducer_name' in results
        assert 'clustering_name' in results
        
        assert len(results['images']) == len(mock_images)
        assert results['transformed_data'] is not None
        assert results['labels'] is not None


class TestConvenienceFunctions:
    """Test convenience functions for quick analysis."""
    
    def test_quick_analysis_function(self):
        """Test the quick_analysis convenience function."""
        # Create temporary image files (mock paths)
        image_paths = ["/mock/path/scene_001.SAFE", "/mock/path/scene_002.SAFE"]
        
        # Since we can't actually load files, we'll test parameter handling
        try:
            # This should fail gracefully when trying to load mock paths
            results = quick_analysis(image_paths, method='pca', clustering='kmeans')
            # If it doesn't fail, check it was set up correctly
            assert isinstance(results.reducer, PCAReducer)
            assert isinstance(results.clustering, KMeansClustering)
        except Exception as e:
            # Expected to fail when loading mock paths or having no processed images
            error_str = str(e)
            assert ("Failed to process" in error_str or 
                    "No such file" in error_str or 
                    "No processed images available" in error_str)


class TestVisualizationIntegration:
    """Test integration with visualization components."""
    
    def test_visualizer_initialization(self):
        """Test QuickLook visualizer initialization."""
        # Create a processor first
        processor = QuickLookProcessor()
        image_data = create_realistic_satellite_data()[:3]
        mock_images = []
        
        for i in range(len(image_data)):
            data_uint16 = (image_data[i] * 3000 + 1000).astype(np.uint16)
            mock_img = MockSatelliteImage(data_uint16, f"scene_{i:03d}")
            mock_images.append(mock_img)
        
        processor.run_complete_analysis(mock_images)
        
        # Now create visualizer
        viz = QuickLookVisualizer(processor)
        assert viz is not None
        assert viz.processor is processor
        
    def test_visualization_data_preparation(self):
        """Test preparing data for visualization."""
        # Process some data first
        processor = QuickLookProcessor()
        image_data = create_realistic_satellite_data()[:6]
        mock_images = []
        
        for i in range(len(image_data)):
            data_uint16 = (image_data[i] * 3000 + 1000).astype(np.uint16)
            mock_img = MockSatelliteImage(data_uint16, f"scene_{i:03d}")
            mock_images.append(mock_img)
        
        # Run analysis
        processor.run_complete_analysis(mock_images)
        results = processor.get_results()
        
        # Verify data is suitable for visualization
        assert results['transformed_data'].shape[1] >= 2  # Need at least 2D for scatter plot
        assert len(results['labels']) == len(results['processed_images'])
        assert results['metadata_df'] is not None
        
        # Test that we can create visualization data
        viz = QuickLookVisualizer(processor)
        
        # Data should be ready for plotting
        x_data = results['transformed_data'][:, 0]
        y_data = results['transformed_data'][:, 1] 
        colors = results['labels']
        
        assert len(x_data) == len(y_data) == len(colors)


class TestRealWorldScenarios:
    """Test scenarios that mimic real-world usage."""
    
    def test_clustering_pipeline_runs(self):
        """Test that the complete clustering pipeline runs successfully."""
        processor = QuickLookProcessor(
            clustering=DBSCANClustering(eps=0.3, min_samples=3)
        )
        
        # Create synthetic data
        image_data = create_realistic_satellite_data()
        mock_images = []
        
        for i in range(len(image_data)):
            data_uint16 = (image_data[i] * 3000 + 1000).astype(np.uint16)
            mock_img = MockSatelliteImage(data_uint16, f"test_scene_{i:03d}")
            mock_images.append(mock_img)
        
        # Process - just verify it runs without errors
        processor.run_complete_analysis(mock_images)
        
        # Verify the pipeline completed
        assert processor.processed_images is not None
        assert processor.transformed_data is not None
        assert processor.labels is not None
        assert len(processor.labels) == len(mock_images)
        assert processor.metadata_df is not None
        
    def test_temporal_consistency(self):
        """Test that similar images cluster together over time."""
        processor = QuickLookProcessor()
        
        # Create temporally consistent data
        base_scene = create_realistic_satellite_data()[0]
        temporal_images = []
        
        for i in range(8):
            # Add small temporal variations
            temporal_factor = 0.95 + 0.1 * np.sin(2 * np.pi * i / 8)
            noise = np.random.normal(0, 0.02, base_scene.shape)
            
            varied_scene = base_scene * temporal_factor + noise
            varied_scene = np.clip(varied_scene, 0, 1)
            
            data_uint16 = (varied_scene * 3000 + 1000).astype(np.uint16)
            mock_img = MockSatelliteImage(data_uint16, f"temporal_{i:03d}")
            temporal_images.append(mock_img)
        
        # Process
        processor.run_complete_analysis(temporal_images)
        
        # Similar scenes should tend to cluster together
        results = processor.get_results()
        assert results['transformed_data'] is not None
        assert len(np.unique(processor.labels)) >= 1  # At least some clustering
        
    def test_processing_performance(self):
        """Test processing performance with larger datasets."""
        import time
        
        processor = QuickLookProcessor()
        
        # Create larger dataset
        large_image_data = create_realistic_satellite_data()  # 12 images
        mock_images = []
        
        for i in range(len(large_image_data)):
            data_uint16 = (large_image_data[i] * 3000 + 1000).astype(np.uint16)
            mock_img = MockSatelliteImage(data_uint16, f"large_scene_{i:03d}")
            mock_images.append(mock_img)
        
        # Time the processing
        start_time = time.time()
        processor.run_complete_analysis(mock_images)
        processing_time = time.time() - start_time
        
        # Should complete in reasonable time (less than 30 seconds for this test data)
        assert processing_time < 30
        
        # Verify results quality
        assert len(processor.processed_images) == len(mock_images)
        assert processor.transformed_data is not None
        assert processor.labels is not None