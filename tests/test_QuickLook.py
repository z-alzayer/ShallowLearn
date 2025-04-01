import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from rasterio.transform import from_origin
from unittest.mock import patch, MagicMock
import os
from PIL import Image

import ShallowLearn.QuickLook as ql
import ShallowLearn.Transform as trf 

NUM_EXAMPLES = 20
# --- Fixtures ---

@pytest.fixture
def mock_pvi_image():
    """Creates a simple mock PVI image (RGB)."""
    return np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)

@pytest.fixture
def mock_l1c_image():
    """Creates a simple mock L1C-like image (4 bands)."""
    # Using a slightly larger range typical for L1C reflectance (scaled)
    return np.random.randint(0, 3000, size=(20, 20, 4), dtype=np.uint16)

@pytest.fixture
def mock_pvi_files(tmp_path, mock_pvi_image):
    """Creates mock PVI image files in a temporary directory as JPGs."""
    files = []
    for i in range(NUM_EXAMPLES):
        file_path = tmp_path / f"mock_pvi_{i}.jpg"
        # Convert numpy array to PIL Image and save as JPG
        # Assumes mock_pvi_image is in (height, width, channels) format
        Image.fromarray(mock_pvi_image).save(file_path)
        files.append(str(file_path))
    return files


@pytest.fixture
def mock_area_dataframe():
    """Creates a mock DataFrame with 20 examples."""
    num_examples = 20  # Hardcoded number of examples
    return pd.DataFrame({
        'FILE_PATH': [f'mock_l1c_path_{i}.SAFE/MTD_MSIL1C.xml' for i in range(num_examples)],
        'Label': [0] * num_examples,
        'DATATAKE_1_DATATAKE_SENSING_START': pd.date_range(
            start='2023-01-01', 
            periods=num_examples, 
            freq='10D'  # 10-day intervals
        ),
        'CLOUD_COVERAGE_ASSESSMENT': [5.0 + i * 5.0 for i in range(num_examples)],
    })


@pytest.fixture
def mock_shapefile():
    """Creates a simple mock GeoDataFrame for clipping."""
    polygon = Polygon([(0, 0), (1, 1), (1, 0)])
    gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs="EPSG:4326")
    # Important: QuickLookArea expects the shapefile CRS to be transformable
    # to the raster CRS, or the same. We might need to adjust this based
    # on how mock L1C loading is handled. For simplicity, assume it works.
    return gdf
# --- Mocks for External Dependencies ---

# Mock LoadSentinel2L1C to avoid actual file loading and processing
@pytest.fixture
def mock_load_sen2(mock_l1c_image):
    """Mocks the LoadSentinel2L1C class and its methods."""
    with patch('ShallowLearn.QuickLook.load_sen2') as mock_loader_class:
        mock_instance = MagicMock()
        # Mock the clip_raster_with_shape method to return a clipped image
        # Needs to return data in the expected format (bands, height, width)
        clipped_img = mock_l1c_image.transpose(2, 0, 1) # Example shape
        mock_instance.clip_raster_with_shape.return_value = clipped_img
        # Mock any other methods needed during initialization if necessary
        mock_loader_class.return_value = mock_instance
        yield mock_loader_class

# Mock metadata extraction if generate_dataframe uses it directly
@pytest.fixture
def mock_extract_metadata(mock_area_dataframe):
    """Mocks metadata extraction functions."""
    # QuickLookArea now modifies the input df directly, so less mocking needed here
    # If combine_metadata_w_pvi_analysis was used, we'd mock it.
    # For generate_metadata_dataframe, if called internally, we might mock it:
    with patch('ShallowLearn.QuickLook.extract_meta.generate_metadata_dataframe') as mock_gen_meta:
         # Return a dataframe similar to mock_area_dataframe but maybe without Label
         mock_gen_meta.return_value = mock_area_dataframe.drop(columns=['Label'])
         yield mock_gen_meta

# --- Test Classes ---

class TestQuickLookModel:
    """Tests for the abstract base class QuickLookModel (via its helper methods)."""

    def test_create_custom_pastel_cmap(self):
        """Test the colormap creation utility."""
        model = ql.QuickLookModel(files=[]) # Dummy instantiation
        labels = [-1, 0, 1, 2]
        cmap = model.create_custom_pastel_cmap(labels)
        assert isinstance(cmap, ql.ListedColormap)
        # Check if the number of colors matches the number of unique labels
        assert len(cmap.colors) == len(labels)

class TestQuickLookPVI:
    """Tests for the QuickLookPVI class."""

    def test_init_with_files(self, mock_pvi_files):
        """Test initialization with a list of file paths."""
        # Reduce complexity for PCA/DBSCAN
        with patch('ShallowLearn.QuickLook.PCA') as mock_pca, \
             patch('ShallowLearn.QuickLook.DBSCAN') as mock_dbscan:

            mock_pca_instance = MagicMock()
            mock_pca_instance.fit_transform.return_value = np.random.rand(len(mock_pvi_files), 2)
            mock_pca.return_value = mock_pca_instance

            mock_dbscan_instance = MagicMock()
            mock_dbscan_instance.fit.return_value = mock_dbscan_instance
            mock_dbscan_instance.labels_ = np.random.randint(-1, 1, size=len(mock_pvi_files))
            mock_dbscan.return_value = mock_dbscan_instance

            instance = ql.QuickLookPVI(files=mock_pvi_files)

            assert instance.PVI is True
            assert len(instance.files) == len(mock_pvi_files)
            assert np.array(instance.imagery).shape[0] == len(mock_pvi_files)
            assert isinstance(instance.imagery[0], np.ndarray)
            assert instance.imagery[0].shape == (10, 10, 3) # From mock_pvi_image
            assert instance.pca_model is mock_pca_instance
            assert hasattr(instance, 'transformed_data')
            assert instance.transformed_data.shape == (len(mock_pvi_files), 2)
            assert hasattr(instance, 'labels')
            assert len(instance.labels) == len(mock_pvi_files)
            # Check if models were called
            mock_pca_instance.fit_transform.assert_called_once()
            mock_dbscan_instance.fit.assert_called_once()

    def test_load_data(self, mock_pvi_files):
        """Test the data loading method."""
        instance = ql.QuickLookPVI(files=mock_pvi_files) # Let init run
        # Optionally re-run load_data if needed, or just check results from init
        # imagery = instance.load_data() # Assumes load_data is safe to call again
        imagery = instance.imagery # Check imagery loaded during init
        assert len(imagery) == 20
        assert isinstance(imagery[0], np.ndarray)
        assert imagery[0].shape == (10, 10, 3) # From mock_pvi_image
