import pytest
import numpy as np
import tempfile
import os
import zipfile
import geopandas as gpd
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry import Point, Polygon
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from ShallowLearn.LoadData import (
    LoadGeoTIFF, LoadSentinel2L1C, PVI_Dataloader, 
    LoadFromCSV, LoadSeasonalData, LoadNumpyArray
)
from ShallowLearn.band_mapping import band_mapping


class TestLoadGeoTIFF:
    @pytest.fixture
    def sample_geotiff(self):
        """Create a temporary GeoTIFF file for testing"""
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            # Create test data with multiple bands
            data = np.random.randint(0, 10000, (4, 10, 10), dtype=np.uint16)
            transform = from_bounds(145.0, -15.0, 146.0, -14.0, 10, 10)
            
            with rasterio.open(
                tmp.name, 'w',
                driver='GTiff',
                height=10, width=10, count=4,
                dtype=data.dtype,
                crs=CRS.from_epsg(32755),
                transform=transform,
                nodata=0
            ) as dst:
                dst.write(data)
            
            yield tmp.name
            os.unlink(tmp.name)

    def test_load_geotiff_initialization(self, sample_geotiff):
        """Test LoadGeoTIFF initialization"""
        loader = LoadGeoTIFF(sample_geotiff)
        assert loader.data_source == sample_geotiff
        assert loader.metadata is None
        assert loader.bounds is None

    def test_load_geotiff_load_method(self, sample_geotiff):
        """Test LoadGeoTIFF load method"""
        loader = LoadGeoTIFF(sample_geotiff)
        data = loader.load()
        
        assert isinstance(data, np.ndarray)
        assert data.shape == (4, 10, 10)  # (bands, height, width)
        assert data.dtype == np.uint16

    def test_load_geotiff_get_metadata(self, sample_geotiff):
        """Test LoadGeoTIFF get_metadata method"""
        loader = LoadGeoTIFF(sample_geotiff)
        metadata = loader.get_metadata()
        
        assert metadata is not None
        assert 'driver' in metadata
        assert 'width' in metadata
        assert 'height' in metadata
        assert 'count' in metadata
        assert metadata['width'] == 10
        assert metadata['height'] == 10
        assert metadata['count'] == 4

    def test_load_geotiff_get_bounds(self, sample_geotiff):
        """Test LoadGeoTIFF get_bounds method"""
        loader = LoadGeoTIFF(sample_geotiff)
        bounds = loader.get_bounds()
        
        assert bounds is not None
        assert len(bounds) == 4
        assert bounds.left == 145.0
        assert bounds.bottom == -15.0
        assert bounds.right == 146.0
        assert bounds.top == -14.0

    def test_load_geotiff_nonexistent_file(self):
        """Test LoadGeoTIFF with nonexistent file"""
        loader = LoadGeoTIFF("nonexistent_file.tif")
        
        with pytest.raises(rasterio.RasterioIOError):
            loader.load()


class TestLoadSentinel2L1C:
    @pytest.fixture
    def mock_sentinel2_zip(self):
        """Create a mock Sentinel-2 ZIP file structure"""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, 'w') as zf:
                # Create minimal Sentinel-2 structure
                zf.writestr('S2A_MSIL1C_20230101T000000_N0400_R000_T32UPU_20230101T000000.SAFE/MTD_MSIL1C.xml', 
                           '<?xml version="1.0" encoding="UTF-8"?><root></root>')
                zf.writestr('S2A_MSIL1C_20230101T000000_N0400_R000_T32UPU_20230101T000000.SAFE/GRANULE/L1C_T32UPU_A000000_20230101T000000/IMG_DATA/T32UPU_20230101T000000_B02.jp2', '')
                zf.writestr('S2A_MSIL1C_20230101T000000_N0400_R000_T32UPU_20230101T000000.SAFE/GRANULE/L1C_T32UPU_A000000_20230101T000000/IMG_DATA/T32UPU_20230101T000000_B03.jp2', '')
                zf.writestr('S2A_MSIL1C_20230101T000000_N0400_R000_T32UPU_20230101T000000.SAFE/GRANULE/L1C_T32UPU_A000000_20230101T000000/IMG_DATA/T32UPU_20230101T000000_B04.jp2', '')
                zf.writestr('S2A_MSIL1C_20230101T000000_N0400_R000_T32UPU_20230101T000000.SAFE/GRANULE/L1C_T32UPU_A000000_20230101T000000/IMG_DATA/T32UPU_20230101T000000_B08.jp2', '')
            
            yield tmp.name
            os.unlink(tmp.name)

    @pytest.fixture
    def sample_shapefile_for_clipping(self):
        """Create a shapefile for clipping tests"""
        with tempfile.NamedTemporaryFile(suffix='.shp', delete=False) as tmp:
            polygon = Polygon([(145.2, -14.8), (145.4, -14.8), (145.4, -14.6), (145.2, -14.6)])
            gdf = gpd.GeoDataFrame([1], geometry=[polygon], crs="EPSG:4326")
            
            base_name = tmp.name[:-4]
            gdf.to_file(f"{base_name}.shp")
            
            yield gdf
            
            # Clean up
            for ext in ['.shp', '.shx', '.dbf', '.prj']:
                try:
                    os.unlink(f"{base_name}{ext}")
                except FileNotFoundError:
                    pass

    def test_load_sentinel2_initialization_zip(self, mock_sentinel2_zip):
        """Test LoadSentinel2L1C initialization with ZIP file"""
        loader = LoadSentinel2L1C(mock_sentinel2_zip)
        
        assert loader.data_source == mock_sentinel2_zip
        assert loader.is_zip == True
        assert loader.band_mapping == band_mapping
        assert len(loader.files) == 1
        assert "MTD_MSIL1C.xml" in loader.files[0]

    def test_load_sentinel2_initialization_with_custom_band_mapping(self, mock_sentinel2_zip):
        """Test LoadSentinel2L1C initialization with custom band mapping"""
        custom_band_mapping = {"B02": {"index": 0}, "B03": {"index": 1}}
        loader = LoadSentinel2L1C(mock_sentinel2_zip, band_mapping=custom_band_mapping)
        
        assert loader.band_mapping == custom_band_mapping
        assert loader.band_mapping != band_mapping

    def test_load_sentinel2_multiple_mtd_files_error(self):
        """Test error handling when multiple MTD files are found"""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, 'w') as zf:
                # Create multiple MTD files - should raise exception
                zf.writestr('MTD_MSIL1C.xml', '<?xml version="1.0"?><root></root>')
                zf.writestr('MTD_MSIL1C_duplicate.xml', '<?xml version="1.0"?><root></root>')
            
            try:
                with pytest.raises(Exception, match="Multiple or no MTD_MSIL1C files found"):
                    LoadSentinel2L1C(tmp.name)
            finally:
                os.unlink(tmp.name)

    def test_load_sentinel2_no_mtd_files_error(self):
        """Test error handling when no MTD files are found"""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, 'w') as zf:
                # Create ZIP without MTD files
                zf.writestr('some_other_file.txt', 'content')
            
            try:
                with pytest.raises(Exception, match="Multiple or no MTD_MSIL1C files found"):
                    LoadSentinel2L1C(tmp.name)
            finally:
                os.unlink(tmp.name)

    @patch('rasterio.open')
    def test_load_sentinel2_load_method(self, mock_rasterio_open, mock_sentinel2_zip):
        """Test LoadSentinel2L1C load method"""
        # Mock rasterio.open behavior
        mock_dataset = Mock()
        mock_dataset.subdatasets = [
            'SENTINEL2_L1C:/path/to/file:10m:EPSG_32755',
            'SENTINEL2_L1C:/path/to/file:20m:EPSG_32755',
            'SENTINEL2_L1C:/path/to/file:60m:EPSG_32755'
        ]
        mock_dataset.tags.return_value = {'PRODUCT_TYPE': 'S2MSI1C'}
        mock_dataset.profile = {'driver': 'GTiff'}
        mock_dataset.meta = {'width': 10980, 'height': 10980}
        mock_dataset.offsets = [0, 0, 0]
        mock_dataset.bounds = rasterio.coords.BoundingBox(600000, 4000000, 709800, 4109800)
        
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset
        
        loader = LoadSentinel2L1C(mock_sentinel2_zip)
        subdatasets = loader.load()
        
        assert len(subdatasets) == 3
        assert '10m' in subdatasets[0]
        assert '20m' in subdatasets[1]
        assert '60m' in subdatasets[2]

    @patch('rasterio.open')
    def test_get_resolution_subdatasets(self, mock_rasterio_open, mock_sentinel2_zip):
        """Test get_resolution_subdatasets method"""
        mock_dataset = Mock()
        mock_dataset.subdatasets = [
            'SENTINEL2_L1C:/path/to/file:10m:EPSG_32755',
            'SENTINEL2_L1C:/path/to/file:20m:EPSG_32755',
            'SENTINEL2_L1C:/path/to/file:60m:EPSG_32755',
            'SENTINEL2_L1C:/path/to/file:TCI:EPSG_32755'
        ]
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset
        
        loader = LoadSentinel2L1C(mock_sentinel2_zip)
        resolutions = loader.get_resolution_subdatasets()
        
        assert '10m' in resolutions
        assert '20m' in resolutions
        assert '60m' in resolutions
        assert 'tci' in resolutions
        assert len(resolutions['10m']) == 1
        assert len(resolutions['20m']) == 1
        assert len(resolutions['60m']) == 1
        assert len(resolutions['tci']) == 1

    @patch('rasterio.open')
    @patch('ShallowLearn.ResamplingMethods.get_raster_dimensions')
    def test_get_selected_bands(self, mock_get_dimensions, mock_rasterio_open, mock_sentinel2_zip):
        """Test get_selected_bands method"""
        # Mock dependencies
        mock_get_dimensions.return_value = (1000, 1000)
        mock_dataset = Mock()
        mock_dataset.subdatasets = ['SENTINEL2_L1C:/path/to/file:10m:EPSG_32755']
        mock_dataset.descriptions = ['B02', 'B03', 'B04', 'B08']
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset
        
        loader = LoadSentinel2L1C(mock_sentinel2_zip)
        
        # Mock the describe_bands method result
        with patch.object(loader, 'describe_bands') as mock_describe:
            mock_describe.return_value = {
                'SENTINEL2_L1C:/path/to/file:10m:EPSG_32755': ['B2', 'B3', 'B4', 'B8']
            }
            
            selected_bands_dict, width, height = loader.get_selected_bands(
                resolution="10m", 
                selected_bands=['B02', 'B03', 'B04', 'B08']
            )
            
            assert width == 1000
            assert height == 1000
            assert len(selected_bands_dict) == 4

    @patch('rasterio.open')
    @patch('ShallowLearn.ResamplingMethods.resample_raster')
    def test_construct_resampled_array(self, mock_resample, mock_rasterio_open, mock_sentinel2_zip):
        """Test construct_resampled_array method"""
        # Mock resampling to return test data
        mock_resample.return_value = np.random.randint(0, 1000, (100, 100))
        
        # Mock get_selected_bands
        mock_selected_bands = {
            0: ['path1', 1, 'B02'],
            1: ['path2', 1, 'B03'],
            2: ['path3', 1, 'B04'],
            3: ['path4', 1, 'B08']
        }
        
        loader = LoadSentinel2L1C(mock_sentinel2_zip)
        
        with patch.object(loader, 'get_selected_bands') as mock_get_bands:
            mock_get_bands.return_value = (mock_selected_bands, 100, 100)
            
            result = loader.construct_resampled_array(
                resolution="10m", 
                selected_bands=['B02', 'B03', 'B04', 'B08']
            )
            
            assert result.shape == (100, 100, 4)
            assert result.dtype in [np.int64, np.float64]  # Depending on mock data

    @patch('rasterio.open')
    @patch('rasterio.mask.mask')
    @patch('rasterio.warp.reproject')
    def test_clip_raster_with_shape_exact_geometry(self, mock_reproject, mock_mask, mock_rasterio_open, 
                                                   mock_sentinel2_zip, sample_shapefile_for_clipping):
        """Test clip_raster_with_shape method with exact geometry (use_mask=True)"""
        # Mock mask operation
        mock_clipped_data = np.random.randint(0, 1000, (1, 50, 50), dtype=np.uint16)
        mock_transform = rasterio.transform.from_bounds(145.2, -14.8, 145.4, -14.6, 50, 50)
        mock_mask.return_value = (mock_clipped_data, mock_transform)
        
        # Mock reproject operation
        mock_reproject.return_value = (np.random.randint(0, 1000, (50, 50)), None)
        
        # Mock rasterio open
        mock_src = Mock()
        mock_src.crs = CRS.from_epsg(32755)
        mock_rasterio_open.return_value.__enter__.return_value = mock_src
        
        # Mock get_selected_bands
        mock_selected_bands = {
            0: ['path_10m', 1, 'B02'],
            1: ['path_10m', 2, 'B03'],
            2: ['path_10m', 3, 'B04'],
            3: ['path_10m', 4, 'B08']
        }
        
        loader = LoadSentinel2L1C(mock_sentinel2_zip)
        
        with patch.object(loader, 'get_selected_bands') as mock_get_bands:
            mock_get_bands.return_value = (mock_selected_bands, 100, 100)
            
            result = loader.clip_raster_with_shape(
                sample_shapefile_for_clipping,
                resolution='10m',
                selected_bands=['B02', 'B03', 'B04', 'B08'],
                use_mask=True
            )
            
            assert isinstance(result, np.ndarray)
            assert result.shape[0] == 4  # Number of bands
            assert result.ndim == 3

    @patch('rasterio.open')
    @patch('rasterio.mask.mask')
    @patch('rasterio.warp.reproject')
    def test_clip_raster_with_shape_bounding_box(self, mock_reproject, mock_mask, mock_rasterio_open,
                                                  mock_sentinel2_zip, sample_shapefile_for_clipping):
        """Test clip_raster_with_shape method with bounding box (use_mask=False)"""
        # Mock mask operation
        mock_clipped_data = np.random.randint(0, 1000, (1, 50, 50), dtype=np.uint16)
        mock_transform = rasterio.transform.from_bounds(145.2, -14.8, 145.4, -14.6, 50, 50)
        mock_mask.return_value = (mock_clipped_data, mock_transform)
        
        # Mock reproject operation
        mock_reproject.return_value = (np.random.randint(0, 1000, (50, 50)), None)
        
        # Mock rasterio open
        mock_src = Mock()
        mock_src.crs = CRS.from_epsg(32755)
        mock_rasterio_open.return_value.__enter__.return_value = mock_src
        
        # Mock get_selected_bands
        mock_selected_bands = {
            0: ['path_10m', 1, 'B02'],
            1: ['path_10m', 2, 'B03']
        }
        
        loader = LoadSentinel2L1C(mock_sentinel2_zip)
        
        with patch.object(loader, 'get_selected_bands') as mock_get_bands:
            mock_get_bands.return_value = (mock_selected_bands, 100, 100)
            
            result = loader.clip_raster_with_shape(
                sample_shapefile_for_clipping,
                resolution='10m',
                selected_bands=['B02', 'B03'],
                use_mask=False  # Use bounding box instead of exact geometry
            )
            
            assert isinstance(result, np.ndarray)
            assert result.shape[0] == 2  # Number of bands
            assert result.ndim == 3

    def test_clip_raster_with_shape_no_resolution_match_error(self, mock_sentinel2_zip, sample_shapefile_for_clipping):
        """Test clip_raster_with_shape error when no resolution matches"""
        # Mock get_selected_bands to return bands without the requested resolution
        mock_selected_bands = {
            0: ['path_20m', 1, 'B02'],  # No 10m resolution paths
            1: ['path_20m', 2, 'B03']
        }
        
        loader = LoadSentinel2L1C(mock_sentinel2_zip)
        
        with patch.object(loader, 'get_selected_bands') as mock_get_bands:
            mock_get_bands.return_value = (mock_selected_bands, 100, 100)
            
            with pytest.raises(ValueError, match="Could not determine the final width and height"):
                loader.clip_raster_with_shape(
                    sample_shapefile_for_clipping,
                    resolution='10m',  # Looking for 10m but only 20m available
                    selected_bands=['B02', 'B03']
                )


class TestLoadFromCSV:
    @pytest.fixture
    def sample_csv_file(self):
        """Create a temporary CSV file for testing"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            csv_data = """reef,full_path,date
reef_001,/path/to/image1.tif,2023-01-01
reef_001,/path/to/image2.tif,2023-02-01
reef_002,/path/to/image3.tif,2023-01-15
reef_002,/path/to/image4.tif,2023-02-15"""
            tmp.write(csv_data)
            tmp.flush()
            
            yield tmp.name
            os.unlink(tmp.name)

    def test_load_from_csv_initialization(self, sample_csv_file):
        """Test LoadFromCSV initialization"""
        loader = LoadFromCSV(sample_csv_file)
        
        assert isinstance(loader.data_source, pd.DataFrame)
        assert len(loader.data_source) == 4
        assert 'reef' in loader.data_source.columns
        assert 'full_path' in loader.data_source.columns

    def test_get_specific_reef(self, sample_csv_file):
        """Test get_specific_reef method"""
        loader = LoadFromCSV(sample_csv_file)
        reef_data = loader.get_specific_reef('reef_001')
        
        assert len(reef_data) == 2
        assert all(reef_data['reef'] == 'reef_001')

    def test_get_specific_reef_nonexistent(self, sample_csv_file):
        """Test get_specific_reef with nonexistent reef"""
        loader = LoadFromCSV(sample_csv_file)
        reef_data = loader.get_specific_reef('reef_999')
        
        assert len(reef_data) == 0

    @patch.object(LoadGeoTIFF, 'load')
    def test_load_specific_reef(self, mock_geotiff_load, sample_csv_file):
        """Test load_specific_reef method"""
        # Mock GeoTIFF loading
        mock_geotiff_load.return_value = np.random.randint(0, 1000, (3, 100, 100))
        
        loader = LoadFromCSV(sample_csv_file)
        images = loader.load_specific_reef('reef_001')
        
        assert len(images) == 2  # reef_001 has 2 images
        assert all(isinstance(img, np.ndarray) for img in images)
        assert mock_geotiff_load.call_count == 2

    @patch.object(LoadGeoTIFF, 'load')
    def test_load_all_data(self, mock_geotiff_load, sample_csv_file):
        """Test load method for all data"""
        # Mock GeoTIFF loading
        mock_geotiff_load.return_value = np.random.randint(0, 1000, (3, 100, 100))
        
        loader = LoadFromCSV(sample_csv_file)
        all_images = loader.load()
        
        assert len(all_images) == 4  # All 4 images in CSV
        assert all(isinstance(img, np.ndarray) for img in all_images)
        assert mock_geotiff_load.call_count == 4


class TestPVIDataloader:
    @pytest.fixture
    def mock_pvi_zip(self):
        """Create a mock PVI ZIP file"""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, 'w') as zf:
                # Create a file with PVI in the name
                zf.writestr('data/PVI_image_20230101.tif', b'fake tiff data')
                zf.writestr('metadata.xml', b'<metadata></metadata>')
            
            yield tmp.name
            os.unlink(tmp.name)

    def test_pvi_dataloader_initialization(self, mock_pvi_zip):
        """Test PVI_Dataloader initialization"""
        loader = PVI_Dataloader(mock_pvi_zip)
        
        assert loader.is_zip == True
        assert 'PVI' in loader.files
        assert loader.zip_path.startswith('zip+file://')

    def test_pvi_dataloader_nonzip_file(self):
        """Test PVI_Dataloader with non-ZIP file"""
        loader = PVI_Dataloader('regular_file.tif')
        assert loader.is_zip == False

    @patch('rasterio.open')
    def test_pvi_dataloader_load(self, mock_rasterio_open, mock_pvi_zip):
        """Test PVI_Dataloader load method"""
        # Mock rasterio dataset
        mock_data = np.random.randint(0, 255, (3, 100, 100), dtype=np.uint8)
        mock_dataset = Mock()
        mock_dataset.read.return_value = mock_data
        mock_rasterio_open.return_value.__enter__.return_value = mock_dataset
        
        loader = PVI_Dataloader(mock_pvi_zip)
        result = loader.load()
        
        # The load method swaps axes, so check the final shape
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)  # After axis swapping

    def test_pvi_dataloader_failed_zip(self):
        """Test PVI_Dataloader with corrupted ZIP file"""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            tmp.write(b'not a valid zip file')
            tmp.flush()
            
            try:
                # Should handle the exception gracefully
                loader = PVI_Dataloader(tmp.name)
                # The __init__ should complete even if ZIP is corrupted
                assert loader.is_zip == True
            finally:
                os.unlink(tmp.name)


class TestLoadNumpyArray:
    @pytest.fixture
    def sample_numpy_file(self):
        """Create a temporary numpy file for testing"""
        data = np.random.randint(0, 1000, (5, 100, 100, 4), dtype=np.uint16)
        
        with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp:
            np.save(tmp.name, data)
            
            yield tmp.name
            os.unlink(tmp.name)

    def test_load_numpy_array_initialization(self, sample_numpy_file):
        """Test LoadNumpyArray initialization"""
        loader = LoadNumpyArray(sample_numpy_file)
        assert loader.data_source == sample_numpy_file

    def test_load_numpy_array_load(self, sample_numpy_file):
        """Test LoadNumpyArray load method"""
        loader = LoadNumpyArray(sample_numpy_file)
        data = loader.load()
        
        assert isinstance(data, np.ndarray)
        assert data.shape == (5, 100, 100, 4)
        assert data.dtype == np.uint16

    def test_load_numpy_array_nonexistent_file(self):
        """Test LoadNumpyArray with nonexistent file"""
        loader = LoadNumpyArray("nonexistent_file.npy")
        
        with pytest.raises(FileNotFoundError):
            loader.load()


class TestBandMappingIntegration:
    """Test integration between data loaders and band mapping"""
    
    def test_sentinel2_uses_band_mapping_parameter(self):
        """Test that Sentinel2 loader uses the band_mapping parameter correctly"""
        custom_mapping = {"B02": {"index": 0}, "B03": {"index": 1}}
        
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, 'w') as zf:
                zf.writestr('MTD_MSIL1C.xml', '<?xml version="1.0"?><root></root>')
            
            try:
                loader = LoadSentinel2L1C(tmp.name, band_mapping=custom_mapping)
                assert loader.band_mapping == custom_mapping
                assert loader.band_mapping != band_mapping  # Not the default
            finally:
                os.unlink(tmp.name)

    def test_sentinel2_default_band_mapping(self):
        """Test that Sentinel2 loader uses default band_mapping when not specified"""
        with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
            with zipfile.ZipFile(tmp.name, 'w') as zf:
                zf.writestr('MTD_MSIL1C.xml', '<?xml version="1.0"?><root></root>')
            
            try:
                loader = LoadSentinel2L1C(tmp.name)
                assert loader.band_mapping == band_mapping  # Uses default
            finally:
                os.unlink(tmp.name)


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_csv_file(self):
        """Test LoadFromCSV with empty CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp.write("reef,full_path\n")  # Header only
            tmp.flush()
            
            try:
                loader = LoadFromCSV(tmp.name)
                assert len(loader.data_source) == 0
                
                # Test methods with empty data
                reef_data = loader.get_specific_reef('any_reef')
                assert len(reef_data) == 0
                
                all_data = loader.load()
                assert len(all_data) == 0
                
            finally:
                os.unlink(tmp.name)

    def test_malformed_csv_file(self):
        """Test LoadFromCSV with malformed CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
            tmp.write("this,is,not,proper,csv,format\nwith,inconsistent,columns")
            tmp.flush()
            
            try:
                # Should still create a DataFrame, pandas is quite forgiving
                loader = LoadFromCSV(tmp.name)
                assert isinstance(loader.data_source, pd.DataFrame)
                
            finally:
                os.unlink(tmp.name)

    def test_geotiff_with_invalid_metadata(self):
        """Test LoadGeoTIFF error handling for corrupted files"""
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            tmp.write(b'not a valid geotiff file')
            tmp.flush()
            
            try:
                loader = LoadGeoTIFF(tmp.name)
                
                # Should raise rasterio error when trying to read
                with pytest.raises(rasterio.RasterioIOError):
                    loader.load()
                    
                with pytest.raises(rasterio.RasterioIOError):
                    loader.get_metadata()
                    
                with pytest.raises(rasterio.RasterioIOError):
                    loader.get_bounds()
                    
            finally:
                os.unlink(tmp.name)