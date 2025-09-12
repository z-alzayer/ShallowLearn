"""
Tests for I/O metadata preservation and correctness.
Ensures that load_image and related functions preserve proper metadata.
"""

import os
import tempfile
import numpy as np
import pytest
import rasterio as rio
from pathlib import Path
from rasterio.crs import CRS
from rasterio.transform import from_bounds

# Import the functions to test
import sys
sys.path.insert(0, "/home/zba21/Documents/ShallowLearn")

from ShallowLearn.io.image_loader import load_image
from ShallowLearn.io.satellite_data import GeoTIFFImage, Sentinel2Image
from ShallowLearn.io.vrt_builder import Sentinel2VRTBuilder


class TestMetadataIntegrity:
    """Test metadata preservation across I/O operations."""
    
    def setup_method(self):
        """Setup test files and directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_crs = CRS.from_epsg(32755)  # UTM Zone 55S
        self.test_transform = from_bounds(300000, 8300000, 310000, 8310000, 100, 100)
        
    def teardown_method(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_geotiff(self, bands: int = 4, dtype='uint16') -> str:
        """Create a test GeoTIFF file with proper metadata."""
        test_file = os.path.join(self.temp_dir, "test_image.tif")
        
        # Create test data
        data = np.random.randint(0, 10000, size=(bands, 100, 100), dtype=dtype)
        
        # Write with proper metadata
        with rio.open(
            test_file,
            'w',
            driver='GTiff',
            height=100,
            width=100,
            count=bands,
            dtype=dtype,
            crs=self.test_crs,
            transform=self.test_transform,
            nodata=0
        ) as dst:
            dst.write(data)
            
        return test_file
    
    def test_geotiff_metadata_preservation(self):
        """Test that GeoTIFF metadata is correctly preserved."""
        test_file = self.create_test_geotiff(bands=12)
        
        # Load using our I/O function
        img, metadata, bounds = load_image(test_file, return_meta=True)
        
        # Verify image properties
        assert img is not None, "Image should not be None"
        assert len(img.shape) == 3, f"Expected 3D array, got shape {img.shape}"
        assert img.shape[2] == 12, f"Expected 12 bands, got {img.shape[2]}"
        
        # Verify metadata
        assert metadata is not None, "Metadata should not be None"
        assert 'crs' in metadata, "CRS should be in metadata"
        assert 'transform' in metadata, "Transform should be in metadata"
        assert 'count' in metadata, "Band count should be in metadata"
        assert 'width' in metadata and 'height' in metadata, "Dimensions should be in metadata"
        
        # Verify specific values
        assert metadata['count'] == 12, f"Expected count=12, got {metadata['count']}"
        assert metadata['crs'] == self.test_crs, f"CRS mismatch: expected {self.test_crs}, got {metadata['crs']}"
        assert metadata['width'] == 100, f"Expected width=100, got {metadata['width']}"
        assert metadata['height'] == 100, f"Expected height=100, got {metadata['height']}"
        
    def test_vrt_metadata_preservation(self):
        """Test that VRT metadata is correctly preserved."""
        # Create a test VRT by first creating a GeoTIFF
        test_geotiff = self.create_test_geotiff(bands=4)
        vrt_file = os.path.join(self.temp_dir, "test.vrt")
        
        # Create VRT
        from osgeo import gdal
        gdal.BuildVRT(vrt_file, [test_geotiff], separate=True)
        
        # Load using our I/O function
        img, metadata, bounds = load_image(vrt_file, return_meta=True)
        
        # Verify VRT handling
        assert img is not None, "VRT image should not be None"
        assert metadata is not None, "VRT metadata should not be None"
        assert metadata['count'] == 4, f"Expected 4 bands from VRT, got {metadata['count']}"
        assert 'driver' in metadata, "Driver should be in metadata"
        
    def test_band_count_accuracy(self):
        """Test that band count is accurately reported across different scenarios."""
        # Test different band counts
        for band_count in [1, 3, 4, 8, 12]:
            test_file = self.create_test_geotiff(bands=band_count)
            
            img, metadata, bounds = load_image(test_file, return_meta=True)
            
            assert metadata['count'] == band_count, \
                f"Band count mismatch for {band_count} bands: expected {band_count}, got {metadata['count']}"
            assert img.shape[2] == band_count, \
                f"Image shape mismatch for {band_count} bands: expected {band_count}, got {img.shape[2]}"
    
    def test_geotiff_loader_direct(self):
        """Test GeoTIFFImage class directly."""
        test_file = self.create_test_geotiff(bands=6, dtype='uint16')
        
        # Load using GeoTIFFImage
        loader = GeoTIFFImage(test_file)
        img = loader.load()
        metadata = loader.get_metadata()
        bounds = loader.get_bounds()
        
        # Verify results
        assert img is not None, "Direct GeoTIFF loading should work"
        assert img.shape[0] == 6, f"Expected 6 bands in bands-first format, got {img.shape[0]}"
        assert metadata['count'] == 6, f"Expected count=6, got {metadata['count']}"
        assert metadata['dtype'] == 'uint16', f"Expected uint16 dtype, got {metadata['dtype']}"
        assert bounds is not None, "Bounds should be available"
    
    def test_sentinel2_detection(self):
        """Test that Sentinel-2 files are properly detected and handled."""
        # Create a test file with Sentinel-2 naming pattern
        s2_file = os.path.join(self.temp_dir, "S2A_MSIL1C_20200101T120000_test.tif")
        
        # Create test data (bands-first format as expected from Sentinel-2)
        data = np.random.randint(0, 10000, size=(4, 100, 100), dtype='uint16')
        
        with rio.open(
            s2_file,
            'w',
            driver='GTiff',
            height=100,
            width=100,
            count=4,
            dtype='uint16',
            crs=self.test_crs,
            transform=self.test_transform
        ) as dst:
            dst.write(data)
        
        # Load and verify format detection (use geotiff for simple TIF files)
        img, metadata, bounds = load_image(s2_file, return_meta=True, file_format="geotiff")
        
        assert img is not None, "Sentinel-2 file should load"
        assert len(img.shape) == 3, "Should be 3D array in channels-last format"
        # Image should be transposed to channels-last format by load_image
        assert img.shape[2] == 4, f"Expected 4 bands in channels-last, got shape {img.shape}"
        
    def test_metadata_types(self):
        """Test that metadata values have correct types."""
        test_file = self.create_test_geotiff()
        
        img, metadata, bounds = load_image(test_file, return_meta=True)
        
        # Check metadata types
        assert isinstance(metadata['width'], int), f"Width should be int, got {type(metadata['width'])}"
        assert isinstance(metadata['height'], int), f"Height should be int, got {type(metadata['height'])}"
        assert isinstance(metadata['count'], int), f"Count should be int, got {type(metadata['count'])}"
        assert isinstance(metadata['crs'], CRS), f"CRS should be CRS object, got {type(metadata['crs'])}"
        
    def test_bounds_calculation(self):
        """Test that bounds are correctly calculated."""
        test_file = self.create_test_geotiff()
        
        img, metadata, bounds = load_image(test_file, return_meta=True)
        
        assert bounds is not None, "Bounds should not be None"
        
        # Verify bounds are reasonable (within our test transform bounds)
        assert 300000 <= bounds.left <= 310000, f"Left bound out of range: {bounds.left}"
        assert 300000 <= bounds.right <= 310000, f"Right bound out of range: {bounds.right}"
        assert 8300000 <= bounds.bottom <= 8310000, f"Bottom bound out of range: {bounds.bottom}"
        assert 8300000 <= bounds.top <= 8310000, f"Top bound out of range: {bounds.top}"
        
    def test_nodata_handling(self):
        """Test that nodata values are properly handled in metadata."""
        test_file = self.create_test_geotiff()
        
        # Manually set nodata value
        with rio.open(test_file, 'r+') as dst:
            dst.nodata = 65535
        
        img, metadata, bounds = load_image(test_file, return_meta=True)
        
        # Check that nodata is preserved
        assert 'nodata' in metadata, "Nodata should be in metadata"
        assert metadata['nodata'] == 65535, f"Expected nodata=65535, got {metadata['nodata']}"
    
    def test_error_handling(self):
        """Test proper error handling for invalid files."""
        nonexistent_file = os.path.join(self.temp_dir, "nonexistent.tif")
        
        # Should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            load_image(nonexistent_file, return_meta=True)
        
    def test_format_detection(self):
        """Test automatic format detection."""
        # Test different file extensions
        formats = [
            ("test.tif", "geotiff"),
            ("test.vrt", "geotiff"), 
            ("S2A_MSIL1C_test.zip", "sentinel2"),
            ("S2B_MSIL2A_test.SAFE", "sentinel2"),
            ("LC08_test.tar", "landsat")
        ]
        
        for filename, expected_format in formats:
            from ShallowLearn.io.image_loader import _detect_file_format
            detected = _detect_file_format(Path(filename))
            assert detected == expected_format, \
                f"Format detection failed for {filename}: expected {expected_format}, got {detected}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])