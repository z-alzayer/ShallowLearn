"""
Comprehensive tests for the IO module using real satellite data.
Tests VRT creation, metadata parsing, and satellite data classes.
"""

import os
import tempfile
import tarfile
import zipfile
from pathlib import Path
import pytest
import numpy as np
import rasterio
from osgeo import gdal

# Add the ShallowLearn directory to sys.path to import our modules
import sys
sys.path.insert(0, '/Users/ziad/Documents/GitHub/ShallowLearn')

from ShallowLearn.io.satellite_data import (
    SatelliteImage, LandsatImage, Sentinel2Image,
    SatelliteImageCollection, LandsatImageCollection, Sentinel2ImageCollection, 
    create_satellite_image, create_satellite_collection
)
from ShallowLearn.io.vrt_builder import (
    VRTBuilder, LandsatVRTBuilder, Sentinel2VRTBuilder
)


class TestRealDataAvailability:
    """Test that real data files exist before running other tests."""
    
    def test_landsat_data_exists(self):
        """Verify Landsat tar files exist."""
        landsat_dir = Path("/mnt/sda_mount/Landsat_CH3/tm")
        assert landsat_dir.exists(), f"Landsat directory {landsat_dir} does not exist"
        
        tar_files = list(landsat_dir.glob("*.tar"))
        assert len(tar_files) >= 1, f"Expected at least 1 Landsat tar file, found {len(tar_files)}"
        
        # Check specific file mentioned in e2e tests
        lt05_file = landsat_dir / "LT05_L1TP_157046_19920109_20200914_02_T1.tar"
        
        assert lt05_file.exists(), f"Landsat 5 file {lt05_file} not found"
    
    def test_sentinel2_data_exists(self):
        """Verify Sentinel-2 zip file exists."""
        sentinel2_dir = Path("/mnt/sda_mount/L1C_Full")
        assert sentinel2_dir.exists(), f"Sentinel-2 directory {sentinel2_dir} does not exist"
        
        zip_files = list(sentinel2_dir.glob("*.zip"))
        assert len(zip_files) >= 1, f"Expected at least 1 Sentinel-2 zip file, found {len(zip_files)}"
        
        # Check specific files mentioned in e2e tests
        s2_n0500_file = sentinel2_dir / "S2A_MSIL1C_20151124T003752_N0500_R059_T55LCD_20231009T132839.zip"
        s2_n0400_file = sentinel2_dir / "S2A_MSIL1C_20221008T003711_N0400_R059_T55LCD_20221008T014306.zip"
        
        # Check that at least one of these files exists
        assert s2_n0500_file.exists() or s2_n0400_file.exists(), f"Neither Sentinel-2 test file found in {sentinel2_dir}"


class TestLandsatVRTBuilder:
    """Test Landsat VRT creation with real data."""
    
    @pytest.fixture
    def landsat_files(self):
        """Get real Landsat tar files."""
        landsat_dir = Path("/Users/ziad/Documents/GitHub/ShallowLearn/data/landsat")
        return list(landsat_dir.glob("*.tar"))
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_landsat_vrt_builder_init(self, landsat_files, temp_output_dir):
        """Test LandsatVRTBuilder initialization."""
        builder = LandsatVRTBuilder(str(temp_output_dir))
        assert isinstance(builder, LandsatVRTBuilder)
        assert hasattr(builder, 'build_vrt')
        assert hasattr(builder, '_parse_metadata')
    
    def test_landsat_metadata_parsing(self, landsat_files, temp_output_dir):
        """Test MTL metadata parsing from real Landsat tar files."""
        builder = LandsatVRTBuilder(str(temp_output_dir))
        
        for tar_file in landsat_files[:1]:  # Test first file
            print(f"Testing metadata parsing for {tar_file}")
            
            # Check if MTL file exists in tar
            with tarfile.open(tar_file, 'r') as tar:
                mtl_files = [m for m in tar.getnames() if 'MTL.txt' in m]
                assert len(mtl_files) > 0, f"No MTL.txt found in {tar_file}"
            
            # Parse metadata
            metadata = builder._parse_metadata(str(tar_file))
            assert isinstance(metadata, dict)
            assert len(metadata) > 0, "Metadata dictionary is empty"
            
            # Check for essential metadata fields
            essential_fields = [
                'L1_METADATA_FILE.SPACECRAFT_ID',
                'L1_METADATA_FILE.IMAGE_ATTRIBUTES.CLOUD_COVER'
            ]
            
            for field in essential_fields:
                # Allow for variations in field naming
                found = any(field.split('.')[-1] in key for key in metadata.keys())
                assert found, f"Essential field pattern '{field}' not found in metadata keys: {list(metadata.keys())[:10]}"
    
    def test_landsat_band_detection(self, landsat_files, temp_output_dir):
        """Test detection of band files in Landsat tar archives."""
        builder = LandsatVRTBuilder(str(temp_output_dir))
        
        for tar_file in landsat_files[:1]:  # Test first file
            print(f"Testing band detection for {tar_file}")
            
            with tarfile.open(tar_file, 'r') as tar:
                all_files = tar.getnames()
                tif_files = [f for f in all_files if f.endswith('.TIF')]
                
                assert len(tif_files) > 0, f"No .TIF files found in {tar_file}"
                
                # Check for band files (should have B1, B2, etc.)
                band_files = [f for f in tif_files if '_B' in f and '_B' in f.split('_')[-1]]
                assert len(band_files) > 5, f"Expected multiple band files, found {len(band_files)}: {band_files}"
    
    def test_landsat_vrt_creation(self, landsat_files, temp_output_dir):
        """Test VRT creation from real Landsat data."""
        builder = LandsatVRTBuilder(str(temp_output_dir))
        
        for tar_file in landsat_files[:1]:  # Test first file to avoid long test times
            print(f"Testing VRT creation for {tar_file}")
            
            output_path = temp_output_dir / f"{tar_file.stem}.vrt"
            
            # Create VRT
            vrt_path = builder.build_vrt(str(tar_file))
            
            assert Path(vrt_path).exists(), f"VRT file {vrt_path} was not created"
            
            # Verify VRT can be opened with GDAL
            ds = gdal.Open(vrt_path)
            assert ds is not None, f"Could not open VRT {vrt_path} with GDAL"
            
            # Check VRT properties
            assert ds.RasterCount > 5, f"Expected multiple bands, got {ds.RasterCount}"
            assert ds.RasterXSize > 0 and ds.RasterYSize > 0, "Invalid raster dimensions"
            
            # Check metadata preservation
            metadata = ds.GetMetadata()
            assert len(metadata) > 0, "No metadata found in VRT"
            
            # Check MTL metadata domain
            mtl_metadata = ds.GetMetadata('MTL')
            assert len(mtl_metadata) > 0, "No MTL metadata found in VRT"
            
            ds = None  # Close dataset


class TestSentinel2VRTBuilder:
    """Test Sentinel-2 VRT creation with real data."""
    
    @pytest.fixture
    def sentinel2_files(self):
        """Get real Sentinel-2 zip files."""
        sentinel2_dir = Path("/Users/ziad/Documents/GitHub/ShallowLearn/data/sentinel2")
        return list(sentinel2_dir.glob("*.zip"))
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary directory for outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)
    
    def test_sentinel2_vrt_builder_init(self, sentinel2_files, temp_output_dir):
        """Test Sentinel2VRTBuilder initialization."""
        builder = Sentinel2VRTBuilder(str(temp_output_dir))
        assert isinstance(builder, Sentinel2VRTBuilder)
        assert hasattr(builder, 'build_vrt')
        assert hasattr(builder, '_parse_metadata')
    
    def test_sentinel2_metadata_parsing(self, sentinel2_files, temp_output_dir):
        """Test XML metadata parsing from real Sentinel-2 zip files."""
        builder = Sentinel2VRTBuilder(str(temp_output_dir))
        
        for zip_file in sentinel2_files[:1]:  # Test first file
            print(f"Testing metadata parsing for {zip_file}")
            
            # Check if XML metadata exists in zip
            with zipfile.ZipFile(zip_file, 'r') as zf:
                xml_files = [f for f in zf.namelist() if f.endswith('MTD_MSIL1C.xml')]
                assert len(xml_files) > 0, f"No MTD_MSIL1C.xml found in {zip_file}"
            
            # Parse metadata
            metadata = builder._parse_metadata(str(zip_file))
            assert isinstance(metadata, dict)
            assert len(metadata) > 0, "Metadata dictionary is empty"
            
            # Check for essential Sentinel-2 metadata fields
            essential_patterns = ['SPACECRAFT_NAME', 'PRODUCT_TYPE', 'CLOUD_COVERAGE']
            
            for pattern in essential_patterns:
                found = any(pattern.lower() in key.lower() for key in metadata.keys())
                assert found, f"Pattern '{pattern}' not found in metadata keys: {list(metadata.keys())[:10]}"
    
    def test_sentinel2_band_detection(self, sentinel2_files, temp_output_dir):
        """Test detection of band files in Sentinel-2 zip archives."""
        builder = Sentinel2VRTBuilder(str(temp_output_dir))
        
        for zip_file in sentinel2_files[:1]:  # Test first file
            print(f"Testing band detection for {zip_file}")
            
            with zipfile.ZipFile(zip_file, 'r') as zf:
                all_files = zf.namelist()
                jp2_files = [f for f in all_files if f.endswith('.jp2')]
                
                assert len(jp2_files) > 0, f"No .jp2 files found in {zip_file}"
                
                # Check for band files (should have B01, B02, etc.)
                band_files = [f for f in jp2_files if '_B' in f and any(f'B{i:02d}' in f for i in range(1, 13))]
                assert len(band_files) > 10, f"Expected multiple band files, found {len(band_files)}"
    
    def test_sentinel2_vrt_creation(self, sentinel2_files, temp_output_dir):
        """Test VRT creation from real Sentinel-2 data."""
        builder = Sentinel2VRTBuilder(str(temp_output_dir))
        
        for zip_file in sentinel2_files[:1]:  # Test first file to avoid long test times
            print(f"Testing VRT creation for {zip_file}")
            
            output_path = temp_output_dir / f"{zip_file.stem}.vrt"
            
            # Create VRT
            vrt_path = builder.build_vrt(str(zip_file))
            
            assert Path(vrt_path).exists(), f"VRT file {vrt_path} was not created"
            
            # Verify VRT can be opened with GDAL
            ds = gdal.Open(vrt_path)
            assert ds is not None, f"Could not open VRT {vrt_path} with GDAL"
            
            # Check VRT properties
            assert ds.RasterCount > 10, f"Expected multiple bands (>10), got {ds.RasterCount}"
            assert ds.RasterXSize > 0 and ds.RasterYSize > 0, "Invalid raster dimensions"
            
            # Check metadata preservation
            metadata = ds.GetMetadata()
            assert len(metadata) > 0, "No metadata found in VRT"
            
            ds = None  # Close dataset


class TestSatelliteImageClasses:
    """Test satellite image classes with real VRT data."""
    
    @pytest.fixture
    def temp_vrt_dir(self):
        """Create temporary VRTs from real data for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create one Landsat VRT
            landsat_dir = Path("/Users/ziad/Documents/GitHub/ShallowLearn/data/landsat")
            landsat_files = list(landsat_dir.glob("*.tar"))
            
            if landsat_files:
                builder = LandsatVRTBuilder(str(temp_path))
                landsat_vrt = builder.build_vrt(str(landsat_files[0]))
            
            # Create one Sentinel-2 VRT
            sentinel2_dir = Path("/Users/ziad/Documents/GitHub/ShallowLearn/data/sentinel2")
            sentinel2_files = list(sentinel2_dir.glob("*.zip"))
            
            if sentinel2_files:
                builder = Sentinel2VRTBuilder(str(temp_path))
                sentinel2_vrt = builder.build_vrt(str(sentinel2_files[0]))
            
            yield temp_path
    
    def test_landsat_image_loading(self, temp_vrt_dir):
        """Test LandsatImage class with real data."""
        landsat_vrt = temp_vrt_dir / "test_landsat.vrt"
        
        if landsat_vrt.exists():
            # Test loading
            img = LandsatImage(str(landsat_vrt))
            
            assert isinstance(img, LandsatImage)
            assert hasattr(img, 'image')
            assert isinstance(img.image, np.ndarray)
            assert len(img.image.shape) == 3, f"Expected 3D array, got shape {img.image.shape}"
            
            # Check band order
            assert hasattr(img, 'band_order')
            assert len(img.band_order) > 0
            
            # Check metadata
            assert hasattr(img, 'metadata')
            assert isinstance(img.metadata, dict)
    
    def test_sentinel2_image_loading(self, temp_vrt_dir):
        """Test Sentinel2Image class with real data."""
        sentinel2_vrt = temp_vrt_dir / "test_sentinel2.vrt"
        
        if sentinel2_vrt.exists():
            # Test loading
            img = Sentinel2Image(str(sentinel2_vrt))
            
            assert isinstance(img, Sentinel2Image)
            assert hasattr(img, 'image')
            assert isinstance(img.image, np.ndarray)
            assert len(img.image.shape) == 3, f"Expected 3D array, got shape {img.image.shape}"
            
            # Check band order
            assert hasattr(img, 'band_order')
            assert len(img.band_order) > 0
            
            # Check metadata
            assert hasattr(img, 'metadata')
            assert isinstance(img.metadata, dict)
    
    def test_auto_detection(self, temp_vrt_dir):
        """Test automatic satellite type detection."""
        for vrt_file in temp_vrt_dir.glob("*.vrt"):
            if vrt_file.exists():
                img = create_satellite_image(str(vrt_file))
                assert isinstance(img, SatelliteImage)
                assert img.__class__.__name__ in ['LandsatImage', 'Sentinel2Image']
    
    def test_satellite_collection(self, temp_vrt_dir):
        """Test SatelliteImageCollection with mixed data."""
        vrt_files = list(temp_vrt_dir.glob("*.vrt"))
        
        if len(vrt_files) > 0:
            collection = SatelliteImageCollection(vrt_files)
            assert len(collection) > 0
            assert len(collection.images) == len(vrt_files)
            
            # Test iteration
            for img in collection:
                assert isinstance(img, SatelliteImage)
            
            # Test indexing
            if len(collection) > 0:
                first_img = collection[0]
                assert isinstance(first_img, SatelliteImage)


class TestIntegrationWorkflows:
    """Test complete workflows with real data."""
    
    def test_end_to_end_landsat_workflow(self):
        """Test complete Landsat processing workflow."""
        landsat_dir = Path("/mnt/sda_mount/Landsat_CH3/tm")
        landsat_files = list(landsat_dir.glob("*.tar"))
        
        if not landsat_files:
            pytest.skip("No Landsat files available")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 1: Build VRT
            builder = LandsatVRTBuilder(str(temp_path))
            vrt_path = builder.build_vrt(str(landsat_files[0]))
            
            # Step 2: Load as satellite image
            img = create_satellite_image(vrt_path)
            assert img.__class__.__name__ == 'LandsatImage'
            
            # Step 3: Verify data integrity
            assert img.image.shape[2] >= 11  # At least 11 bands for Landsat
            assert not np.all(np.isnan(img.image))  # Should have some valid data
    
    def test_end_to_end_sentinel2_workflow(self):
        """Test complete Sentinel-2 processing workflow."""
        sentinel2_dir = Path("/mnt/sda_mount/L1C_Full")
        sentinel2_files = list(sentinel2_dir.glob("*.zip"))
        
        if not sentinel2_files:
            pytest.skip("No Sentinel-2 files available")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Step 1: Build VRT
            builder = Sentinel2VRTBuilder(str(temp_path))
            vrt_path = builder.build_vrt(str(sentinel2_files[0]))
            
            # Step 2: Load as satellite image
            img = create_satellite_image(vrt_path)
            assert img.__class__.__name__ == 'Sentinel2Image'
            
            # Step 3: Verify data integrity
            assert img.image.shape[2] >= 12  # At least 12 bands for Sentinel-2
            assert not np.all(np.isnan(img.image))  # Should have some valid data
    
    def test_batch_processing(self):
        """Test batch processing of multiple files."""
        # Get available files
        landsat_dir = Path("/mnt/sda_mount/Landsat_CH3/tm")
        sentinel2_dir = Path("/mnt/sda_mount/L1C_Full")
        
        landsat_files = list(landsat_dir.glob("*.tar"))[:1]  # Limit for speed
        sentinel2_files = list(sentinel2_dir.glob("*.zip"))[:1]
        
        all_files = [(f, 'landsat') for f in landsat_files] + [(f, 'sentinel2') for f in sentinel2_files]
        
        if not all_files:
            pytest.skip("No satellite files available")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            created_vrts = []
            
            # Process all files
            for file_path, satellite_type in all_files:
                if satellite_type == 'landsat':
                    builder = LandsatVRTBuilder(str(temp_path))
                else:
                    builder = Sentinel2VRTBuilder(str(temp_path))
                
                vrt_path = builder.build_vrt(str(file_path))
                created_vrts.append(vrt_path)
            
            # Verify all VRTs were created
            assert len(created_vrts) == len(all_files)
            
            # Test collection creation using factory function
            collection = create_satellite_collection(str(temp_path))
            assert len(collection) == len(all_files)
            
            # Verify each image loads correctly
            for img in collection:
                assert isinstance(img, SatelliteImage)
                assert img.__class__.__name__ in ['LandsatImage', 'Sentinel2Image']
                assert img.image.shape[2] > 10  # Should have multiple bands


def test_vrt_reading_with_rasterio():
    """Test that created VRT files can actually be read using rasterio."""
    landsat_dir = Path("/Users/ziad/Documents/GitHub/ShallowLearn/data/landsat")
    sentinel2_dir = Path("/Users/ziad/Documents/GitHub/ShallowLearn/data/sentinel2")
    
    landsat_files = list(landsat_dir.glob("*.tar"))[:1]
    sentinel2_files = list(sentinel2_dir.glob("*.zip"))[:1]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test Landsat VRT reading
        if landsat_files:
            builder = LandsatVRTBuilder(str(temp_path))
            vrt_path = builder.build_vrt(str(landsat_files[0]))
            
            with rasterio.open(vrt_path) as src:
                # Check basic properties
                assert src.count > 0
                assert src.width > 0
                assert src.height > 0
                
                # Try to read a band - this should work without errors
                band_data = src.read(1)
                assert band_data.shape == (src.height, src.width)
                
                # Try to read all bands (handle mixed dtypes)
                try:
                    all_data = src.read()
                    assert all_data.shape == (src.count, src.height, src.width)
                except ValueError as e:
                    if "more than one 'dtype' found" in str(e):
                        print(f"Mixed dtypes in Landsat VRT: {e}")
                        # Read bands individually instead
                        for i in range(1, src.count + 1):
                            band_data = src.read(i)
                            assert band_data.shape == (src.height, src.width)
                    else:
                        raise
        
        # Test Sentinel-2 VRT reading
        if sentinel2_files:
            builder = Sentinel2VRTBuilder(str(temp_path))
            vrt_path = builder.build_vrt(str(sentinel2_files[0]))
            
            print(f"Created Sentinel-2 VRT: {vrt_path}")
            
            # Print VRT content for debugging
            with open(vrt_path, 'r') as f:
                vrt_content = f.read()
                print("First 1500 chars of VRT:")
                print(vrt_content[:1500])
            
            with rasterio.open(vrt_path) as src:
                # Check basic properties
                assert src.count == 13  # Should have 13 bands
                assert src.width > 0
                assert src.height > 0
                
                print(f"Sentinel-2 VRT properties: {src.count} bands, {src.width}x{src.height}")
                
                # Try to read each band individually to identify any problematic bands
                for band_idx in range(1, src.count + 1):
                    try:
                        band_data = src.read(band_idx)
                        assert band_data.shape == (src.height, src.width)
                        print(f"Successfully read band {band_idx} shape: {band_data.shape}")
                    except Exception as e:
                        print(f"Error reading band {band_idx}: {e}")
                        raise
                
                # Try to read all bands
                try:
                    all_data = src.read()
                    assert all_data.shape == (13, src.height, src.width)
                    print(f"Successfully read all bands shape: {all_data.shape}")
                except Exception as e:
                    print(f"Error reading all bands: {e}")
                    raise


def test_vrt_with_random_crop():
    """Test VRT creation with random crop from within image bounds using GeoDataFrame."""
    import geopandas as gpd
    from shapely.geometry import Polygon
    import random
    
    landsat_dir = Path("/Users/ziad/Documents/GitHub/ShallowLearn/data/landsat")
    sentinel2_dir = Path("/Users/ziad/Documents/GitHub/ShallowLearn/data/sentinel2")
    
    landsat_files = list(landsat_dir.glob("*.tar"))[:1]
    sentinel2_files = list(sentinel2_dir.glob("*.zip"))[:1]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test Landsat cropping
        if landsat_files:
            print(f"\nTesting Landsat cropping with {landsat_files[0].name}")
            
            # First get the full image bounds
            builder = LandsatVRTBuilder(str(temp_path))
            full_vrt_path = builder.build_vrt(str(landsat_files[0]))
            
            with rasterio.open(full_vrt_path) as src:
                full_bounds = src.bounds
                full_crs = src.crs
                
                print(f"Full image bounds: {full_bounds}")
                print(f"Full image CRS: {full_crs}")
                print(f"Full image size: {src.width}x{src.height}")
                
                # Create a random crop within the image bounds (50% of original size)
                crop_width = (full_bounds.right - full_bounds.left) * 0.5
                crop_height = (full_bounds.top - full_bounds.bottom) * 0.5
                
                # Random starting position (ensuring crop stays within bounds)
                max_left = full_bounds.right - crop_width
                max_bottom = full_bounds.top - crop_height
                
                crop_left = random.uniform(full_bounds.left, max_left)
                crop_bottom = random.uniform(full_bounds.bottom, max_bottom)
                crop_right = crop_left + crop_width
                crop_top = crop_bottom + crop_height
                
                print(f"Random crop bounds: left={crop_left:.2f}, bottom={crop_bottom:.2f}, right={crop_right:.2f}, top={crop_top:.2f}")
                
                # Create GeoDataFrame with crop bounds using Polygon instead of box
                crop_geometry = Polygon([
                    (crop_left, crop_bottom),
                    (crop_right, crop_bottom), 
                    (crop_right, crop_top),
                    (crop_left, crop_top),
                    (crop_left, crop_bottom)
                ])
                crop_gdf = gpd.GeoDataFrame([1], geometry=[crop_geometry], crs=full_crs)
                
                # Create cropped VRT
                cropped_vrt_path = builder.build_vrt(str(landsat_files[0]), bounds=crop_gdf)
                
                # Verify cropped VRT
                with rasterio.open(cropped_vrt_path) as cropped_src:
                    cropped_bounds = cropped_src.bounds
                    
                    print(f"Cropped VRT bounds: {cropped_bounds}")
                    print(f"Cropped VRT size: {cropped_src.width}x{cropped_src.height}")
                    
                    # Verify bounds are approximately correct (allowing for pixel alignment)
                    assert cropped_bounds.left >= crop_left - 100  # Allow some tolerance for pixel alignment
                    assert cropped_bounds.bottom >= crop_bottom - 100
                    assert cropped_bounds.right <= crop_right + 100
                    assert cropped_bounds.top <= crop_top + 100
                    
                    # Verify cropped image is smaller than original
                    assert cropped_src.width < src.width
                    assert cropped_src.height < src.height
                    
                    # Test reading data from cropped VRT
                    try:
                        cropped_data = cropped_src.read(1)
                        assert cropped_data.shape == (cropped_src.height, cropped_src.width)
                        assert not np.all(cropped_data == 0)  # Should have actual data, not all zeros
                        print(f"Successfully read cropped Landsat data: {cropped_data.shape}")
                    except ValueError as e:
                        if "more than one 'dtype' found" in str(e):
                            # Handle mixed dtypes by reading bands individually
                            for i in range(1, min(4, cropped_src.count + 1)):
                                band_data = cropped_src.read(i)
                                assert band_data.shape == (cropped_src.height, cropped_src.width)
                        else:
                            raise
        
        # Test Sentinel-2 cropping
        if sentinel2_files:
            print(f"\nTesting Sentinel-2 cropping with {sentinel2_files[0].name}")
            
            # First get the full image bounds
            builder = Sentinel2VRTBuilder(str(temp_path))
            full_vrt_path = builder.build_vrt(str(sentinel2_files[0]))
            
            with rasterio.open(full_vrt_path) as src:
                full_bounds = src.bounds
                full_crs = src.crs
                
                print(f"Full image bounds: {full_bounds}")
                print(f"Full image CRS: {full_crs}")
                print(f"Full image size: {src.width}x{src.height}")
                
                # Create a random crop within the image bounds (30% of original size)
                crop_width = (full_bounds.right - full_bounds.left) * 0.3
                crop_height = (full_bounds.top - full_bounds.bottom) * 0.3
                
                # Random starting position
                max_left = full_bounds.right - crop_width
                max_bottom = full_bounds.top - crop_height
                
                crop_left = random.uniform(full_bounds.left, max_left)
                crop_bottom = random.uniform(full_bounds.bottom, max_bottom)
                crop_right = crop_left + crop_width
                crop_top = crop_bottom + crop_height
                
                print(f"Random crop bounds: left={crop_left:.2f}, bottom={crop_bottom:.2f}, right={crop_right:.2f}, top={crop_top:.2f}")
                
                # Create GeoDataFrame with crop bounds using Polygon instead of box
                crop_geometry = Polygon([
                    (crop_left, crop_bottom),
                    (crop_right, crop_bottom), 
                    (crop_right, crop_top),
                    (crop_left, crop_top),
                    (crop_left, crop_bottom)
                ])
                crop_gdf = gpd.GeoDataFrame([1], geometry=[crop_geometry], crs=full_crs)
                
                # Create cropped VRT
                cropped_vrt_path = builder.build_vrt(str(sentinel2_files[0]), bounds=crop_gdf)
                
                # Verify cropped VRT
                with rasterio.open(cropped_vrt_path) as cropped_src:
                    cropped_bounds = cropped_src.bounds
                    
                    print(f"Cropped VRT bounds: {cropped_bounds}")
                    print(f"Cropped VRT size: {cropped_src.width}x{cropped_src.height}")
                    
                    # Verify bounds are approximately correct
                    assert cropped_bounds.left >= crop_left - 50  # Allow tolerance for pixel alignment
                    assert cropped_bounds.bottom >= crop_bottom - 50
                    assert cropped_bounds.right <= crop_right + 50
                    assert cropped_bounds.top <= crop_top + 50
                    
                    # Verify cropped image is smaller than original
                    assert cropped_src.width < src.width
                    assert cropped_src.height < src.height
                    
                    # Verify all 13 bands are present in cropped VRT
                    assert cropped_src.count == 13
                    
                    # Test reading data from cropped VRT
                    cropped_data = cropped_src.read(2)  # Read band 2 (B02)
                    assert cropped_data.shape == (cropped_src.height, cropped_src.width)
                    assert not np.all(cropped_data == 0)  # Should have actual data
                    print(f"Successfully read cropped Sentinel-2 data: {cropped_data.shape}")
                    
                    # Test reading all bands
                    all_cropped_data = cropped_src.read()
                    assert all_cropped_data.shape == (13, cropped_src.height, cropped_src.width)
                    print(f"Successfully read all cropped Sentinel-2 bands: {all_cropped_data.shape}")
                    
                    print("Cropping test passed successfully!")


def test_vrt_bounds_validation():
    """Test that VRT builder properly validates bounds and rejects non-intersecting crops."""
    import geopandas as gpd
    from shapely.geometry import Polygon
    
    sentinel2_dir = Path("/mnt/sda_mount/L1C_Full")
    sentinel2_files = list(sentinel2_dir.glob("*.zip"))[:1]
    
    if not sentinel2_files:
        pytest.skip("No Sentinel-2 files available")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test with Sentinel-2 file
        zip_file = sentinel2_files[0]
        print(f"\\nTesting bounds validation with {zip_file.name}")
        
        builder = Sentinel2VRTBuilder(str(temp_path))
        
        # First, get the actual image bounds
        full_vrt_path = builder.build_vrt(str(zip_file))
        
        with rasterio.open(full_vrt_path) as src:
            image_bounds = src.bounds
            image_crs = src.crs
            
            print(f"Image bounds: {image_bounds}")
            print(f"Image CRS: {image_crs}")
        
        # Test 1: Create bounds that DO intersect (should succeed)
        intersecting_bounds = gpd.GeoDataFrame(
            [1],
            geometry=[Polygon([
                (image_bounds.left + 1000, image_bounds.bottom + 1000),
                (image_bounds.left + 2000, image_bounds.bottom + 1000),
                (image_bounds.left + 2000, image_bounds.bottom + 2000),
                (image_bounds.left + 1000, image_bounds.bottom + 2000),
                (image_bounds.left + 1000, image_bounds.bottom + 1000)
            ])],
            crs=image_crs
        )
        
        print("\\n1️⃣ Testing with intersecting bounds...")
        intersecting_vrt = builder.build_vrt(str(zip_file), bounds=intersecting_bounds)
        assert intersecting_vrt is not None, "VRT creation should succeed for intersecting bounds"
        print(f"✅ VRT created successfully: {Path(intersecting_vrt).name}")
        
        # Verify the created VRT has reasonable dimensions
        with rasterio.open(intersecting_vrt) as src:
            assert src.width < 1000 and src.height < 1000, "Cropped VRT should be small"
            print(f"✅ Cropped VRT dimensions: {src.width} x {src.height}")
        
        # Test 2: Create bounds that do NOT intersect (should return None)
        non_intersecting_bounds = gpd.GeoDataFrame(
            [1], 
            geometry=[Polygon([
                (image_bounds.right + 10000, image_bounds.top + 10000),
                (image_bounds.right + 11000, image_bounds.top + 10000),
                (image_bounds.right + 11000, image_bounds.top + 11000),
                (image_bounds.right + 10000, image_bounds.top + 11000),
                (image_bounds.right + 10000, image_bounds.top + 10000)
            ])],
            crs=image_crs
        )
        
        print("\\n2️⃣ Testing with non-intersecting bounds...")
        non_intersecting_vrt = builder.build_vrt(str(zip_file), bounds=non_intersecting_bounds)
        assert non_intersecting_vrt is None, "VRT creation should return None for non-intersecting bounds"
        print("✅ VRT creation correctly rejected for non-intersecting bounds")
        
        # Test 3: Create bounds in different CRS that don't intersect (should also return None)
        # Create bounds in WGS84 that are far from the image
        wgs84_non_intersecting = gpd.GeoDataFrame(
            [1],
            geometry=[Polygon([
                (0.0, 0.0),  # Somewhere near Africa (far from Australia)
                (0.1, 0.0),
                (0.1, 0.1),
                (0.0, 0.1),
                (0.0, 0.0)
            ])],
            crs='EPSG:4326'
        )
        
        print("\\n3️⃣ Testing with non-intersecting bounds in different CRS...")
        different_crs_vrt = builder.build_vrt(str(zip_file), bounds=wgs84_non_intersecting)
        assert different_crs_vrt is None, "VRT creation should return None for non-intersecting bounds in different CRS"
        print("✅ VRT creation correctly rejected for non-intersecting bounds in different CRS")


def test_l2a_auxiliary_bands_filtering():
    """Test that L2A auxiliary bands can be included/excluded properly."""
    import geopandas as gpd
    from shapely.geometry import Polygon
    
    sentinel2_dir = Path("/mnt/sda_mount/L2A_Full")
    sentinel2_files = list(sentinel2_dir.glob("*.zip"))[:1]
    
    if not sentinel2_files:
        pytest.skip("No Sentinel-2 L2A files available")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        zip_file = sentinel2_files[0]
        print(f"\\nTesting L2A auxiliary bands with {zip_file.name}")
        
        # Test 1: Default behavior (exclude auxiliary bands)
        print("1️⃣ Testing default behavior (exclude auxiliary bands)...")
        builder_default = Sentinel2VRTBuilder(str(temp_path), include_auxiliary_bands=False)
        
        # Get band files and check filtering
        band_files_default = builder_default._get_band_files(str(zip_file))
        auxiliary_count_default = sum(1 for f in band_files_default 
                                     if any(aux in os.path.basename(f) for aux in ['AOT', 'WVP', 'SCL']))
        
        print(f"   Total bands: {len(band_files_default)}")
        print(f"   Auxiliary bands: {auxiliary_count_default}")
        assert auxiliary_count_default == 0, "Should exclude auxiliary bands by default"
        
        # Create VRT and test loading
        vrt_default = builder_default.build_vrt(str(zip_file))
        assert vrt_default is not None, "VRT creation should succeed"
        
        # Test loading the VRT
        with rasterio.open(vrt_default) as src:
            print(f"   VRT dimensions: {src.width} x {src.height}")
            print(f"   VRT band count: {src.count}")
            
            # Should have reasonable number of bands (around 12-13 for spectral only)
            assert src.count <= 15, f"Too many bands ({src.count}) - should be ~12-13 for spectral only"
            assert src.count >= 10, f"Too few bands ({src.count}) - should have at least 10 spectral bands"
        
        # Test 2: Include auxiliary bands
        print("2️⃣ Testing with auxiliary bands included...")
        builder_with_aux = Sentinel2VRTBuilder(str(temp_path), include_auxiliary_bands=True)
        
        band_files_with_aux = builder_with_aux._get_band_files(str(zip_file))
        auxiliary_count_with_aux = sum(1 for f in band_files_with_aux 
                                      if any(aux in os.path.basename(f) for aux in ['AOT', 'WVP', 'SCL']))
        
        print(f"   Total bands: {len(band_files_with_aux)}")
        print(f"   Auxiliary bands: {auxiliary_count_with_aux}")
        
        # Should include auxiliary bands now
        assert auxiliary_count_with_aux > 0, "Should include auxiliary bands when requested"
        assert len(band_files_with_aux) > len(band_files_default), "Should have more bands when including auxiliary"
        
        print(f"✅ L2A auxiliary bands filtering works correctly")
        print(f"   Default: {len(band_files_default)} bands (0 auxiliary)")
        print(f"   With auxiliary: {len(band_files_with_aux)} bands ({auxiliary_count_with_aux} auxiliary)")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])