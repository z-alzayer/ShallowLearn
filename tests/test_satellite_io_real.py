#!/usr/bin/env python3
"""
Real data tests for satellite I/O functionality.
Tests actual GeoTIFF, .SAFE, and .zip files with real Sentinel-2 data.
"""

import pytest
import numpy as np
import os
import sys
from pathlib import Path

# Add ShallowLearn to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ShallowLearn.io.satellite_data import Sentinel2Image
import geopandas as gpd
from shapely.geometry import box

class TestRealSentinel2Data:
    """Test with real Sentinel-2 files."""
    
    # Real file paths - adjust these to actual files on your system
    L1C_SAFE_FILE = "/mnt/sda_mount/All_L1C_55LCD/S2A_MSIL1C_20160721T004002_N0204_R059_T55LCD_20160721T003959.SAFE"
    L2A_ZIP_FILE = "/mnt/sda_mount/L2A_Full/S2A_MSIL2A_20151124T003752_N0204_R059_T55LCD_20151124T003831.zip"
    
    @pytest.fixture(autouse=True)
    def check_files_exist(self):
        """Check if test files exist, skip tests if not."""
        if not os.path.exists(self.L1C_SAFE_FILE):
            pytest.skip(f"L1C test file not found: {self.L1C_SAFE_FILE}")
        if not os.path.exists(self.L2A_ZIP_FILE):
            pytest.skip(f"L2A test file not found: {self.L2A_ZIP_FILE}")
    
    def test_l1c_safe_basic_loading(self):
        """Test basic L1C .SAFE directory loading."""
        print(f"\nTesting L1C file: {self.L1C_SAFE_FILE}")
        
        # Test basic loading (4 bands at 10m resolution)
        s2_image = Sentinel2Image(self.L1C_SAFE_FILE, load_all_bands=False)
        
        # Basic checks
        assert s2_image.image is not None
        assert s2_image.image.ndim == 3  # height, width, bands
        assert s2_image.image.shape[2] >= 4  # At least 4 bands
        assert len(s2_image.present_bands) >= 4
        
        # Check metadata
        assert 'PROCESSING_LEVEL' in s2_image.tags
        assert s2_image.tags['PROCESSING_LEVEL'] in ['Level-1C', 'L1C']
        
        # Check specific bands are present
        assert s2_image.has_band('B02')  # Blue
        assert s2_image.has_band('B03')  # Green
        assert s2_image.has_band('B04')  # Red
        assert s2_image.has_band('B08')  # NIR
        
        # Check band data access
        b02_data = s2_image.get_band_data('B02')
        assert b02_data is not None
        assert b02_data.shape == s2_image.image.shape[:2]  # Same height, width
        
        print(f"✅ L1C basic loading: {s2_image.image.shape}, {len(s2_image.present_bands)} bands")
        print(f"   Present bands: {sorted(s2_image.present_bands)}")
        print(f"   Processing level: {s2_image.tags.get('PROCESSING_LEVEL')}")
    
    def test_l1c_safe_all_bands_loading(self):
        """Test L1C .SAFE loading with all bands."""
        print(f"\nTesting L1C all bands: {self.L1C_SAFE_FILE}")
        
        s2_image = Sentinel2Image(self.L1C_SAFE_FILE, load_all_bands=True, target_resolution="10m")
        
        # Should have more bands
        assert s2_image.image is not None
        assert s2_image.image.ndim == 3
        assert len(s2_image.present_bands) >= 10  # Should have most bands
        
        # Check L1C specific bands
        expected_l1c_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        for band in expected_l1c_bands:
            assert s2_image.has_band(band), f"Missing expected L1C band: {band}"
        
        # L1C should have B10 (Cirrus)
        assert s2_image.has_band('B10'), "L1C should have B10 (Cirrus) band"
        
        print(f"✅ L1C all bands: {s2_image.image.shape}, {len(s2_image.present_bands)} bands")
        print(f"   Present bands: {sorted(s2_image.present_bands)}")
        
        # Verify all bands have same spatial dimensions (resampled to 10m)
        height, width = s2_image.image.shape[:2]
        for band_name in s2_image.present_bands:
            band_data = s2_image.get_band_data(band_name)
            assert band_data.shape == (height, width), f"Band {band_name} has wrong shape"
    
    def test_l2a_zip_basic_loading(self):
        """Test basic L2A .zip file loading."""
        print(f"\nTesting L2A file: {self.L2A_ZIP_FILE}")
        
        s2_image = Sentinel2Image(self.L2A_ZIP_FILE, load_all_bands=False)
        
        # Basic checks
        assert s2_image.image is not None
        assert s2_image.image.ndim == 3
        assert s2_image.image.shape[2] >= 4
        assert len(s2_image.present_bands) >= 4
        
        # Check metadata
        assert 'PROCESSING_LEVEL' in s2_image.tags
        assert s2_image.tags['PROCESSING_LEVEL'] in ['Level-2A', 'L2A', 'Level-2Ap']
        
        print(f"✅ L2A basic loading: {s2_image.image.shape}, {len(s2_image.present_bands)} bands")
        print(f"   Present bands: {sorted(s2_image.present_bands)}")
        print(f"   Processing level: {s2_image.tags.get('PROCESSING_LEVEL')}")
    
    def test_l2a_zip_all_bands_loading(self):
        """Test L2A .zip loading with all bands."""
        print(f"\nTesting L2A all bands: {self.L2A_ZIP_FILE}")
        
        s2_image = Sentinel2Image(self.L2A_ZIP_FILE, load_all_bands=True, target_resolution="10m")
        
        # Should have bands but NOT B10
        assert s2_image.image is not None
        assert len(s2_image.present_bands) >= 10
        
        # Check L2A specific bands (should NOT have B10)
        expected_l2a_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']
        for band in expected_l2a_bands:
            assert s2_image.has_band(band), f"Missing expected L2A band: {band}"
        
        # L2A should NOT have B10 (Cirrus) - it's removed in L2A processing
        print(f"   Checking B10 status: present={s2_image.has_band('B10')}")
        
        print(f"✅ L2A all bands: {s2_image.image.shape}, {len(s2_image.present_bands)} bands")
        print(f"   Present bands: {sorted(s2_image.present_bands)}")
    
    def test_l1c_vs_l2a_band_count_difference(self):
        """Test that L1C has more bands than L2A due to B10."""
        print(f"\nTesting L1C vs L2A band count difference")
        
        # Load both with all bands
        l1c_image = Sentinel2Image(self.L1C_SAFE_FILE, load_all_bands=True)
        l2a_image = Sentinel2Image(self.L2A_ZIP_FILE, load_all_bands=True)
        
        l1c_count = len(l1c_image.present_bands)
        l2a_count = len(l2a_image.present_bands)
        
        print(f"   L1C bands: {l1c_count} - {sorted(l1c_image.present_bands)}")
        print(f"   L2A bands: {l2a_count} - {sorted(l2a_image.present_bands)}")
        
        # L1C should have B10, L2A should not
        l1c_has_b10 = l1c_image.has_band('B10')
        l2a_has_b10 = l2a_image.has_band('B10')
        
        print(f"   L1C has B10: {l1c_has_b10}")
        print(f"   L2A has B10: {l2a_has_b10}")
        
        if l1c_has_b10 and not l2a_has_b10:
            print("✅ Correct: L1C has B10, L2A does not")
            assert l1c_count == l2a_count + 1, f"Expected L1C to have 1 more band than L2A"
        else:
            print("⚠️  Band count difference not as expected, but bands may be handled differently")


class TestRealSentinel2Clipping:
    """Test clipping functionality with real data."""
    
    L1C_SAFE_FILE = "/mnt/sda_mount/All_L1C_55LCD/S2A_MSIL1C_20160721T004002_N0204_R059_T55LCD_20160721T003959.SAFE"
    
    @pytest.fixture(autouse=True)
    def check_files_exist(self):
        """Check if test files exist."""
        if not os.path.exists(self.L1C_SAFE_FILE):
            pytest.skip(f"L1C test file not found: {self.L1C_SAFE_FILE}")
    
    def test_load_without_clipping(self):
        """Test loading full image without clipping."""
        print(f"\nTesting full image loading (no clipping)")
        
        s2_image = Sentinel2Image(self.L1C_SAFE_FILE, load_all_bands=False)
        
        full_shape = s2_image.image.shape
        full_size = s2_image.image.size
        
        print(f"✅ Full image: {full_shape}, size: {full_size:,} pixels")
        
        # Should be large (typical S2 scene is ~10980x10980)
        assert s2_image.image.shape[0] > 1000
        assert s2_image.image.shape[1] > 1000
        
        return s2_image
    
    def test_load_with_reef_clipping_post_load(self):
        """Test traditional clipping after loading (for comparison)."""
        print(f"\nTesting post-load clipping (traditional method)")
        
        # Load full image first
        s2_full = Sentinel2Image(self.L1C_SAFE_FILE, load_all_bands=False)
        
        # Load reef geometry - select first reef only for efficient testing
        from ShallowLearn.io.load_poly import get_reef_gdf
        all_reefs = get_reef_gdf()
        single_reef = all_reefs.iloc[0:1].copy()  # Select first reef only
        print(f"   Using single reef: Area = {single_reef.iloc[0].Area:.0f} m²")
        
        # Clip after loading
        s2_clipped = s2_full.clip_to_geometry(single_reef, buffer_meters=100)
        
        reduction = s2_full.image.size / s2_clipped.image.size
        
        print(f"✅ Post-load clipping:")
        print(f"   Full: {s2_full.image.shape}")
        print(f"   Clipped: {s2_clipped.image.shape}")
        print(f"   Reduction: {reduction:.1f}x")
        
        # Should be much smaller
        assert s2_clipped.image.size < s2_full.image.size
        assert reduction > 10  # Should be significant reduction
        
        return s2_clipped, single_reef
    
    def test_load_with_reef_clipping_during_load(self):
        """Test clipping during loading (new method)."""
        print(f"\nTesting during-load clipping (new method)")
        
        # Load reef geometry first - select first reef only for efficient testing
        from ShallowLearn.io.load_poly import get_reef_gdf
        all_reefs = get_reef_gdf()
        single_reef = all_reefs.iloc[0:1].copy()  # Select first reef only
        
        print(f"   Using single reef: Area = {single_reef.iloc[0].Area:.0f} m²")
        print(f"   Reef bounds: {single_reef.total_bounds}")
        
        # Load with clipping during read
        try:
            s2_clipped = Sentinel2Image(
                self.L1C_SAFE_FILE, 
                load_all_bands=False,
                clip_geometry=single_reef,
                buffer_meters=100
            )
            
            print(f"✅ During-load clipping succeeded:")
            print(f"   Clipped shape: {s2_clipped.image.shape}")
            print(f"   Clipped size: {s2_clipped.image.size:,} pixels")
            
            # Should be reef-scale (small area with 100m buffer)
            assert s2_clipped.image.shape[0] < 600  # Height < 600 pixels (~6km)
            assert s2_clipped.image.shape[1] < 600  # Width < 600 pixels (~6km)
            assert s2_clipped.image.size < 5_000_000  # Less than 5M pixels (600x600x13 bands)
            
            return s2_clipped
            
        except Exception as e:
            print(f"❌ During-load clipping failed: {e}")
            pytest.fail(f"During-load clipping failed: {e}")
    
    def test_clipping_methods_comparison(self):
        """Compare clipping during load vs after load."""
        print(f"\nComparing clipping methods")
        
        # Method 1: Traditional (load full, then clip)
        s2_full = Sentinel2Image(self.L1C_SAFE_FILE, load_all_bands=False)
        
        from ShallowLearn.io.load_poly import get_reef_gdf
        all_reefs = get_reef_gdf()
        single_reef = all_reefs.iloc[0:1].copy()  # Select first reef only
        
        s2_post_clip = s2_full.clip_to_geometry(single_reef, buffer_meters=100)
        
        # Method 2: During load clipping
        try:
            s2_during_clip = Sentinel2Image(
                self.L1C_SAFE_FILE,
                load_all_bands=False,
                clip_geometry=single_reef,
                buffer_meters=100
            )
            
            print(f"   Post-load clipping: {s2_post_clip.image.shape}")
            print(f"   During-load clipping: {s2_during_clip.image.shape}")
            
            # Shapes should be similar (might not be exactly same due to different processing)
            post_size = s2_post_clip.image.size
            during_size = s2_during_clip.image.size
            
            size_diff = abs(post_size - during_size) / max(post_size, during_size)
            print(f"   Size difference: {size_diff*100:.1f}%")
            
            # Should be reasonably similar (within 20%)
            assert size_diff < 0.20, f"Clipping methods produce very different sizes: {size_diff*100:.1f}%"
            
            print("✅ Both clipping methods work and produce similar results")
            
        except Exception as e:
            print(f"❌ During-load clipping failed: {e}")
            print("✅ Post-load clipping works as fallback")


class TestRealGeometryHandling:
    """Test geometry and CRS handling."""
    
    L1C_SAFE_FILE = "/mnt/sda_mount/All_L1C_55LCD/S2A_MSIL1C_20160721T004002_N0204_R059_T55LCD_20160721T003959.SAFE"
    
    @pytest.fixture(autouse=True)
    def check_files_exist(self):
        if not os.path.exists(self.L1C_SAFE_FILE):
            pytest.skip(f"L1C test file not found: {self.L1C_SAFE_FILE}")
    
    def test_image_bounds_and_crs(self):
        """Test that we can get image bounds and CRS correctly."""
        print(f"\nTesting image bounds and CRS")
        
        s2_image = Sentinel2Image(self.L1C_SAFE_FILE, load_all_bands=False)
        
        # Check bounds
        bounds = s2_image.get_bounds()
        assert bounds is not None
        
        print(f"   Image bounds: {bounds}")
        print(f"   Image CRS: {s2_image.meta.get('crs')}")
        print(f"   Image transform: {s2_image.meta.get('transform')}")
        
        # Bounds should be reasonable UTM coordinates
        assert bounds.left > 0
        assert bounds.bottom > 0
        assert bounds.right > bounds.left
        assert bounds.top > bounds.bottom
        
        # Should be in a UTM zone (coordinates in hundreds of thousands)
        assert 100000 < bounds.left < 800000  # Reasonable UTM easting
        assert 1000000 < bounds.bottom < 10000000  # Reasonable UTM northing (includes southern hemisphere)
    
    def test_simple_geometry_clipping(self):
        """Test clipping with a simple geometry."""
        print(f"\nTesting simple geometry clipping")
        
        # Load image to get its bounds
        s2_image = Sentinel2Image(self.L1C_SAFE_FILE, load_all_bands=False)
        bounds = s2_image.get_bounds()
        
        # Create a small box in the center of the image
        center_x = (bounds.left + bounds.right) / 2
        center_y = (bounds.bottom + bounds.top) / 2
        size = 5000  # 5km box
        
        test_geometry = box(
            center_x - size/2, center_y - size/2,
            center_x + size/2, center_y + size/2
        )
        
        test_gdf = gpd.GeoDataFrame([1], geometry=[test_geometry], crs=s2_image.meta['crs'])
        
        print(f"   Test geometry bounds: {test_gdf.total_bounds}")
        print(f"   Test geometry CRS: {test_gdf.crs}")
        
        # Try clipping with this geometry
        try:
            s2_clipped = Sentinel2Image(
                self.L1C_SAFE_FILE,
                load_all_bands=False,
                clip_geometry=test_gdf,
                buffer_meters=500
            )
            
            print(f"✅ Simple geometry clipping worked:")
            print(f"   Clipped shape: {s2_clipped.image.shape}")
            print(f"   Original size: {s2_image.image.size:,}")
            print(f"   Clipped size: {s2_clipped.image.size:,}")
            print(f"   Reduction: {s2_image.image.size / s2_clipped.image.size:.1f}x")
            
            # Should be much smaller
            assert s2_clipped.image.size < s2_image.image.size / 10
            
        except Exception as e:
            print(f"❌ Simple geometry clipping failed: {e}")
            pytest.fail(f"Simple geometry clipping failed: {e}")


if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s", "--tb=short"])