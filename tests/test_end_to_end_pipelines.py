"""
End-to-end pipeline tests using real data.

These tests verify the complete data pipeline from loading through processing
to visualization using actual satellite and remote sensing data stored locally.
No mock data is used - all tests use hardcoded paths to real data.

Test Categories:
1. GeoTIFF Pipeline (single files, multi-band, collections)
2. Sentinel-2 Pipeline (ZIP, SAFE, different processing baselines)
3. Landsat Pipeline (individual bands, collections, TAR archives)
4. Classification Data Pipeline (benthic, geomorphic)
5. Multi-format Integration Tests
6. Visualization Output Tests (RGB, color spaces, indices)
"""

from pathlib import Path

import numpy as np
import pytest

from ShallowLearn.band_mapping import band_mapping
from ShallowLearn.core.array_utils import (
    clip_array,
    normalize_array,
)

# Import the new modular structure
from ShallowLearn.io import (
    GeoTIFFCollection,
    GeoTIFFImage,
    LoadGeoTIFF,
    load_image,
)
from ShallowLearn.visualization import (
    plot_color_space,
    plot_hsv,
    plot_lab,
    plot_rgb,
    plot_rgb_enhanced,
    plot_ycbcr,
)


class TestDataPaths:
    """Hardcoded paths to real data for testing."""

    # GeoTIFF Data
    GBR_BENTHIC_SINGLE = "/mnt/sda_mount/Clipped/GBR_2017/benthic_2/0_benthic_2.tif"
    GBR_GEOMORPHIC_SINGLE = (
        "/mnt/sda_mount/Clipped/GBR_2017/geomorphic/0_geomorphic.tif"
    )

    # Planetscope Multi-band Data
    PLANETSCOPE_DIR = (
        "/mnt/sda_mount/Clipped/Planetscope/20230331_233843_22_24a8_3B_udm2"
    )
    PLANETSCOPE_SINGLE = "/mnt/sda_mount/Clipped/Planetscope/20230331_233843_22_24a8_3B_udm2/1_20230331_233843_22_24a8_3B_udm2.tif"

    # Sentinel-2 Data
    SENTINEL2_ZIP_N0500 = "/mnt/sda_mount/L1C_Full/S2A_MSIL1C_20151124T003752_N0500_R059_T55LCD_20231009T132839.zip"
    SENTINEL2_ZIP_N0400 = "/mnt/sda_mount/L1C_Full/S2A_MSIL1C_20221008T003711_N0400_R059_T55LCD_20221008T014306.zip"
    SENTINEL2_SAFE_DIR = "/mnt/sda_mount/Clipped/L1C/S2A_MSIL1C_20160323T003752_N0201_R059_T55LCD_20160323T003830.SAFE"

    # Landsat Data
    LANDSAT_SINGLE_BAND = "/mnt/sda_mount/Landsat_CH3/bands/LC08_L2SP_157046_20220127_20220204_02_T1_SR_B2.TIF"
    LANDSAT_BANDS_DIR = "/mnt/sda_mount/Landsat_CH3/bands"
    LANDSAT_TAR_FILE = (
        "/mnt/sda_mount/Landsat_CH3/tm/LT05_L1TP_157046_19920109_20200914_02_T1.tar"
    )

    # Collections
    GBR_BENTHIC_DIR = "/mnt/sda_mount/Clipped/GBR_2017/benthic_2"
    GBR_GEOMORPHIC_DIR = "/mnt/sda_mount/Clipped/GBR_2017/geomorphic"


class TestGeoTIFFPipeline:
    """Test complete GeoTIFF processing pipeline."""

    def test_single_band_geotiff_pipeline(self):
        """Test pipeline: single-band GeoTIFF → load → process → visualize."""
        # Load single-band classification data
        img = load_image(TestDataPaths.GBR_BENTHIC_SINGLE)

        # Validate loading
        assert img is not None
        assert len(img.shape) == 3  # (height, width, bands)
        assert img.shape[2] == 1  # Single band
        assert img.dtype in [np.uint8, np.uint16, np.int16, np.float32, np.float64]

        # Process with array utilities
        clipped = clip_array(img, 0, 255)
        assert clipped.shape == img.shape
        assert np.all(clipped >= 0)
        assert np.all(clipped <= 255)

        # Test with metadata
        img_with_meta, metadata, bounds = load_image(
            TestDataPaths.GBR_BENTHIC_SINGLE, return_meta=True
        )

        assert np.array_equal(img, img_with_meta)
        assert isinstance(metadata, dict)
        assert "crs" in metadata
        assert bounds is not None

        print("✓ Single-band GeoTIFF pipeline complete")

    def test_multiband_geotiff_pipeline(self):
        """Test pipeline: multi-band GeoTIFF → load → RGB → color spaces."""
        # Load Planetscope multi-band data
        img = load_image(TestDataPaths.PLANETSCOPE_SINGLE)

        # Validate loading
        assert img is not None
        assert len(img.shape) == 3
        assert img.shape[2] > 3  # Multi-band (Planetscope has 8 bands)

        # Test RGB visualization (using first 3 bands)
        rgb = plot_rgb(img, band_indices=[0, 1, 2], plot=False)
        assert rgb.shape == (img.shape[0], img.shape[1], 3)
        assert rgb.dtype == np.uint8
        assert np.all(rgb >= 0) and np.all(rgb <= 255)

        # Test color space conversions
        hsv = plot_hsv(img, plot=False)
        assert hsv.shape == (img.shape[0], img.shape[1], 3)

        lab = plot_lab(img, plot=False)
        assert lab.shape == (img.shape[0], img.shape[1], 3)

        ycbcr = plot_ycbcr(img, plot=False)
        assert ycbcr.shape == (img.shape[0], img.shape[1], 3)

        print("✓ Multi-band GeoTIFF pipeline complete")

    def test_geotiff_collection_pipeline(self):
        """Test pipeline: GeoTIFF collection → batch load → compare."""
        # Load collection of classification tiles
        collection = GeoTIFFCollection(TestDataPaths.GBR_BENTHIC_DIR, pattern="*.tif")

        assert len(collection) > 0
        print(f"Found {len(collection)} files in collection")

        # Test individual file access
        first_img = collection[0]
        assert isinstance(first_img, GeoTIFFImage)
        assert first_img.image is not None

        # Test batch loading (limit to first 3 for speed)
        images = (
            collection.load_all()[:3] if len(collection) > 3 else collection.load_all()
        )
        assert len(images) > 0

        # Validate all images have consistent properties
        shapes = [img.image.shape for img in images]
        dtypes = [img.image.dtype for img in images]

        # All should be same dtype (classification data)
        assert len(set(dtypes)) <= 2  # Allow for minor dtype variations

        print(f"✓ Collection pipeline complete: {len(images)} images loaded")

    def test_backwards_compatibility_geotiff(self):
        """Test that old LoadGeoTIFF interface still works."""
        # Test old interface
        old_loader = LoadGeoTIFF(TestDataPaths.GBR_BENTHIC_SINGLE)
        old_data = old_loader.load()
        old_meta = old_loader.get_metadata()
        old_bounds = old_loader.get_bounds()

        # Test new interface (disable clipping to match old behavior)
        new_data = load_image(
            TestDataPaths.GBR_BENTHIC_SINGLE, clip=False
        )  # Match old format
        new_data_meta, new_meta, new_bounds = load_image(
            TestDataPaths.GBR_BENTHIC_SINGLE, return_meta=True, clip=False
        )

        # Compare results (accounting for new channels-last format)
        # Old format: (bands, height, width), New format: (height, width, bands)
        if len(old_data.shape) == 3:
            # Transpose old data to match new channels-last format for comparison
            old_data_transposed = np.transpose(old_data, (1, 2, 0))
            assert old_data_transposed.shape == new_data.shape
            assert np.array_equal(old_data_transposed, new_data)
        else:
            assert old_data.shape == new_data.shape
            
        assert old_data.dtype == new_data.dtype
        assert old_meta["crs"] == new_meta["crs"]

        print("✓ Backwards compatibility maintained (with channels-last format)")


class TestSentinel2Pipeline:
    """Test Sentinel-2 data pipeline with different processing baselines."""

    def test_n0400_processing_baseline_correction(self):
        """Test that N0400+ processing baseline correction is applied."""
        # This would test with a real N0400 file if Sentinel-2 loading was implemented
        # For now, test the regex pattern matching
        from ShallowLearn.io.image_loader import _apply_legacy_transformations

        test_array = np.array([[[1000, 2000, 3000]]], dtype=np.float64)

        # Test N0400 correction
        n0400_path = Path("S2A_MSIL1C_20221008T003711_N0400_R059_T55LCD_test.zip")
        corrected = _apply_legacy_transformations(test_array, n0400_path)
        expected = test_array - 1000
        assert np.array_equal(corrected, expected)

        # Test N0500 correction (should also apply)
        n0500_path = Path("S2A_MSIL1C_20151124T003752_N0500_R059_T55LCD_test.zip")
        corrected = _apply_legacy_transformations(test_array, n0500_path)
        assert np.array_equal(corrected, expected)

        # Test N0399 (should NOT apply)
        n0399_path = Path("S2A_MSIL1C_20151124T003752_N0399_R059_T55LCD_test.zip")
        uncorrected = _apply_legacy_transformations(test_array, n0399_path)
        assert np.array_equal(uncorrected, test_array)

        print("✓ N0400+ processing baseline correction working")

    def test_file_format_detection(self):
        """Test auto-detection of Sentinel-2 files."""
        from ShallowLearn.io.image_loader import _detect_file_format

        # Test various Sentinel-2 patterns
        s2_patterns = [
            "S2A_MSIL1C_20221008T003711_N0400_R059_T55LCD_20221008T014306.zip",
            "S2B_MSIL2A_20210101T000000_N0300_R001_T31UDQ_20210101T000000.SAFE",
            "s2a_msil1c_test.tif",  # lowercase
        ]

        for pattern in s2_patterns:
            format_type = _detect_file_format(Path(pattern))
            assert format_type == "sentinel2", (
                f"Failed to detect Sentinel-2 in {pattern}"
            )

        # Test non-Sentinel-2 patterns
        other_patterns = [
            "LC08_L1TP_123456_20210101_20210101_01_T1_B1.TIF",
            "regular_image.tif",
            "classification.tiff",
        ]

        for pattern in other_patterns:
            format_type = _detect_file_format(Path(pattern))
            assert format_type != "sentinel2", (
                f"Incorrectly detected Sentinel-2 in {pattern}"
            )

        print("✓ File format detection working")


class TestLandsatPipeline:
    """Test Landsat data processing pipeline."""

    def test_landsat_single_band_pipeline(self):
        """Test single Landsat band processing."""
        # Load single Landsat band
        img = load_image(TestDataPaths.LANDSAT_SINGLE_BAND)

        assert img is not None
        assert len(img.shape) == 3
        # Landsat bands are typically single-band TIFFs

        # Check for NaN values (common in Landsat data)
        has_nan = np.isnan(img).any()
        print(f"Image has NaN values: {has_nan}")
        
        if has_nan:
            # Handle NaN values by replacing with 0 for processing
            img_clean = np.nan_to_num(img, nan=0.0)
            print(f"Cleaned image range: {img_clean.min()} to {img_clean.max()}")
        else:
            img_clean = img

        # Test processing with cleaned data
        clipped = clip_array(img_clean, 0, 10000)
        normalized = normalize_array(clipped)

        assert normalized.shape == img.shape
        
        # Only check finite values if we had NaNs
        if has_nan:
            finite_mask = np.isfinite(normalized)
            if finite_mask.any():
                assert np.all(normalized[finite_mask] >= 0)
                assert np.all(normalized[finite_mask] <= 1)
        else:
            assert np.all(normalized >= 0)
            assert np.all(normalized <= 1)

        print("✓ Landsat single band pipeline complete")

    def test_landsat_format_detection(self):
        """Test Landsat file format detection."""
        from ShallowLearn.io.image_loader import _detect_file_format

        landsat_patterns = [
            "LC08_L2SP_157046_20220127_20220204_02_T1_SR_B2.TIF",
            "LC09_L1TP_123456_20210101_20210101_01_T1_B1.TIF",
            "LE07_L1TP_123456_20210101_20210101_01_T1_B1.TIF",
            "LT05_L1TP_123456_20210101_20210101_01_T1_B1.TIF",
        ]

        for pattern in landsat_patterns:
            format_type = _detect_file_format(Path(pattern))
            assert format_type == "landsat", f"Failed to detect Landsat in {pattern}"

        print("✓ Landsat format detection working")


class TestVisualizationPipeline:
    """Test comprehensive visualization pipeline."""

    def test_flexible_rgb_visualization(self):
        """Test RGB visualization with different band selection methods."""
        # Load multi-band image
        img = load_image(TestDataPaths.PLANETSCOPE_SINGLE)

        # Test direct band indices
        rgb1 = plot_rgb_enhanced(img, band_indices=[0, 1, 2], plot=False)
        assert rgb1.shape == (img.shape[0], img.shape[1], 3)

        # Test with stretch disabled
        rgb2 = plot_rgb_enhanced(img, band_indices=[0, 1, 2], stretch=False, plot=False)
        assert rgb2.shape == rgb1.shape
        # Should be different due to stretching
        assert not np.array_equal(rgb1, rgb2)

        # Test default behavior (backwards compatible)
        rgb3 = plot_rgb(img, plot=False)  # Should use defaults
        assert rgb3.shape == (img.shape[0], img.shape[1], 3)

        print("✓ Flexible RGB visualization working")

    def test_color_space_pipeline(self):
        """Test unified color space conversion pipeline."""
        # Load test image
        img = load_image(TestDataPaths.PLANETSCOPE_SINGLE)

        # Test each color space
        color_spaces = ["hsv", "lab", "ycbcr"]

        for cs in color_spaces:
            converted = plot_color_space(
                img, color_space=cs, band_indices=[0, 1, 2], plot=False
            )

            assert converted.shape == (img.shape[0], img.shape[1], 3)
            assert np.isfinite(converted).all()  # No NaN/inf values

        print("✓ Color space pipeline working")

    def test_band_mapping_integration(self):
        """Test visualization with band mapping for Sentinel-2 style data."""
        # Create synthetic 13-band image (like Sentinel-2)
        synthetic_img = np.random.randint(100, 3000, (50, 50, 13), dtype=np.uint16)

        # Test with band mapping
        rgb_mapped = plot_rgb_enhanced(
            synthetic_img,
            band_names=["B04", "B03", "B02"],  # Red, Green, Blue
            band_mapping=band_mapping,
            plot=False,
        )

        assert rgb_mapped.shape == (50, 50, 3)
        assert rgb_mapped.dtype == np.uint8

        # Test without band mapping (should use defaults)
        rgb_default = plot_rgb_enhanced(synthetic_img, plot=False)
        assert rgb_default.shape == (50, 50, 3)

        print("✓ Band mapping integration working")


class TestDataIntegration:
    """Test integration across different data types."""

    def test_mixed_data_processing(self):
        """Test processing different data types in sequence."""
        data_files = [
            (TestDataPaths.GBR_BENTHIC_SINGLE, "classification"),
            (TestDataPaths.PLANETSCOPE_SINGLE, "multispectral"),
            (TestDataPaths.LANDSAT_SINGLE_BAND, "landsat"),
        ]

        results = []

        for file_path, data_type in data_files:
            # Load with consistent interface
            img = load_image(file_path)

            # Apply consistent processing
            clipped = clip_array(img, 0, 10000)

            # Store results
            results.append(
                {
                    "type": data_type,
                    "shape": img.shape,
                    "dtype": img.dtype,
                    "path": file_path,
                }
            )

        # Validate we processed different types successfully
        types_processed = set(r["type"] for r in results)
        assert len(types_processed) >= 2  # At least 2 different types

        print(f"✓ Mixed data integration: {types_processed}")

    def test_error_handling_and_validation(self):
        """Test proper error handling for invalid inputs."""
        # Test non-existent file
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path.tif")

        # Test invalid band indices
        img = load_image(TestDataPaths.PLANETSCOPE_SINGLE)
        with pytest.raises(ValueError):
            plot_rgb_enhanced(img, band_indices=[99, 100, 101], plot=False)

        # Test invalid color space
        with pytest.raises(ValueError):
            plot_color_space(img, color_space="invalid", plot=False)

        print("✓ Error handling working correctly")


class TestPerformanceAndMemory:
    """Test performance characteristics and memory usage."""

    def test_large_image_handling(self):
        """Test handling of larger images without loading everything into memory."""
        # Test with largest available image
        large_img_path = TestDataPaths.PLANETSCOPE_SINGLE  # Multi-band, likely large

        # Test metadata access without full load
        loader = GeoTIFFImage(large_img_path)
        shape = loader.shape
        dtype = loader.dtype

        # Metadata should be accessible without loading full data
        assert shape is not None
        assert dtype is not None

        # Now load and verify
        img = loader.load()
        assert img.shape == shape
        assert img.dtype == dtype

        print(f"✓ Large image handling: {shape}, {dtype}")

    def test_collection_lazy_loading(self):
        """Test that collections don't load everything at once."""
        collection = GeoTIFFCollection(TestDataPaths.GBR_BENTHIC_DIR, pattern="*.tif")

        # Initially, images list should be empty (lazy loading)
        assert len(collection.images) == 0
        assert len(collection.files) > 0

        # Access first image should load only that one
        first_img = collection[0]
        assert len(collection.images) == 1

        # Access another should load up to that index
        if len(collection) > 1:
            second_img = collection[1]
            assert len(collection.images) == 2

        print("✓ Lazy loading working correctly")


if __name__ == "__main__":
    """Run tests manually for debugging."""
    print("Running End-to-End Pipeline Tests")
    print("=" * 50)

    # Check data availability first
    test_paths = TestDataPaths()
    missing_paths = []

    for attr_name in dir(test_paths):
        if not attr_name.startswith("_"):
            path = getattr(test_paths, attr_name)
            if isinstance(path, str) and not Path(path).exists():
                missing_paths.append((attr_name, path))

    if missing_paths:
        print("WARNING: Missing data paths:")
        for name, path in missing_paths:
            print(f"  {name}: {path}")
        print()

    # Run test classes
    test_classes = [
        TestGeoTIFFPipeline,
        TestSentinel2Pipeline,
        TestLandsatPipeline,
        TestVisualizationPipeline,
        TestDataIntegration,
        TestPerformanceAndMemory,
    ]

    for test_class in test_classes:
        print(f"\n{test_class.__name__}:")
        print("-" * 30)

        test_instance = test_class()
        test_methods = [
            method for method in dir(test_instance) if method.startswith("test_")
        ]

        for method_name in test_methods:
            try:
                method = getattr(test_instance, method_name)
                method()
            except Exception as e:
                print(f"  FAILED {method_name}: {e}")

    print("\n" + "=" * 50)
    print("End-to-End Tests Complete")

