"""Pytest configuration and shared fixtures."""

from pathlib import Path

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "requires_data: mark test as requiring external data files"
    )


@pytest.fixture(scope="session")
def test_data_paths():
    """Fixture providing test data paths with availability checks."""
    
    class TestDataPaths:
        """Container for test data paths with existence checking."""
        
        # Define all paths
        GBR_BENTHIC_SINGLE = "/mnt/sda_mount/Clipped/GBR_2017/benthic_2/0_benthic_2.tif"
        GBR_GEOMORPHIC_SINGLE = "/mnt/sda_mount/Clipped/GBR_2017/geomorphic/0_geomorphic.tif"
        GBR_BENTHIC_DIR = "/mnt/sda_mount/Clipped/GBR_2017/benthic_2"
        PLANETSCOPE_DIR = "/mnt/sda_mount/Clipped/Planetscope/20230331_233843_22_24a8_3B_udm2"
        PLANETSCOPE_SINGLE = "/mnt/sda_mount/Clipped/Planetscope/20230331_233843_22_24a8_3B_udm2/1_20230331_233843_22_24a8_3B_udm2.tif"
        SENTINEL2_ZIP_N0500 = "/mnt/sda_mount/L1C_Full/S2A_MSIL1C_20151124T003752_N0500_R059_T55LCD_20231009T132839.zip"
        SENTINEL2_ZIP_N0400 = "/mnt/sda_mount/L1C_Full/S2A_MSIL1C_20221008T003711_N0400_R059_T55LCD_20221008T014306.zip"
        SENTINEL2_SAFE_DIR = "/mnt/sda_mount/Clipped/L1C/S2A_MSIL1C_20160323T003752_N0201_R059_T55LCD_20160323T003830.SAFE"
        
        @classmethod
        def exists(cls, attr_name):
            """Check if a data path exists."""
            path = getattr(cls, attr_name, None)
            if path is None:
                return False
            return Path(path).exists()
        
        @classmethod
        def skip_if_missing(cls, *attr_names):
            """Skip test if any of the specified data paths don't exist."""
            missing = []
            for attr_name in attr_names:
                if not cls.exists(attr_name):
                    path = getattr(cls, attr_name, "unknown")
                    missing.append(f"{attr_name}: {path}")
            
            if missing:
                pytest.skip(f"Test data not available: {', '.join(missing)}")
    
    return TestDataPaths


@pytest.fixture(autouse=True)
def check_data_availability(request, test_data_paths):
    """Automatically skip tests marked with requires_data if data is missing."""
    marker = request.node.get_closest_marker("requires_data")
    if marker:
        # Get the required data paths from the marker
        required_paths = marker.args
        if required_paths:
            test_data_paths.skip_if_missing(*required_paths)