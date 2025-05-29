"""
Comprehensive tests for the PyTorch satellite data loader.
Tests band alignment, resizing, and data loading functionality.
"""

import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List
import warnings

from ShallowLearn.torch_dataloaders import SatelliteDataset, create_satellite_dataloader, collate_satellite_batch

# Mock satellite image classes for testing
class MockSatelliteImage:
    """Mock satellite image for testing purposes."""
    
    def __init__(self, satellite_type: str, shape: tuple = (100, 100), bands: List[str] = None):
        self.satellite_type = satellite_type
        self.shape = shape
        
        if satellite_type == 'sentinel2':
            self.band_order = {
                "B01": 0, "B02": 1, "B03": 2, "B04": 3, "B05": 4,
                "B06": 5, "B07": 6, "B08": 7, "B8A": 8, "B09": 9,
                "B10": 10, "B11": 11, "B12": 12
            }
            self.available_bands = bands if bands else list(self.band_order.keys())
        else:  # landsat
            self.band_order = {
                "B1": 0, "B2": 1, "B3": 2, "B4": 3, "B5": 4,
                "B6": 5, "B7": 6, "B8": 7, "B9": 8, "B10": 9, "B11": 10
            }
            self.available_bands = bands if bands else ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
        
        # Create fake image data
        num_bands = len(self.available_bands)
        self.image = np.random.rand(*shape, num_bands).astype(np.float32)
    
    def has_band(self, band_name: str) -> bool:
        return band_name in self.available_bands
    
    def get_band_data(self, band_name: str) -> np.ndarray:
        if self.has_band(band_name):
            band_idx = self.band_order[band_name]
            if band_idx < self.image.shape[2]:
                return self.image[:, :, band_idx]
        return None


def create_fake_vrt_files(temp_dir: Path, satellite_type: str, count: int = 3) -> List[str]:
    """Create fake VRT files for testing."""
    files = []
    
    for i in range(count):
        if satellite_type == 'sentinel2':
            filename = f"S2A_MSIL2A_2021010{i+1}T100000_N0400_R073_T32TPS_{i+1}.vrt"
        else:  # landsat
            filename = f"LC08_L2SP_180033_2021010{i+1}_20210110_02_T1_{i}.vrt"
        
        file_path = temp_dir / filename
        
        # Create a minimal VRT file structure
        vrt_content = f"""<VRTDataset rasterXSize="100" rasterYSize="100">
  <SRS>EPSG:4326</SRS>
  <VRTRasterBand dataType="Float32" band="1">
    <Description>B01</Description>
  </VRTRasterBand>
</VRTDataset>"""
        
        file_path.write_text(vrt_content)
        files.append(str(file_path))
    
    return files


def test_band_mapping():
    """Test band nomenclature unification."""
    print("Testing band mapping...")
    
    # Test Sentinel-2 to Landsat mapping
    dataset = SatelliteDataset.__new__(SatelliteDataset)  # Create without __init__
    dataset.BAND_MAPPING = SatelliteDataset.BAND_MAPPING
    
    # Test some key mappings
    assert dataset.BAND_MAPPING['B02'] == 'B2', "Blue band mapping failed"
    assert dataset.BAND_MAPPING['B03'] == 'B3', "Green band mapping failed"
    assert dataset.BAND_MAPPING['B04'] == 'B4', "Red band mapping failed"
    assert dataset.BAND_MAPPING['B08'] == 'B5', "NIR band mapping failed"
    assert dataset.BAND_MAPPING['B11'] == 'B6', "SWIR1 band mapping failed"
    assert dataset.BAND_MAPPING['B12'] == 'B7', "SWIR2 band mapping failed"
    
    print("✓ Band mapping test passed")


def test_dataset_initialization():
    """Test dataset initialization with different parameters."""
    print("Testing dataset initialization...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create fake directories and files
        sen2_dir = temp_path / "sen2_crop"
        lsat_dir = temp_path / "cropped_lsat" 
        sen2_dir.mkdir()
        lsat_dir.mkdir()
        
        sen2_files = create_fake_vrt_files(sen2_dir, 'sentinel2', 2)
        lsat_files = create_fake_vrt_files(lsat_dir, 'landsat', 2)
        
        # Mock the create_satellite_image function
        original_create_image = None
        try:
            import ShallowLearn.torch_dataloaders as tdl
            original_create_image = tdl.create_satellite_image
            
            def mock_create_image(file_path):
                if 'S2A' in file_path:
                    return MockSatelliteImage('sentinel2')
                else:
                    return MockSatelliteImage('landsat')
            
            tdl.create_satellite_image = mock_create_image
            
            # Test initialization with directories
            dataset = SatelliteDataset(
                sentinel_dir=str(sen2_dir),
                landsat_dir=str(lsat_dir),
                target_size=(256, 256),
                filter_invalid=False  # Disable filtering for test
            )
            
            assert len(dataset) == 4, f"Expected 4 files, got {len(dataset)}"
            assert dataset.target_size == (256, 256), "Target size not set correctly"
            
            # Test initialization with file paths
            dataset2 = SatelliteDataset(
                sentinel_paths=sen2_files,
                landsat_paths=lsat_files,
                target_size=(128, 128),
                bands=['B2', 'B3', 'B4'],
                filter_invalid=False  # Disable filtering for test
            )
            
            assert len(dataset2) == 4, f"Expected 4 files, got {len(dataset2)}"
            assert dataset2.bands == ['B2', 'B3', 'B4'], "Custom bands not set correctly"
            
        finally:
            if original_create_image:
                tdl.create_satellite_image = original_create_image
    
    print("✓ Dataset initialization test passed")


def test_image_resizing():
    """Test image resizing functionality."""
    print("Testing image resizing...")
    
    dataset = SatelliteDataset.__new__(SatelliteDataset)
    dataset.target_size = (64, 64)
    
    # Test single band resizing
    original_image = np.random.rand(100, 100).astype(np.float32)
    resized = dataset._resize_image(original_image)
    
    assert resized.shape == (64, 64), f"Single band resize failed: {resized.shape}"
    
    # Test multi-band resizing
    multi_band_image = np.random.rand(100, 100, 5).astype(np.float32)
    resized_multi = dataset._resize_image(multi_band_image)
    
    assert resized_multi.shape == (64, 64, 5), f"Multi-band resize failed: {resized_multi.shape}"
    
    # Test no resizing needed
    correct_size = np.random.rand(64, 64, 3).astype(np.float32)
    no_resize = dataset._resize_image(correct_size)
    
    assert np.array_equal(correct_size, no_resize), "No-resize case failed"
    
    print("✓ Image resizing test passed")


def test_band_extraction():
    """Test band extraction and alignment."""
    print("Testing band extraction...")
    
    dataset = SatelliteDataset.__new__(SatelliteDataset)
    dataset.bands = ['B2', 'B3', 'B4', 'B5']
    dataset.BAND_MAPPING = SatelliteDataset.BAND_MAPPING
    
    # Test Sentinel-2 band extraction
    s2_img = MockSatelliteImage('sentinel2', shape=(50, 50))
    s2_bands = dataset._extract_bands(s2_img, 'sentinel2')
    
    assert s2_bands.shape == (50, 50, 4), f"S2 band extraction failed: {s2_bands.shape}"
    
    # Test Landsat band extraction
    lsat_img = MockSatelliteImage('landsat', shape=(50, 50))
    lsat_bands = dataset._extract_bands(lsat_img, 'landsat')
    
    assert lsat_bands.shape == (50, 50, 4), f"Landsat band extraction failed: {lsat_bands.shape}"
    
    print("✓ Band extraction test passed")


def test_dataset_getitem():
    """Test dataset __getitem__ functionality."""
    print("Testing dataset __getitem__...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create fake files
        sen2_dir = temp_path / "sen2"
        sen2_dir.mkdir()
        sen2_files = create_fake_vrt_files(sen2_dir, 'sentinel2', 1)
        
        # Mock the create_satellite_image function
        try:
            import ShallowLearn.torch_dataloaders as tdl
            original_create_image = tdl.create_satellite_image
            
            def mock_create_image(file_path):
                return MockSatelliteImage('sentinel2', shape=(80, 80))
            
            tdl.create_satellite_image = mock_create_image
            
            dataset = SatelliteDataset(
                sentinel_paths=sen2_files,
                target_size=(64, 64),
                bands=['B2', 'B3', 'B4'],
                filter_invalid=False
            )
            
            # Test getting an item
            item = dataset[0]
            
            # Check return structure
            assert 'image' in item, "Missing 'image' key"
            assert 'satellite_type' in item, "Missing 'satellite_type' key"
            assert 'file_path' in item, "Missing 'file_path' key"
            assert 'bands' in item, "Missing 'bands' key"
            
            # Check tensor properties
            image_tensor = item['image']
            assert isinstance(image_tensor, torch.Tensor), "Image should be a tensor"
            assert image_tensor.shape == (3, 64, 64), f"Wrong tensor shape: {image_tensor.shape}"
            assert item['satellite_type'] == 'sentinel2', "Wrong satellite type"
            assert item['bands'] == ['B2', 'B3', 'B4'], "Wrong bands"
            
        finally:
            if original_create_image:
                tdl.create_satellite_image = original_create_image
    
    print("✓ Dataset __getitem__ test passed")


def test_dataloader_creation():
    """Test DataLoader creation and functionality."""
    print("Testing DataLoader creation...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create fake directories
        sen2_dir = temp_path / "sen2_crop"
        lsat_dir = temp_path / "cropped_lsat"
        sen2_dir.mkdir()
        lsat_dir.mkdir()
        
        create_fake_vrt_files(sen2_dir, 'sentinel2', 2)
        create_fake_vrt_files(lsat_dir, 'landsat', 2)
        
        # Mock the create_satellite_image function
        try:
            import ShallowLearn.torch_dataloaders as tdl
            original_create_image = tdl.create_satellite_image
            
            def mock_create_image(file_path):
                if 'S2A' in file_path:
                    return MockSatelliteImage('sentinel2', shape=(100, 100))
                else:
                    return MockSatelliteImage('landsat', shape=(120, 120))
            
            tdl.create_satellite_image = mock_create_image
            
            # Create DataLoader
            dataloader = create_satellite_dataloader(
                sentinel_dir=str(sen2_dir),
                landsat_dir=str(lsat_dir),
                batch_size=2,
                target_size=(64, 64),
                bands=['B2', 'B3', 'B4'],
                num_workers=0,  # Avoid multiprocessing issues in tests
                filter_invalid=False
            )
            
            # Test iteration
            for batch in dataloader:
                assert 'images' in batch, "Missing 'images' in batch"
                assert 'satellite_types' in batch, "Missing 'satellite_types' in batch"
                assert 'file_paths' in batch, "Missing 'file_paths' in batch"
                assert 'bands' in batch, "Missing 'bands' in batch"
                
                images = batch['images']
                assert images.shape == (2, 3, 64, 64), f"Wrong batch shape: {images.shape}"
                assert len(batch['satellite_types']) == 2, "Wrong number of satellite types"
                assert len(batch['file_paths']) == 2, "Wrong number of file paths"
                
                break  # Only test first batch
            
        finally:
            if original_create_image:
                tdl.create_satellite_image = original_create_image
    
    print("✓ DataLoader creation test passed")


def test_band_statistics():
    """Test band statistics calculation."""
    print("Testing band statistics...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create fake files
        sen2_dir = temp_path / "sen2"
        sen2_dir.mkdir()
        sen2_files = create_fake_vrt_files(sen2_dir, 'sentinel2', 2)
        
        # Mock the create_satellite_image function
        try:
            import ShallowLearn.torch_dataloaders as tdl
            original_create_image = tdl.create_satellite_image
            
            def mock_create_image(file_path):
                # Create predictable data for statistics
                img = MockSatelliteImage('sentinel2', shape=(50, 50))
                img.image = np.ones((50, 50, 13)) * 0.5  # All values = 0.5
                return img
            
            tdl.create_satellite_image = mock_create_image
            
            dataset = SatelliteDataset(
                sentinel_paths=sen2_files,
                target_size=(50, 50),
                bands=['B2', 'B3'],
                filter_invalid=False
            )
            
            stats = dataset.get_band_statistics()
            
            assert 'B2' in stats, "Missing B2 statistics"
            assert 'B3' in stats, "Missing B3 statistics"
            
            # Since all values are 0.5, mean should be ~0.5 and std should be ~0
            assert abs(stats['B2']['mean'] - 0.5) < 0.1, f"Wrong B2 mean: {stats['B2']['mean']}"
            assert stats['B2']['std'] < 0.1, f"Wrong B2 std: {stats['B2']['std']}"
            
        finally:
            if original_create_image:
                tdl.create_satellite_image = original_create_image
    
    print("✓ Band statistics test passed")


def test_collate_function():
    """Test custom collate function."""
    print("Testing collate function...")
    
    # Create fake batch data
    batch = [
        {
            'image': torch.randn(3, 64, 64),
            'satellite_type': 'sentinel2',
            'file_path': '/path/to/s2.vrt',
            'bands': ['B2', 'B3', 'B4']
        },
        {
            'image': torch.randn(3, 64, 64),
            'satellite_type': 'landsat',
            'file_path': '/path/to/lsat.vrt',
            'bands': ['B2', 'B3', 'B4']
        }
    ]
    
    collated = collate_satellite_batch(batch)
    
    assert 'images' in collated, "Missing 'images' key"
    assert 'satellite_types' in collated, "Missing 'satellite_types' key"
    assert 'file_paths' in collated, "Missing 'file_paths' key"
    assert 'bands' in collated, "Missing 'bands' key"
    
    assert collated['images'].shape == (2, 3, 64, 64), f"Wrong images shape: {collated['images'].shape}"
    assert len(collated['satellite_types']) == 2, "Wrong satellite_types length"
    assert len(collated['file_paths']) == 2, "Wrong file_paths length"
    assert collated['bands'] == ['B2', 'B3', 'B4'], "Wrong bands"
    
    print("✓ Collate function test passed")


def run_integration_test():
    """Run a full integration test simulating real usage."""
    print("\nRunning integration test...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create directory structure like the real data
        sen2_dir = temp_path / "sen2_crop"
        lsat_dir = temp_path / "cropped_lsat"
        sen2_dir.mkdir()
        lsat_dir.mkdir()
        
        # Create multiple files
        create_fake_vrt_files(sen2_dir, 'sentinel2', 3)
        create_fake_vrt_files(lsat_dir, 'landsat', 3)
        
        try:
            import ShallowLearn.torch_dataloaders as tdl
            original_create_image = tdl.create_satellite_image
            
            def mock_create_image(file_path):
                if 'S2A' in file_path:
                    return MockSatelliteImage('sentinel2', shape=(200, 200))
                else:
                    return MockSatelliteImage('landsat', shape=(180, 180))
            
            tdl.create_satellite_image = mock_create_image
            
            # Create dataset like in your example
            dataloader = create_satellite_dataloader(
                sentinel_dir=str(sen2_dir),
                landsat_dir=str(lsat_dir),
                batch_size=4,
                shuffle=True,
                target_size=(512, 512),
                bands=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
                num_workers=0,
                filter_invalid=False
            )
            
            print(f"Created dataloader with {len(dataloader.dataset)} items")
            
            # Test full iteration
            total_samples = 0
            for batch_idx, batch in enumerate(dataloader):
                images = batch['images']
                total_samples += images.shape[0]
                
                print(f"Batch {batch_idx}: {images.shape}, types: {set(batch['satellite_types'])}")
                
                # Verify tensor properties
                assert images.dtype == torch.float32, "Wrong tensor dtype"
                assert images.shape[1] == 7, f"Wrong number of bands: {images.shape[1]}"
                assert images.shape[2:] == (512, 512), f"Wrong spatial size: {images.shape[2:]}"
                
                # Check for NaN handling
                if torch.isnan(images).any():
                    print(f"  Warning: Found NaN values in batch {batch_idx}")
            
            print(f"Processed {total_samples} total samples")
            assert total_samples == 6, f"Expected 6 samples, got {total_samples}"
            
        finally:
            if original_create_image:
                tdl.create_satellite_image = original_create_image
    
    print("✓ Integration test passed")


if __name__ == "__main__":
    print("Testing PyTorch Satellite DataLoader")
    print("=" * 50)
    
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")
    
    try:
        # Run individual tests
        test_band_mapping()
        test_dataset_initialization()
        test_image_resizing()
        test_band_extraction()
        test_dataset_getitem()
        test_dataloader_creation()
        test_band_statistics()
        test_collate_function()
        
        # Run integration test
        run_integration_test()
        
        print("\n" + "=" * 50)
        print("🎉 ALL TESTS PASSED!")
        print("The PyTorch satellite dataloader is working correctly.")
        
        # Example usage
        print("\n" + "=" * 50)
        print("Example Usage:")
        print("""
# Create a dataloader for your data directories
dataloader = create_satellite_dataloader(
    sentinel_dir="../data/sen2_crop/",
    landsat_dir="../data/cropped_lsat/", 
    batch_size=8,
    target_size=(512, 512),
    bands=['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
    shuffle=True
)

# Iterate through batches
for batch in dataloader:
    images = batch['images']          # Shape: (batch_size, 7, 512, 512) 
    satellite_types = batch['satellite_types']  # ['sentinel2', 'landsat', ...]
    file_paths = batch['file_paths']  # ['/path/to/file1.vrt', ...]
    bands = batch['bands']            # ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
    
    # Your training/inference code here
    pass
        """)
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()