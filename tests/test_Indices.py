import numpy as np
from ShallowLearn.core.band_mapping import band_mapping
import ShallowLearn.features.indices as band_indices
import pytest

# Fixture for a sample multi-band image (e.g., 3x3 with 13 channels)
@pytest.fixture
def sample_indices_image():
    # Create a simple image where values increase with band number
    img = np.zeros((3, 3, 13), dtype=np.float32)
    for i in range(13):
        # Assign distinct values, avoiding zero except where intended
        img[:, :, i] = (i + 1) * 100.0 + np.random.rand(3, 3) * 10
    img[0, 0, 0] = 0 # Force a zero
    img[0, 1, 1] = np.nan # Force a NaN
    return img

# --- Test Functions ---

def test_get_band_numbers():
    bands = ['B04', 'B03', 'B02']
    expected = [band_mapping[b]['index'] for b in bands]
    assert band_indices.get_band_numbers(bands, band_mapping) == expected

def test_validate_band_shape_valid(sample_indices_image):
    # Test should pass
    band_indices.validate_band_shape(sample_indices_image, [0, 1, 12])

def test_validate_band_shape_invalid_index(sample_indices_image):
    with pytest.raises(ValueError):
        band_indices.validate_band_shape(sample_indices_image, [0, 1, 13]) # 13 is out of bounds

def test_validate_band_shape_invalid_dims():
    img_2d = np.zeros((3,3))
    with pytest.raises(ValueError):
        band_indices.validate_band_shape(img_2d, [0, 1])


@pytest.mark.parametrize("index_func, bands", [
    (band_indices.ci, ['B04', 'B05', 'B06', 'B07']),
    (band_indices.oci, ['B04', 'B08', 'B12']),
    (band_indices.cl_oci, ['B02','B03','B04']),
    (band_indices.ssi, ['B04', 'B05', 'B08']),
    (band_indices.ti, ['B04', 'B08']),
    (band_indices.wqi, ['B03', 'B04', 'B05', 'B06']),
    (band_indices.ndci, ['B03', 'B05']),
    (band_indices.cloud_index, ['B08', 'B11']),
    (band_indices.bgr, ['B02', 'B03']),
    (band_indices.calculate_water_surface_index, ['B03', 'B04', 'B08', 'B11', 'B12']),
    (band_indices.calculate_pseudo_subsurface_depth, ['B02', 'B03']),
])
def test_index_calculation(index_func, bands, sample_indices_image):
    result = index_func(sample_indices_image, bands=bands)
    assert result.shape == (3, 3) # Expect 2D output
    assert result.dtype == np.float32 or result.dtype == np.float64
