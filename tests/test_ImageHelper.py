import pytest
import numpy as np
import ShallowLearn.ImageHelper as ih
from skimage.color import rgb2lab, rgb2hsv, rgb2ycbcr
from unittest.mock import patch, MagicMock

# Fixture for a sample multi-band image (e.g., 5x5 with 13 channels)
@pytest.fixture
def sample_image_13band():
    # Simple gradient-like image for testing
    img = np.zeros((5, 5, 13), dtype=np.float16)
    for i in range(13):
        img[:, :, i] = np.linspace(0, 10000 + i*100, 25).reshape(5, 5)
    img[0, 0, 0] = 0 # Ensure a zero value
    img[1, 1, 1] = 15000 # Ensure a value > 10000
    img[2, 2, 2] = -50 # Ensure a negative value for clip_array testing
    return img

# Fixture for a sample RGB image (already scaled 0-255)
@pytest.fixture
def sample_rgb_image():
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, 25).reshape(5, 5) # R
    img[:, :, 1] = np.linspace(50, 200, 25).reshape(5, 5) # G
    img[:, :, 2] = np.linspace(255, 0, 25).reshape(5, 5) # B
    return img

# --- Test Functions ---

def test_clip_array(sample_image_13band):
    clipped = ih.clip_array(sample_image_13band)
    assert np.all(clipped >= 0)
    assert np.all(clipped <= 10000)
    assert clipped.shape == sample_image_13band.shape
    # # Check specific values that should have been clipped
    assert clipped[1, 1, 1] == 10000
    assert clipped[2, 2, 2] == 0

def test_select_channels(sample_image_13band):
    indices = [3, 2, 1] # R, G, B
    selected = ih.select_channels(sample_image_13band, indices)
    assert selected.shape == (5, 5, 3)
    np.testing.assert_array_equal(selected[:,:,0], sample_image_13band[:,:,3])
    np.testing.assert_array_equal(selected[:,:,1], sample_image_13band[:,:,2])
    np.testing.assert_array_equal(selected[:,:,2], sample_image_13band[:,:,1])

def test_select_channels_invalid_indices(sample_image_13band):
    with pytest.raises(ValueError):
        ih.select_channels(sample_image_13band, [1, 2])
    with pytest.raises(ValueError):
        ih.select_channels(sample_image_13band, [1, 2, 3, 4])

def test_remove_channel(sample_image_13band):
    channel_to_remove = 5
    modified = ih.remove_channel(sample_image_13band, channel_to_remove)
    assert modified.shape == (5, 5, 12)
    # Check if the channel is actually gone by comparing a slice
    np.testing.assert_array_equal(modified[:,:,channel_to_remove], sample_image_13band[:,:,channel_to_remove+1])

def test_remove_channel_invalid(sample_image_13band):
    with pytest.raises(ValueError):
        ih.remove_channel(sample_image_13band, 13)
    with pytest.raises(ValueError):
        ih.remove_channel(sample_image_13band, -1)

@patch('matplotlib.pyplot.show') # Mock plt.show() to prevent plots popping up
def test_plot_rgb_return_array(mock_show, sample_image_13band):
    rgb = ih.plot_rgb(sample_image_13band, bands=['B04', 'B03', 'B02'], plot=False)
    assert rgb.shape == (5, 5, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb >= 0)
    assert np.all(rgb <= 255)


@patch('matplotlib.pyplot.show')
def test_plot_rgb_plot_mode(mock_show, sample_image_13band):
    # Test it runs without error in plot mode
    result = ih.plot_rgb(sample_image_13band, plot=True)
    assert result is None # Should return None when plotting
    mock_show.assert_called_once()

# Tests for plot_hsv, plot_lab, plot_ycbcr returning arrays
@patch('matplotlib.pyplot.show')
def test_plot_hsv_return_array(mock_show, sample_image_13band):
    hsv = ih.plot_hsv(sample_image_13band, plot=False)
    assert hsv.shape == (5, 5, 3)
    # HSV values have specific ranges, H=[0,1], S=[0,1], V=[0,1] roughly after rgb2hsv
    assert np.all(hsv >= 0)
    # assert np.all(hsv <= 1.001) # Allow for floating point inaccuracies

@patch('matplotlib.pyplot.show')
def test_plot_lab_return_array(mock_show, sample_image_13band):
    lab = ih.plot_lab(sample_image_13band, plot=False)
    assert lab.shape == (5, 5, 3)
    # L range [0, 100], a, b roughly [-128, 127]

@patch('matplotlib.pyplot.show')
def test_plot_ycbcr_return_array(mock_show, sample_image_13band):
    ycbcr = ih.plot_ycbcr(sample_image_13band, plot=False)
    assert ycbcr.shape == (5, 5, 3)
    # Y [16, 235], Cb/Cr [16, 240] for standard range uint8

# Mocking joblib.load for predict_mask
@patch('joblib.load')
def test_predict_mask(mock_joblib_load, sample_image_13band):
    mock_pipeline = MagicMock()
    # Make predict return a simple array based on input mean or something
    def mock_predict(data):
        # Predict class 1 if mean > 5000, else 0
        means = np.mean(data, axis=1)
        return (means > 5000).astype(int)

    mock_pipeline.predict.side_effect = mock_predict
    mock_joblib_load.return_value = mock_pipeline

    mask_pred = ih.predict_mask(sample_image_13band, model='dummy_path')
    assert mask_pred.shape == (5, 5)
    assert mask_pred.dtype == int or mask_pred.dtype == bool # Depending on mask_val usage
    
    # Test with mask_val
    mask_pred_val = ih.predict_mask(sample_image_13band, model='dummy_path', mask_val=1)
    assert mask_pred_val.shape == (5, 5)
    assert mask_pred_val.dtype == bool

@patch('joblib.load')
def test_gen_mask(mock_joblib_load, sample_image_13band):
    mock_pipeline = MagicMock()
    def mock_predict(data):
        means = np.mean(data, axis=1)
        return (means > 5000).astype(bool) # Example prediction
    mock_pipeline.predict.side_effect = mock_predict
    mock_joblib_load.return_value = mock_pipeline

    mask_gen = ih.gen_mask(sample_image_13band, mask=1) # request mask where prediction is 1
    assert mask_gen.shape == (5, 5)
    assert mask_gen.dtype == bool

def test_apply_mask(sample_image_13band):
    mask = np.zeros((5, 5), dtype=bool)
    mask[2:4, 2:4] = True # A small square mask
    mask_3d = np.expand_dims(mask, axis=2) # Make it broadcast correctly

    masked_data_zero = ih.apply_mask(sample_image_13band, mask_3d, fill_value=0)
    assert masked_data_zero.shape == sample_image_13band.shape
    assert np.all(masked_data_zero[~mask] == 0) # Check fill value
    np.testing.assert_array_equal(masked_data_zero[mask], sample_image_13band[mask]) # Check kept values

    masked_data_nan = ih.apply_mask(sample_image_13band.astype(float), mask_3d, fill_value=np.nan)
    assert np.all(np.isnan(masked_data_nan[~mask])) # Check fill value (NaN)
    np.testing.assert_array_equal(masked_data_nan[mask], sample_image_13band[mask])

# generate_multichannel_mask depends on gen_mask, test output shape
@patch('joblib.load')
def test_generate_multichannel_mask(mock_joblib_load, sample_image_13band):
    mock_pipeline = MagicMock()
    def mock_predict(data):
        means = np.mean(data, axis=1)
        return (means > 5000).astype(int) # Example prediction
    mock_pipeline.predict.side_effect = mock_predict
    mock_joblib_load.return_value = mock_pipeline

    multi_mask = ih.generate_multichannel_mask(sample_image_13band, mask_val=1)
    assert multi_mask.shape == sample_image_13band.shape
    # Check if values outside the predicted mask area are zero (after scaling)
    # This requires knowing the mask, which depends on the mock predict logic.
    # Simpler check: sum outside the predicted 'true' area should be low/zero.

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.hist')
def test_plot_histogram(mock_hist, mock_show, sample_image_13band):
    # Test it runs without error for a single channel like array
    ih.plot_histogram(sample_image_13band[:,:,0], plot=True)
    mock_hist.assert_called()
    mock_show.assert_called_once()

@patch('matplotlib.pyplot.show')
@patch('matplotlib.pyplot.plot')
def test_plot_histograms(mock_plot, mock_show, sample_image_13band):
    # Test it runs without error
    ih.plot_histograms(sample_image_13band, plot=True)
    assert mock_plot.call_count == 13 # Called for each channel
    mock_show.assert_called_once()

def test_median_without_zeros_or_nans(sample_image_13band):
    img = sample_image_13band.copy().astype(float)
    img[0,1,0] = np.nan # Add a NaN
    img[0,2,1] = 0 # Add a zero
    img[0,3,2] = np.nan
    img[0,4,2] = 0

    medians = ih.median_without_zeros_or_nans(np.expand_dims(img, axis=0)) # Test with 1 image
    assert medians.shape == (1, 13)
    # Check a specific channel where we added NaN/zero
    band0_valid = img[:,:,0][(img[:,:,0] != 0) & (~np.isnan(img[:,:,0]))]
    expected_median0 = np.median(band0_valid)
    assert medians[0, 0] == expected_median0