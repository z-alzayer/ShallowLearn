"""
Tests for time series plotting utilities.
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ShallowLearn.visualization.time_series_plots import (
    animate_images_and_timeseries,
    extract_point_spectra,
    plot_image_with_marker,
    plot_spectral_timeseries,
)


@pytest.fixture
def sample_images():
    """Create sample image sequence."""
    return [np.random.rand(100, 100, 3) for _ in range(5)]


@pytest.fixture
def sample_timeseries():
    """Create sample timeseries data."""
    dates = pd.date_range('2023-01-01', periods=5, freq='D')
    data = {
        'B4': np.random.rand(5),
        'B3': np.random.rand(5),
        'B2': np.random.rand(5)
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_image():
    """Create a single sample image."""
    return np.random.rand(100, 100, 3)


@pytest.fixture
def sample_spectra():
    """Create sample spectral data."""
    return np.random.rand(10, 4)  # 10 time steps, 4 bands


@pytest.fixture
def sample_dates():
    """Create sample date sequence."""
    return pd.date_range('2023-01-01', periods=10, freq='D')


class TestAnimateImagesAndTimeseries:
    """Test the animation function."""
    
    def test_animate_basic(self, sample_images, sample_timeseries):
        """Test basic animation creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_animation.gif"
            
            result_path = animate_images_and_timeseries(
                sample_images, 
                sample_timeseries,
                output_path=str(output_path)
            )
            
            assert Path(result_path).exists()
            assert result_path == str(output_path)
    
    def test_animate_custom_parameters(self, sample_images, sample_timeseries):
        """Test animation with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "custom_animation.gif"
            
            result_path = animate_images_and_timeseries(
                sample_images,
                sample_timeseries,
                point_coords=(25, 75),
                fps=1,
                title_prefix="Test Scene",
                bands=['B4', 'B3'],
                band_colors=['blue', 'green'],
                output_path=str(output_path)
            )
            
            assert Path(result_path).exists()
    
    def test_animate_mismatched_lengths(self, sample_images, sample_timeseries):
        """Test error handling for mismatched input lengths."""
        # Remove one image to create mismatch
        short_images = sample_images[:-1]
        
        with pytest.raises(ValueError, match="Number of images .* must match timeseries length"):
            animate_images_and_timeseries(short_images, sample_timeseries)
    
    def test_animate_mismatched_bands_colors(self, sample_images, sample_timeseries):
        """Test error handling for mismatched bands and colors."""
        with pytest.raises(ValueError, match="Number of bands must match number of colors"):
            animate_images_and_timeseries(
                sample_images, 
                sample_timeseries,
                bands=['B4', 'B3'],
                band_colors=['red']  # Only one color for two bands
            )


class TestPlotImageWithMarker:
    """Test the image marker plotting function."""
    
    def test_plot_basic(self, sample_image):
        """Test basic image plotting with marker."""
        fig = plot_image_with_marker(sample_image, (50, 50))
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)
    
    def test_plot_custom_parameters(self, sample_image):
        """Test plotting with custom parameters."""
        fig = plot_image_with_marker(
            sample_image,
            point_coords=(30, 70),
            arrow_offset=(10, -30),
            marker_color='blue',
            marker_size=12,
            title="Custom Title"
        )
        
        assert isinstance(fig, plt.Figure)
        assert fig.axes[0].get_title() == "Custom Title"
        plt.close(fig)
    
    def test_plot_with_save(self, sample_image):
        """Test saving the plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_plot.png"
            
            fig = plot_image_with_marker(
                sample_image, 
                (50, 50),
                save_path=str(save_path)
            )
            
            assert save_path.exists()
            plt.close(fig)


class TestPlotSpectralTimeseries:
    """Test the spectral time series plotting function."""
    
    def test_plot_basic(self, sample_spectra, sample_dates):
        """Test basic spectral plotting."""
        fig = plot_spectral_timeseries(sample_spectra, sample_dates)
        
        assert isinstance(fig, plt.Figure)
        assert len(fig.axes) == 1
        plt.close(fig)
    
    def test_plot_with_labels(self, sample_spectra, sample_dates):
        """Test plotting with custom band labels."""
        band_labels = {0: 'Red', 1: 'Green', 2: 'Blue', 3: 'NIR'}
        
        fig = plot_spectral_timeseries(
            sample_spectra, 
            sample_dates,
            band_labels=band_labels,
            title="Custom Spectral Plot"
        )
        
        assert isinstance(fig, plt.Figure)
        assert fig.axes[0].get_title() == "Custom Spectral Plot"
        plt.close(fig)
    
    def test_plot_string_dates(self, sample_spectra):
        """Test plotting with string dates."""
        string_dates = ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', 
                       '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08',
                       '2023-01-09', '2023-01-10']
        
        fig = plot_spectral_timeseries(sample_spectra, string_dates)
        
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
    
    def test_plot_with_save(self, sample_spectra, sample_dates):
        """Test saving the spectral plot."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "spectral_plot.png"
            
            fig = plot_spectral_timeseries(
                sample_spectra, 
                sample_dates,
                save_path=str(save_path)
            )
            
            assert save_path.exists()
            plt.close(fig)
    
    def test_plot_invalid_shape(self, sample_dates):
        """Test error handling for invalid spectra shape."""
        invalid_spectra = np.random.rand(10)  # 1D array
        
        with pytest.raises(ValueError, match="Spectra must be 2D array"):
            plot_spectral_timeseries(invalid_spectra, sample_dates)
    
    def test_plot_mismatched_dates(self, sample_spectra):
        """Test error handling for mismatched dates length."""
        short_dates = pd.date_range('2023-01-01', periods=5, freq='D')  # Only 5 dates for 10 time steps
        
        with pytest.raises(ValueError, match="Dates length .* must match time steps"):
            plot_spectral_timeseries(sample_spectra, short_dates)


class TestExtractPointSpectra:
    """Test the point spectra extraction function."""
    
    def test_extract_basic(self):
        """Test basic spectra extraction."""
        # Create 4D image stack: (time=5, height=10, width=10, bands=3)
        image_stack = np.random.rand(5, 10, 10, 3)
        
        spectra = extract_point_spectra(image_stack, x=5, y=5)
        
        assert spectra.shape == (5, 3)  # 5 time steps, 3 bands
        
        # Verify extracted values match original
        np.testing.assert_array_equal(spectra, image_stack[:, 5, 5, :])
    
    def test_extract_invalid_shape(self):
        """Test error handling for invalid image stack shape."""
        invalid_stack = np.random.rand(10, 10, 3)  # 3D instead of 4D
        
        with pytest.raises(ValueError, match="Image stack must be 4D"):
            extract_point_spectra(invalid_stack, 5, 5)
    
    def test_extract_out_of_bounds(self):
        """Test error handling for out-of-bounds coordinates."""
        image_stack = np.random.rand(5, 10, 10, 3)
        
        # Test x coordinate out of bounds
        with pytest.raises(ValueError, match="Coordinates .* out of bounds"):
            extract_point_spectra(image_stack, x=15, y=5)
        
        # Test y coordinate out of bounds  
        with pytest.raises(ValueError, match="Coordinates .* out of bounds"):
            extract_point_spectra(image_stack, x=5, y=15)
        
        # Test negative coordinates
        with pytest.raises(ValueError, match="Coordinates .* out of bounds"):
            extract_point_spectra(image_stack, x=-1, y=5)
    
    def test_extract_edge_coordinates(self):
        """Test extraction at image edges."""
        image_stack = np.random.rand(3, 10, 10, 2)
        
        # Test corner coordinates
        spectra_corner = extract_point_spectra(image_stack, x=0, y=0)
        assert spectra_corner.shape == (3, 2)
        
        # Test edge coordinates
        spectra_edge = extract_point_spectra(image_stack, x=9, y=9)
        assert spectra_edge.shape == (3, 2)


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_full_workflow(self):
        """Test a complete workflow from data to visualization."""
        # Create synthetic data
        time_steps = 6
        height, width, bands = 50, 50, 4
        
        # Image stack
        image_stack = np.random.rand(time_steps, height, width, bands)
        
        # Extract spectra at a point
        x, y = 25, 25
        spectra = extract_point_spectra(image_stack, x, y)
        
        # Create date index
        dates = pd.date_range('2023-01-01', periods=time_steps, freq='W')
        
        # Plot spectral time series
        fig_spectra = plot_spectral_timeseries(spectra, dates)
        assert isinstance(fig_spectra, plt.Figure)
        plt.close(fig_spectra)
        
        # Plot image with marker
        sample_image = image_stack[0, :, :, :3]  # First image, RGB bands
        fig_marker = plot_image_with_marker(sample_image, (x, y))
        assert isinstance(fig_marker, plt.Figure)
        plt.close(fig_marker)
        
        # Create timeseries DataFrame for animation
        band_names = ['B1', 'B2', 'B3', 'B4']
        df = pd.DataFrame(spectra, columns=band_names, index=dates)
        
        # Create animation (test without saving to avoid file I/O in test)
        images = [image_stack[i, :, :, :3] for i in range(time_steps)]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "workflow_test.gif"
            result_path = animate_images_and_timeseries(
                images, df, point_coords=(x, y),
                bands=['B1', 'B2', 'B3'],
                output_path=str(output_path)
            )
            
            assert Path(result_path).exists()