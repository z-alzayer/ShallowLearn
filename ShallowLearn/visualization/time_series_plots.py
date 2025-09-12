"""
Time series visualization utilities for satellite imagery analysis.
Provides functions for animating image sequences and plotting spectral data over time.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
from matplotlib.animation import PillowWriter


def animate_images_and_timeseries(
    images: List[np.ndarray],
    timeseries_data: pd.DataFrame,
    point_coords: Tuple[int, int] = (50, 50),
    fps: int = 2,
    output_path: str = "animation.gif",
    title_prefix: str = "Patch reef",
    bands: List[str] = None,
    band_colors: List[str] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> str:
    """
    Create an animated visualization showing satellite images with spectral time series.
    
    Shows images on the left with a red dot marking the analysis point, and time series
    plots on the right showing spectral bands over time with a moving vertical line.
    
    Parameters
    ----------
    images : List[np.ndarray]
        List of satellite images to animate (should match timeseries length)
    timeseries_data : pd.DataFrame
        DataFrame with datetime index and spectral band columns
    point_coords : Tuple[int, int], optional
        (x, y) coordinates for the analysis point marker, by default (50, 50)
    fps : int, optional
        Frames per second for animation, by default 2
    output_path : str, optional
        Path to save the animation GIF, by default "animation.gif"
    title_prefix : str, optional
        Prefix for frame titles, by default "Patch reef"
    bands : List[str], optional
        Band names to plot, by default ['B4', 'B3', 'B2']
    band_colors : List[str], optional
        Colors for each band, by default ['r', 'g', 'b']
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 5)
        
    Returns
    -------
    str
        Path to the saved animation file
        
    Raises
    ------
    ValueError
        If images and timeseries data lengths don't match
    """
    if len(images) != len(timeseries_data):
        raise ValueError(f"Number of images ({len(images)}) must match timeseries length ({len(timeseries_data)})")
    
    if bands is None:
        bands = ['B4', 'B3', 'B2']
    if band_colors is None:
        band_colors = ['r', 'g', 'b']
        
    if len(bands) != len(band_colors):
        raise ValueError("Number of bands must match number of colors")
    
    x, y = point_coords
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    
    def update_frame(frame: int) -> None:
        """Update function for animation frames."""
        axs[0].clear()
        axs[1].clear()
        
        # Display satellite image with analysis point
        img = images[frame]
        axs[0].imshow(img)
        axs[0].plot(x, y, 'ro', markersize=8)
        axs[0].arrow(x, y-30, 0, 15, head_width=5, head_length=5, fc='red', ec='red')
        axs[0].set_title('Satellite Image')
        axs[0].axis('off')
        
        # Plot time series with current frame highlighted
        for band, color in zip(bands, band_colors):
            if band in timeseries_data.columns:
                axs[1].scatter(timeseries_data.index, timeseries_data[band], 
                             label=band, color=color, alpha=0.7)
        
        # Add vertical line for current date
        current_date = timeseries_data.index[frame]
        axs[1].axvline(x=current_date, color='k', linestyle='--', alpha=0.8)
        
        # Format time series plot
        axs[1].legend()
        axs[1].set_ylim(0, 1)
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Normalized Reflectance')
        axs[1].set_title('Spectral Time Series')
        axs[1].tick_params(axis='x', rotation=45)
        
        # Set main title with current date
        plt.suptitle(f"{title_prefix} - {current_date.strftime('%Y-%m-%d')}", fontsize=14)
        plt.tight_layout()
    
    # Create and save animation
    ani = animation.FuncAnimation(fig, update_frame, frames=len(images), repeat=False)
    writer = PillowWriter(fps=fps)
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    ani.save(str(output_path), writer=writer)
    plt.close(fig)
    
    return str(output_path)


def plot_image_with_marker(
    image: np.ndarray,
    point_coords: Tuple[int, int],
    arrow_offset: Tuple[int, int] = (0, -20),
    marker_color: str = 'red',
    marker_size: int = 8,
    figsize: Tuple[int, int] = (10, 8),
    title: str = None,
    save_path: str = None
) -> plt.Figure:
    """
    Display an image with a marker point and directional arrow.
    
    Parameters
    ----------
    image : np.ndarray
        The image to display
    point_coords : Tuple[int, int]
        (x, y) coordinates for the marker point
    arrow_offset : Tuple[int, int], optional
        (dx, dy) offset for arrow start position, by default (0, -20)
    marker_color : str, optional
        Color for marker and arrow, by default 'red'
    marker_size : int, optional
        Size of the marker point, by default 8
    figsize : Tuple[int, int], optional
        Figure size, by default (10, 8)
    title : str, optional
        Plot title, by default None
    save_path : str, optional
        Path to save the plot, by default None
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object
    """
    x, y = point_coords
    dx, dy = arrow_offset
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display image
    ax.imshow(image)
    
    # Add marker point
    ax.plot(x, y, 'o', color=marker_color, markersize=marker_size)
    
    # Add directional arrow
    ax.arrow(x + dx, y - dy * 2, -dx, dy, 
             head_width=5, head_length=5, 
             fc=marker_color, ec=marker_color)
    
    # Format plot
    ax.set_title(title or f'Image with marker at ({x}, {y})')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_spectral_timeseries(
    spectra: np.ndarray,
    dates: Union[pd.DatetimeIndex, List[str]],
    band_labels: Dict[int, Union[str, int]] = None,
    title: str = "Spectral Time Series",
    figsize: Tuple[int, int] = (12, 8),
    colors: List[str] = None,
    save_path: str = None,
    show_legend: bool = True
) -> plt.Figure:
    """
    Plot spectral values across time for multiple bands.
    
    Parameters
    ----------
    spectra : np.ndarray
        Spectral data with shape (time_steps, bands)
    dates : Union[pd.DatetimeIndex, List[str]]
        Dates corresponding to each time step
    band_labels : Dict[int, Union[str, int]], optional
        Mapping from band index to display label
    title : str, optional
        Plot title, by default "Spectral Time Series"
    figsize : Tuple[int, int], optional
        Figure size, by default (12, 8)
    colors : List[str], optional
        Colors for each band line
    save_path : str, optional
        Path to save the plot, by default None
    show_legend : bool, optional
        Whether to show legend, by default True
        
    Returns
    -------
    plt.Figure
        The matplotlib figure object
        
    Raises
    ------
    ValueError
        If spectra is not 2D or dates length doesn't match time dimension
    """
    if len(spectra.shape) != 2:
        raise ValueError(f"Spectra must be 2D array, got shape {spectra.shape}")
    
    time_steps, n_bands = spectra.shape
    
    if len(dates) != time_steps:
        raise ValueError(f"Dates length ({len(dates)}) must match time steps ({time_steps})")
    
    # Convert dates to strings if they're datetime objects
    if hasattr(dates[0], 'strftime'):
        date_strings = [d.strftime('%Y-%m-%d') for d in dates]
    else:
        date_strings = dates
    
    # Set up colors
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_bands))
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each band
    for band_idx in range(n_bands):
        if band_labels and band_idx in band_labels:
            label = f"Band {band_labels[band_idx]}"
        else:
            label = f"Band {band_idx}"
            
        color = colors[band_idx % len(colors)]
        ax.plot(date_strings, spectra[:, band_idx], 
               label=label, color=color, marker='o', linewidth=2)
    
    # Format plot
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Spectral Value', fontsize=12)
    ax.set_title(title, fontsize=14)
    
    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    if show_legend:
        ax.legend(loc='best')
    
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def extract_point_spectra(
    image_stack: np.ndarray,
    x: int,
    y: int
) -> np.ndarray:
    """
    Extract spectral values at a specific point across all time steps.
    
    Parameters
    ----------
    image_stack : np.ndarray
        Image stack with shape (time, height, width, bands)
    x : int
        X coordinate of the point
    y : int
        Y coordinate of the point
        
    Returns
    -------
    np.ndarray
        Spectral values with shape (time, bands)
        
    Raises
    ------
    ValueError
        If image_stack is not 4D or coordinates are out of bounds
    """
    if len(image_stack.shape) != 4:
        raise ValueError(f"Image stack must be 4D (time, height, width, bands), got {image_stack.shape}")
    
    time_steps, height, width, n_bands = image_stack.shape
    
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Coordinates ({x}, {y}) out of bounds for image size ({width}, {height})")
    
    return image_stack[:, y, x, :]