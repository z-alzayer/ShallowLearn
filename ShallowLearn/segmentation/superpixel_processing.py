"""
Superpixel processing utilities for analysis and visualization.
Refactored from SuperPixelProcessing.py to follow module organization.
"""

import re
from typing import Optional, Tuple, Union

import numpy as np
from scipy import stats

from .superpixels import (
    generate_sup_pixel_labels,
    pad_slice_segments_w_0pads,
    pca_segments,
    process_superpixel_dii_pipeline,
)


def extract_date_from_filename(filename: str) -> str:
    """
    Extract the date from the filename string.
    
    Parameters:
    -----------
    filename : str
        Filename containing date in YYYYMMDD format
        
    Returns:
    --------
    str
        Extracted date or 'Unknown Date'
    """
    match = re.search(r"\d{8}", filename)
    return match.group(0) if match else "Unknown Date"


def calculate_image_statistics(image: np.ndarray, statistic_type: str = "mean") -> np.ndarray:
    """
    Calculate statistics of an image per channel.
    
    Parameters:
    -----------
    image : np.ndarray
        Input image (height, width, channels)
    statistic_type : str
        Type of statistic ('mean', 'mode', 'median')
        
    Returns:
    --------
    np.ndarray
        Statistics for each channel
    """
    statistics = np.zeros(image.shape[2])
    
    for channel in range(image.shape[2]):
        if statistic_type == "mean":
            statistics[channel] = image[:, :, channel].mean()
        elif statistic_type == "median":
            statistics[channel] = np.median(image[:, :, channel])
        elif statistic_type == "mode":
            mode_result = stats.mode(image[:, :, channel], axis=None)
            statistics[channel] = mode_result.mode[0]
        else:
            raise ValueError(
                "Invalid statistic type specified. Choose 'mean', 'mode', or 'median'."
            )
    
    return statistics


def calculate_nonzero_percentile(data: np.ndarray, percentile: int) -> float:
    """
    Calculate percentile of non-zero values in data.
    
    Parameters:
    -----------
    data : np.ndarray
        Input data array
    percentile : int
        Percentile to calculate (0-100)
        
    Returns:
    --------
    float
        Percentile value
    """
    non_zero = data[data != 0]
    if len(non_zero) > 0:
        return np.percentile(non_zero, percentile)
    return 0.0


def create_and_pad_superpixels_v2(
    img: np.ndarray,
    n_segments: int = 300,
    compactness: float = 10.0,
    pad_shape: Tuple[int, int, int] = (32, 32, 13)
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create superpixels and extract padded patches.
    
    Parameters:
    -----------
    img : np.ndarray
        Input image
    n_segments : int
        Number of superpixel segments
    compactness : float
        Compactness parameter for SLIC
    pad_shape : Tuple[int, int, int]
        Shape for padded patches
        
    Returns:
    --------
    Tuple[np.ndarray, np.ndarray]
        Superpixel labels and padded patches
    """
    from .superpixels import slic_segmentation
    
    # Generate superpixels
    super_pixel = slic_segmentation(img, n_segments=n_segments, compactness=compactness)
    
    # Extract padded patches
    padded_segments = pad_slice_segments_w_0pads(img, super_pixel, shape=pad_shape)
    
    return super_pixel, padded_segments


def relabel_super_pixels(super_pixel: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Relabel superpixel segments based on cluster labels.
    
    Parameters:
    -----------
    super_pixel : np.ndarray
        Original superpixel segmentation
    labels : np.ndarray
        New labels for each segment
        
    Returns:
    --------
    np.ndarray
        Relabeled superpixel array
    """
    relabeled = np.zeros_like(super_pixel)
    unique_segments = np.unique(super_pixel)
    
    for idx, segment_id in enumerate(unique_segments):
        if idx < len(labels):
            relabeled[super_pixel == segment_id] = labels[idx]
    
    return relabeled


def process_single_image_from_array(
    img: np.ndarray,
    get_deep: bool = False,
    n_segments: int = 300
) -> dict:
    """
    Process a single image array through the superpixel DII pipeline.
    
    Parameters:
    -----------
    img : np.ndarray
        Input image array
    get_deep : bool
        Whether to return deep water mask
    n_segments : int
        Number of superpixel segments
        
    Returns:
    --------
    dict
        Processing results including DII stack and masks
    """
    # Generate superpixels
    segments = generate_sup_pixel_labels(img, no_segments=n_segments)
    
    # Process through DII pipeline
    results = process_superpixel_dii_pipeline(
        img,
        segments,
        bands=[0, 1, 2] if img.shape[2] >= 3 else list(range(img.shape[2]))
    )
    
    if get_deep:
        results['deep_water_mask'] = results['deep_mask']
    
    return results