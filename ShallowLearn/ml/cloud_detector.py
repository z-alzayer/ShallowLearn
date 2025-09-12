"""
Cloud detection utilities using XGBoost models.
Refactored from CloudDetector.py to follow module organization.
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import xgboost as xgb
from scipy.ndimage import binary_erosion

from ..core.array_utils import apply_mask
from ..io import load_image


def detect_clouds(
    datacube: np.ndarray, 
    threshold: float = 0.0, 
    window_size: int = 8
) -> np.ndarray:
    """
    Detect clouds in a time series of images by comparing each pixel to the mean of the surrounding window.

    Parameters:
    -----------
    datacube : np.ndarray
        4D numpy array (time, x, y, channels)
    threshold : float
        Threshold for cloud detection
    window_size : int
        Number of images in the sliding window

    Returns:
    --------
    np.ndarray
        4D boolean array (time, x, y, channels) indicating cloud presence
    """
    datacube = np.array(datacube)
    time, x, y, channels = datacube.shape
    half_window = window_size // 2

    # Initialize cloud mask with the same shape as datacube
    cloud_mask = np.zeros((time, x, y, channels), dtype=bool)

    # Pad the datacube to handle the boundaries
    padded_datacube = np.pad(
        datacube, ((half_window, half_window), (0, 0), (0, 0), (0, 0)), mode="reflect"
    )

    for t in range(time):
        window = padded_datacube[t : t + window_size]
        window_mean = np.mean(window, axis=0)
        cloud_mask[t] = datacube[t] > (window_mean + threshold)

    return cloud_mask


def add_nan_buffer(arr: np.ndarray, dilation_size: int = 3) -> np.ndarray:
    """
    Add a buffer around NaN values in the given array using morphological dilation.

    Parameters:
    -----------
    arr : np.ndarray
        Input array
    dilation_size : int
        Size of the dilation structure (controls buffer thickness)

    Returns:
    --------
    np.ndarray
        Modified array with buffer around NaN values
    """
    # Create a mask for non-NaN values
    non_nan_mask = ~np.isnan(arr)
    
    # Erode the non-NaN mask (which dilates the NaN regions)
    structure = np.ones((dilation_size, dilation_size))
    eroded_mask = binary_erosion(non_nan_mask, structure=structure)
    
    # Set the newly eroded areas to NaN
    result = arr.copy()
    result[~eroded_mask] = np.nan
    
    return result


def get_default_model_path() -> Path:
    """Get the default path to the cloud detection model."""
    # Look for model in package directory structure
    package_dir = Path(__file__).parent.parent.parent  # ShallowLearn root
    model_path = package_dir / "Models" / "CloudDetectXGB.json"
    
    if model_path.exists():
        return model_path
    
    # Fallback to original absolute path if it exists
    fallback_path = Path("/home/zba21/Documents/ShallowLearn/Models/CloudDetectXGB.json")
    if fallback_path.exists():
        return fallback_path
    
    raise FileNotFoundError(
        f"Cloud detection model not found. Expected at {model_path} or {fallback_path}"
    )


def cloud_regressor(
    img: Union[np.ndarray, str],
    return_mask: bool = False,
    threshold: float = 500,
    planet: bool = False,
    dilation: bool = True,
    dilation_size: int = 10,
    model_path: Optional[Union[str, Path]] = None,
    processed: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Apply cloud detection using XGBoost regressor.

    Parameters:
    -----------
    img : Union[np.ndarray, str]
        Input image array or path
    return_mask : bool
        If True, return only the cloud mask
    threshold : float
        Threshold for cloud detection
    planet : bool
        If True, use Planet satellite band ordering
    dilation : bool
        If True, apply morphological dilation to cloud mask
    dilation_size : int
        Size of dilation structure
    model_path : Optional[Union[str, Path]]
        Path to XGBoost model (uses default if None)
    processed : bool
        If True, image is already preprocessed

    Returns:
    --------
    Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
        Masked image or cloud mask depending on return_mask parameter
    """
    # Load model
    if model_path is None:
        model_path = get_default_model_path()
    
    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    
    # Prepare image
    if planet:
        img_prepared, shape, original = load_img_planet(img, processed=processed)
    else:
        img_prepared, shape, original = load_img_model(img, processed=processed)

    # Generate predictions
    mask = np.expand_dims(model.predict(img_prepared).reshape(shape[:2]), axis=2)
    
    if return_mask:
        return mask

    # Apply threshold and dilation
    if dilation:
        mask = mask < threshold
        mask = add_nan_buffer(mask[:, :, 0], dilation_size=dilation_size)
        mask = np.expand_dims(mask, axis=2)

    return apply_mask(original, mask)


def percentile_without_zeros(arr: np.ndarray, q: float) -> float:
    """
    Calculate percentile excluding zero values.

    Parameters:
    -----------
    arr : np.ndarray
        Input array
    q : float
        Percentile to compute (0-100)

    Returns:
    --------
    float
        Percentile value or NaN if all zeros
    """
    non_zero_arr = arr[arr != 0]
    if len(non_zero_arr) == 0:
        return np.nan
    return np.percentile(non_zero_arr, q)


def percentile_without_zeros_and_first(arr: np.ndarray, q: float) -> float:
    """
    Calculate percentile excluding zeros and repeated smallest values.

    Parameters:
    -----------
    arr : np.ndarray
        Input array
    q : float
        Percentile to compute (0-100)

    Returns:
    --------
    float
        Percentile value or NaN if insufficient data
    """
    non_zero_arr = arr[arr != 0]
    if len(non_zero_arr) == 0:
        return np.nan
    # Remove repeated smallest values
    non_zero_arr = np.unique(np.sort(non_zero_arr))[1:]
    if len(non_zero_arr) == 0:
        return np.nan
    return np.percentile(non_zero_arr, q)


def load_img_model(
    img: Union[np.ndarray, str], 
    processed: bool = False
) -> Tuple[np.ndarray, Tuple[int, ...], np.ndarray]:
    """
    Load and prepare image for cloud detection model (Sentinel-2 band ordering).

    Parameters:
    -----------
    img : Union[np.ndarray, str]
        Input image array or path
    processed : bool
        If True, image is already in correct format

    Returns:
    --------
    Tuple[np.ndarray, Tuple[int, ...], np.ndarray]
        Prepared image, original shape, original image
    """
    if processed:
        return img.reshape(-1, 4), img.shape, img
    
    if isinstance(img, str):
        img = load_image(img)

    # Select bands: B5, B4, B3, B9 (indices 4, 3, 2, 8 for Sentinel-2)
    return img[:, :, [4, 3, 2, 8]].reshape(-1, 4), img.shape, img


def load_img_planet(
    img: Union[np.ndarray, str], 
    processed: bool = False
) -> Tuple[np.ndarray, Tuple[int, ...], np.ndarray]:
    """
    Load and prepare image for cloud detection model (Planet satellite band ordering).

    Parameters:
    -----------
    img : Union[np.ndarray, str]
        Input image array or path
    processed : bool
        If True, image is already in correct format

    Returns:
    --------
    Tuple[np.ndarray, Tuple[int, ...], np.ndarray]
        Prepared image, original shape, original image
    """
    if processed:
        return img.reshape(-1, 4), img.shape, img
    
    if isinstance(img, str):
        img = load_image(img)

    # Select bands for Planet ordering
    return img[:, :, [3, 2, 1, 0]].reshape(-1, 4), img.shape, img