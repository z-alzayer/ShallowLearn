"""
Core array utilities for image processing.
Minimal dependencies, reusable functions for array manipulation.
"""
import numpy as np
from typing import Union, Tuple, Optional


def clip_array(arr: np.ndarray, min_val: float = 0, max_val: float = 10000) -> np.ndarray:
    """
    Clips an input multiband array to values between min_val and max_val.

    Parameters:
    -----------
    arr : np.ndarray
        A NumPy array of any shape
    min_val : float, default=0
        Minimum value for clipping
    max_val : float, default=10000
        Maximum value for clipping

    Returns:
    --------
    np.ndarray
        A clipped NumPy array with the same shape as input
    """
    return np.clip(arr, min_val, max_val)


def select_channels(arr: np.ndarray, indices: list) -> np.ndarray:
    """
    Selects specific channels based on the given indices from a 3D numpy array.

    Parameters:
    -----------
    arr : np.ndarray
        A numpy array of shape (x, y, z)
    indices : list
        A list containing the indices of channels to be selected.
        Its length must be 3 or it will raise a ValueError.

    Returns:
    --------
    np.ndarray
        A numpy array of shape (x, y, 3)
    """
    if len(indices) != 3:
        raise ValueError("The length of indices must be 3.")
    
    return arr[:, :, indices]


def remove_channel(img: np.ndarray, channel: int) -> np.ndarray:
    """
    Removes a specified channel from a 3D image array.

    Parameters:
    -----------
    img : np.ndarray
        The 3D image array with shape (height, width, channels)
    channel : int
        The index of the channel to remove

    Returns:
    --------
    np.ndarray
        The image array with the specified channel removed

    Raises:
    -------
    ValueError
        If the channel index is out of bounds
    """
    if channel < 0 or channel >= img.shape[2]:
        raise ValueError("Channel index is out of bounds.")

    return np.concatenate((img[:, :, :channel], img[:, :, channel+1:]), axis=2)


def apply_mask(data: np.ndarray, mask: np.ndarray, fill_value: float = 0) -> np.ndarray:
    """
    Applies a mask to the data array.

    Parameters:
    -----------
    data : np.ndarray
        The input data array
    mask : np.ndarray
        The mask array
    fill_value : float, default=0
        The value to use where the mask is False

    Returns:
    --------
    np.ndarray
        The masked data array
    """
    return np.where(mask, data, fill_value)


def generate_multichannel_mask(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Generates a multichannel mask for the input image.

    Parameters:
    -----------
    img : np.ndarray
        The input image array
    mask : np.ndarray
        2D boolean mask

    Returns:
    --------
    np.ndarray
        The multichannel mask array
    """
    reshaped_mask = np.repeat(mask[:, :, np.newaxis], img.shape[2], axis=2)
    final_mask = img * reshaped_mask
    rescaled_image = final_mask.copy()

    for i in range(final_mask.shape[2]):
        channel_min = final_mask[:, :, i].min()
        channel_max = final_mask[:, :, i].max()
        if channel_max > channel_min:  # Avoid division by zero
            rescaled_image[:, :, i] = (final_mask[:, :, i] - channel_min) / (channel_max - channel_min) * 255
        else:
            rescaled_image[:, :, i] = 0
            
    return rescaled_image


def mask_out_zeros(arr: np.ndarray) -> np.ndarray:
    """
    Masks out zero values in an array.

    Parameters:
    -----------
    arr : np.ndarray
        Input array

    Returns:
    --------
    np.ndarray
        Array with zeros masked out
    """
    return np.ma.masked_where(arr == 0, arr)


def validate_band_shape(image: np.ndarray, band_numbers: list) -> None:
    """
    Validates that the image has enough bands for the given band numbers.

    Parameters:
    -----------
    image : np.ndarray
        Input image array with shape (height, width, bands)
    band_numbers : list
        List of band indices to validate

    Raises:
    -------
    ValueError
        If any band number is out of bounds
    """
    if len(image.shape) != 3:
        raise ValueError("Image must be 3D array with shape (height, width, bands)")
    
    max_band = max(band_numbers) if band_numbers else -1
    if max_band >= image.shape[2]:
        raise ValueError(f"Band index {max_band} is out of bounds for image with {image.shape[2]} bands")


def median_without_zeros_or_nans(images: np.ndarray) -> np.ndarray:
    """
    Computes the median of each band in each image, excluding zeros and NaN values.

    Parameters:
    -----------
    images : np.ndarray
        Input 4D array of images with shape (i, x, y, z)

    Returns:
    --------
    np.ndarray
        Median values for each band in each image, shape (i, z)
    """
    num_images = images.shape[0]
    num_bands = images.shape[3]
    medians = np.zeros((num_images, num_bands))

    for i in range(num_images):
        for z in range(num_bands):
            band_data = images[i, :, :, z]
            # Mask zeros and NaNs
            masked_data = np.ma.masked_where((band_data == 0) | np.isnan(band_data), band_data)
            medians[i, z] = np.ma.median(masked_data)

    return medians


def pad_patches(patches: list, target_shape: Tuple[int, int] = (20, 20)) -> np.ndarray:
    """
    Pads patches to a uniform size.

    Parameters:
    -----------
    patches : list
        List of patch arrays
    target_shape : tuple, default=(20, 20)
        Target shape for padding

    Returns:
    --------
    np.ndarray
        Array of padded patches
    """
    padded_patches = []
    
    for patch in patches:
        if patch.shape[:2] == target_shape:
            padded_patches.append(patch)
        else:
            # Calculate padding needed
            pad_height = max(0, target_shape[0] - patch.shape[0])
            pad_width = max(0, target_shape[1] - patch.shape[1])
            
            # If patch is larger than target, crop it
            if patch.shape[0] > target_shape[0] or patch.shape[1] > target_shape[1]:
                patch = patch[:target_shape[0], :target_shape[1]]
                pad_height = pad_width = 0
            
            # Pad if necessary
            if len(patch.shape) == 3:
                pad_config = ((0, pad_height), (0, pad_width), (0, 0))
            else:
                pad_config = ((0, pad_height), (0, pad_width))
                
            padded_patch = np.pad(patch, pad_config, mode='constant', constant_values=0)
            padded_patches.append(padded_patch)
    
    return np.array(padded_patches)


def get_band_numbers(bands: list, band_mapping: dict) -> list:
    """
    Converts band names to band numbers using a band mapping dictionary.

    Parameters:
    -----------
    bands : list
        List of band names (e.g., ['B02', 'B03', 'B04'])
    band_mapping : dict
        Dictionary mapping band names to indices

    Returns:
    --------
    list
        List of band indices

    Raises:
    -------
    KeyError
        If a band name is not found in the mapping
    """
    band_numbers = []
    for band in bands:
        if band not in band_mapping:
            raise KeyError(f"Band '{band}' not found in band mapping")
        band_numbers.append(band_mapping[band]['index'])
    
    return band_numbers


def normalize_array(arr: np.ndarray, method: str = 'minmax') -> np.ndarray:
    """
    Normalize array using different methods.

    Parameters:
    -----------
    arr : np.ndarray
        Input array
    method : str, default='minmax'
        Normalization method ('minmax', 'zscore', 'unit')

    Returns:
    --------
    np.ndarray
        Normalized array
    """
    if method == 'minmax':
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        if arr_max == arr_min:
            return np.zeros_like(arr)
        return (arr - arr_min) / (arr_max - arr_min)
    
    elif method == 'zscore':
        mean = np.mean(arr)
        std = np.std(arr)
        if std == 0:
            return np.zeros_like(arr)
        return (arr - mean) / std
    
    elif method == 'unit':
        norm = np.linalg.norm(arr)
        if norm == 0:
            return arr
        return arr / norm
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")