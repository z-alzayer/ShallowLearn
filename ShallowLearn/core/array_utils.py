"""
Core array utilities for image processing.
Minimal dependencies, reusable functions for array manipulation.
"""
import numpy as np
import colorsys
import math
from math import pi
from typing import Union, Tuple, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler


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


# Transform functions moved from Transform.py

def mutliband_standard_scaler(image: np.ndarray) -> np.ndarray:
    """
    Apply standard scaling to a multi-band image.
    
    Parameters:
    -----------
    image : np.ndarray
        Multi-band image array
        
    Returns:
    --------
    np.ndarray
        Standard scaled image
    """
    original_shape = image.shape
    img = image.reshape(-1, original_shape[-1])
    scaler = StandardScaler()
    rescaled_img = scaler.fit_transform(img)
    return rescaled_img.reshape(original_shape)


def mutliband_robust_scaler(image: np.ndarray) -> np.ndarray:
    """
    Apply robust scaling to a multi-band image.
    
    Parameters:
    -----------
    image : np.ndarray
        Multi-band image array
        
    Returns:
    --------
    np.ndarray
        Robust scaled image
    """
    original_shape = image.shape
    img = image.reshape(-1, original_shape[-1])
    scaler = RobustScaler()
    rescaled_img = scaler.fit_transform(img)
    return rescaled_img.reshape(original_shape)


def BCET(image: np.ndarray, min_value: int = 0, max_value: int = 255, desired_mean: int = 110) -> np.ndarray:
    """
    Applies a Bias Correction and Enhancement Technique (BCET) to an image.
    
    This technique stretches the image data with a parabolic function, which transforms
    the pixel intensity values to a specified range (default: 0 to 255) with a specified mean (default: 110).

    Parameters:
    -----------
    image : np.ndarray
        The input image as a numpy array
    min_value : int, default=0
        The minimum pixel intensity value in the output image
    max_value : int, default=255
        The maximum pixel intensity value in the output image
    desired_mean : int, default=110
        The mean pixel intensity value for the output image

    Returns:
    --------
    np.ndarray
        The transformed image as a numpy array with dtype 'int'
    """
    input_min = np.min(image)
    input_max = np.max(image)
    input_mean = np.mean(image)
    input_mean_sq = np.mean(image ** 2)

    parabola_vertex = (
        (input_max ** 2 * (desired_mean - min_value) - input_mean_sq * (max_value - min_value) +
         input_min ** 2 * (max_value - desired_mean)) /
        (2 * (input_max * (desired_mean - min_value) - input_mean * (max_value - min_value) + input_min * (max_value - desired_mean)))
    )

    parabola_coefficient = (max_value - min_value) / ((input_max - input_min) * (input_max + input_min - 2 * parabola_vertex))
    parabola_constant = min_value - parabola_coefficient * (input_min - parabola_vertex) ** 2

    transformed_image = parabola_coefficient * (image - parabola_vertex) ** 2 + parabola_constant

    return transformed_image.astype(int)  # Ensure the output values are integers for plotting


def linear_contrast_enhancement(image: np.ndarray, max_value: int = 255) -> np.ndarray:
    """
    Applies a Linear Contrast Enhancement (LCE) to an image.

    This technique linearly rescales the image pixel intensity values to the full range of possible values (0-255).

    Parameters:
    -----------
    image : np.ndarray
        The input image as a numpy array
    max_value : int, default=255
        Maximum value for the output range

    Returns:
    --------
    np.ndarray
        The rescaled image as a numpy array

    Raises:
    -------
    ValueError
        If all valid pixel values in the image are the same
    """
    # Create a copy to avoid side effects
    image = image.copy()
    
    # Identify NaN values
    mask_nan = np.isnan(image)

    # Replace NaN values with 0
    image[mask_nan] = 0

    # Get the minimum value from non-zero elements of the image
    original_min_intensity = np.min(image[np.nonzero(image)])
    min_intensity = original_min_intensity + 0.001

    # Get the maximum value from the image
    max_intensity = np.max(image)

    # Check if maximum and minimum are the same
    if max_intensity == original_min_intensity:
        raise ValueError("Cannot apply linear contrast enhancement: all pixel values in the image are the same.")

    # Apply linear contrast enhancement and clip to keep values within the desired range
    enhanced_image = np.clip((image - min_intensity) * (max_value / (max_intensity - min_intensity)), 0, max_value)

    enhanced_image[mask_nan] = np.nan  # Replace NaN values with NaN

    return enhanced_image


def BCET_multi(image: np.ndarray, min_value: int = 0, max_value: int = 255, desired_mean: int = 110) -> np.ndarray:
    """
    Applies BCET to each channel of a multi-channel image separately.
    
    Parameters:
    -----------
    image : np.ndarray
        Multi-channel image array
    min_value : int, default=0
        Minimum output value
    max_value : int, default=255
        Maximum output value  
    desired_mean : int, default=110
        Desired mean value
        
    Returns:
    --------
    np.ndarray
        BCET enhanced multi-channel image
    """
    # Get the number of channels in the image
    num_channels = image.shape[2] if len(image.shape) == 3 else 1

    # Initialize an empty array for the output image
    output_image = np.zeros_like(image)

    # Apply BCET to each channel separately
    for channel in range(num_channels):
        output_image[:, :, channel] = BCET(image[:, :, channel], min_value, max_value, desired_mean)

    return output_image


def LCE_multi(image: np.ndarray) -> np.ndarray:
    """
    Applies linear contrast enhancement to each channel of a multi-channel image separately.
    
    Parameters:
    -----------
    image : np.ndarray
        Multi-channel image array
        
    Returns:
    --------
    np.ndarray
        LCE enhanced multi-channel image
    """
    # Get the number of channels in the image
    num_channels = image.shape[2] if len(image.shape) == 3 else 1

    # Initialize an empty array for the output image
    output_image = np.zeros_like(image)

    # Apply linear contrast enhancement to each channel separately
    for channel in range(num_channels):
        output_image[:, :, channel] = linear_contrast_enhancement(image[:, :, channel])

    return output_image


def rgb_to_hsi(rgb: np.ndarray) -> np.ndarray:
    """
    Convert RGB image to HSI color space.
    
    Parameters:
    -----------
    rgb : np.ndarray
        RGB image array
        
    Returns:
    --------
    np.ndarray
        HSI image array
    """
    # Create copy to avoid side effects
    rgb = rgb.copy()
    
    # Normalize the RGB values
    rgb = rgb / 255.0

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    # Calculate Intensity
    I = np.nanmean(rgb, axis=-1)

    # Calculate Saturation
    num = np.nanmean((rgb - I[:, :, np.newaxis]) ** 2, axis=-1)
    den = 2 * I * (1 - I)
    den = np.where(np.isnan(den), 1, den) # handle NaN
    S = np.sqrt(num / den)

    # Calculate Hue
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    den = np.where(np.isnan(den), 0.00001, den) # handle NaN
    theta = np.arccos(num + 0.0001 / den + 0.0001)

    H = theta.copy()
    H[b > g] = 2 * np.pi - H[b > g]

    H = H / (2 * np.pi)  # normalize to [0, 1]
    H = H * 360  # scale to degrees

    hsi = np.dstack((H, S, I))
    
    return hsi


def hsi_to_rgb(array: np.ndarray) -> np.ndarray:
    """
    Convert an array of HSI (Hue, Saturation, Intensity) values to an array of RGB values.
    
    Parameters:
    -----------
    array : np.ndarray
        HSI image array
        
    Returns:
    --------
    np.ndarray
        RGB image array
    """
    # Extract the HSI values from the array
    h = array[:, :, 0]
    s = array[:, :, 1]
    i = array[:, :, 2]

    # Check if the hue is outside the range [0, 360)
    h = np.where(h < 0, h + 360, h)
    h = np.where(h >= 360, h - 360, h)

    # Check if the saturation is outside the range [0, 1]
    s = np.clip(s, 0, 1)

    # Check if the intensity is outside the range [0, 1]
    i = np.clip(i, 0, 1)

    # Convert the hue to the range [0, 6)
    h = h / 60

    # Calculate the chroma
    c = (1 - np.abs(2*i - 1)) * s

    # Calculate the x value
    x = c * (1 - np.abs(h % 2 - 1))

    # Calculate the m value
    m = i - c/2

    # Calculate the RGB values
    r, g, b = np.zeros_like(h), np.zeros_like(h), np.zeros_like(h)
    idx = np.where((0 <= h) & (h < 1))
    r[idx], g[idx], b[idx] = c[idx], x[idx], 0
    idx = np.where((1 <= h) & (h < 2))
    r[idx], g[idx], b[idx] = x[idx], c[idx], 0
    idx = np.where((2 <= h) & (h < 3))
    r[idx], g[idx], b[idx] = 0, c[idx], x[idx]
    idx = np.where((3 <= h) & (h < 4))
    r[idx], g[idx], b[idx] = 0, x[idx], c[idx]
    idx = np.where((4 <= h) & (h < 5))
    r[idx], g[idx], b[idx] = x[idx], 0, c[idx]
    idx = np.where((5 <= h) & (h < 6))
    r[idx], g[idx], b[idx] = c[idx], 0, x[idx]

    # Add the m value to each RGB value
    r, g, b = r + m, g + m, b + m

    # Convert the RGB values to the range [0, 255]
    r, g, b = (r * 255).astype(np.uint8), (g * 255).astype(np.uint8), (b * 255).astype(np.uint8)

    # Stack the RGB values into a single array
    rgb_array = np.stack((r, g, b), axis=2)

    return rgb_array


def transform_multiband_lab(arr: np.ndarray, bands: Tuple[int, int, int] = (3, 2, 1)) -> np.ndarray:
    """
    Reindex the bands of a multiband image to match the order of the LAB color space.
    
    Parameters:
    -----------
    arr : np.ndarray
        Input multiband array
    bands : tuple of int, default=(3, 2, 1)
        Band indices for Red, Green, Blue channels
        
    Returns:
    --------
    np.ndarray
        LAB transformed multiband image
    """
    from skimage.color import rgb2lab, lab2rgb
    
    arr_copy = arr.copy()
    # Convert arr_copy to float64
    arr_copy = arr_copy.astype(np.float64)
    
    # Extract RGB channels and apply LAB stretch
    rgb_arr = arr_copy[:, :, [bands[0], bands[1], bands[2]]]
    lab_array = rgb2lab(rgb_arr)
    lab_array[:, :, 0] = linear_contrast_enhancement(lab_array[:, :, 0])
    rgb_arr = lab2rgb(lab_array)
    
    # Update the original array with transformed RGB
    arr_copy[:, :, bands[0]] = rgb_arr[:, :, 0]
    arr_copy[:, :, bands[1]] = rgb_arr[:, :, 1]
    arr_copy[:, :, bands[2]] = rgb_arr[:, :, 2]

    return arr_copy


def transform_multiband_hsv(arr: np.ndarray, bands: Tuple[int, int, int] = (3, 2, 1), max_value: int = 100) -> np.ndarray:
    """
    Reindex the bands of a multiband image to match the order of the HSV color space.
    
    Parameters:
    -----------
    arr : np.ndarray
        Input multiband array
    bands : tuple of int, default=(3, 2, 1)
        Band indices for Red, Green, Blue channels
    max_value : int, default=100
        Maximum value for enhancement
        
    Returns:
    --------
    np.ndarray
        HSV transformed multiband image
    """
    from skimage.color import rgb2hsv, hsv2rgb
    
    arr_copy = arr.copy()
    # Convert arr_copy to float64
    arr_copy = arr_copy.astype(np.float64)
    
    # Extract RGB channels and apply HSV stretch
    rgb_arr = arr_copy[:, :, [bands[0], bands[1], bands[2]]]
    hsv_array = rgb2hsv(rgb_arr)
    hsv_array[:, :, 1] = linear_contrast_enhancement(hsv_array[:, :, 1], max_value=max_value)
    rgb_arr = hsv2rgb(hsv_array)
    
    # Update the original array with transformed RGB
    arr_copy[:, :, bands[0]] = rgb_arr[:, :, 0]
    arr_copy[:, :, bands[1]] = rgb_arr[:, :, 1]
    arr_copy[:, :, bands[2]] = rgb_arr[:, :, 2]
    
    return arr_copy