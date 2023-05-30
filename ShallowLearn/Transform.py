import numpy.ma as ma
import numpy as np
import colorsys
import math
from math import pi

def BCET(image, min_value=0, max_value=255, desired_mean=110):
    """
    Applies a Bias Correction and Enhancement Technique (BCET) to an image.
    
    This technique stretches the image data with a parabolic function, which transforms
    the pixel intensity values to a specified range (default: 0 to 255) with a specified mean (default: 110).

    Parameters:
    - image: The input image as a numpy array.
    - min_value: The minimum pixel intensity value in the output image.
    - max_value: The maximum pixel intensity value in the output image.
    - desired_mean: The mean pixel intensity value for the output image.

    Returns:
    - The transformed image as a numpy array with dtype 'int'.
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


def linear_contrast_enhancement(image):
    """
    Applies a Linear Contrast Enhancement (LCE) to an image.

    This technique linearly rescales the image pixel intensity values to the full range of possible values (0-255).

    Parameters:
    - image: The input image as a numpy array.

    Returns:
    - The rescaled image as a numpy array.

    Raises:
    - ValueError: If all valid pixel values in the image are the same.
    """
    # Identify NaN values
    mask_nan = np.isnan(image)

    # Replace NaN values with 0
    image[mask_nan] = 0

    # Get the minimum value from non-zero elements of the image
    min_intensity = np.min(image[np.nonzero(image)])

    # Get the maximum value from the image
    max_intensity = np.max(image)

    # Check if maximum and minimum are the same
    if max_intensity == min_intensity:
        raise ValueError("Cannot apply linear contrast enhancement: all pixel values in the image are the same.")

    # Apply linear contrast enhancement and clip to keep values within the desired range
    enhanced_image = np.clip((image - min_intensity) * (255 / (max_intensity - min_intensity)), 0, 255)

    return enhanced_image

def BCET_multi(image, min_value=0, max_value=255, desired_mean=110):
    """
    Applies BCET to each channel of a multi-channel image separately.
    """
    # Get the number of channels in the image
    num_channels = image.shape[2] if len(image.shape) == 3 else 1

    # Initialize an empty array for the output image
    output_image = np.zeros_like(image)

    # Apply BCET to each channel separately
    for channel in range(num_channels):
        output_image[:, :, channel] = BCET(image[:, :, channel], min_value, max_value, desired_mean)

    return output_image


def LCE_multi(image):
    """
    Applies linear contrast enhancement to each channel of a multi-channel image separately.
    """
    # Get the number of channels in the image
    num_channels = image.shape[2] if len(image.shape) == 3 else 1

    # Initialize an empty array for the output image
    output_image = np.zeros_like(image)

    # Apply linear contrast enhancement to each channel separately
    for channel in range(num_channels):
        output_image[:, :, channel] = linear_contrast_enhancement(image[:, :, channel])

    return output_image



def rgb_to_hsi(rgb):
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
    theta = np.arccos(num / den)

    H = theta.copy()
    H[b > g] = 2 * np.pi - H[b > g]

    H = H / (2 * np.pi)  # normalize to [0, 1]
    H = H * 360  # scale to degrees

    hsi = np.dstack((H, S, I))
    
    return hsi

def hsi_to_rgb(hsi):
    H = hsi[:, :, 0]  # Hue
    S = hsi[:, :, 1]  # Saturation
    I = hsi[:, :, 2]  # Intensity

    # Create NaN mask for each H, S, I 
    H_nan_mask = np.isnan(H)
    S_nan_mask = np.isnan(S)
    I_nan_mask = np.isnan(I)

    # Where there are NaNs, set values to a safe number to prevent issues in calculation
    H[H_nan_mask] = 0
    S[S_nan_mask] = 0
    I[I_nan_mask] = 0

    R = np.zeros(H.shape)
    G = np.zeros(H.shape)
    B = np.zeros(H.shape)

    # RG sector (0 <= H < 120)
    idx = (0 <= H) & (H < 120)
    B[idx] = I[idx] * (1 - S[idx])
    R[idx] = I[idx] * (1 + S[idx] * np.cos(np.deg2rad(H[idx])) / np.cos(np.deg2rad(60 - H[idx])))
    G[idx] = 3*I[idx] - (R[idx] + B[idx])

    # GB sector (120 <= H < 240)
    idx = (120 <= H) & (H < 240)
    H[idx] = H[idx] - 120
    R[idx] = I[idx] * (1 - S[idx])
    G[idx] = I[idx] * (1 + S[idx] * np.cos(np.deg2rad(H[idx])) / np.cos(np.deg2rad(60 - H[idx])))
    B[idx] = 3*I[idx] - (R[idx] + G[idx])

    # BR sector (240 <= H < 360)
    idx = (240 <= H) & (H <= 360)
    H[idx] = H[idx] - 240
    G[idx] = I[idx] * (1 - S[idx])
    B[idx] = I[idx] * (1 + S[idx] * np.cos(np.deg2rad(H[idx])) / np.cos(np.deg2rad(60 - H[idx])))
    R[idx] = 3*I[idx] - (G[idx] + B[idx])

    R = np.clip(R, 0, 1) * 255.0
    G = np.clip(G, 0, 1) * 255.0
    B = np.clip(B, 0, 1) * 255.0

    # Restore NaN values
    R[H_nan_mask | S_nan_mask | I_nan_mask] = np.nan
    G[H_nan_mask | S_nan_mask | I_nan_mask] = np.nan
    B[H_nan_mask | S_nan_mask | I_nan_mask] = np.nan

    rgb = np.dstack((R, G, B)).astype(np.uint8)

    return rgb