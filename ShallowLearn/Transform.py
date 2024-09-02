import numpy.ma as ma
import numpy as np
import colorsys
import math
from math import pi
from skimage.color import rgb2lab, lab2rgb, hsv2rgb
from ShallowLearn.ImageHelper import plot_lab, plot_hsv, generate_multichannel_mask
from sklearn.preprocessing import StandardScaler, RobustScaler

def mutliband_standard_scaler(image):
    original_shape = image.shape
    img = image.reshape(-1, original_shape[-1])
    scaler = StandardScaler()
    rescaled_img = scaler.fit_transform(img)
    return rescaled_img.reshape(original_shape)


def mutliband_robust_scaler(image):
    original_shape = image.shape
    img = image.reshape(-1, original_shape[-1])
    scaler = RobustScaler()
    rescaled_img = scaler.fit_transform(img)
    return rescaled_img.reshape(original_shape)




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


def linear_contrast_enhancement(image, max_value=255):
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
    min_intensity = np.min(image[np.nonzero(image)]) + 0.001

    # Get the maximum value from the image
    max_intensity = np.max(image)

    # Check if maximum and minimum are the same
    if max_intensity == min_intensity:
        raise ValueError("Cannot apply linear contrast enhancement: all pixel values in the image are the same.")

    # Apply linear contrast enhancement and clip to keep values within the desired range
    enhanced_image = np.clip((image - min_intensity) * (max_value / (max_intensity - min_intensity)), 0, max_value)

    enhanced_image[mask_nan] = np.nan  # Replace NaN values with NaN

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


import numpy as np

def hsi_to_rgb(array):
    """
    Convert an array of HSI (Hue, Saturation, Intensity) values to an array of RGB values.
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

def transform_lab_stretch(array):
    """
    Transform an Image into LAB space and stretch the contrast of the L channel.
    """
    
    lab_array = plot_lab(array)
    lab_array[:, :, 0] = linear_contrast_enhancement(lab_array[:, :, 0])
    rgb_array = lab2rgb(lab_array)
    return rgb_array

def transform_multiband_lab(arr, bands = None):
    """
    Reindex the bands of a multiband image to match the order of the LAB color space.
    Input is a multiband array
    If bands is None, the default is [3,2,1] - where 3 is the Red band, 2 is the Green band, and 1 is the Blue band.
    """
    if bands is None:
        bands = [3,2,1]

    arr_copy = arr.copy()
    #convert arr_copy to float64
    arr_copy = arr_copy.astype(np.float64)
    rgb_arr = transform_lab_stretch(arr_copy)
    
    arr_copy[:,:,bands[0]] = rgb_arr[:,:,0]
    arr_copy[:,:,bands[1]] = rgb_arr[:,:,1]
    arr_copy[:,:,bands[2]] = rgb_arr[:,:,2]

    return arr_copy



def transform_hsv_stretch(array, max_value = 100):
    """
    Transform an Image into HSV space and stretch the contrast of the S channel.
    """
    
    hsv_array = plot_hsv(array)
    hsv_array[:, :, 1] = linear_contrast_enhancement(hsv_array[:, :, 1], max_value = max_value)
    rgb_array = hsv2rgb(hsv_array)
    return rgb_array

def transform_multiband_hsv(arr, bands = None, max_value = 100, mask = None):
    """
    Reindex the bands of a multiband image to match the order of the HSV color space.
    Input is a multiband array
    If bands is None, the default is [3,2,1] - where 3 is the Red band, 2 is the Green band, and 1 is the Blue band.
    """
    if bands is None:
        bands = [3,2,1]

    arr_copy = arr.copy()
    #convert arr_copy to float64
    arr_copy = arr_copy.astype(np.float64)
    rgb_arr = transform_hsv_stretch(arr_copy, max_value = max_value)
    
    arr_copy[:,:,bands[0]] = rgb_arr[:,:,0]
    arr_copy[:,:,bands[1]] = rgb_arr[:,:,1]
    arr_copy[:,:,bands[2]] = rgb_arr[:,:,2]
    return arr_copy