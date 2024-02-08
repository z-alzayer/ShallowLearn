import numpy as np
import cv2
from skimage.feature import local_binary_pattern
from skimage.filters import gabor
from skimage.feature import hog
from skimage.color import rgb2gray

def edge_density(image):
    """
    Computes the edge density using the Canny Edge Detector.
    
    Purpose:
    Edge density can be used to quantify the amount of texture or detail in an image, which can be useful in various image processing tasks.
    
    Parameters:
    - image: numpy array, shape (height, width, num_channels)
      The input image array.
    
    Returns:
    - edge_density_map: numpy array, shape (height, width)
      The computed edge density.
    """
    # Convert image to grayscale if it has multiple channels
    if image.shape[2] > 1:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image
    
    # Apply Canny Edge Detector
    edges = cv2.Canny(gray_image, 50, 200)
    
    # Compute edge density
    edge_density_map = edges / 255.0
    return edge_density_map



def texture_features(image, P=8, R=1):
    """
    Computes texture features using Local Binary Pattern (LBP).
    
    Purpose:
    LBP is a powerful texture descriptor and can be used for texture classification.
    
    Parameters:
    - image: numpy array, shape (height, width, num_channels)
      The input image array.
    - P: int, optional
      Number of circularly symmetric neighbor set points (default is 8).
    - R: int, optional
      Radius of circle (default is 1).
    
    Returns:
    - lbp_texture: numpy array, shape (height, width)
      The computed texture features using LBP.
    """
    # Convert image to grayscale if it has multiple channels
    if image.shape[2] > 1:
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray_image = image

    # Apply Local Binary Pattern
    lbp_texture = local_binary_pattern(gray_image, P, R, method="uniform")
    return lbp_texture

def color_histogram(image, bins=32):
    """
    Computes the color histogram for each channel in the image.
    
    Purpose:
    Color histograms represent the distribution of colors in an image and can be used for color-based image analysis.
    
    Parameters:
    - image: numpy array, shape (height, width, num_channels)
      The input image array.
    - bins: int, optional
      Number of bins for the histogram (default is 32).
    
    Returns:
    - hist: numpy array, shape (num_channels, bins)
      The computed color histograms for each channel.
    """
    num_channels = image.shape[2]
    hist = np.zeros((num_channels, bins))

    for channel in range(num_channels):
        hist[channel, :], _ = np.histogram(image[:, :, channel], bins=bins, range=[0, 256])

    return hist

def sobel_edge_detection(image):
    """
    Applies Sobel edge detection to an image.

    Parameters:
    - image: numpy array, shape (height, width, num_channels)
      The input image array.

    Returns:
    - edge_image: numpy array, shape (height, width)
      The edge-detected image.
    """
    # Convert to grayscale if necessary
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] > 1 else image

    # Apply Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    edge_image = np.sqrt(sobelx**2 + sobely**2)

    return edge_image



def gabor_features(image, frequency=0.6):
    """
    Applies Gabor filter to an image.

    Parameters:
    - image: numpy array, shape (height, width, num_channels)
      The input image array.
    - frequency: float
      The frequency of the sinusoidal function.

    Returns:
    - gabor_response: numpy array, shape (height, width)
      The filtered image.
    """
    if image.shape[2] > 1:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Applying Gabor filter
    gabor_response, _ = gabor(image, frequency=frequency)
    return gabor_response



def histogram_of_oriented_gradients(image, pixels_per_cell=(16, 16), cells_per_block=(4, 4), orientations=9):
    """
    Computes the Histogram of Oriented Gradients (HOG) feature descriptor.

    Parameters:
    - image: numpy array, shape (height, width, num_channels)
      The input image array.
    - pixels_per_cell: tuple of int
      Size (in pixels) of a cell.
    - cells_per_block: tuple of int
      Number of cells in each block.
    - orientations: int
      Number of orientation bins.

    Returns:
    - hog_features: numpy array
      The HOG feature descriptor for the image.
    """
    # if image.shape[2] > 1:
    #     image = rgb2gray(image)

    fd, hog_features = hog(image, orientations=orientations,channel_axis=-1,  
                           pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, block_norm='L1-sqrt', visualize = True)
    
    return hog_features

def sobel_edge_detection(image):
    """
    Applies Sobel edge detection to an image.

    Parameters:
    - image: numpy array, shape (height, width, num_channels)
      The input image array.

    Returns:
    - edge_image: numpy array, shape (height, width)
      The edge-detected image.
    """
    # Convert to grayscale if necessary
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.shape[2] > 1 else image

    # Apply Sobel edge detection
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    edge_image = np.sqrt(sobelx**2 + sobely**2)

    return edge_image

