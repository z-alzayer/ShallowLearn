import numpy as np
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import joblib
from skimage.color import rgb2lab, rgb2hsv, rgb2ycbcr
from ShallowLearn.band_mapping import band_mapping

from ShallowLearn import LoadData
from matplotlib.colors import ListedColormap, to_rgba
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

def load_img(path, return_meta=False):
    """
    Loads an image from the specified path.

    Args:
        path (str): The path to the image file.

    Returns:
        numpy.ndarray: The loaded image array.

    """
    img = LoadData.LoadGeoTIFF(path).load()
    
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)

    if return_meta:
        return img, LoadData.LoadGeoTIFF(path).get_metadata(), LoadData.LoadGeoTIFF(path).get_bounds()
    return img

def select_channels(arr, indices):
    """
    Selects specific channels based on the given indices from a 3D numpy array.

    :param arr: A numpy array of shape (x, y, z).
    :param indices: A list or array-like containing the indices of channels to be selected.
                    Its length must be 3 or it will raise a ValueError.
    :return: A numpy array of shape (x, y, 3).
    """

    if len(indices) != 3:
        raise ValueError("The length of indices must be 3.")

    return arr[:, :, indices]


def remove_channel(img, channel):
    """
    Removes a specified channel from a 3D image array.

    Args:
        img (numpy.ndarray): The 3D image array with shape (height, width, channels).
        channel (int): The index of the channel to remove.

    Returns:
        numpy.ndarray: The image array with the specified channel removed.

    Raises:
        ValueError: If the channel index is out of bounds.
    """
    if channel < 0 or channel >= img.shape[2]:
        raise ValueError("Channel index is out of bounds.")

    return np.concatenate((img[:, :, :channel], img[:, :, channel+1:]), axis=2)

def plot_rgb(img, bands=None, plot=False):
    """
    Plots the RGB image using the specified bands.

    Args:
        img (numpy.ndarray): The input image array.
        bands (list): The list of band codes to use for the RGB channels. Default is ['B04', 'B03', 'B02'].
        plot (bool): Whether to display the image plot using matplotlib. Default is False.

    Returns:
        numpy array with selected bands: The RGB image array if plot is set to False.

    """
    if bands is None:
        bands = ['B04', 'B03', 'B02']  # Default band codes

    band_numbers = [band_mapping[band]['index'] for band in bands]

    img_shape = img.shape
    r = np.uint8(minmax_scale(img[:, :, band_numbers[0]].flatten(), feature_range=(0, 255), axis=0, copy=True)).reshape(
        img_shape[0], img_shape[1])
    g = np.uint8(minmax_scale(img[:, :, band_numbers[1]].flatten(), feature_range=(0, 255), axis=0, copy=True)).reshape(
        img_shape[0], img_shape[1])
    b = np.uint8(minmax_scale(img[:, :, band_numbers[2]].flatten(), feature_range=(0, 255), axis=0, copy=True)).reshape(
        img_shape[0], img_shape[1])
    rgb = np.dstack((r, g, b))

    if plot:
        plt.imshow(rgb)
        plt.show()
        return None

    return rgb

def plot_hsv(img, plot=False):
    """
    Converts the input image to the HSV color space and optionally plots the H channel.

    Args:
        img (numpy.ndarray): The input image array.
        plot (bool): Flag to indicate whether to plot the H channel or not. Default is False.

    Returns:
        numpy.ndarray or None: The hsv array if plot=False, otherwise None.

    """
    hsv_img = rgb2hsv(plot_rgb(img))
    h_channel = hsv_img[:, :, 0]
    
    if plot == False:
        return hsv_img
    # Plot the H channel
    plt.imshow(h_channel, cmap='hsv')
    plt.axis('off')
    plt.title('H Channel')
    plt.show()
    return None

def plot_ycbcr(img, plot=False):
    """
    Converts the input image to the YCbCr color space and optionally plots the Y channel.

    Args:
        img (numpy.ndarray): The input image array.
        plot (bool): Flag to indicate whether to plot the Y channel or not. Default is False.

    Returns:
        numpy.ndarray or None: The ycbcr array if plot=False, otherwise None.

    """
    ycbcr_img = rgb2ycbcr(plot_rgb(img))
    y_channel = ycbcr_img[:, :, 0]
    
    if plot == False:
        return ycbcr_img
    # Plot the Y channel
    plt.imshow(y_channel, cmap='gray')
    plt.axis('off')
    plt.title('Y Channel')
    plt.show()
    return None




def plot_lab(img, plot=False):
    """
    Converts the input image to the LAB color space and optionally plots the L channel.

    Args:
        img (numpy.ndarray): The input image array.
        plot (bool): Flag to indicate whether to plot the L channel or not. Default is False.

    Returns:
        numpy.ndarray or None: The lab array if plot=False, otherwise None.

    """
    lab_img = rgb2lab(plot_rgb(img, plot=False))
    l_channel = lab_img[:, :, 0]
    if plot == False:
        return lab_img
    # Plot the L channel
    plt.imshow(l_channel, cmap='gray')
    plt.axis('off')
    plt.title('L Channel')
    plt.show()
    return None

def predict_mask(img, model = None,  mask_val=None, estimator = None):
    """
    Predicts the mask for the input image using a pre-trained model.

    Args:
        img (numpy.ndarray): The input image array.
        model (str): The path to the pre-trained model. Default is None and uses all 13 bands.
        mask_val (int): The mask value to consider. Default is 9.

    Returns:
        numpy.ndarray: The predicted mask for the image.

    """
    if model is None:
        loaded_pipeline = joblib.load('../Models/pipeline_pca2_k10.pkl')
    else:
        loaded_pipeline = joblib.load(model)
    
    if estimator is not None:
        loaded_pipeline = estimator

    img_shape = img.shape
    temp = img.reshape(img_shape[0] * img_shape[1], img_shape[2])
    if mask_val is None:
        pred = loaded_pipeline.predict(temp) 
    else:
        pred = loaded_pipeline.predict(temp) == mask_val
    return pred.reshape(img_shape[0], img_shape[1])

def gen_mask(img, mask=9):
    """
    Generates a mask for the input image based on the predicted mask.

    Args:
        img (numpy.ndarray): The input image array.
        mask (int): The mask value to consider. Default is 9.

    Returns:
        numpy.ndarray: The mask array.

    """
    mask = predict_mask(img, mask)
    return mask

def apply_mask(data, mask, fill_value=0):
    """
    Applies a mask to the data array.

    Args:
        data (numpy.ndarray): The input data array.
        mask (numpy.ndarray): The mask array.
        fill_value (float): The value to use where the mask is False.

    Returns:
        numpy.ndarray: The masked data array.
    """
    masked_data = np.where(mask, data, fill_value)
    return masked_data


def generate_multichannel_mask(img, mask=None, mask_val=9):
    """
    Generates a multichannel mask for the input image based on the predicted mask.
    Can also be used to generate a multichannel mask for a given mask.

    Args:
        img (numpy.ndarray): The input image array.
        mask (int): The mask value to consider. Default is 9.

    Returns:
        numpy.ndarray: The multichannel mask array.

    """
    if mask is None:
        mask = gen_mask(img, mask_val)
    
    reshaped_mask = np.repeat(mask[:, :, np.newaxis], img.shape[2], axis=2)
    final_mask = img * reshaped_mask
    rescaled_image = final_mask.copy()

    for i in range(final_mask.shape[2]):
        channel_min = final_mask[:, :, i].min()
        channel_max = final_mask[:, :, i].max()
        rescaled_image[:, :, i] = (final_mask[:, :, i] - channel_min) / (channel_max - channel_min) * 255
    return rescaled_image


def plot_with_legend(array, value_dict):
    """
    Plots a 2D array with a legend using distinct colors for discrete class labels.

    Parameters:
    array (2D numpy array): The array to be plotted.
    value_dict (dict): A dictionary mapping values in the array to labels.
    """
    
    # Create a colormap with a distinct color for each class
    n_classes = len(value_dict)
    cmap = plt.cm.get_cmap('Set3', n_classes)  # 'tab10' or 'Set3'

    # Create the plot
    plt.imshow(array, cmap=cmap)

    # Create a color map index for each discrete value
    colors = [cmap(i) for i in range(n_classes)]

    # Create a legend
    patches = [mpatches.Patch(color=colors[i], label=label) for i, (value, label) in enumerate(value_dict.items())]
    
    # Add the legend to the plot
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    # Show the plot
    plt.show()


def plot_histogram(img,  plot=True, bins=50, min_value=1, title=None):
    """
    Plots histogram for a specific channel in the input image.

    Args:
        img (numpy.ndarray): The input image array.
        channel (int): The index of the channel to be plotted. Default is 0.
        plot (bool): Flag to indicate whether to plot the histogram or not. Default is True.
        bins (int): The number of bins for the histogram. Default is 50.
        min_value (int): The minimum value to consider for the histogram. Values below this threshold will be removed. Default is 1.
        title (str): Title of the plot. Default is None.

    Returns:
        None
    """

    
    if plot:
        channel_data = img.flatten()
        channel_data = channel_data[channel_data >= min_value]
        
        # Plotting the histogram
        plt.hist(channel_data, bins=bins, range=(0, np.max(channel_data)))
        
        # Customize the plot
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title(title if title else f'Histogram')
        plt.show()



def plot_histograms(img, plot=True, bins=50, min_value=1, channel_legend=None, title = None):
    """
    Plots histograms for each channel in the input image using line plots.

    Args:
        img (numpy.ndarray): The input image array.
        plot (bool): Flag to indicate whether to plot the histograms or not. Default is True.
        bins (int): The number of bins for the histograms. Default is 50.
        min_value (int): The minimum value to consider for the histograms. Values below this threshold will be removed. Default is 1.
        channel_legend (dict): Dictionary for channel legend. Keys are channel indices and values are channel names. Default is None.

    Returns:
        None

    """
    if len(img.shape) == 2:
        plot_histogram(img, plot=plot, bins=bins, min_value=min_value, title=title)
        return

    num_channels = img.shape[2]

    if plot:
        x = np.linspace(0, np.max(img), bins)
        for i in range(num_channels):
            channel_data = img[:, :, i].flatten()
            channel_data = channel_data[channel_data >= min_value]
            histogram, _ = np.histogram(channel_data, bins=bins, range=(0, np.max(img)))

            # Use the dictionary for legend if provided, else use default labeling
            label = channel_legend[i] if channel_legend and i in channel_legend else f'Channel {i + 1}'

            # Plot the histogram using line plot
            plt.plot(x, histogram, label=label)

        # Customize the plot
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        if title is None:
            plt.title('Histogram of Each Channel')
        else:
            plt.title(title)
        plt.legend()
        plt.show()

def discrete_implot(arr, change_labels=None, change_colors=None, pixel_scale=10, title = None, return_fig = False):
    if len(arr.shape) == 1:
        arr = np.reshape(arr, (-1, 1))  # Reshape 1D array to 2D

    unique_labels = np.unique(arr)
    num_labels = len(unique_labels)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_arr = np.vectorize(label_to_int.get)(arr)  # Convert labels to integers

    # Create color list using the 'viridis' colormap
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, num_labels))
    
    # If change_labels and change_colors are specified, modify the corresponding colors
    if change_labels is not None and change_colors is not None:
        for label, color in zip(change_labels, change_colors):
            if label in label_to_int:
                colors[label_to_int[label]] = to_rgba(color)
            else:
                print(f"Label {label} not found in array.")

    cmap = ListedColormap(colors)

    fig, ax = plt.subplots()
    im = ax.imshow(int_arr, cmap=cmap)

    # Create a colorbar with discrete levels
    cbar = fig.colorbar(im, ticks=np.arange(num_labels), drawedges=True)
    cbar.set_label('Labels')
    cbar.set_ticklabels(unique_labels)  # Set the tick labels to the unique_labels

    # Add scale bar of 1 km
    scalebar = AnchoredSizeBar(ax.transData,
                               10 * pixel_scale, '1 km', 'lower right', 
                               pad=0.25,
                               color='white',
                               frameon=False,
                               size_vertical=1)
    ax.add_artist(scalebar)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title('Discrete Plot')
    if return_fig:
        return fig
    plt.show()
label_dict = {}  # Global dictionary to ensure label consistency between plots

def discrete_implotv2(arr, ax=None, string_labels=None, change_colors=None, pixel_scale=10, title=None, return_fig=False):
    global label_dict

    if len(arr.shape) == 1:
        arr = np.reshape(arr, (-1, 1))  # Reshape 1D array to 2D

    if not label_dict:
        unique_labels = np.unique(arr)
        label_to_int = {label: i for i, label in enumerate(unique_labels)}
        label_dict = label_to_int  # Store the label dictionary for future reference
    else:
        label_to_int = label_dict  # Use existing label dictionary

    int_arr = np.vectorize(label_to_int.get)(arr)  # Convert labels to integers
    
    num_labels = len(label_to_int)

    # Create color list using the 'viridis' colormap
    colors = plt.get_cmap('viridis')(np.linspace(0, 1, num_labels))
    
    # If change_colors are specified, modify the corresponding colors
    if change_colors is not None:
        for label, color in zip(label_to_int.keys(), change_colors):
            if label in label_to_int:
                colors[label_to_int[label]] = color
            else:
                print(f"Label {label} not found in array.")

    cmap = ListedColormap(colors)

    if ax is None:
        fig, ax = plt.subplots()
        show_plot = True
    else:
        fig = ax.figure
        show_plot = False

    im = ax.imshow(int_arr, cmap=cmap)

    # Create a colorbar with discrete levels
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(num_labels), drawedges=True)
    cbar.set_label('Labels')
    
    # Set the tick labels using string_labels if provided, else use label_to_int keys
    if string_labels:
        cbar.set_ticklabels([string_labels.get(label, label) for label in label_to_int.keys()])
    else:
        cbar.set_ticklabels(list(label_to_int.keys()))
    
    # Other plot settings here...

    if return_fig:
        return fig
    if show_plot:
        plt.show()
        
def add_north_arrow(ax, relative_position=(0.05, 0.05), arrow_length=0.05, text_offset=-0.02):
    """Add a north arrow to the axis."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    
    x = xlim[0] + (xlim[1] - xlim[0]) * relative_position[0]
    y = ylim[0] + (ylim[1] - ylim[0]) * relative_position[1]
    
    ax.arrow(x, y, 0, arrow_length, head_width=0.02 * (xlim[1] - xlim[0]), head_length=0.03 * (ylim[1] - ylim[0]), fc='black', ec='black')
    ax.text(x, y + text_offset * (ylim[1] - ylim[0]), 'N', horizontalalignment='center', verticalalignment='center', fontsize=12, fontweight='bold', c = 'black')


def plot_geotiff(image_data, bounds, title="Map with coordinates", ax = None):
    """
    Plot GeoTIFF with UTM Coordinates and a scale bar.

    Parameters:
    - image_data: 2D or 3D array containing the raster data
    - bounds: Tuple containing the bounding coordinates (left, right, bottom, top)
    """
    from matplotlib_scalebar.scalebar import ScaleBar
    from matplotlib.ticker import ScalarFormatter

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    left, bottom, right, top = bounds

    # If the image data is RGB and contains NaN values, convert it to RGBA
    if image_data.ndim == 3 and image_data.shape[2] == 3 and np.isnan(image_data).any():
        # Create an alpha channel with the same shape as one of the RGB channels
        alpha = np.ones_like(image_data[..., 0])

        # Where any of the RGB channels is NaN, set the alpha to 0
        alpha[np.isnan(image_data).any(axis=-1)] = 0

        # Create an RGBA image by stacking the RGB channels and the alpha channel along the last dimension
        image_data = np.dstack((image_data, alpha))

    ax.imshow(image_data, extent=[left, right, bottom, top])  # Adjust colormap as needed for non-RGBA data

    # Use ScalarFormatter to force no scientific notation
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    # Add a scale bar
    scalebar = ScaleBar(2, location='lower right', scale_loc='bottom', box_alpha=0)  # 1 pixel = 1 meter (assuming your image_data is in meters)
    ax.add_artist(scalebar)

    add_north_arrow(ax)

    ax.set_title(title)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    
def median_without_zeros_or_nans(images):
    """
    Computes the median of each band in each image, excluding zeros and NaN values.

    Parameters:
    - images: numpy array, shape (i, x, y, z)
      The input 4D array of images.

    Returns:
    - medians: numpy array, shape (i, z)
      The median values for each band in each image, excluding zeros and NaNs.
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