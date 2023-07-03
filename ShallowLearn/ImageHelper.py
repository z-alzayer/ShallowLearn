import numpy as np
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import joblib
from skimage.color import rgb2lab, rgb2hsv, rgb2ycbcr
from ShallowLearn.band_mapping import band_mapping

from ShallowLearn import LoadData
from matplotlib.colors import ListedColormap, to_rgba
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

def load_img(path):
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
    return img

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


def plot_histograms(img, plot=True, bins=50, min_value=1):
    """
    Plots histograms for each channel in the input image using line plots.

    Args:
        img (numpy.ndarray): The input image array.
        plot (bool): Flag to indicate whether to plot the histograms or not. Default is True.
        bins (int): The number of bins for the histograms. Default is 50.
        min_value (int): The minimum value to consider for the histograms. Values below this threshold will be removed. Default is 1.

    Returns:
        None

    """
    num_channels = img.shape[2]

    if plot:
        x = np.linspace(0, np.max(img), bins)
        for i in range(num_channels):
            channel_data = img[:, :, i].flatten()
            channel_data = channel_data[channel_data >= min_value]
            histogram, _ = np.histogram(channel_data, bins=bins, range=(0, np.max(img)))

            # Plot the histogram using line plot
            plt.plot(x, histogram, label=f'Channel {i + 1}')

        # Customize the plot
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.title('Histogram of Each Channel')
        plt.legend()
        plt.show()


    def standard__minmax_scaler_(img):
        # Test if this actually helps at a later time 

        # from sklearn.preprocessing import scale, minmax_scale
        # test_img = LoadData.LoadGeoTIFF(dir_list[1]).load()
        # test_img = np.swapaxes(test_img, 0, 2)
        # test_img_original = test_img.copy()
        # r = np.float16(scale(test_img[:,:,4].flatten(), with_mean=True, with_std=True, copy=True )).reshape(534, 725)
        # g = np.float16(scale(test_img[:,:,3].flatten(), with_mean=True, with_std=True, copy=True )).reshape(534, 725)
        # b = np.float16(scale(test_img[:,:,2].flatten(), with_mean=True, with_std=True, copy=True )).reshape(534, 725)
        # # Ensure the values are positive
        # r = np.uint8(minmax_scale(r.flatten(),feature_range=(0,255), axis=0, copy=True )).reshape(534, 725)
        # g = np.uint8(minmax_scale(g.flatten(),feature_range=(0,255), axis=0, copy=True )).reshape(534, 725)
        # b = np.uint8(minmax_scale(b.flatten(),feature_range=(0,255),  axis=0, copy=True )).reshape(534, 725)
        # rescaled_img = np.swapaxes(np.array([r,g,b]), 0, 2)
        pass


def discrete_implot(arr, change_labels=None, change_colors=None, pixel_scale=10, title = None):
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
    plt.show()