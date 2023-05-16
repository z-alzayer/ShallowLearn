import numpy as np
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt
import joblib
from skimage.color import rgb2lab

from ShallowLearn import LoadData

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

def plot_rgb(img, bands=[4, 3, 2], plot = False):
    """
    Plots the RGB image using the specified bands.

    Args:
        img (numpy.ndarray): The input image array.
        bands (list): The list of bands to use for the RGB channels. Default is [4, 3, 2].

    Returns:
        numpy array with selected bands: The RGB image array if plot set to False.

    """
    img_shape = img.shape
    r = np.uint8(minmax_scale(img[:, :, bands[0]].flatten(), feature_range=(0, 255), axis=0, copy=True)).reshape(
        img_shape[0], img_shape[1])
    g = np.uint8(minmax_scale(img[:, :, bands[1]].flatten(), feature_range=(0, 255), axis=0, copy=True)).reshape(
        img_shape[0], img_shape[1])
    b = np.uint8(minmax_scale(img[:, :, bands[2]].flatten(), feature_range=(0, 255), axis=0, copy=True)).reshape(
        img_shape[0], img_shape[1])
    rgb = np.dstack((r, g, b))
    if plot == True:
        plt.imshow(rgb)
        return None
    return rgb



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

def predict_mask(img, mask_val=9):
    """
    Predicts the mask for the input image using a pre-trained model.

    Args:
        img (numpy.ndarray): The input image array.
        mask_val (int): The mask value to consider. Default is 9.

    Returns:
        numpy.ndarray: The predicted mask for the image.

    """
    loaded_pipeline = joblib.load('../Models/pipeline_pca2_k10.pkl')
    img_shape = img.shape
    temp = img.reshape(img_shape[0] * img_shape[1], img_shape[2])
    pred = loaded_pipeline.predict(temp) == mask_val
    return pred.reshape(img_shape[0], img_shape[1])

def generate_multichannel_mask(img, mask=9):
    """
    Generates a multichannel mask for the input image based on the predicted mask.

    Args:
        img (numpy.ndarray): The input image array.
        mask (int): The mask value to consider. Default is 9.

    Returns:
        numpy.ndarray: The multichannel mask array.

    """
    mask = predict_mask(img, mask)
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
    